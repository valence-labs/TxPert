from typing import Any, Union, List, Dict, Tuple
from omegaconf import DictConfig, OmegaConf

import torch
import random
import pandas as pd
import numpy as np

import gspp.constants as cs


class GSPGraph:
    """
    Class to create graphs from raw data and downsample them. Support for multiple graph types.

    Args:
        pert2id (dict): Dictionary mapping perturbations to IDs.
        gene2id (dict): Master dictionary mapping genes to IDs regardless if they are in the perturbation set.
        graph_cfg (Union[str, List, Dict]): The type of graph including optional arguments.
            - str: Provide graph type (e.g. "go", "string").
            - List: Provide a list of graph types, e.g., ["go", "string"].
            - Dict: Provide a dictionary with graph identifiers as keys and optional arguments as values, e.g., {"graph1": {"graph_type": "string",...}, "graph2": {"graph_type": "go",...}}.
        cache_dir (str): The directory to read raw graph data.
    """

    def __init__(
        self,
        pert2id: Dict[str, int],
        gene2id: Dict[str, int],
        graph_cfg: Union[str, List, Dict] = "go",
        cache_dir: str = cs.DATA_DIR / "graphs",
    ):
        self.pert2id = pert2id
        self.gene2id = gene2id
        self.cache_dir = cache_dir

        if isinstance(graph_cfg, str):
            graph_cfg = {graph_cfg: {"graph_type": graph_cfg}}
        elif isinstance(graph_cfg, List):
            graph_cfg = {
                graph_type: {"graph_type": graph_type} for graph_type in graph_cfg
            }

        self.graph_dict = {}
        for graph_name, graph_args in graph_cfg.items():
            if isinstance(graph_args, DictConfig):
                graph_args = OmegaConf.to_container(graph_args, resolve=True)

            graph_type = graph_args.pop("graph_type", "string")
            p_downsample = graph_args.pop("p_downsample", 1.0)
            p_rewire_src = graph_args.pop("p_rewire_src", 0.0)
            p_rewire_tgt = graph_args.pop("p_rewire_tgt", 0.0)
            random_seed = graph_args.pop("random_seed", 42)
            seed_downsample = graph_args.pop("seed_downsample", None)

            graph = self.create_graph(graph_type, graph_args)

            # Donwsample graph if parameter is set
            if p_downsample < 1.0:
                graph = self.downsample_graph(graph, p_downsample, seed_downsample)

            # Rewire graph if parameter is set
            if p_rewire_src > 0.0 or p_rewire_tgt > 0.0:
                print('Rewiring graph with p_rewires:', (p_rewire_src, p_rewire_tgt))
                graph = self.random_rewire(graph, p_rewire_src=p_rewire_src, p_rewire_tgt=p_rewire_tgt, random_seed=random_seed)

            self.graph_dict[graph_name] = graph


    def create_graph(self, graph_type: str, graph_args: Dict[str, Any] = None):
        """
        Load the graph based on the graph type.
        """
        if graph_type == "go":
            network = pd.read_csv(f"{self.cache_dir}/go/go_top_50.csv")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "string":
            network = pd.read_parquet(f"{self.cache_dir}/string/v11.5.parquet")
            graph = self.process_graph(network, **graph_args)

        elif graph_type == "dense":
            num_nodes = len(self.pert2id)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            graph = edge_index, edge_weight, len(self.pert2id)

        else:
            raise ValueError("Invalid graph mode")

        return graph

    def process_graph(
        self,
        network: pd.DataFrame,
        reduce2perts: bool = True,
        reduce2positive: bool = False,
        norm_weights: bool = False,
        mode: str = None,
    ):
        """
        Allows processing the network and converting it to PyG format.

        Args:
            network (pd.DataFrame): The network to process.
            reduce2perts (bool): Whether to reduce the network to only perturbation genes.
            reduce2positive (bool): Whether to reduce the network to only positive weights.
            norm_weights (bool): Whether to normalize the weights to [0,1].
            mode (str): The mode of edge selection. Currently supports "top_n" edges per target, "percentile_q" edges per target, edges with weight about "threshold".
        """
        # Check which column naming convention is used
        if "regulator" in network.columns and "target" in network.columns:
            source_col = "regulator"
            target_col = "target"
        elif "source" in network.columns and "target" in network.columns:
            source_col = "source"
            target_col = "target"
        else:
            raise ValueError(
                "Network must have either regulator/target or gene1/gene2 columns"
            )

        # Check and rename importance column if it exists
        if "importance" in network.columns:
            network = network.rename(columns={"importance": "weight"})

        # Rename columns to standard format
        if source_col != "regulator":
            network = network.rename(
                columns={source_col: "regulator", target_col: "target"}
            )

        # Reduce to edges between perturbation genes
        if reduce2perts:
            network["regulator"] = network["regulator"].map(self.pert2id)
            network["target"] = network["target"].map(self.pert2id)
            network = network.dropna()
            num_nodes = len(self.pert2id)
        else:
            network["regulator"] = network["regulator"].map(self.gene2id)
            network["target"] = network["target"].map(self.gene2id)
            network = network.dropna()
            num_nodes = len(self.gene2id)

        # Reduce to positive weights only
        if reduce2positive:
            network = network[network["weight"] > 0]
            network = network.reset_index(drop=True)

        # Normalize the weights to [0,1] - added abs() to fix negative weights from Ph/Tx in this case
        if norm_weights:
            network["weight"] = abs(network["weight"].transform(lambda x: x / x.max()))

        # Determine the mode of edge selection
        if mode is not None:
            mode, arg = mode.split("_")
            arg = int(arg)

            if mode == "top":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x.nlargest(arg, ["weight"]))
                    .reset_index(drop=True)
                )

            # Per target gene, only keep edges with weights above a certain threshold; be carful when also using `norm_weights`
            elif mode == "threshold":
                network = (
                    network.groupby("target")
                    .apply(lambda x: x[x["weight"] >= arg])
                    .reset_index(drop=True)
                )

            # Per target gene, only use edges with a cosine similarity in the specified percentile of absolute values
            elif "percentile" in mode:
                # Calculate threshold value based on absolute similarities if desired
                if "abs" in mode:
                    threshold = np.percentile(network["weight"].abs(), arg)
                    network = network[network["weight"].abs() >= threshold]
                else:
                    threshold = np.percentile(network["weight"], arg)
                    network = network[network["weight"] >= threshold]

                network = network.reset_index(drop=True)

        # Package the graph into PyG format
        edge_index = torch.tensor(
            [network["regulator"].to_numpy(), network["target"].to_numpy()],
            dtype=torch.long,
        )
        edge_weight = torch.tensor(network["weight"].to_numpy(), dtype=torch.float)

        return edge_index, edge_weight, num_nodes


    def downsample_graph(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_downsample: float = 1.0,
        seed_downsample: int = None,
    ):
        """
        Downsample the graph to a random fraction of the original edges.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_indes, edge_weight & num_nodes) to downsample.
            p_downsample (float): The fraction of edges to keep.
            seed_downsample (int): The seed for the downsample operation.
        """
        edge_index, edge_weight, num_nodes = graph

        n = len(edge_weight)
        n_downsample = int(p_downsample * n)

        perm = list(range(n))
        if seed_downsample is not None:
            random.seed(seed_downsample)
        random.shuffle(perm)

        edge_index = edge_index[:, perm]
        edge_weight = edge_weight[perm]

        edge_index = edge_index[:, :n_downsample]
        edge_weight = edge_weight[:n_downsample]

        return edge_index, edge_weight, num_nodes
    

    def random_rewire(
        self,
        graph: Tuple[torch.Tensor, torch.Tensor],
        p_rewire_src: float = 0.0,
        p_rewire_tgt: float = 0.0,
        random_seed: int = None,
    ):
        """
        Rewire each edge with a given probability.

        Args:
            graph (Tuple[torch.Tensor, torch.Tensor, num_nodes]): The graph (edge_index, edge_weight, and num_nodes) to rewire.
            p_rewire_src (float): The probability of randomly rewiring the source node of each edge.
            p_rewire_tgt (float): The probability of randomly rewiring the target node of each edge.
            random_seed (int): The seed for the rewiring operation to ensure reproducibility.
        """
        edge_index, edge_weight, num_nodes = graph

        num_edges = edge_index.size(1)
        if random_seed is not None:
            random_gen = random.Random(random_seed)
        else:
            random_gen = random.Random()
        for i in range(num_edges):
            if random_gen.random() < p_rewire_src:
                new_source = random_gen.randint(0, num_nodes - 1)
                edge_index[0, i] = new_source
            if random_gen.random() < p_rewire_tgt:
                new_target = random_gen.randint(0, num_nodes - 1)
                edge_index[1, i] = new_target

        return edge_index, edge_weight, num_nodes