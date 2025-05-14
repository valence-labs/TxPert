from typing import Union, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gspp.data.graphmodule import GSPGraph

from gspp.constants import GNN_DICT


class GNN(nn.Module):
    """
    Basic GNN model for perturbation prediction

    Args:
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        in_dim (int): Dimension of the input data.
        layer_type (str): Type of GNN layer to use (see GNN_DICT for few options).
        num_layers (int): Number of GNN layers.
        skip (str): Type of skip connection to use.
        hidden_dim (int): Dimension of the hidden layers.
        out_dim (int): Dimension of the output data.
        dropout (float): Dropout rate.
        num_heads (int): Number of heads for multi-head attention.
        concat (bool): Whether to concatenate the multi-head attention outputs (sum aggregation if `False`).
        add_self_loops (bool): Whether to add self-loops to the graph.
        use_edge_weight (bool): Whether to use edge weights.
        use_struc_feat (bool): Whether to use structural features. TODO (Fred): Not implemented yet
        device (str): Device to run the model on.
    """

    def __init__(
        self,
        graph: Union[GSPGraph, Tuple],
        in_dim: int = None,
        layer_type: str = "gat_v2",
        num_layers: int = 4,
        skip: Literal[
            "none", "skip_cat", "skip_cat_learned", "skip_sum_learned"
        ] = "none",
        hidden_dim: int = 64,
        out_dim: int = 64,
        dropout: float = 0.0,
        num_heads: int = 2,
        concat: bool = False,
        add_self_loops: bool = False,
        use_edge_weight: bool = False,
        use_struc_feat: bool = False,
        device: str = "cpu",
    ):
        super(GNN, self).__init__()

        out_dim = hidden_dim if out_dim is None else out_dim

        if "gat" not in layer_type:
            concat = False

        if isinstance(graph, Tuple):
            self.num_perts = graph[-1]
        else:
            self.num_perts = next(iter(graph.graph_dict.values()))[-1]

        self.num_layers = num_layers
        self.skip = skip
        self.concat = concat
        self.heads = num_heads
        self.use_edge_weight = use_edge_weight
        self.dropout = dropout
        self.device = device

        self.use_cntr = False
        self.cat_cntr = False

        # Learned perturbation embeddings for each perturbation gene (these are the input node features)
        self.pert_embeddings = nn.Embedding(
            num_embeddings=self.num_perts,
            embedding_dim=hidden_dim,
            device=device,
        )

        gnn_layer = GNN_DICT[layer_type]

        self.gnn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        in_dim = hidden_dim if in_dim is None else in_dim

        res_dim = 0
        for idx in range(num_layers):
            self.gnn_layers.append(
                gnn_layer(
                    in_dim,
                    hidden_dim,
                    edge_dim=1 if use_edge_weight else None,
                    heads=num_heads,
                    concat=concat,
                    add_self_loops=add_self_loops,
                )
            )

            if "skip" in skip:
                res_dim = in_dim
                if "learned" in skip:
                    self.skip_layers.append(nn.Linear(in_dim, hidden_dim))
                    res_dim = hidden_dim
                if "cat" not in skip:
                    res_dim = 0

            in_dim = (
                res_dim + num_heads * hidden_dim if concat else res_dim + hidden_dim
            )

        self.out_nn = nn.Linear(in_dim, out_dim)

    def forward(self, edge_index, edge_weight, z_intrinsic=None):
        x = self.pert_embeddings(torch.arange(self.num_perts, device=self.device))

        if not self.use_edge_weight:
            edge_weight = None

        for idx, layer in enumerate(self.gnn_layers):
            if "skip" in self.skip:
                res = x

                if "learned" in self.skip:
                    res = self.skip_layers[idx](res)

            x = layer(x, edge_index, edge_attr=edge_weight)

            if self.skip == "none":
                pass
            elif "sum" in self.skip:
                x = (
                    x + res
                    if not self.concat
                    else x + torch.cat(self.heads * [res], dim=-1)
                )
            elif "cat" in self.skip:
                x = torch.cat([x, res], dim=-1)

            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_nn(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
