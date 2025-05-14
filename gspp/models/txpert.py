from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import MLP

from gspp.data.graphmodule import GSPGraph

from gspp.constants import CONTROL_LABEL

from gspp.models.basal_state_models.vae import VAE
from gspp.models.basal_state_models.mlp import BasalMLP
from gspp.models.basal_state_models.moe import MoE, MoA
from gspp.models.pert_models.basic_gnn import GNN
from gspp.models.pert_models.hybrid_wrapper import HybridWrapper
from gspp.models.pert_models.multi_graph import MultiGraph
from gspp.models.pert_models.exphormer import ExphormerModel


BASAL_STATE_MODEL_DICT = {
    "vae": VAE,
    "mlp": BasalMLP,
    "moe": MoE,
    "moa": MoA,
}

PERT_MODEL_DICT = {
    "mlp": MLP,
    "gnn": GNN,
    "hybrid_wrapper": HybridWrapper,
    "multilayer": MultiGraph,
    "exphormer": ExphormerModel,
}


class TxPert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adata_output_dim: int,
        graph: GSPGraph,
        cntr_model_args: Dict[str, Any],
        pert_model_args: Dict[str, Any],
        hidden_dim: int = 512,
        latent_dim: int = 64,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.2,
        omit_cntr: bool = False,
        no_basal_model: bool = False,
        no_pert_model: bool = False,
        pert_input_dim: int = None,
        device: str = "cpu",
        mse_weight: float = 1.0,
    ):
        super(TxPert, self).__init__()
        """
        This type of model decomposes into two parts:
        1. Control model (one of BASAL_STATE_MODEL_DICT): A model that learns an intrinsic representation of the control.
        2. Perturbation model (one of PERT_MODEL_DICT): A model that learns the perturbation effect.
        A final decoder combines the information of both model parts into final prediction of perturbed expression.
        Further, the loss function is defined here.

        Args:
            input_dim (int): Dimension of the input data.
            output_dim (int): Dimension of the output data.
            graph (GSPGraph): Graph object containing the edge index and edge weight.
            cntr_model_args (Dict[str, Any]): Arguments for the control model.
            pert_model_args (Dict[str, Any]): Arguments for the perturbation model.
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            use_batch_norm (bool): Whether to use batch normalization.
            use_layer_norm (bool): Whether to use layer normalization.
            dropout (float): Dropout rate.
            omit_cntr (bool): Whether to omit the control when combining basal state and perturbation effect.
            device (str): Device to run the model on.
            mse_weight (float): Weight of the MSE loss in the hybrid reconstruction loss.
        """
        self.mse_weight = mse_weight

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.no_basal_model = no_basal_model
        self.no_pert_model = no_pert_model

        self.adata_output_dim = adata_output_dim

        # Basal/control model
        if self.no_basal_model:
            self.cntr_model = None
        else:
            self.cntr_model_type = cntr_model_args.pop("model_type", "vae")

            if self.cntr_model_type not in ["moe", "moa"]:
                cntr_model_args.pop("rank", None)

            self.cntr_model = BASAL_STATE_MODEL_DICT[self.cntr_model_type](
                input_dim=input_dim,
                latent_dim=latent_dim,
                **cntr_model_args,
            ).to(device)

        # Perturbation model
        self.pert_model_type = pert_model_args.pop("model_type", "gnn")
        self.pert_model = PERT_MODEL_DICT[self.pert_model_type](
            out_dim=latent_dim,
            graph=graph,
            device=device,
            **pert_model_args,
        ).to(device)

        if self.no_pert_model:
            self.pert_model = None

        self.omit_cntr = omit_cntr

        # Decoder
        decoder_layers_intrinsic = [
            nn.Linear(latent_dim + (self.input_dim - self.output_dim), hidden_dim),
            (
                nn.BatchNorm1d(hidden_dim)
                if use_batch_norm
                else nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            ),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.adata_output_dim),
        ]
        self.decoder = nn.Sequential(*decoder_layers_intrinsic)

        # Graph
        if "hybrid" in self.pert_model_type:
            self.edge_index, self.edge_weight = {}, {}
            for graph_type, g in graph.graph_dict.items():
                edge_index, edge_weight, _ = g

                self.edge_index[graph_type] = torch.Tensor(edge_index).to(device)
                self.edge_weight[graph_type] = torch.Tensor(edge_weight).to(device)

        elif "multilayer" in self.pert_model_type:
            self.graphs = []
            for g in graph.graph_dict.values():
                edge_index, edge_weight, num_nodes = g
                self.graphs.append(
                    (
                        torch.Tensor(edge_index).to(device),
                        torch.Tensor(edge_weight).to(device),
                        num_nodes,
                    )
                )
            num_layers = len(self.graphs)
            self.p = [
                [j for j in range(num_layers) if j != i] for i in range(num_layers)
            ]
            # Create supra-adjacency matrix for GNN
            supra_edge_index, supra_edge_weight, total_nodes = self.create_supra_adj(
                self.graphs
            )
            self.g_supra = (supra_edge_index, supra_edge_weight, total_nodes)

        else:
            edge_index, edge_weight, _ = next(iter(graph.graph_dict.values()))
            self.edge_index = torch.Tensor(edge_index).to(device)
            self.edge_weight = torch.Tensor(edge_weight).to(device)

        self.device = device

    def forward(self, cntr, pert_idxs, p_emb):
        batch_size = cntr.size(0)

        # Control model
        if self.no_basal_model:
            if self.no_pert_model:
                return cntr, None, None

            z_intrinsic, intrinsic_mean, intrinsic_log_var = 0.0, None, None
        else:
            z_intrinsic, intrinsic_mean, intrinsic_log_var = self.cntr_model.forward(
                cntr
            )

        if self.no_pert_model:
            # Decoder
            prediction = self.decoder(
                torch.cat([z_intrinsic, cntr[:, self.output_dim :]], dim=-1)
            )

            if self.no_basal_model:
                prediction = cntr[:, : self.output_dim] + prediction

            return prediction, intrinsic_mean, intrinsic_log_var

        else:
            # Perturbation model
            if self.pert_model_type == "multilayer":
                z_p = self.pert_model.forward(
                    graphs=self.graphs, p=self.p, g_supra=self.g_supra
                )
            elif self.pert_model_type == "exphormer":
                z_p = self.pert_model.forward()
            else:
                z_p = self.pert_model.forward(
                    self.edge_index, self.edge_weight, z_intrinsic
                )

            self.pert_z = torch.zeros(batch_size, z_p.size(-1), device=self.device)

            for i, perts in enumerate(pert_idxs):
                for p in perts:
                    if self.omit_cntr and p in [-1, CONTROL_LABEL]:
                        continue

                    if self.pert_model.use_cntr or self.pert_model.cat_cntr:
                        self.pert_z[i] = self.pert_z[i] + z_p[p][i]
                    else:
                        self.pert_z[i] = self.pert_z[i] + z_p[p]

        # Decoder
        prediction = self.decoder(
            torch.cat([z_intrinsic + self.pert_z, cntr[:, self.output_dim :]], dim=-1)
        )

        if self.no_basal_model:
            prediction = cntr[:, : self.output_dim] + prediction

        return prediction, intrinsic_mean, intrinsic_log_var

    def loss(
        self,
        prediction,
        intrinsic_mean,
        intrinsic_log_var,
        target,
    ):
        """
        Computes the overall loss function of the model.
        """
        recon_loss = self.reconstruction_loss(prediction, target)

        if self.no_basal_model:
            kl_div_intrinsic = 0.0
        else:
            kl_div_intrinsic = (
                self.kl_divergence(intrinsic_mean, intrinsic_log_var)
                if self.cntr_model_type == "vae"
                else 0.0
            )

        logs = {
            "recon_loss": recon_loss,
            "kl_div_intrinsic": kl_div_intrinsic,
        }

        return recon_loss + kl_div_intrinsic, logs

    def reconstruction_loss(self, x_hat, x):
        """
        Computes a hybrid reconstruction loss (MSE + cossim) between the predicted and the target expression.
        """
        mse_loss = F.mse_loss(x_hat, x, reduction="mean")
        cos_loss = 1 - F.cosine_similarity(x_hat, x, dim=-1).mean()

        return self.mse_weight * mse_loss + (1 - self.mse_weight) * cos_loss

    def kl_divergence(self, mean, log_var):
        """
        Computes the KL divergence between the learned latent distribution (computed by the basal state model) and the standard normal distribution.
        """
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def create_supra_adj(self, graphs):
        """
        Create supra-adjacency matrix for multilayer GNN.
        """
        device = graphs[0][0].device
        nodes_per_layer = [g[2] for g in graphs]
        total_nodes = sum(nodes_per_layer)

        edges_list = []
        weights_list = []
        offset = 0

        for i, (edge_index, edge_weight, num_nodes) in enumerate(graphs):
            # Check edge index bounds before offset
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                raise ValueError(
                    f"Layer {i}: Edge index {max_idx} exceeds number of nodes {num_nodes}"
                )

            # Create new edge index with offset
            new_edge_index = edge_index.clone()
            new_edge_index += offset

            # Validate after offset
            new_max_idx = new_edge_index.max().item()
            if new_max_idx >= total_nodes:
                raise ValueError(
                    f"Layer {i}: Offset edge index {new_max_idx} exceeds total nodes {total_nodes}"
                )

            edges_list.append(new_edge_index)
            weights_list.append(edge_weight)

            offset += num_nodes

        # Combine edges and weights
        supra_edge_index = torch.cat(edges_list, dim=1)
        supra_edge_weight = torch.cat(weights_list)

        return supra_edge_index, supra_edge_weight, total_nodes
