from typing import Any, List, Dict, Literal, Tuple
from collections import OrderedDict

from torch import Tensor

import torch
import torch.nn as nn

from gspp.data.graphmodule import GSPGraph

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP
from torch_geometric.nn.inits import glorot

from gspp.models.pert_models.basic_gnn import GNN
from gspp.constants import ACTIVATION_DICT


class HybridWrapper(nn.Module):
    """
    Wrapper class to combine multiple GNNs into a single model.

    Args:
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        gnns_config (Dict[str, Dict[str, Any]]): The configuration of the different GNNs.
        in_dim (int): The input dimension of the data.
        out_dim (int): The output dimension of the data.
        combine_fn (Literal["cat", "att", "att_bias"]): The combine function for the different GNNs.
        use_cntr (bool): Whether to use the counterfactual information.
        cat_cntr (bool): Whether to concatenate the counterfactual information.
        hidden_dim (int): The hidden dimension of the model.
        num_heads (int): The number of heads.
        depth_mlp (int): The depth of the MLP.
        activation (str): The activation function.
        activation_att (str): The activation function for the attention.
        device (str): The device.
    """

    def __init__(
        self,
        graph: GSPGraph,
        gnns_config: Dict[str, Dict[str, Any]],
        in_dim: int = None,
        out_dim: int = 64,
        combine_fn: Literal["cat", "att", "att_bias"] = "cat",
        use_cntr: bool = False,
        cat_cntr: bool = False,
        hidden_dim: int = 64,
        num_heads: int = 1,
        depth_mlp: int = 2,
        activation: str = "relu",
        activation_att: str = "leaky_relu",
        device: str = "cpu",
    ):
        super(HybridWrapper, self).__init__()

        self.combine_fn = combine_fn
        self.num_heads = num_heads
        self.device = device

        self.use_cntr = use_cntr
        self.cat_cntr = cat_cntr

        self.num_nodes = min(
            [num_nodes for _, _, num_nodes in graph.graph_dict.values()]
        )

        in_dim = hidden_dim if in_dim is None else in_dim

        self.gnns = nn.ModuleDict()
        for mode, gnn_args in gnns_config.items():
            layer_type, graph_type = mode.split("+")
            self.gnns[f"{layer_type}/{graph_type}"] = GNN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=num_heads * hidden_dim,
                graph=graph.graph_dict[graph_type],
                layer_type=layer_type,
                device=device,
                **gnn_args,
            )

        if self.combine_fn != "cat":
            self.linear_k = Linear(
                out_dim, num_heads * hidden_dim, weight_initializer="glorot"
            )
            self.att_k = nn.ParameterList(
                [nn.Parameter(torch.empty(1, hidden_dim)) for _ in range(num_heads)]
            )
            self.att_q = nn.ParameterList(
                [nn.Parameter(torch.empty(1, hidden_dim)) for _ in range(num_heads)]
            )

        self.activation = ACTIVATION_DICT[activation]
        self.activation_att = ACTIVATION_DICT[activation_att]

        if depth_mlp > 0:
            m = 0
            if self.combine_fn == "cat":
                m += num_heads * len(self.gnns)
                in_dim = m * hidden_dim

            if self.combine_fn == "att":
                in_dim = in_dim

            if self.cat_cntr:
                in_dim += out_dim

            self.mlp_out = MLP(
                [in_dim] + (depth_mlp - 1) * [hidden_dim] + [out_dim], activation=activation, norm=None
            )
        else:
            self.mlp_out = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.combine_fn != "cat":
            for param in self.att_k:
                glorot(param)
            for param in self.att_q:
                glorot(param)
            self.linear_k.reset_parameters()

        if self.mlp_out is not None:
            self.mlp_out.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        x_k: Tensor = None,
    ) -> Tensor:
        x_channel_dict = OrderedDict()

        if self.use_cntr and x_k is not None:
            x_channel_dict["x_k"] = x_k

        for key, gnn in self.gnns.items():
            _, graph_type = key.split("/")
            x_channel_dict[key] = gnn(edge_index[graph_type], edge_weight[graph_type])[
                : self.num_nodes, ...
            ]

        if self.combine_fn == "cat":
            x = torch.cat(list(x_channel_dict.values()), dim=-1)
        elif self.combine_fn == "att":
            x = self.channel_attention(x_channel_dict)
        else:
            raise ValueError(f"Un-supported `combine_fn` {self.combine_fn}.")

        if self.mlp_out is not None:
            if self.cat_cntr:
                x = torch.cat(
                    [
                        (
                            x
                            if self.use_cntr
                            else x.unsqueeze(1).expand(-1, x_k.size(0), -1)
                        ),
                        x_k.unsqueeze(0).expand(self.num_nodes, -1, -1),
                    ],
                    dim=-1,
                )

            x = self.mlp_out(x)

        return x

    def channel_attention(self, x_channel_dict: Dict[str, Tensor]) -> Tensor:
        """
        Attention mechanism to combine the different channels of the hybrid layer.
        """
        x_k = x_channel_dict.pop("x_k", None)

        if x_k is not None:
            batch_size = x_k.size(0)

            x_k = self.activation_att(self.linear_k(x_k))
            # x_k = x_k.view(self.num_heads, self.num_nodes, -1)
            x_k = x_k.view(batch_size, self.num_heads, -1)
            e_k = torch.cat(
                [
                    torch.matmul(
                        x_k[:, idx].unsqueeze(-2), self.att_k[idx].unsqueeze(-1)
                    )
                    for idx in range(self.num_heads)
                ],
                dim=1,
            )
        else:
            batch_size = 1
            e_k = Tensor([0.0]).view(1, 1, 1).to(self.device)

        e_list = []
        for key, x_q in x_channel_dict.items():
            x_q = self.activation_att(x_q)  # TODO: Check if activation is needed
            x_q = x_q.view(self.num_nodes, self.num_heads, -1)
            e_q = torch.cat(
                [
                    torch.matmul(
                        x_q[:, idx].unsqueeze(-2), self.att_q[idx].unsqueeze(-1)
                    )
                    for idx in range(self.num_heads)
                ],
                dim=1,
            )
            e_list.append(e_q.unsqueeze(1) + e_k.unsqueeze(0))

        e = torch.stack(e_list, dim=0)
        channel_weights = torch.softmax(e, dim=0)
        weighted_channels = torch.mul(
            channel_weights,
            torch.stack(
                [
                    x.view(self.num_nodes, self.num_heads, -1)
                    for x in x_channel_dict.values()
                ],
                dim=0,
            ).unsqueeze(2),
        )
        x = weighted_channels.sum(dim=0).sum(dim=-2) / self.num_heads

        return x.squeeze()
