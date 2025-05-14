from typing import Union, Literal, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter

from gspp.data.graphmodule import GSPGraph
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import is_undirected


class ExphormerModel(nn.Module):

    """
    This model implements the Exphormer architecture, as introduced in arXiv:2303.06147.
    Exphormer is a sparse graph transformer model that enhances message propagation
    by incorporating an expander graph layout on top of the inherent graph edges. This
    design endows the model with robust universal message propagation capabilities.

    Args:
        graph (Union[GSPGraph, Tuple]): Input graph structure, either as a GSPGraph or a tuple.
        layer_type (str): Type of layer to use, either "exphormer" (only Transformer part) or "exphormer_w_mpnn" (hybrid Transformer/message-passing-neural-network).
        num_layers (int): Number of ExphormerLayer modules in the model.
        hidden_dim (int): Hidden feature dimension for the model.
        in_dim (int): Input feature dimension for the model.
        out_dim (int): Output feature dimension; defaults to `hidden_dim` if not specified.
        dropout (float): Dropout probability for regularization.
        num_heads (int): Number of attention heads in each ExphormerLayer.
        add_self_loops (bool): Whether to add self-loops to the graph.
        use_edge_weight (bool): Whether to use edge weights in the model.
        expander_degree (int): Degree of the expander graph used for additional edges.
        add_reverse_edges (bool): Whether to add reverse edges to the graph if it is directed.
        pos_enc (str): Type of positional encoding to use ("none", "RWSE", "LapPE", or "RWSE+LapPE").
        union_edge_type (str): Type of edge feature representation, either "multihot" or "torch_emb".
        edge_feat_map_type (str): Type of edge feature mapping, either "linear", "mlp", or "mlp_w_nodes".
        gate_v (bool): Whether to use a gating mechanism for v mapping using edge features.
        device (str): The device on which the model will run (e.g., "cpu" or "cuda").
    """

    def __init__(
        self,
        graph: Union[GSPGraph, Tuple],
        layer_type: Literal[
            "exphormer", "exphormer_w_mpnn"
        ] = "exphormer",
        num_layers: int = 4,
        skip: Literal[
            "none", "skip_cat", "skip_cat_learned", "skip_sum_learned"
        ] = "none", # Not actually used in this model
        hidden_dim: int = 64,
        in_dim: int = None,
        out_dim: int = None,
        dropout: float = 0.0,
        num_heads: int = 2,
        concat: bool = False, # Not actually used in this model
        add_self_loops: bool = True, 
        use_edge_weight: bool = False,
        use_struc_feat: bool = False, # Not actually used in this model
        expander_degree: int = 3,
        add_reverse_edges: bool = True,
        pos_enc: Literal["none", "RWSE", "LapPE", "RWSE+LapPE"] = "none", # No posenc support in the current version
        union_edge_type: Literal["multihot", "torch_emb"] = "multihot",
        edge_feat_map_type: Literal["linear", "mlp", "mlp_w_nodes"] = "mlp",
        gate_v: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.dropout = dropout
        self.add_mpnn = True if layer_type == 'exphormer_w_mpnn' else False

        self.use_cntr = False
        self.cat_cntr = False

        self.expander_degree = expander_degree
        self.add_self_loops = add_self_loops
        self.add_reverese_edges = add_reverse_edges
        self.union_edge_type = union_edge_type

        out_dim = in_dim if out_dim is None else out_dim

        self.layers = nn.ModuleList()

        if isinstance(graph, Tuple):
            self.num_perts = graph[-1]
        else:
            self.num_perts = next(iter(graph.graph_dict.values()))[-1]

        for _ in range(num_layers):
            self.layers.append(ExphormerLayer(hidden_dim, hidden_dim, num_heads, dropout, 
                                                edge_feat_map_type=edge_feat_map_type, 
                                                add_mpnn=self.add_mpnn, gate_v=gate_v))
        self.out_nn = nn.Linear(hidden_dim, out_dim)

        # Learned perturbation embeddings for each perturbation gene (these are the input node features)
        self.pert_embeddings = nn.Embedding(
            num_embeddings=self.num_perts,
            embedding_dim=hidden_dim,
            device=device,
        )

        edge_set_list = []
        edge_type_list = []
        self.local_edge_index = None
        self.local_edge_weight = None

        for idx, G in enumerate(graph.graph_dict.values()):
            edge_index, edge_weight, _ = G
            edge_index = edge_index.to(device, dtype=torch.long)
            edge_weight = edge_weight.view(-1, 1).to(device, dtype=torch.float)

            # in case we are using exphorme_w_mpnn, we use only the first graph edges as local edges, since the mpnn right now not designed to work with multiple graphs
            if  self.add_mpnn and idx == 0:
                self.local_edge_index = edge_index
            if use_edge_weight and idx == 0:
                self.local_edge_weight = edge_weight

            edge_set_list.append(edge_index)
            edge_type_list.append(torch.zeros(edge_index.size(1)) + len(edge_type_list))
            
            is_undirected_graph = is_undirected(edge_index)
            if self.add_reverese_edges and not is_undirected_graph:
                rev_edge_index = edge_index.flip(0)
                edge_set_list.append(rev_edge_index)
                edge_type_list.append(torch.zeros(rev_edge_index.size(1)) + len(edge_type_list))


        if self.add_self_loops:
            self_loops = torch.arange(self.num_perts).view(1, -1).repeat(2, 1).to(device, dtype=torch.long)
            edge_set_list.append(self_loops)
            edge_type_list.append(torch.zeros(self_loops.size(1)) + len(edge_type_list))
        
        if self.expander_degree > 0:
            self.expander_edge_index = generate_random_graph_with_hamiltonian_cycles(self.num_perts, self.expander_degree, rng=None).to(device, dtype=torch.long)
            edge_set_list.append(self.expander_edge_index)
            edge_type_list.append(torch.zeros(self.expander_edge_index.size(1)) + len(edge_type_list))

        if self.union_edge_type == "multihot":
            edge_sets = [set(zip(edge_index[0].tolist(), edge_index[1].tolist())) for edge_index in edge_set_list]
            all_edges = set.union(*edge_sets)

            edge_to_index = {edge: idx for idx, edge in enumerate(all_edges)}
            self.edge_type_multihot = torch.zeros(len(all_edges), len(edge_sets), device=device, dtype=torch.float)

            for set_idx, edge_set in enumerate(edge_sets):
                for edge in edge_set:
                    edge_idx = edge_to_index[edge]
                    self.edge_type_multihot[edge_idx, set_idx] = 1

            sorted_edges = sorted(all_edges, key=lambda edge: edge_to_index[edge])
            self.edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().to(device)
            
            self.edge_feat_map = nn.Linear(self.edge_type_multihot.size(1), hidden_dim, bias=False)
        elif self.union_edge_type == "torch_emb":
            self.edge_index = torch.cat(edge_set_list, dim=1).contiguous()
            self.edge_type = torch.cat(edge_type_list, dim=0).to(device, dtype=torch.long).contiguous()
            
            self.edge_type_emb = nn.Embedding(
                num_embeddings=len(edge_type_list),
                embedding_dim=hidden_dim,
                device=device,
            )

            self.use_edge_weight = use_edge_weight
            if self.use_edge_weight:
                self.edge_weight_map = torch.nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        else:
            raise ValueError(f"Unknown union edge type: {self.union_edge_type}")


    def forward(self):
        x = self.pert_embeddings(torch.arange(self.num_perts, device=self.device))

        if self.union_edge_type == "multihot":
            edge_attr = self.edge_feat_map(self.edge_type_multihot)
        else:
            edge_attr = self.edge_type_emb(self.edge_type)
            if self.use_edge_weight:
                edge_weight_emb = self.edge_weight_map(self.local_edge_weight)
                edge_attr[self.edge_type == 0] += edge_weight_emb

        for layer in self.layers:
            x, edge_attr = layer(x, self.edge_index, edge_attr, self.local_edge_index)

        x = self.out_nn(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x



class ExphormerLayer(nn.Module):

    '''
    This layer implements the original Exphormer layer with some adaptations from the follow-up work Spexphormer (arXiv:2411.16278).

    Args:
        in_dim (int): Input feature dimension of the nodes.
        out_dim (int): Output feature dimension of the nodes.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for regularization.
        use_bias (bool, optional): Whether to use bias in linear layers. Default is False.
        dim_edge (int, optional): Input feature dimension of the edges. If None, defaults to `in_dim`.
        edge_feat_map_type (str, optional): Type of edge feature mapping. Options are:
            - "linear": Linear transformation of edge features.
            - "mlp": Multi-layer perceptron for edge feature mapping.
            - "mlp_w_nodes": MLP with node features included in edge feature mapping.
        add_mpnn (bool, optional): Whether to include a local graph convolution 
            (GATv2Conv) for additional message passing. Default is False.
        gate_v (bool, optional): Whether to use a gating mechanism for v mapping using edge features. Default is False.
    
    Forward Args:
        h (torch.Tensor): Node feature matrix of shape (num_nodes, in_dim).
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges), 
            defining the graph connectivity.
        edge_attr (torch.Tensor): Edge feature matrix of shape (num_edges, dim_edge).
        local_edge_index (torch.Tensor, optional): Edge index tensor for local 
            graph convolution. Required if `add_mpnn` is True.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Updated node feature matrix of shape (num_nodes, hidden_dim).
            - Updated edge feature matrix of shape (num_edges, hidden_dim) 
              (if edge feature mapping is MLP-based).
    '''

    def __init__(self, in_dim, out_dim, num_heads, dropout, use_bias=False, 
                 dim_edge=None, edge_feat_map_type="linear", add_mpnn=False, gate_v=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.edge_feat_map_type = edge_feat_map_type

        if dim_edge is None:
            dim_edge = in_dim

        dim_h = self.out_dim * num_heads

        self.Q = nn.Linear(in_dim, dim_h, bias=use_bias)
        self.K = nn.Linear(in_dim, dim_h, bias=use_bias)
        self.V = nn.Linear(in_dim, dim_h, bias=use_bias)

        if self.edge_feat_map_type == "linear":
            self.E = nn.Linear(dim_edge, dim_h, bias=use_bias)
        elif edge_feat_map_type == "mlp":
            self.E = nn.Sequential(
                nn.Linear(dim_edge, dim_edge),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_edge, dim_h),
            )
            self.E_norm = nn.BatchNorm1d(dim_h)
        elif edge_feat_map_type == "mlp_w_nodes":
            self.E_1 = nn.Linear(dim_edge, dim_edge, bias=use_bias)
            self.E_q = nn.Linear(in_dim, dim_edge, bias=use_bias)
            self.E_k = nn.Linear(in_dim, dim_edge, bias=use_bias)
            self.E_activation = F.relu
            self.E_dropout1 = nn.Dropout(dropout)
            self.E_2 = nn.Linear(dim_edge, dim_h, bias=use_bias)
            self.E_norm = nn.BatchNorm1d(dim_h)
        else:
            raise ValueError(f"Unknown edge feature mapping: {self.edge_feat_map_type}")
        
        self.E_bias = nn.Linear(dim_edge, num_heads, bias=True)

        self.gate_v = gate_v
        if self.gate_v:
            self.E_V = nn.Linear(dim_edge, dim_h, bias=use_bias)

        self.add_mpnn = add_mpnn
        if self.add_mpnn:
            self.local_conv = GATv2Conv(in_channels=in_dim, out_channels=self.out_dim * num_heads, 
                                        heads=2, dropout=dropout, concat=False, add_self_loops=True)

        self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr, local_edge_index=None):
        h_in = h
        
        Q_h = self.Q(h).view(-1, self.num_heads, self.out_dim)
        K_h = self.K(h).view(-1, self.num_heads, self.out_dim)
        V_h = self.V(h).view(-1, self.num_heads, self.out_dim)

        if self.edge_feat_map_type == "mlp_w_nodes":
            E = self.E_1(edge_attr) + self.E_q(h)[edge_index[1]] + self.E_k(h)[edge_index[0]]
            E = self.E_dropout1(self.E_activation(E))
            E = self.E_2(E).view(-1, self.num_heads, self.out_dim)
        else:
            E = self.E(edge_attr).view(-1, self.num_heads, self.out_dim)
        E_bias = self.E_bias(edge_attr).view(-1, self.num_heads, 1)

        if self.gate_v:
            E_V = self.E_V(edge_attr).view(-1, self.num_heads, self.out_dim)

        # self.propagate_attention(batch, edge_index)
        score = K_h[edge_index[0].to(torch.long)] * Q_h[edge_index[1].to(torch.long)] * E  # element-wise multiplication

        score = score / np.sqrt(self.out_dim)

        score = score.sum(-1, keepdim=True)
        score = score + E_bias
        score = torch.exp(score.clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        if self.gate_v:
            msg = msg * E_V 
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        wV = torch.zeros_like(V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        Z = score.new_zeros(V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        h_attn = wV / (Z + 1e-6)

        h_attn = h_attn.view(-1, self.out_dim * self.num_heads)

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in + h_attn  # Residual connection.
        h = self.norm1_attn(h_attn)

        if self.add_mpnn:
            h_local = self.local_conv(h_in, local_edge_index)
            h = h + h_local
        
        # Feed Forward block.
        h = self.norm2(h + self._ff_block(h))

        if self.edge_feat_map_type.startswith("mlp"):
            edge_attr = self.E_norm(E.view(-1, self.out_dim * self.num_heads))

        return h, edge_attr

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))



def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):
    """Generates a 2d-regular graph with n nodes using d random hamiltonian cycles.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y, x).
    
    Args:
        num_nodes: Number of nodes in the desired graph.
        degree: Desired degree.
        rng: random number generator
        
    Returns:
        edges: Tensor of shape (2, num_edges) where the first row is the senders and the second is the receivers.
    """
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(0)
    
    edges = []
    for _ in range(degree):
        perm = torch.randperm(num_nodes, generator=rng)
        for i in range(num_nodes):
            u, v = perm[i].item(), perm[(i - 1) % num_nodes].item()
            edges.append((u, v))
            edges.append((v, u))

    return torch.tensor(edges).t()