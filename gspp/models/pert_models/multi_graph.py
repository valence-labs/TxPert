import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_add


class GatedCombiner(nn.Module):
    def __init__(self, struct_emb_dim, gnn_emb_dim, output_dim):
        super().__init__()

        self.struct_emb_dim = struct_emb_dim
        self.gnn_emb_dim = gnn_emb_dim
        self.output_dim = output_dim

        # Gate network takes concatenated embeddings as input
        self.gate = nn.Sequential(
            nn.Linear(struct_emb_dim + gnn_emb_dim, output_dim), nn.Sigmoid()
        )

        # Transformation layers for each embedding type - project to common output dimension
        self.struct_transform = nn.Linear(struct_emb_dim, output_dim)
        self.gnn_transform = nn.Linear(gnn_emb_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, struct_emb, gnn_emb):
        # Validate input dimensions
        if struct_emb.size(-1) != self.struct_emb_dim:
            raise ValueError(
                f"Expected struct_emb dimension {self.struct_emb_dim}, got {struct_emb.size(-1)}"
            )
        if gnn_emb.size(-1) != self.gnn_emb_dim:
            raise ValueError(
                f"Expected gnn_emb dimension {self.gnn_emb_dim}, got {gnn_emb.size(-1)}"
            )

        # Transform embeddings to common dimension
        struct_transformed = self.struct_transform(struct_emb)
        gnn_transformed = self.gnn_transform(gnn_emb)

        # Compute gate values
        gate_input = torch.cat([struct_emb, gnn_emb], dim=-1)
        z = self.gate(gate_input)

        # Combine embeddings using gate
        combined = z * struct_transformed + (1 - z) * gnn_transformed

        # Add residual connections if dimensions permit
        if struct_emb.size(-1) == self.output_dim:
            combined = combined + struct_emb
        if gnn_emb.size(-1) == self.output_dim:
            combined = combined + gnn_emb

        # Normalize
        combined = self.layer_norm(combined)

        return combined


class MGAT(nn.Module):
    _multi_head = {"concat", "avg"}

    def __init__(
        self,
        num_hidden,
        in_dim,
        hidden_dim,
        output_dim,
        heads,
        activation,
        aggregation="concat",
        drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
    ):
        super(MGAT, self).__init__()
        aggregation = aggregation.lower()
        if aggregation not in self._multi_head:
            raise ValueError(
                "Unrecognized aggregation mode for attention heads : {} ".format(
                    aggregation
                )
            )

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.attention_weights = []
        self.aggregation = aggregation
        self.num_layers = num_hidden
        self.layers = nn.ModuleList()

        # First layer
        if num_hidden == 1:
            # If only one layer, use output_dim directly
            out_dim = output_dim
        else:
            out_dim = hidden_dim

        self.layers.append(
            GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads[0],
                dropout=drop,
                negative_slope=negative_slope,
                add_self_loops=True,
                bias=True,
            )
        )

        # Hidden layers
        for l in range(1, num_hidden):
            if self.aggregation == "concat":
                inf = hidden_dim * heads[l - 1]
            else:
                inf = hidden_dim
            if l == num_hidden - 1:
                drop = 0.0
                out_dim = output_dim
            else:
                out_dim = hidden_dim
            self.layers.append(
                GATv2Conv(
                    in_channels=inf,
                    out_channels=out_dim,
                    heads=heads[l],
                    dropout=drop,
                    negative_slope=negative_slope,
                    add_self_loops=True,
                    bias=True,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        for gnn in self.layers:
            gnn.reset_parameters()

    def forward(self, graph_tuple, inputs):
        h = inputs

        self.attention_weights = []
        layer_outputs = []

        edge_index, edge_weight, total_nodes = graph_tuple

        for i, layer in enumerate(self.layers):
            edge_index = edge_index.long()

            # GAT layer
            h = layer(h, edge_index)

            # Attention aggregation
            if self.aggregation == "concat":
                h = h.flatten(1)
            else:
                h = h.mean(1)

            # Activation
            if self.activation is not None:
                h = self.activation(h)

            layer_outputs.append(h)

        return layer_outputs, [output.shape for output in layer_outputs]


class MLStruct(torch.nn.Module):
    def __init__(
        self,
        edge_dim,
        node_dim,
        phi_dim,
        n_layers,
        beta=1.0,
        dropout=0.2,
        f_dropout=0.7,
        eps=1e-8,
    ):
        super(MLStruct, self).__init__()

        self.n_layers = n_layers
        self.beta = beta
        self.eps = eps
        self.f_dropout = f_dropout
        self.cns = nn.ModuleList()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.cached_adjs = [None] * n_layers

        for _ in range(n_layers):
            f_edge = torch.nn.Sequential(
                torch.nn.Linear(1, edge_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(edge_dim, node_dim),
            )

            f_node = torch.nn.Sequential(
                torch.nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
            )

            g_phi = torch.nn.Sequential(
                torch.nn.Linear(1, phi_dim), nn.ReLU(), nn.Linear(phi_dim, node_dim)
            )

            cns_l = nn.ModuleList([f_edge, f_node, g_phi])
            self.cns.append(cns_l)

        self.reset_parameters()

    def reset_parameters(self):
        for cnl in self.cns:
            for l in cnl:
                l.apply(self.weight_reset)
        self.cached_adjs = [None] * self.n_layers

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def random_fill(self, mat, dtype=torch.float32):
        mat.mul_(1 + F.dropout(torch.full_like(mat, self.eps), p=self.f_dropout))
        return mat

    def chunked_sparse_dense_matmul(self, mat_src, mat_dst, chunk_size=1000):
        """Perform matrix multiplication in chunks to save memory."""
        device = mat_src.storage.value().device
        n = mat_src.sparse_sizes()[0]
        result = torch.zeros((n,), device=device)

        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            src_chunk = mat_src[i:end].to_dense()
            dst_chunk = mat_dst[i:end].to_dense()
            result[i:end] = torch.sum(src_chunk * dst_chunk, dim=1)

        return result

    def forward(self, graphs, edge_w=False):
        out_structs = []
        node_struct_feats = []

        for l_id in range(self.n_layers):
            edge_index, edge_weight, num_nodes = graphs[l_id]

            if edge_index.size(1) == 0:
                out_structs.append(None)
                node_struct_feats.append(None)
                continue

            try:
                f_edge = self.cns[l_id][0]
                f_node = self.cns[l_id][1]
                g_phi = self.cns[l_id][2]

                # Process edge weights to node_dim features
                edge_weights = edge_weight if edge_w else torch.ones_like(edge_weight)
                edge_weight_A = f_edge(edge_weights.unsqueeze(-1))

                # Compute node features maintaining node_dim
                node_struct_feat = scatter_add(
                    edge_weight_A, edge_index[1], dim=0, dim_size=num_nodes
                )

                # Process node features while keeping node_dim
                node_struct_feat = f_node(node_struct_feat)
                node_struct_feats.append(node_struct_feat)

            finally:
                if "edge_weights" in locals():
                    del edge_weights
                torch.cuda.empty_cache()

        return node_struct_feats


class MultiGraph(nn.Module):
    def __init__(self, graph, dropout, device="cpu", eps=1e-8, **args):
        super(MultiGraph, self).__init__()
        self.device = device
        self.dropout = dropout
        self.eps = eps
        self.edge_w = False
        self.beta = 1.0
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

        self.num_perts = max(g[2] for g in self.graphs)

        self.n_layers = len(self.graphs)
        self.struct = not args["no_struct"]

        # Structural component
        if not args["no_struct"]:
            self.psi = args["psi"]
            self.edge_dim = args["edge_dim"]
            self.node_dim = args["node_dim"]
            self.phi_dim = args["phi_dim"]
            self.attn_dropout = args["attn_dropout"]
            self.f_dropout = args["f_dropout"]
            self.n_heads = args["n_heads"]

            self.struct_model = MLStruct(
                self.edge_dim,
                self.node_dim,
                self.phi_dim,
                self.n_layers,
                beta=self.beta,
                dropout=self.dropout,
                eps=self.eps,
                f_dropout=self.f_dropout,
            )

        self.struct_dim = args["phi_dim"]

        # GNN component
        self.input_dim = args["input_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.output_dim = args["output_dim"]
        self.num_hidden = args["num_hidden"]
        self.heads = [args["n_heads"]] * self.num_hidden
        self.activation = args["activation"]
        self.attn_dropout = 0.0
        if "attn_dropout" in args:
            self.attn_dropout = args["attn_dropout"]
        self.residual = False
        if "residual" in args:
            self.residual = args["residual"]
        self.aggregation = "avg"
        if "aggregation" in args:
            self.aggregation = args["aggregation"]

        self.gnn_model = MGAT(
            self.num_hidden,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.heads,
            self.activation,
            self.aggregation,
            self.dropout,
            self.attn_dropout,
            residual=self.residual,
        )

        # Calculate the output dimension of GNN model based on aggregation and heads
        if self.aggregation == "concat":
            self.gnn_dim = self.output_dim * self.heads[-1]
        else:  # 'avg' or other aggregation
            self.gnn_dim = self.output_dim

        # Add GatedCombiners for each layer
        self.combiners = nn.ModuleList()
        if self.struct:
            for _ in range(self.n_layers):
                self.combiners.append(
                    GatedCombiner(
                        struct_emb_dim=self.struct_dim,
                        gnn_emb_dim=self.gnn_dim,
                        output_dim=self.output_dim,  # Standardize output dimension
                    )
                )
        else:
            self.gnn_projections = nn.ModuleList()
            for _ in range(self.n_layers):
                # Only need projection if dimensions differ
                if self.gnn_dim != self.output_dim:
                    self.gnn_projections.append(
                        nn.Sequential(
                            nn.Linear(self.gnn_dim, self.output_dim),
                            nn.LayerNorm(self.output_dim),
                        )
                    )
                else:
                    # Identity if dimensions already match
                    self.gnn_projections.append(nn.Identity())

        self.use_cntr = False
        self.cat_cntr = False

        # Learned perturbation embeddings for each perturbation gene (these are the input node features)
        self.features = nn.Embedding(
            num_embeddings=self.num_perts,
            embedding_dim=self.input_dim,
            device=device,
        )

    def forward(self, graphs, p, g_supra):
        """
        Args:
            graphs: List of (edge_index, edge_weight, num_nodes) tuples for each layer
            p: List of layer indices for cross-layer attention
            g_supra: Supra-adjacency matrix for GNN component
        """
        # Get features from embedding layer
        features = self.features(torch.arange(self.num_perts, device=self.device))

        # Replicate features for each layer
        features_list = []
        for graph in graphs:
            num_nodes = graph[2]  # Get actual number of nodes for this layer
            layer_features = self.features(torch.arange(num_nodes, device=self.device))
            features_list.append(layer_features)

        # Concatenate for supra-adjacency usage
        features_supra = torch.cat(features_list, dim=0)

        node_embeddings = {}

        # Structural component
        if self.struct:
            node_struct_feats = self.struct_model(graphs, edge_w=self.edge_w)
            node_embeddings["struct_feats"] = node_struct_feats

        # GNN component
        out_gnn, out_gnn_shapes = self.gnn_model(g_supra, features_supra)
        node_embeddings["gnn_feats"] = out_gnn

        # Extract features for each layer
        nodes_offset = 0
        node_embeddings["layer_wise"] = []
        for graph in graphs:
            num_nodes = graph[2]
            node_embeddings["layer_wise"].append(
                out_gnn[-1][nodes_offset : nodes_offset + num_nodes]
            )
            nodes_offset += num_nodes

        combined_embeddings = []
        if self.struct:
            # Combine the two components
            for l_id in range(self.n_layers):
                struct_emb = (
                    node_embeddings["struct_feats"][l_id]
                    if "struct_feats" in node_embeddings
                    else None
                )
                gnn_emb = (
                    node_embeddings["layer_wise"][l_id]
                    if "layer_wise" in node_embeddings
                    else None
                )

                if struct_emb is not None and gnn_emb is not None:
                    # Use gated combination
                    combined = self.combiners[l_id](struct_emb, gnn_emb)
                    combined_embeddings.append(combined)
                else:
                    combined_embeddings.append(None)
        else:
            for l_id in range(self.n_layers):
                gnn_emb = (
                    node_embeddings["layer_wise"][l_id]
                    if "layer_wise" in node_embeddings
                    else None
                )
                projected_emb = self.gnn_projections[l_id](gnn_emb)
                combined_embeddings.append(projected_emb)

        # Return layer 0 representation
        return combined_embeddings[0]
