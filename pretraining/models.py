import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pretraining.utils import MaskingStrategy, global_similarity_kl_loss, route_contrastive_loss, node_sequence_loss, edge_sequence_loss, pos_neg_neighborhood_contrastive_loss, transport_consistency_loss, route_order_contrastive_loss, soft_clustering_loss


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SequenceSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, pos_emb=None):
        super().__init__()
        if pos_emb is not None:
            self.pos_emb = pos_emb
        else:
            self.pos_emb = nn.Embedding(1024, embed_dim)
        self.local_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + self.dropout(local))

        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
    

class ModePredictor(nn.Module):
    def __init__(self, dim, num_modes):
        super().__init__()
        self.proj = nn.Linear(dim, num_modes)

    def forward(self, x):
        return self.proj(x)





class RNNHypergraphEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers,
        num_heads=4,
        max_len=512,
        dropout=0.1,
        pos_emb=None,
        use_pos_emb=True,
        use_local_seq=True,
        hg_attn = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_pos_emb = use_pos_emb
        self.hg_attn = hg_attn
        self.dropout = nn.Dropout(dropout)

        if use_pos_emb:
            self.pos_emb = pos_emb if pos_emb is not None else nn.Embedding(max_len, embed_dim)

        if self.hg_attn:
            self.seq_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )
            self.seq_norm = nn.LayerNorm(embed_dim)


        self.same_mlps = nn.ModuleList([
            MLP(embed_dim * 2, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.other_mlps = nn.ModuleList([
            MLP(embed_dim * 2, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.same_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])

        self.other_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])

        self.route_rnns = nn.ModuleList([
            nn.GRU(embed_dim, embed_dim, batch_first=True)
            for _ in range(num_layers)
        ])

        self.route_update_cells = nn.ModuleList([
            nn.GRUCell(embed_dim, embed_dim)
            for _ in range(num_layers)
        ])

        self.route_to_token_mlps = nn.ModuleList([
            MLP(embed_dim * 2, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.route_to_token_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])

        self.token_norm1 = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

        self.token_norm2 = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

        self.route_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

    def _masked_route_rnn(self, rnn, x, mask_valid):
        B, L, D = x.shape

        lengths = mask_valid.sum(dim=1).long()
        lengths_cpu = lengths.clamp(min=1).cpu()

        packed = pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = rnn(packed)

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=L,
        )

        valid = mask_valid.float().unsqueeze(-1)
        pooled = (out * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)

        return pooled.masked_fill(lengths.eq(0).unsqueeze(-1), 0.0)

    def forward(self, token_feat, edge_ids, mask_valid):
        B, L, D = token_feat.shape
        device = token_feat.device

        mask_valid = mask_valid.bool()
        x = token_feat

        # ---------------------------------------------------------
        # 0. Sequence self-attention before hypergraph MP
        # ---------------------------------------------------------
        if self.use_pos_emb:
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos)

        key_padding_mask = ~mask_valid

        if self.hg_attn:
            attn_out, _ = self.seq_attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            x = self.seq_norm(x + self.dropout(attn_out))
        x = x.masked_fill(~mask_valid.unsqueeze(-1), 0.0)

        # ---------------------------------------------------------
        # Flatten valid edge occurrences
        # ---------------------------------------------------------
        flat_mask = mask_valid.reshape(-1)
        flat_edge_ids = edge_ids.reshape(-1)[flat_mask]

        route_indices = (
            torch.arange(B, device=device)
            .unsqueeze(1)
            .expand(B, L)
            .reshape(-1)[flat_mask]
        )

        _, edge_indices = torch.unique(flat_edge_ids, return_inverse=True)
        num_edges = int(edge_indices.max().item()) + 1 if edge_indices.numel() > 0 else 0

        edge_route_pair = torch.stack([edge_indices, route_indices], dim=1)

        unique_pairs, pair_indices = torch.unique(
            edge_route_pair,
            return_inverse=True,
            dim=0,
        )

        num_pairs = unique_pairs.size(0)
        pair_edge_indices = unique_pairs[:, 0]

        edge_route_count = scatter_sum(
            torch.ones(num_pairs, 1, device=device),
            pair_edge_indices,
            dim=0,
            dim_size=num_edges,
        )

        idf = torch.log((B + 1.0) / (edge_route_count + 1.0)) + 1.0

        route_nodes = torch.zeros(B, D, device=device)

        # ---------------------------------------------------------
        # Hypergraph message passing
        # ---------------------------------------------------------
        for layer in range(self.num_layers):
            flat_x = x.reshape(-1, D)[flat_mask]
            ones = torch.ones(flat_x.size(0), 1, device=device)

            # Intra-route context:
            # same edge identity, same route, excluding current occurrence.
            pair_sum = scatter_sum(
                flat_x,
                pair_indices,
                dim=0,
                dim_size=num_pairs,
            )

            pair_count = scatter_sum(
                ones,
                pair_indices,
                dim=0,
                dim_size=num_pairs,
            )

            same_sum = pair_sum[pair_indices] - flat_x
            same_count = pair_count[pair_indices] - 1.0

            same_context = same_sum / same_count.clamp(min=1.0)
            same_context = same_context.masked_fill(same_count.le(0.0), 0.0)

            # Inter-route context:
            # same edge identity, other routes only.
            edge_sum = scatter_sum(
                flat_x,
                edge_indices,
                dim=0,
                dim_size=num_edges,
            )

            edge_count = scatter_sum(
                ones,
                edge_indices,
                dim=0,
                dim_size=num_edges,
            )

            other_sum = edge_sum[edge_indices] - pair_sum[pair_indices]
            other_count = edge_count[edge_indices] - pair_count[pair_indices]

            other_context = other_sum / other_count.clamp(min=1.0)
            other_context = other_context.masked_fill(other_count.le(0.0), 0.0)

            same_delta = self.same_mlps[layer](
                torch.cat([flat_x, same_context], dim=-1)
            )

            other_delta = self.other_mlps[layer](
                torch.cat([flat_x, other_context], dim=-1)
            )

            same_gate = self.same_gates[layer](
                torch.cat([flat_x, same_context, same_delta], dim=-1)
            )

            other_gate = self.other_gates[layer](
                torch.cat([flat_x, other_context, other_delta], dim=-1)
            )

            other_gate = other_gate * idf[edge_indices]

            flat_x = flat_x + same_gate * self.dropout(same_delta) + other_gate * self.dropout(other_delta)

            x_next = torch.zeros_like(x.reshape(-1, D))
            x_next[flat_mask] = flat_x
            x = x_next.reshape(B, L, D)
            x = self.token_norm1[layer](x)
            x = x.masked_fill(~mask_valid.unsqueeze(-1), 0.0)

            # Edge-route message passing: route sequence update.
            route_delta = self._masked_route_rnn(
                self.route_rnns[layer],
                x,
                mask_valid,
            )

            route_nodes = self.route_update_cells[layer](
                route_delta,
                route_nodes,
            )

            route_nodes = self.route_norms[layer](route_nodes)
            route_nodes = self.dropout(route_nodes)

            # Route-edge message passing.
            flat_x = x.reshape(-1, D)[flat_mask]
            route_context = route_nodes[route_indices]

            token_delta = self.route_to_token_mlps[layer](
                torch.cat([flat_x, route_context], dim=-1)
            )

            token_gate = self.route_to_token_gates[layer](
                torch.cat([flat_x, route_context, token_delta], dim=-1)
            )

            flat_x = flat_x + token_gate * self.dropout(token_delta)

            x_next = torch.zeros_like(x.reshape(-1, D))
            x_next[flat_mask] = flat_x
            x = x_next.reshape(B, L, D)
            x = self.token_norm2[layer](x)
            x = x.masked_fill(~mask_valid.unsqueeze(-1), 0.0)

        # Final edge-identity embeddings, optional for caller.
        flat_x = x.reshape(-1, D)[flat_mask]
        token_weight = idf[edge_indices]

        edge_sum = scatter_sum(
            flat_x * token_weight,
            edge_indices,
            dim=0,
            dim_size=num_edges,
        )

        edge_den = scatter_sum(
            token_weight,
            edge_indices,
            dim=0,
            dim_size=num_edges,
        )

        edge_nodes = edge_sum / (edge_den + 1e-8)

        route_embedding = self.output_norm(route_nodes)

        return route_embedding, edge_nodes, edge_indices
    

class InterRoute2Vec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.eval_ = vars(args).get("eval", False)
        self.losses = args.losses

        node_dim = args.node_dim
        cell_dim = args.cell_dim
        edge_dim = args.edge_dim

        num_hyper_layers = args.num_hyper_layers
        num_heads = args.num_heads
        dropout = vars(args).get("dropout", 0.0)
        self.embed_dim = embed_dim = args.embed_dim

        self.view1_strategies = eval(args.view1_strategies)
        self.view2_strategies = eval(args.view2_strategies)
        self.view1_ratios = eval(args.view1_ratios)
        self.view2_ratios = eval(args.view2_ratios)

        self.node_proj = nn.Linear(node_dim + cell_dim, embed_dim)
        self.edge_proj = nn.Linear(edge_dim, embed_dim)

        max_len = vars(args).get("max_len", 500)

        self.shared_pos_emb = nn.Embedding(max_len, embed_dim)


        self.context_drift_weight = vars(args).get("context_drift_weight", 0.01)

        self.final_route_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3)
        )

        self.segment_route_mapping = MLP(embed_dim * 2, embed_dim, dropout=dropout)

        
        self.node_attn = SequenceSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, pos_emb=self.shared_pos_emb)
        self.edge_attn = SequenceSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, pos_emb=self.shared_pos_emb)

        self.fusion = MLP(embed_dim * 2, embed_dim, dropout=dropout)

        
        self.route_hypergraph = RNNHypergraphEncoder(
            embed_dim=embed_dim,
            num_layers=num_hyper_layers,
            num_heads=num_heads,
            pos_emb=self.shared_pos_emb,
            dropout=dropout,
            use_pos_emb=vars(args).get("use_pos_emb", False),
            use_local_seq=vars(args).get("use_local_seq", False),
            hg_attn=vars(args).get("hg_attn", True),
        )


        self.route_mapping = MLP(embed_dim * 2, embed_dim, dropout=dropout)
        self.node_output = MLP(embed_dim * 2, embed_dim, dropout=dropout)
        self.edge_output = MLP(embed_dim * 2, embed_dim, dropout=dropout)

        self.mode_predictor_nodes = ModePredictor(embed_dim, args.num_modes_nodes)
        self.mode_predictor_edges = ModePredictor(embed_dim, args.num_modes_edges)

        self.temperature = vars(args).get("temperature", 0.5)
        self.regularizer_weights = eval(vars(args).get("regularizer_weights", "{'node': 0.05, 'edge': 0.05, 'cons': 0.01}"))  # node, edgeand cons are regularizers, 
        self.loss_weights = eval(vars(args).get("eval_metrics_weights_global", "{'retrieval': 0.3, 'ranking': 0.5, 'clustering': 0.2}")) # route --> ranking, sim --> retieval, clust --> clustering

        self.num_clusters = vars(args).get("num_clusters", 10)

        self.cluster_centers = torch.nn.Parameter(torch.randn(self.num_clusters, embed_dim))

        self.loss_ema = {}          # stores running averages
        self.ema_beta = 0.98        # smoothing factor (good default)
        self.ema_eps = 1e-8

    def get_route_embedding(
        self,
        node_sequence_emb,
        cell_sequence_emb,
        edge_sequence_emb,
        edge_ids,
        mask_valid,
    ):
        B, L, _ = edge_sequence_emb.shape

        proj_node_seq = self.node_proj(torch.cat([node_sequence_emb, cell_sequence_emb], dim=-1))
        proj_edge_seq = self.edge_proj(edge_sequence_emb)

        padding_mask = ~mask_valid.bool()
        node_feat = self.node_attn(proj_node_seq, padding_mask=padding_mask)
        edge_feat = self.edge_attn(proj_edge_seq, padding_mask=padding_mask)


        token_feat = self.fusion(torch.cat([node_feat, edge_feat], dim=-1))
        token_feat = token_feat.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, L, H)

        hyper_route_embedding, _, edge_indices = self.route_hypergraph(
            token_feat=token_feat,
            edge_ids=edge_ids,
            mask_valid=mask_valid,
        )

        return hyper_route_embedding, node_feat, edge_feat, edge_indices

    def forward(self, batch_data, eval_mode=None):
        node_sequence_emb = batch_data["node_seq_emb"].to(self.device)
        cell_sequence_emb = batch_data["cell_seq_emb"].to(self.device)
        edge_sequence_emb = batch_data["edge_seq_emb"].to(self.device)

        edge_ids = batch_data["edge_ids"].to(self.device)
        node_feasible_modes = batch_data["node_feasible_modes"].to(self.device)
        edge_mode = batch_data["edge_modes"].to(self.device)
        mask_valid = batch_data["mask"].to(self.device).bool()

        B, L, _ = edge_sequence_emb.shape

        if self.eval_ or eval_mode:
            with torch.no_grad():
                route_embedding, _, _, _ = self.get_route_embedding(
                    node_sequence_emb,
                    cell_sequence_emb,
                    edge_sequence_emb,
                    edge_ids,
                    mask_valid,
                )
            return route_embedding

        route_emb, node_feat, edge_feat, edge_indices = self.get_route_embedding(
            node_sequence_emb,
            cell_sequence_emb,
            edge_sequence_emb,
            edge_ids,
            mask_valid,
        )

        route_rep = route_emb.unsqueeze(1).expand(-1, L, -1)

        node_out = self.node_output(torch.cat([node_feat, route_rep], dim=-1))
        edge_out = self.edge_output(torch.cat([edge_feat, route_rep], dim=-1))

        mask_base = torch.ones(B, L, 1, device=self.device)

        mask1 = MaskingStrategy.apply_masking_stack(
            mask_base,
            self.view1_strategies,
            self.view1_ratios,
        )
        mask2 = MaskingStrategy.apply_masking_stack(
            mask_base,
            self.view2_strategies,
            self.view2_ratios,
        )

        mask1 = mask1 * mask_valid.unsqueeze(-1)
        mask2 = mask2 * mask_valid.unsqueeze(-1)

        route_emb_view1, node_feat1, edge_feat1, _ = self.get_route_embedding(
            node_sequence_emb * mask1,
            cell_sequence_emb * mask1,
            edge_sequence_emb * mask1,
            edge_ids,
            mask_valid,
        )

        route_emb_view2, node_feat2, edge_feat2, _ = self.get_route_embedding(
            node_sequence_emb * mask2,
            cell_sequence_emb * mask2,
            edge_sequence_emb * mask2,
            edge_ids,
            mask_valid,
        )

        route_rep1 = route_emb_view1.unsqueeze(1).expand(-1, L, -1)
        route_rep2 = route_emb_view2.unsqueeze(1).expand(-1, L, -1)

        node_out_view1 = self.node_output(torch.cat([node_feat1, route_rep1], dim=-1))
        node_out_view2 = self.node_output(torch.cat([node_feat2, route_rep2], dim=-1))

        edge_out_view1 = self.edge_output(torch.cat([edge_feat1, route_rep1], dim=-1))
        edge_out_view2 = self.edge_output(torch.cat([edge_feat2, route_rep2], dim=-1))

        losses = []
        losses_dict = {f"L_{loss_name}": None for loss_name in self.losses}
        mapping_loss_pos = {}
        pos = 0

        if "node" in self.losses:
            node_mask1 = (mask1.repeat(1, 1, self.embed_dim) > 0)
            node_mask2 = (mask2.repeat(1, 1, self.embed_dim) > 0)

            L_node1 = node_sequence_loss(node_out_view1, node_out, node_mask1)
            L_node2 = node_sequence_loss(node_out_view2, node_out, node_mask2)
            L_node = L_node1 + L_node2

            # weight
            L_node = self.regularizer_weights['node'] * L_node

            losses.append(L_node)
            losses_dict["L_node"] = L_node.item()
            mapping_loss_pos["L_node"] = pos
            pos += 1

        if "edge" in self.losses:
            edge_mask1 = (mask1.repeat(1, 1, self.embed_dim) > 0)
            edge_mask2 = (mask2.repeat(1, 1, self.embed_dim) > 0)

            L_edge1 = edge_sequence_loss(edge_out_view1, edge_out, edge_mask1)
            L_edge2 = edge_sequence_loss(edge_out_view2, edge_out, edge_mask2)
            L_edge = L_edge1 + L_edge2

            # weight
            L_edge = self.regularizer_weights['edge'] * L_edge

            losses.append(L_edge)
            losses_dict["L_edge"] = L_edge.item()
            mapping_loss_pos["L_edge"] = pos
            pos += 1

        if "route" in self.losses:
            L_route = (
                route_contrastive_loss(route_emb_view1, route_emb)
                + route_contrastive_loss(route_emb_view2, route_emb)
                + route_contrastive_loss(route_emb_view1, route_emb_view2)
            )

            # weight
            L_route = self.loss_weights['ranking'] * L_route

            losses.append(L_route)
            losses_dict["L_route"] = L_route.item()
            mapping_loss_pos["L_route"] = pos
            pos += 1

        if "cons" in self.losses:
            L_cons = transport_consistency_loss(
                node_out,
                edge_out,
                edge_mode=edge_mode,
                node_feasible_modes=node_feasible_modes,
                mode_predictor_nodes=self.mode_predictor_nodes,
                mode_predictor_edges=self.mode_predictor_edges,
            )

            # weight
            L_cons = self.regularizer_weights['cons'] * L_cons

            losses.append(L_cons)
            losses_dict["L_cons"] = L_cons.item()
            mapping_loss_pos["L_cons"] = pos
            pos += 1

        if "order" in self.losses:
            L_order = route_order_contrastive_loss(
                self,
                node_sequence_emb,
                cell_sequence_emb,
                edge_sequence_emb,
                edge_ids,
                mask_valid,
                temperature=0.1,
            )

            L_order = self.loss_weights.get("ranking", 1.0) * L_order
            losses.append(L_order)
            losses_dict["L_order"] = L_order.item()
            mapping_loss_pos["L_order"] = pos
            pos += 1
                      

        struct_feat = None
        sem_feat = None
        text_feat = None

        if any(x in self.losses for x in ["sim", "struct", "sem", "text"]):
            sim_losses = []

            if "struct" in self.losses or "sim" in self.losses:
                struct_feat = batch_data["edge_seq_features"]["struct_features"].to(self.device)
                struct_feat = (struct_feat * mask_valid.unsqueeze(-1)).sum(dim=1) / (
                    mask_valid.sum(dim=1, keepdim=True) + 1e-8
                )

                L_struct = pos_neg_neighborhood_contrastive_loss(route_emb, struct_feat, pos_k=5, neg_k=20, temperature=0.1, margin=0.2)

                sim_losses.append(L_struct)
                losses_dict["L_struct"] = L_struct.item()

            if "sem" in self.losses or "sim" in self.losses:
                scalar = batch_data["edge_seq_features"]["scalar_features"].to(self.device)
                categ = batch_data["edge_seq_features"]["categorical_features"].to(self.device)

                sem_raw = torch.cat([scalar, categ], dim=-1)
                sem_feat = (sem_raw * mask_valid.unsqueeze(-1)).sum(dim=1) / (
                    mask_valid.sum(dim=1, keepdim=True) + 1e-8
                )

                L_sem = pos_neg_neighborhood_contrastive_loss(route_emb, sem_feat, pos_k=5, neg_k=20, temperature=0.1, margin=0.2)

                sim_losses.append(L_sem)
                losses_dict["L_sem"] = L_sem.item()

            if "text" in self.losses or "sim" in self.losses:
                text_feat = batch_data["edge_seq_features"]["textual_features"].to(self.device)
                text_feat = (text_feat * mask_valid.unsqueeze(-1)).sum(dim=1) / (
                    mask_valid.sum(dim=1, keepdim=True) + 1e-8
                )

                L_text = pos_neg_neighborhood_contrastive_loss(route_emb, text_feat, pos_k=5, neg_k=20, temperature=0.1, margin=0.2)
                
                sim_losses.append(L_text)
                losses_dict["L_text"] = L_text.item()

            L_sim = sum(sim_losses)

            # weight
            L_sim = self.loss_weights['retrieval'] * L_sim

            losses.append(L_sim)
            losses_dict["L_sim"] = L_sim.item()
            mapping_loss_pos["L_sim"] = pos
            pos += 1

        if "dist" in self.losses and struct_feat is not None and sem_feat is not None:
            L_dist = global_similarity_kl_loss(route_emb, struct_feat, sem_feat)

            # weight (useful for clustering)
            L_dist = self.loss_weights['clustering'] * L_dist

            losses.append(L_dist)
            losses_dict["L_dist"] = L_dist.item()
            mapping_loss_pos["L_dist"] = pos
            pos += 1

        if "clust" in self.losses:
            weak_labels = batch_data["weak_labels"].to(self.device)

            L_clust = soft_clustering_loss(
                route_emb,
                weak_labels,
                self.cluster_centers,
            )

            # weight
            L_clust = self.loss_weights['clustering'] * L_clust

            losses.append(L_clust)
            losses_dict["L_clust"] = L_clust.item()
            mapping_loss_pos["L_clust"] = pos
            pos += 1
        
        # Normalize losses
        raw_losses_dict = losses_dict.copy()
        normalized_losses = []
        normalized_losses_dict = {}

        L_dict = {"ranking": 0.0, "retrieval": 0.0, "clustering": 0.0, "regularizer": 0.0}
        
        mapping_pos = {}
        pos = 0
        for loss_name, pos_idx in mapping_loss_pos.items():
            if "route" in loss_name or "order" in loss_name:
                L_dict["ranking"] += losses[pos_idx]
                if "ranking" not in mapping_pos:
                    mapping_pos["ranking"] = pos
                    pos += 1

                # Optional: keep route contrastive loss also contributing to retrieval.
                # I would not include "order" here, because order is a sequence/ranking signal.
                if "route" in loss_name:
                    L_dict["retrieval"] += losses[pos_idx]
                    if "retrieval" not in mapping_pos:
                        mapping_pos["retrieval"] = pos
                        pos += 1

            elif "sim" in loss_name:
                L_dict["retrieval"] += losses[pos_idx]
                if "retrieval" not in mapping_pos:
                    mapping_pos["retrieval"] = pos
                    pos += 1

            elif "clust" in loss_name or "dist" in loss_name:
                L_dict["clustering"] += losses[pos_idx]
                if "clustering" not in mapping_pos:
                    mapping_pos["clustering"] = pos
                    pos += 1

            else:
                L_dict["regularizer"] += losses[pos_idx]
                if "regularizer" not in mapping_pos:
                    mapping_pos["regularizer"] = pos
                    pos += 1


        for loss_name, L in L_dict.items():
            if "regularizer" in loss_name:
                # regularizers are not normalized, just weighted
                normalized_losses.append(L)
                normalized_losses_dict[loss_name] = L.item()
                continue
            # --- Initialize EMA if first time ---
            if loss_name not in self.loss_ema:
                self.loss_ema[loss_name] = L.detach()

            # --- Update EMA ---
            self.loss_ema[loss_name] = (
                self.ema_beta * self.loss_ema[loss_name]
                + (1 - self.ema_beta) * L.detach()
            )

            # --- Normalize ---
            ema_val = self.loss_ema[loss_name]

            L_norm = (L / (ema_val + self.ema_eps)) ** self.temperature

            # clip normalized loss to prevent extreme values during early training
            L_norm = torch.clamp(L_norm, 0.2, 5.0) # default [0, 10]

            normalized_losses.append(L_norm)
            normalized_losses_dict[loss_name] = L_norm.item()

        return {"route_emb": route_emb, "normalized_losses": normalized_losses, "normalized_losses_dict": normalized_losses_dict, "mapping_loss_pos": mapping_pos, "raw_losses_dict": raw_losses_dict}
