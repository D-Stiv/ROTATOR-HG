
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
import torch
import pickle
import numpy as np
from tqdm import tqdm

from dataloader.loader_utils import get_modes_vec, fix_feat, get_modes_ohe, get_node_indices, get_cell_indices, get_edge_indices


@dataclass
class RouteDatasetBuildConfig:
    include_cells: bool = True
    include_edges: bool = True
    include_node_features: bool = True
    include_edge_features: bool = True
    include_time_sequence: bool = False
    include_even_odd_views: bool = False
    include_mode_tables: bool = False
    include_msahg_tables: bool = False



class RouteDatasetBase(Dataset):
    """
    High-performance route dataset base:
    - precomputes deterministic route tensors once
    - stores data as contiguous tensors [N, ...]
    - supports fast-index collate by gathering whole batch tensors directly
    """

    def __init__(self, instances_df, config, build_cfg: RouteDatasetBuildConfig, device="cpu"):
        super().__init__()

        self.device = device
        self.config = config
        self.build_cfg = build_cfg

        self.max_route_length = config.max_route_length
        self.num_modes = config.num_modes

        self.instances_df = instances_df
        self.instances = instances_df.to_dict("records")
        self.N = len(self.instances)

        self.score_names = [
            c for c in instances_df.columns
            if c.endswith("score") and c != "accident_score"
        ]
        self.label_names = [
            c for c in instances_df.columns
            if c.endswith("label") and c not in ["accidents_label", "mode_label"]
        ]

        print("Loading embeddings...")
        self._load_embeddings(config)

        print("Loading features...")
        self._load_features(config)

        if self.build_cfg.include_mode_tables or self.build_cfg.include_msahg_tables:
            print("Building mode tables...")
            self._build_global_mode_tables()

        print("Allocating tensors...")
        self._allocate_storage()

        print("Precomputing dataset tensors...")
        self._build_all()

    # =========================================================
    # LOAD EMBEDDINGS
    # =========================================================

    def _load_embeddings(self):        
        emb_file = self.config.graph_embeddings_path

        with open(emb_file, "rb") as f:
            emb = pickle.load(f)

        self.node_id_to_idx = emb["node_id_to_idx"]
        self.edge_id_to_idx = emb["edge_id_to_idx"]

        self.node_embeddings = torch.tensor(emb["node_embeddings"], dtype=torch.float32)
        self.edge_embeddings = torch.tensor(emb["edge_embeddings"], dtype=torch.float32)
        self.hidden_size = self.node_embeddings.shape[1]

        
        cell_file = self.config.cell_embeddings_path
        with open(cell_file, "rb") as f:
            d = pickle.load(f)
        self.cell_embeddings = torch.tensor(d["embeddings"], dtype=torch.float32)

        node2cell_file = self.config.node2cell_path
        with open(node2cell_file, "rb") as f:
            d = pickle.load(f)
        self.node2cell = d["int_mapping"]

    # =========================================================
    # LOAD FEATURES
    # =========================================================

    def _load_features(self):
        edge_path = self.config.edge_features_path
        with open(edge_path, "rb") as f:
            edge_data = pickle.load(f)

        edge_dims = {
            "struct_features": len(edge_data["struct_features"][0]),
            "scalar_features": len(edge_data["scalar_features"][0]),
            "categorical_features": len(edge_data["categorical_features"][0]),
            "textual_features": len(edge_data["textual_features"][0]),
        }
        edge_data = self._normalize_feature_table(edge_data, edge_dims)

        self.edge_struct_features = torch.from_numpy(
            np.stack(edge_data["struct_features"].values)
        ).float()
        self.edge_scalar_features = torch.from_numpy(
            np.stack(edge_data["scalar_features"].values)
        ).float()
        self.edge_categorical_features = torch.from_numpy(
            np.stack(edge_data["categorical_features"].values)
        ).float()
        self.edge_textual_features = torch.from_numpy(
            np.stack(edge_data["textual_features"].values)
        ).float()

        node_path = self.config.node_features_path
        with open(node_path, "rb") as f:
            node_data = pickle.load(f)

        node_dims = {
            "struct_features": len(node_data["struct_features"][0]),
            "scalar_features": len(node_data["scalar_features"][0]),
            "categorical_features": len(node_data["categorical_features"][0]),
            "textual_features": len(node_data["textual_features"][0]),
        }
        node_data = self._normalize_feature_table(node_data, node_dims)

        self.node_struct_features = torch.from_numpy(
            np.stack(node_data["struct_features"].values)
        ).float()
        self.node_scalar_features = torch.from_numpy(
            np.stack(node_data["scalar_features"].values)
        ).float()
        self.node_categorical_features = torch.from_numpy(
            np.stack(node_data["categorical_features"].values)
        ).float()
        self.node_textual_features = torch.from_numpy(
            np.stack(node_data["textual_features"].values)
        ).float()

    def _normalize_feature_table(self, table, dims):
        if fix_feat is None:
            return table

        for col, dim in dims.items():
            table[col] = table[col].apply(lambda x: fix_feat(x, dim))
        return table

    # =========================================================
    # MODE TABLES
    # =========================================================

    def _build_global_mode_tables(self):
        self.node_mode_mapping = {}
        self.edge_mode_mapping = {}
        self.edge_runtime_to_idx = {}

        for r in self.instances:
            node_modes = get_modes_ohe(r["seq_node_mode"], self.num_modes, node=True)
            for nid, m in zip(r["seq_node_id"], node_modes):
                t = torch.tensor(m, dtype=torch.float32)
                if nid not in self.node_mode_mapping:
                    self.node_mode_mapping[nid] = t
                else:
                    self.node_mode_mapping[nid] = torch.maximum(self.node_mode_mapping[nid], t)

            edge_modes = get_modes_ohe(r["seq_edge_mode"], self.num_modes)
            for eid, m in zip(r["seq_edge_id"], edge_modes):
                t = torch.tensor(m, dtype=torch.float32)
                if eid not in self.edge_runtime_to_idx:
                    self.edge_runtime_to_idx[eid] = len(self.edge_runtime_to_idx)
                if eid not in self.edge_mode_mapping:
                    self.edge_mode_mapping[eid] = t
                else:
                    self.edge_mode_mapping[eid] = torch.maximum(self.edge_mode_mapping[eid], t)

    # =========================================================
    # STORAGE
    # =========================================================

    def _allocate_storage(self):
        N = self.N
        L = self.max_route_length
        Dn = self.node_embeddings.shape[1]
        Dc = self.cell_embeddings.shape[1]
        De = self.edge_embeddings.shape[1]

        ns = self.node_scalar_features.shape[1]
        nc = self.node_categorical_features.shape[1]
        nt = self.node_textual_features.shape[1]

        es = self.edge_struct_features.shape[1]
        esc = self.edge_scalar_features.shape[1]
        ec = self.edge_categorical_features.shape[1]
        et = self.edge_textual_features.shape[1]

        K = len(self.score_names)
        C = len(self.label_names)

        self.node_seq_emb_all = torch.empty(N, L, Dn)
        self.cell_seq_emb_all = torch.empty(N, L, Dc)
        self.edge_seq_emb_all = torch.empty(N, L, De)

        self.mask_all = torch.empty(N, L, dtype=torch.bool)
        self.seq_len_all = torch.empty(N, dtype=torch.long)
        self.edge_len_all = torch.empty(N, dtype=torch.long)

        self.node_scalar_all = torch.empty(N, L, ns)
        self.node_cat_all = torch.empty(N, L, nc)
        self.node_txt_all = torch.empty(N, L, nt)

        self.edge_struct_all = torch.empty(N, L, es)
        self.edge_scalar_all = torch.empty(N, L, esc)
        self.edge_cat_all = torch.empty(N, L, ec)
        self.edge_txt_all = torch.empty(N, L, et)

        self.weak_labels_all = torch.empty(N, dtype=torch.long)
        self.accident_score_all = torch.empty(N)
        self.accident_label_all = torch.empty(N, dtype=torch.long)
        self.criteria_scores_all = torch.empty(N, K)
        self.criteria_labels_all = torch.empty(N, C, dtype=torch.long)

        # total_time_h, total_distance_km, route_length, num_transfers
        self.total_time_all = torch.empty(N, dtype=torch.float32)
        self.total_distance_all = torch.empty(N, dtype=torch.float32)
        self.route_length_all = torch.empty(N, dtype=torch.long)
        self.num_transfers_all = torch.empty(N, dtype=torch.long)

        if self.build_cfg.include_time_sequence:
            self.time_seq_all = torch.empty(N, L, 1)

        if self.build_cfg.include_even_odd_views:
            L_even = (L + 1) // 2
            L_odd = L // 2
            self.node_seq_even_all = torch.empty(N, L_even, Dn)
            self.node_seq_odd_all = torch.empty(N, L_odd, Dn)
            self.cell_seq_even_all = torch.empty(N, L_even, Dc)
            self.cell_seq_odd_all = torch.empty(N, L_odd, Dc)
            self.edge_seq_even_all = torch.empty(N, L_even, De)
            self.edge_seq_odd_all = torch.empty(N, L_odd, De)
            self.even_len_all = torch.empty(N, dtype=torch.long)
            self.odd_len_all = torch.empty(N, dtype=torch.long)
            if self.build_cfg.include_time_sequence:
                self.time_seq_even_all = torch.empty(N, L_even, 1)
                self.time_seq_odd_all = torch.empty(N, L_odd, 1)

        if self.build_cfg.include_mode_tables:
            self.node_ids_all = torch.empty(N, L, dtype=torch.long)
            self.edge_ids_all = torch.empty(N, L, dtype=torch.long)
            self.node_modes_all = torch.empty(N, L, self.num_modes)
            self.edge_modes_all = torch.empty(N, L, self.num_modes)
            self.route_modes_all = torch.empty(N, self.num_modes)
        elif self.build_cfg.include_msahg_tables:
            self.edge_ids_all = torch.empty(N, L, dtype=torch.long)
            self.route_modes_all = torch.empty(N, self.num_modes)

    # =========================================================
    # BUILD ALL
    # =========================================================

    def _build_all(self):
        seq_node_ids_all = self.instances_df["seq_node_id"].tolist()
        seq_edge_ids_all = self.instances_df["seq_edge_id"].tolist()
        seq_edge_modes_all = self.instances_df["seq_edge_mode"].tolist()

        self.weak_labels_all.copy_(
            torch.as_tensor(self.instances_df["mode_label"].to_numpy(), dtype=torch.long)
        )
        self.accident_score_all.copy_(
            torch.as_tensor(self.instances_df["accident_score"].to_numpy(), dtype=torch.float32)
        )
        self.accident_label_all.copy_(
            torch.as_tensor(self.instances_df["accidents_label"].to_numpy(), dtype=torch.long)
        )

        self.total_time_all.copy_(
            torch.as_tensor(self.instances_df["total_time"].to_numpy(), dtype=torch.float32)
        )
        self.total_distance_all.copy_(
            torch.as_tensor(self.instances_df["total_distance"].to_numpy(), dtype=torch.float32)
        )
        self.route_length_all.copy_(
            torch.as_tensor(self.instances_df["path_length_nodes"].to_numpy(), dtype=torch.long)
        )
        self.num_transfers_all.copy_(
            torch.as_tensor(self.instances_df["num_transfers"].to_numpy(), dtype=torch.long)
        )

        if len(self.score_names) > 0:
            self.criteria_scores_all.copy_(
                torch.as_tensor(self.instances_df[self.score_names].to_numpy(), dtype=torch.float32)
            )
        if len(self.label_names) > 0:
            self.criteria_labels_all.copy_(
                torch.as_tensor(self.instances_df[self.label_names].to_numpy(), dtype=torch.long)
            )

        if self.build_cfg.include_time_sequence:
            time_seq = torch.arange(self.max_route_length, dtype=torch.float32).unsqueeze(-1)
            self.time_seq_all[:] = time_seq.unsqueeze(0)

        node_idx_cache = {}
        cell_idx_cache = {}
        edge_idx_cache = {}
        route_modes_cache = {}

        node_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)
        cell_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)
        edge_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)

        seq_len_all = torch.empty(self.N, dtype=torch.long)
        edge_len_all = torch.empty(self.N, dtype=torch.long)

        if self.build_cfg.include_mode_tables:
            node_ids_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)
            edge_ids_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)
            node_modes_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)
            edge_modes_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)

        elif self.build_cfg.include_msahg_tables:
            raw_edge_ids_take = torch.empty((self.N, self.max_route_length), dtype=torch.long)

        for i, (node_seq, edge_seq, edge_mode_seq) in enumerate(
            tqdm(
                zip(seq_node_ids_all, seq_edge_ids_all, seq_edge_modes_all),
                total=self.N,
                desc="Indexing routes",
            )
        ):
            node_key = tuple(node_seq)
            edge_key = (tuple(edge_seq), tuple(edge_mode_seq))

            node_idx = node_idx_cache.get(node_key)
            if node_idx is None:
                node_idx = get_node_indices(self.node_id_to_idx, node_seq)
                node_idx_cache[node_key] = node_idx

            cell_idx = cell_idx_cache.get(node_key)
            if cell_idx is None:
                cell_idx = get_cell_indices(self.node2cell, node_seq)
                cell_idx_cache[node_key] = cell_idx

            edge_idx = edge_idx_cache.get(edge_key)
            if edge_idx is None:
                edge_idx = get_edge_indices(self.edge_id_to_idx, edge_seq, edge_mode_seq)
                edge_idx_cache[edge_key] = edge_idx

            seq_len = min(len(node_idx), self.max_route_length)
            edge_len = min(len(edge_idx), self.max_route_length)

            seq_len_all[i] = seq_len
            edge_len_all[i] = edge_len

            self.mask_all[i].zero_()
            self.mask_all[i, :seq_len] = True

            node_take[i] = self._build_take_index(node_idx, self.max_route_length)
            cell_take[i] = self._build_take_index(cell_idx, self.max_route_length)
            edge_take[i] = self._build_take_index(edge_idx, self.max_route_length)

            if self.build_cfg.include_mode_tables:
                seq_nodes = node_seq[: self.max_route_length]
                seq_edges = edge_seq[: self.max_route_length]

                if len(seq_nodes) == 0:
                    node_ids_take[i].zero_()
                    node_modes_take[i].zero_()
                else:
                    node_ids_take[i] = self._build_take_index(seq_nodes, self.max_route_length)

                    # reuse the same sequence-level cache key when node_seq repeats
                    cached_node_modes_take = node_idx_cache.get(("node_modes_take", node_key))
                    if cached_node_modes_take is None:
                        node_mode_rows = [self.node_mode_mapping[n] for n in seq_nodes[: self.max_route_length]]
                        node_mode_rows = torch.stack(node_mode_rows, dim=0)
                        node_modes_take_tensor = self._build_repeat_index(node_mode_rows.size(0), self.max_route_length)
                        node_idx_cache[("node_modes_take", node_key)] = node_modes_take_tensor
                    else:
                        node_modes_take_tensor = cached_node_modes_take
                    node_modes_take[i] = node_modes_take_tensor

                if len(seq_edges) == 0:
                    edge_ids_take[i].zero_()
                    edge_modes_take[i].zero_()
                else:
                    runtime_edge_ids = [self.edge_runtime_to_idx[e] for e in seq_edges[: self.max_route_length]]
                    edge_ids_take[i] = self._build_take_index(runtime_edge_ids, self.max_route_length)

                    cached_edge_modes_take = edge_idx_cache.get(("edge_modes_take", edge_key))
                    if cached_edge_modes_take is None:
                        edge_mode_rows = [self.edge_mode_mapping[e] for e in seq_edges[: self.max_route_length]]
                        edge_mode_rows = torch.stack(edge_mode_rows, dim=0)
                        edge_modes_take_tensor = self._build_repeat_index(edge_mode_rows.size(0), self.max_route_length)
                        edge_idx_cache[("edge_modes_take", edge_key)] = edge_modes_take_tensor
                    else:
                        edge_modes_take_tensor = cached_edge_modes_take
                    edge_modes_take[i] = edge_modes_take_tensor

                mode_key = tuple(edge_mode_seq[: self.max_route_length])
                route_modes = route_modes_cache.get(mode_key)
                if route_modes is None:
                    route_modes = self._build_route_modes({"seq_edge_mode": list(mode_key)})
                    route_modes_cache[mode_key] = route_modes
                self.route_modes_all[i] = route_modes

            elif self.build_cfg.include_msahg_tables:
                seq_edges = edge_seq[: self.max_route_length]

                if len(seq_edges) == 0:
                    raw_edge_ids_take[i].zero_()
                else:
                    runtime_edge_ids = [self.edge_runtime_to_idx[e] for e in seq_edges]
                    raw_edge_ids_take[i] = self._build_take_index(runtime_edge_ids, self.max_route_length)

                mode_key = tuple(edge_mode_seq[: self.max_route_length])
                route_modes = route_modes_cache.get(mode_key)
                if route_modes is None:
                    route_modes = self._build_route_modes({"seq_edge_mode": list(mode_key)})
                    route_modes_cache[mode_key] = route_modes
                self.route_modes_all[i] = route_modes

        self.seq_len_all.copy_(seq_len_all)
        self.edge_len_all.copy_(edge_len_all)

        self.node_seq_emb_all.copy_(self.node_embeddings[node_take])
        self.cell_seq_emb_all.copy_(self.cell_embeddings[cell_take])
        self.edge_seq_emb_all.copy_(self.edge_embeddings[edge_take])

        self.node_scalar_all.copy_(self.node_scalar_features[node_take])
        self.node_cat_all.copy_(self.node_categorical_features[node_take])
        self.node_txt_all.copy_(self.node_textual_features[node_take])

        self.edge_struct_all.copy_(self.edge_struct_features[edge_take])
        self.edge_scalar_all.copy_(self.edge_scalar_features[edge_take])
        self.edge_cat_all.copy_(self.edge_categorical_features[edge_take])
        self.edge_txt_all.copy_(self.edge_textual_features[edge_take])

        if self.build_cfg.include_mode_tables:
            self.node_ids_all.copy_(node_ids_take)
            self.edge_ids_all.copy_(edge_ids_take)

            node_modes_values = torch.stack(
                [self.node_mode_mapping[n] for n in self.node_mode_mapping],
                dim=0,
            )
            node_mode_row_lookup = {k: idx for idx, k in enumerate(self.node_mode_mapping.keys())}

            edge_modes_values = torch.stack(
                [self.edge_mode_mapping[e] for e in self.edge_mode_mapping],
                dim=0,
            )
            edge_mode_row_lookup = {k: idx for idx, k in enumerate(self.edge_mode_mapping.keys())}

            # materialize node_modes_all
            for i, node_seq in enumerate(tqdm(seq_node_ids_all, desc="Materializing node/edge mode tensors")):
                seq_nodes = node_seq[: self.max_route_length]
                if len(seq_nodes) == 0:
                    self.node_modes_all[i].zero_()
                else:
                    node_mode_rows = torch.as_tensor(
                        [node_mode_row_lookup[n] for n in seq_nodes],
                        dtype=torch.long,
                    )
                    self.node_modes_all[i].copy_(
                        node_modes_values[self._build_take_index(node_mode_rows.tolist(), self.max_route_length)]
                    )

            # materialize edge_modes_all
            print('[Building dataset tensors] Materializing mode tables...')
            for i, edge_seq in enumerate(seq_edge_ids_all):
                seq_edges = edge_seq[: self.max_route_length]
                if len(seq_edges) == 0:
                    self.edge_modes_all[i].zero_()
                else:
                    edge_mode_rows = torch.as_tensor(
                        [edge_mode_row_lookup[e] for e in seq_edges],
                        dtype=torch.long,
                    )
                    self.edge_modes_all[i].copy_(
                        edge_modes_values[self._build_take_index(edge_mode_rows.tolist(), self.max_route_length)]
                    )

        elif self.build_cfg.include_msahg_tables:
            self.edge_ids_all.copy_(raw_edge_ids_take)

        if self.build_cfg.include_even_odd_views:
            print('[Building dataset tensors] Building even/odd views...')
            for i in tqdm(range(self.N), desc="Building even/odd views"):
                self._store_even_odd_views(
                    i,
                    int(self.seq_len_all[i]),
                    self.node_seq_emb_all[i],
                    self.cell_seq_emb_all[i],
                    self.edge_seq_emb_all[i],
                )

    # =========================================================
    # HELPERS
    # =========================================================

    def _build_take_index(self, idx, length):
        if len(idx) == 0:
            return torch.zeros(length, dtype=torch.long)

        idx = torch.as_tensor(idx[:length], dtype=torch.long)
        if idx.numel() < length:
            pad = idx[-1].repeat(length - idx.numel())
            idx = torch.cat([idx, pad], dim=0)
        return idx

    def _build_repeat_index(self, n, length):
        if n == 0:
            return torch.zeros(length, dtype=torch.long)

        idx = torch.arange(min(n, length), dtype=torch.long)
        if idx.numel() < length:
            pad = idx[-1].repeat(length - idx.numel())
            idx = torch.cat([idx, pad], dim=0)
        return idx


    def _pad_last(self, x):
        n = x.shape[0]
        L = self.max_route_length

        if n == 0:
            return torch.zeros((L,) + x.shape[1:], dtype=x.dtype)

        if n >= L:
            return x[:L]

        pad = x[-1:].repeat(L - n, *([1] * (x.dim() - 1)))
        return torch.cat([x, pad], dim=0)

    def _embed_route(self, r):
        node_idx = get_node_indices(self.node_id_to_idx, r["seq_node_id"])
        cell_idx = get_cell_indices(self.node2cell, r["seq_node_id"])
        edge_idx = get_edge_indices(self.edge_id_to_idx, r["seq_edge_id"], r["seq_edge_mode"])

        return (
            self._pad_last(self.node_embeddings[node_idx]),
            self._pad_last(self.cell_embeddings[cell_idx]),
            self._pad_last(self.edge_embeddings[edge_idx]),
        )

    def _feats_route(self, r):
        node_idx = get_node_indices(self.node_id_to_idx, r["seq_node_id"])
        edge_idx = get_edge_indices(self.edge_id_to_idx, r["seq_edge_id"], r["seq_edge_mode"])

        node_feats = {
            "scalar_features": self._pad_last(self.node_scalar_features[node_idx]),
            "categorical_features": self._pad_last(self.node_categorical_features[node_idx]),
            "textual_features": self._pad_last(self.node_textual_features[node_idx]),
        }
        edge_feats = {
            "struct_features": self._pad_last(self.edge_struct_features[edge_idx]),
            "scalar_features": self._pad_last(self.edge_scalar_features[edge_idx]),
            "categorical_features": self._pad_last(self.edge_categorical_features[edge_idx]),
            "textual_features": self._pad_last(self.edge_textual_features[edge_idx]),
        }
        return node_feats, edge_feats

    def _build_node_sequences(self, r):
        seq = r["seq_node_id"][:self.max_route_length]

        if len(seq) == 0:
            ids = torch.zeros(0, dtype=torch.long)
            modes = torch.zeros(0, self.num_modes)
        else:
            ids = torch.tensor(seq, dtype=torch.long)
            modes = torch.stack([self.node_mode_mapping[n] for n in seq])

        return self._pad_last(ids), self._pad_last(modes)

    def _build_edge_sequences(self, r):
        seq = r["seq_edge_id"][:self.max_route_length]

        if len(seq) == 0:
            ids = torch.zeros(0, dtype=torch.long)
            modes = torch.zeros(0, self.num_modes)
        else:
            ids = torch.tensor([self.edge_runtime_to_idx[e] for e in seq], dtype=torch.long)
            modes = torch.stack([self.edge_mode_mapping[e] for e in seq])

        return self._pad_last(ids), self._pad_last(modes)

    def _build_edge_ids_sequence(self, r):
        edge_idx = get_edge_indices(self.edge_id_to_idx, r["seq_edge_id"], r["seq_edge_mode"])
        ids = torch.tensor(edge_idx[:self.max_route_length], dtype=torch.long)

        return self._pad_last(ids)

    def _build_route_modes(self, r):
        return torch.tensor(
            get_modes_vec(r["seq_edge_mode"], self.num_modes, union=None),
            dtype=torch.float32,
        )

    def _store_even_odd_views(self, i, seq_len, node_emb, cell_emb, edge_emb):
        node_even = node_emb[::2]
        node_odd = node_emb[1::2]
        cell_even = cell_emb[::2]
        cell_odd = cell_emb[1::2]
        edge_even = edge_emb[::2]
        edge_odd = edge_emb[1::2]

        even_len = (seq_len + 1) // 2
        odd_len = seq_len // 2

        self.node_seq_even_all[i] = node_even
        self.node_seq_odd_all[i] = node_odd
        self.cell_seq_even_all[i] = cell_even
        self.cell_seq_odd_all[i] = cell_odd
        self.edge_seq_even_all[i] = edge_even
        self.edge_seq_odd_all[i] = edge_odd
        self.even_len_all[i] = even_len
        self.odd_len_all[i] = odd_len

        if self.build_cfg.include_time_sequence:
            time_seq = self.time_seq_all[i]
            self.time_seq_even_all[i] = time_seq[::2]
            self.time_seq_odd_all[i] = time_seq[1::2]

    # =========================================================
    # DATASET API
    # =========================================================

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Fast path: let collate_fn gather from contiguous storage in one shot.
        return int(idx)

    def get_sample(self, idx):
        """
        Optional helper for debugging / ad hoc inspection.
        Not used by the fast DataLoader path.
        """
        idx = int(idx)

        sample = {
            "node_seq_emb": self.node_seq_emb_all[idx],
            "cell_seq_emb": self.cell_seq_emb_all[idx],
            "edge_seq_emb": self.edge_seq_emb_all[idx],
            "mask": self.mask_all[idx],
            "seq_len": self.seq_len_all[idx],
            "edge_len": self.edge_len_all[idx],
            "node_seq_features": {
                "scalar_features": self.node_scalar_all[idx],
                "categorical_features": self.node_cat_all[idx],
                "textual_features": self.node_txt_all[idx],
            },
            "edge_seq_features": {
                "struct_features": self.edge_struct_all[idx],
                "scalar_features": self.edge_scalar_all[idx],
                "categorical_features": self.edge_cat_all[idx],
                "textual_features": self.edge_txt_all[idx],
            },
            "total_time": self.total_time_all[idx],
            "total_distance": self.total_distance_all[idx],
            "route_length": self.route_length_all[idx],
            "num_transfers": self.num_transfers_all[idx],
            "weak_labels": self.weak_labels_all[idx],
            "accident_score": self.accident_score_all[idx],
            "accident_label": self.accident_label_all[idx],
            "criteria_scores": self.criteria_scores_all[idx],
            "criteria_labels": self.criteria_labels_all[idx],
        }

        if self.build_cfg.include_time_sequence:
            sample["time_seq"] = self.time_seq_all[idx]

        if self.build_cfg.include_even_odd_views:
            sample["node_seq_even"] = self.node_seq_even_all[idx]
            sample["node_seq_odd"] = self.node_seq_odd_all[idx]
            sample["cell_seq_even"] = self.cell_seq_even_all[idx]
            sample["cell_seq_odd"] = self.cell_seq_odd_all[idx]
            sample["edge_seq_even"] = self.edge_seq_even_all[idx]
            sample["edge_seq_odd"] = self.edge_seq_odd_all[idx]
            sample["even_len"] = self.even_len_all[idx]
            sample["odd_len"] = self.odd_len_all[idx]
            if self.build_cfg.include_time_sequence:
                sample["time_seq_even"] = self.time_seq_even_all[idx]
                sample["time_seq_odd"] = self.time_seq_odd_all[idx]

        if self.build_cfg.include_mode_tables:
            sample["node_ids"] = self.node_ids_all[idx]
            sample["edge_ids"] = self.edge_ids_all[idx]
            sample["node_feasible_modes"] = self.node_modes_all[idx]
            sample["edge_modes"] = self.edge_modes_all[idx]
            sample["route_modes"] = self.route_modes_all[idx]
        elif self.build_cfg.include_msahg_tables:
            sample["edge_ids"] = self.edge_ids_all[idx]
            sample["route_modes"] = self.route_modes_all[idx]

        return sample

    def get_batch_by_indices(self, batch_indices):
        """
        Gather a full batch from contiguous storage using tensor indexing.
        This is the fast path used by the collate_fn.
        """
        idx = torch.as_tensor(batch_indices, dtype=torch.long)

        batch = {
            "node_seq_emb": self.node_seq_emb_all[idx],
            "cell_seq_emb": self.cell_seq_emb_all[idx],
            "edge_seq_emb": self.edge_seq_emb_all[idx],
            "mask": self.mask_all[idx],
            "seq_len": self.seq_len_all[idx],
            "edge_len": self.edge_len_all[idx],
            "node_seq_features": {
                "scalar_features": self.node_scalar_all[idx],
                "categorical_features": self.node_cat_all[idx],
                "textual_features": self.node_txt_all[idx],
            },
            "edge_seq_features": {
                "struct_features": self.edge_struct_all[idx],
                "scalar_features": self.edge_scalar_all[idx],
                "categorical_features": self.edge_cat_all[idx],
                "textual_features": self.edge_txt_all[idx],
            },
            "total_time": self.total_time_all[idx],
            "total_distance": self.total_distance_all[idx],
            "route_length": self.route_length_all[idx],
            "num_transfers": self.num_transfers_all[idx],
            "weak_labels": self.weak_labels_all[idx],
            "accident_score": self.accident_score_all[idx],
            "accident_label": self.accident_label_all[idx],
            "criteria_scores": self.criteria_scores_all[idx],
            "criteria_labels": self.criteria_labels_all[idx],
        }

        if self.build_cfg.include_time_sequence:
            batch["time_seq"] = self.time_seq_all[idx]

        if self.build_cfg.include_even_odd_views:
            batch["node_seq_even"] = self.node_seq_even_all[idx]
            batch["node_seq_odd"] = self.node_seq_odd_all[idx]
            batch["cell_seq_even"] = self.cell_seq_even_all[idx]
            batch["cell_seq_odd"] = self.cell_seq_odd_all[idx]
            batch["edge_seq_even"] = self.edge_seq_even_all[idx]
            batch["edge_seq_odd"] = self.edge_seq_odd_all[idx]
            batch["even_len"] = self.even_len_all[idx]
            batch["odd_len"] = self.odd_len_all[idx]
            if self.build_cfg.include_time_sequence:
                batch["time_seq_even"] = self.time_seq_even_all[idx]
                batch["time_seq_odd"] = self.time_seq_odd_all[idx]

        if self.build_cfg.include_mode_tables:
            batch["node_ids"] = self.node_ids_all[idx]
            batch["edge_ids"] = self.edge_ids_all[idx]
            batch["node_feasible_modes"] = self.node_modes_all[idx]
            batch["edge_modes"] = self.edge_modes_all[idx]
            batch["route_modes"] = self.route_modes_all[idx]
        elif self.build_cfg.include_msahg_tables:
            batch["edge_ids"] = self.edge_ids_all[idx]
            batch["route_modes"] = self.route_modes_all[idx]

        return batch

    def collate_fn(self, batch_indices):
        """
        DataLoader collate function.
        `batch_indices` is a list of ints because __getitem__ returns indices only.
        """
        return self.get_batch_by_indices(batch_indices)



class RouteDataset(RouteDatasetBase):
    """
    IR2V-ready dataset:
    - same high-performance storage pattern
    - adds route ids, feasible node modes, edge modes and route mode vectors
    - keeps default DataLoader collation compatible
    """

    def __init__(self, instances_df, config=None, device="cpu"):
        super().__init__(
            instances_df=instances_df,
            config=config,
            build_cfg=RouteDatasetBuildConfig(
                include_cells=True,
                include_edges=True,
                include_node_features=True,
                include_edge_features=True,
                include_time_sequence=True,
                include_even_odd_views=True,
                include_mode_tables=True,
            ),
            device=device,
        )


