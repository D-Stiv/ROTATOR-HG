import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataloader.dataset import RouteDataset
import torch


def build_dataset_for_model(instances_df, config):
    # sort columns in instances_df to ensure consistent order across models (important for caching)
    instances_df = instances_df.reindex(sorted(instances_df.columns), axis=1)
    print(f'Columns: {instances_df.columns.tolist()}')


    return RouteDataset(instances_df, config=config)



def build_loaders(train_dataset, val_dataset, test_dataset, args):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None,
    )
    
    return train_loader, val_loader, test_loader


def get_model_data_split(args):
    dst_dir = getattr(args, "dataset_cache_dir", "data/cache")
    data_splits_dir = getattr(args, "data_splits_dir", "data/splits")

    os.makedirs(dst_dir, exist_ok=True)
    datasets = {}

    for split in ["val", "test", "train"]: # from smaller to bigger
        file_path = os.path.join(data_splits_dir, f"{split}.pkl")
        df = pd.read_pickle(file_path)
        print(f"Loaded {split} dataset with {len(df)} samples.")

        print(f"Preparing {args.model_name} {split} dataset...")
        dataset = build_dataset_for_model(df, args)

        dataset_path = os.path.join(dst_dir, f"{split}.pkl")

        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Saved datasets for {split} to {dataset_path} for future use.")
        datasets[split] = dataset

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    return build_loaders(train_dataset, val_dataset, test_dataset, args=args)

all_modes = ['bike', 'bus', 'drive', 'train', 'tram', 'walk']
mode_to_idx = {m: i for i, m in enumerate(all_modes)}

def get_modes_vec(modes, num_modes=None, union=True):
    """
    Parameters
    ----------
    modes : list of str
        List of transport modes.
    num_modes : int or None
        If provided and numeric, pad or truncate output dimension to this size.
    union : bool
        If True → return single vector (OR over modes).
        If False → return matrix (len(modes), dim) with one-hot rows.

    Returns
    -------
    np.ndarray
    """
    # Determine base dimension
    base_dim = len(all_modes)
    dim = base_dim

    if num_modes is not None:
        dim = int(num_modes)

    # Filter valid modes and get indices
    indices = [mode_to_idx[m] for m in modes if m in mode_to_idx]

    if union:
        vec = np.zeros(dim, dtype=float)
        for idx in indices:
            if idx < dim:  # avoid overflow if truncated
                vec[idx] = 1.0
        return vec

    elif union == False:
        mat = np.zeros((len(modes), dim), dtype=float)
        for row, mode in enumerate(modes):
            idx = mode_to_idx.get(mode, None)
            if idx is not None and idx < dim:
                mat[row, idx] = 1.0
        return mat
    else:
        # union is None: we sum the one-hot vectors without thresholding to get a count of modes
        vec = np.zeros(dim, dtype=float)
        for idx in indices:
            if idx < dim:  # avoid overflow if truncated
                vec[idx] += 1.0
        return vec
    


def fix_feat(x, dim=6):
    if x is None:
        return np.zeros(dim, dtype=np.float32)

    x = np.asarray(x, dtype=np.float32).flatten()

    if x.shape[0] < dim:
        x = np.pad(x, (0, dim - x.shape[0]))

    elif x.shape[0] > dim:
        x = x[:dim]

    return x

def get_modes_ohe(modes, num_modes, node=False):
    if not node:
        modes_vec = get_modes_vec(modes, num_modes=num_modes, union=False)
        return torch.tensor(modes_vec, dtype=torch.float)
    # nodes can have multiple modes, so we take the feasible modes for each node and do an OR operation
    node_modes_vec = []
    for node_modes in modes:
        if isinstance(node_modes, str):
            node_modes = [node_modes]
        modes_vec = get_modes_vec(node_modes, num_modes=num_modes, union=True)
        node_modes_vec.append(torch.tensor(modes_vec))
    node_modes_vec = torch.stack(node_modes_vec, dim=0)  # (L, num_modes)
    return node_modes_vec


def _lookup_index(mapping, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
        str_key = str(key)
        if str_key in mapping:
            return mapping[str_key]
    raise KeyError(f"Could not find any of these keys in index mapping: {keys}")


def get_node_indices(node_id_to_idx, nodes_seq):
    return [_lookup_index(node_id_to_idx, node_id) for node_id in nodes_seq]

def get_edge_indices(edge_id_to_idx, edges_seq, edge_modes_seq=None):
    if edge_modes_seq is None:
        return [_lookup_index(edge_id_to_idx, edge_id) for edge_id in edges_seq]

    return [
        _lookup_index(edge_id_to_idx, (edge_id, mode), f"{edge_id}_{mode}", edge_id)
        for edge_id, mode in zip(edges_seq, edge_modes_seq)
    ]

def get_cell_indices(cell_id_to_idx, cells_seq):
    return [_lookup_index(cell_id_to_idx, cell_id) for cell_id in cells_seq]
