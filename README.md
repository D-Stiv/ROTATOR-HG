# ROTATOR+HG

This repository contains the code accompanying the paper "ROTATOR+HG: Representation Learning for Intermodal Routes with Hypergraph".

## Overview

ROTATOR+HG is an intermodal route representation learning approach. It jointly captures high-order route dependencies, intermodal characteristics, and context-dependent semantics within a unified model based on hypergraph neural networks.

The codebase supports pretraining route embeddings and evaluating them in unsupervised, semi-supervised, and supervised settings.

## Repository Structure

```text
dataloader/     Dataset construction and data loader utilities
pretraining/    ROTATOR+HG model, losses, metrics, configuration, and training entry point
evaluation/     Embedding evaluation, fine-tuning heads, metrics, and evaluation entry point
```

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`torch-scatter` must match the installed PyTorch and CUDA versions. If the default installation fails, install the wheel recommended for your local PyTorch environment.

## Dataset

We generate synthetic routes over the intermodal graph $\mathcal{G}$ from the Wangerland region in Germany. The routes represent touristic day trips comprising 1 to 7 points of interest (e.g., restaurants, parks, museums, hotels), with an average duration of approximately 1.5 to 2 hours (max: around 4.5 hours). Travel distances range from a few meters to several kilometers.

The dataset will be made available upon paper publication.

## Data and Artifacts

The current code expects several external data and artifact paths to be configured before running pretraining or evaluation. These paths are marked in the source code with `Insert: ...` placeholders.

Expected inputs for dataset construction:

- Route split files: `train.pkl`, `val.pkl`, and `test.pkl`, loaded from the split-data directory configured in `dataloader/loader_utils.py`.
- Node and edge graph embeddings: a pickle file containing `node_id_to_idx`, `edge_id_to_idx`, `node_embeddings`, and `edge_embeddings`, configured in `dataloader/dataset.py`.
- Cell embeddings: a pickle file containing grid or cell embeddings, configured in `dataloader/dataset.py`.
- Node-to-cell mapping: a pickle file containing the node-to-cell integer mapping, configured in `dataloader/dataset.py`.
- Edge feature table: a pickle file with structural, scalar, categorical, and textual edge features, configured in `dataloader/dataset.py`.
- Node feature table: a pickle file with structural, scalar, categorical, and textual node features, configured in `dataloader/dataset.py`.

Expected outputs and reusable artifacts:

- Cached dataset objects are written to the dataset-cache directory configured in `dataloader/loader_utils.py`.
- Pretraining checkpoints and the best ROTATOR+HG model path are configured in `pretraining/main.py`.
- Evaluation loads the pretrained model arguments and checkpoint from the paths configured in `evaluation/eval_utils.py`.
- Supervised fine-tuning checkpoints are configured in `evaluation/eval_main.py` and `evaluation/finetune.py`.

## Spatial-Structural Embedding

Before context-aware learning, ROTATOR+HG uses spatial-structural embeddings to represent graph elements in the intermodal network.

Grid or cell embeddings are pretrained with node2vec, following Grover and Leskovec (2016). Node and edge embeddings are pretrained with a directed multigraph neural network, following Egressy et al. (2024). These embeddings are then consumed by the ROTATOR+HG pretraining pipeline as route-level input features.

## Pipeline

The ROTATOR+HG workflow follows this order:

1. Build or obtain the intermodal graph for the Wangerland region.
2. Generate route instances and split them into training, validation, and test sets.
3. Pretrain spatial-structural graph embeddings: node2vec for grid or cell embeddings, and directed multigraph GNN embeddings for graph nodes and edges.
4. Prepare node and edge feature tables, including structural, scalar, categorical, and textual features.
<!-- 5. Configure the data and artifact paths marked by `Insert: ...` placeholders in the source code. -->
5. Run ROTATOR+HG pretraining to learn context-aware route embeddings.
6. Evaluate the learned embeddings with unsupervised, semi-supervised, or supervised fine-tuning protocols.

## Pretraining

Pretraining configuration is defined in `pretraining/config.py`.

```bash
python -m pretraining.main --max_epochs 500 --patience 50 --batch_size 512
```

## Evaluation

Evaluation configuration is defined in `evaluation/eval_config.py`.

```bash
# Unsupervised and semi-supervised evaluation
python -m evaluation.eval_main --eval_type un_semi --batch_size 2048

# Supervised fine-tuning evaluation
python -m evaluation.eval_main --eval_type finetune-supervised --finetune_epochs 500 --finetune_patience 50 --batch_size 2048
```

## Citation

if you find this work relevant, please cite us:

```bibtex
@misc{gounoue2026rotatorhg,
  title = {ROTATOR+HG: Representation Learning for Intermodal Routes with Hypergraph},
  author = {Gounoue, Steve and Mann, Genivika and Dadwal, Rajjat and Demidova, Elena},
  year = {2026},
  note = {Manuscript in preparation}
}
```

<!-- Related methods:

```bibtex
@article{Grover2016node2vecSF,
  title = {node2vec: Scalable Feature Learning for Networks},
  author = {Grover, Aditya and Leskovec, Jure},
  journal = {Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining, KDD},
  year = {2016}
}

@inproceedings{DBLP:conf/aaai/EgressyNBAWA24,
  author = {Egressy, B{\'{e}}ni and von Niederh{\"{a}}usern, Luc and Blanusa, Jovan and Altman, Erik R. and Wattenhofer, Roger and Atasu, Kubilay},
  title = {Provably Powerful Graph Neural Networks for Directed Multigraphs},
  booktitle = {Proceedings of the 38th Conference on Artificial Intelligence},
  pages = {11838--11846},
  year = {2024},
  doi = {10.1609/AAAI.V38I10.29069}
}
``` -->

## License

This project is released under the license included in `LICENSE`.
