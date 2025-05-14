from pathlib import Path

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GATConv


ACTIVATION_DICT = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "none": None,
    "abs": torch.abs,
    "id": lambda x: x,
}

GNN_DICT = {
    "gcn": GCNConv,
    "gin": GINConv,
    "gat": GATConv,
    "gat_v2": GATv2Conv,
}

SCHEDULER_DICT = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "linear": LinearLR,
    "sequential": SequentialLR,
    "none": None,
}

# File locations
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = Path(__file__).parent.parent / "cache"
CHECKPOINT_DIR = Path(__file__).parent.parent / "cache" / "checkpoints"

DEMO_DATASET_CELL_TYPES = {
    "K562_single_cell_line": ("K562", None, "K562"),
    "K562_cross_cell_lines": ("all", None, "K562"),
}

PUBLIC_CELL_TYPES = [
    "K562",
    "RPE1",
    "hepg2",
    "jurkat",
]

PUBLIC_CELL_TYPE_DATASETS = {
    "K562": "replogle_k562_essential",
    "RPE1": "replogle_rpe1_essential",
    "hepg2": "nadig_hepg2",
    "jurkat": "nadig_jurkat",
}

GENE_NAME = "gene_name"
CONTROL = "control"
CELL_TYPE = "cell_line"
CONDITION_NAME = "condition_name"
CONDITION = "condition"
IS_CONTROL = "is_control"
BATCH = "batch"
BATCH_RAW = "batch_num"
CONTROL_LABEL = "ctrl"

TRAIN = "train"
TEST = "test"
VAL = "val"

FAST = "fast"
EXTENDED = "extended"
SLOW = "slow"

RES_PRED = "pred"
RES_TRUTH = "truth"
RES_PERT_CAT = "pert_cat"
RES_DOSE_CAT = "dose_cat"
RES_CELL_TYPE = "cell_type"
RES_EXP_BATCH = "experimental_batch"


class ObsmKey:
    RAW = "raw"
    SCGPT = "scgpt"
