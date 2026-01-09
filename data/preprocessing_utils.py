import numpy as np
import scanpy as sc
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def filter_cells_by_pert_effect(
    adata: sc.AnnData, k: int = 10
) -> tuple[list[int], sc.AnnData]:
    """
    Filter cells by perturbation effect.

    Args:
        adata: AnnData object
        k: Percentile threshold

    Returns:
        Tuple: List of cells under threshold and filtered AnnData object
    """
    perc_underk = []
    subset_idxs = []
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]

    gene_names = adata.obs["condition"].unique()
    gene_locs = {
        g: np.where(adata.var.gene_name == g)[0][0]
        for g in gene_names
        if g in adata.var.gene_name.values
    }
    ctrl_expr = ctrl_adata.X.toarray()
    thresh = np.percentile(ctrl_expr, k, axis=0)

    print("filtering cells by pert effect")
    for g in tqdm(gene_names):
        subset = adata[adata.obs["condition"] == g]

        if g == "ctrl":
            subset_idxs.append(subset.obs.index.values)
            continue

        if g in gene_locs:
            gene_loc = gene_locs[g]
            subset_expr = subset.X[:, gene_loc].toarray().flatten()
            perc_underk.append(np.sum(subset_expr > thresh[gene_loc]))
            subset_idxs.append(subset.obs.index[subset_expr <= thresh[gene_loc]].values)

        else:
            subset_idxs.append(subset.obs.index.values)

    subset_idxs = [item for sublist in subset_idxs for item in sublist]
    filtered_adata = adata[subset_idxs, :].copy()

    return perc_underk, filtered_adata


def define_splits_singles(conditions: list[str], 
                          test_size: float = 0.25, 
                          val_size: float = 0.1,
                          random_state: int = 42) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """
    Define train, test, val splits; and a place holder for the subgroup
    """

    train_val, test = train_test_split(sorted(conditions), test_size=test_size, random_state=random_state)
    train, val = train_test_split(sorted(train_val), test_size=val_size / (1 - test_size), random_state=random_state)
    train_test_val = {"train": train, "test": test, "val": val}

    # use same placeholder keys as for doubles / from gears standard
    subgroup = {"test_subgroup": {"combo_seen0": [], 
                                  "combo_seen1": [], 
                                  "combo_seen2": [], 
                                  "unseen_single": test},
                "val_subgroup": {"combo_seen0": [], 
                                 "combo_seen1": [], 
                                 "combo_seen2": [], 
                                 "unseen_single": val}}

    return train_test_val, subgroup
