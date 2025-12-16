import numpy as np
import scanpy as sc
from tqdm import tqdm


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