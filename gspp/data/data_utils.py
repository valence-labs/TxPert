from typing import Dict, Union
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm

import torch
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from loguru import logger

import gspp.constants as cs

warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stdout, level="INFO")

sc.settings.verbosity = 0


@dataclass
class Payload:
    """
    Dataclass for collate function
    """

    x: torch.Tensor
    control: torch.Tensor
    p: torch.Tensor
    pert_idxs: pd.Series
    pert_cond_names: pd.Series
    pert_dose: pd.Series
    idx: pd.Series
    cell_types: pd.Series
    experimental_batches: pd.Series


def collate_fn(batch) -> Payload:
    """
    Custom collate function for DataLoader

    Args:
        batch: Batch of data

    Returns:
        Payload: Collated data
    """
    (
        x,
        control,
        p,
        pert_idxs,
        pert_cond_names,
        pert_dose,
        idx,
        cell_types,
        experimental_batches,
    ) = zip(*batch)
    return Payload(
        x=torch.stack(x),
        control=torch.stack(control),
        p=torch.stack(p),
        pert_idxs=pert_idxs,
        pert_cond_names=pert_cond_names,
        pert_dose=pert_dose,
        idx=idx,
        cell_types=cell_types,
        experimental_batches=experimental_batches,
    )


def add_huvec_metadata(adata: sc.AnnData) -> None:
    """
    Add metadata to adata object for HUVEC dataset

    Args:
        adata: AnnData object
    """
    metadata_base_path = Path(
        "/mnt/ps/home/CORP/shawn.whitfield/project/outgoing/data/perturbseq/internal/"
    )
    obs = pd.read_parquet(metadata_base_path / "internal_huvec_raw_obs.parquet")
    var = pd.read_parquet(metadata_base_path / "internal_huvec_raw_var.parquet")
    adata.obs = obs
    adata.var = var


def calculate_de_for_eval(adata: sc.AnnData) -> sc.AnnData:
    """
    Calculate DEGs for evaluation

    Args:
        adata: AnnData object

    Returns:
        AnnData: Updated AnnData object with DEGs
    """
    adata = get_DE_genes(adata, skip_calc_de=False)
    adata = get_dropout_non_zero_genes(adata)
    return adata


def get_DE_genes(adata: sc.AnnData, skip_calc_de: bool = False) -> sc.AnnData:
    """
    Get DE genes

    Args:
        adata: AnnData object
        skip_calc_de: Whether to skip DE calculation

    Returns:
        AnnData: Updated AnnData object with DE genes
    """
    if not skip_calc_de:
        _rank_genes_groups_by_cov(
            adata,
            groupby=cs.CONDITION_NAME,
            covariate=cs.CELL_TYPE,
            control_group=f"{cs.CONTROL_LABEL}_1",
            n_genes=len(adata.var),
            key_added="rank_genes_groups_cov_all",
        )
    return adata


def get_dropout_non_zero_genes(adata: sc.AnnData) -> sc.AnnData:
    """
    Get dropout and non-zero genes

    Args:
        adata: AnnData object

    Returns:
        AnnData: Updated AnnData object with dropout and non-zero genes
    """
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(
        len(adata.obs.condition.unique()), adata.X.toarray().shape[1]
    )
    ctrl = mean_expression[np.where(pert_list == cs.CONTROL_LABEL)[0]]

    # in silico modeling and upperbounding
    pert_full_id2pert = dict(adata.obs[[cs.CONDITION_NAME, cs.CONDITION]].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in tqdm(adata.uns["rank_genes_groups_cov_all"].keys()):
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis=0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns["rank_genes_groups_cov_all"][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)

    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))

    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20

    return adata


def _rank_genes_groups_by_cov(
    adata: sc.AnnData,
    groupby: str,
    control_group: str,
    covariate: str,
    n_genes: int = 50,
    rankby_abs: bool = True,
    key_added: str = "rank_genes_groups_cov",
    return_dict: bool = False,
) -> Union[Dict, None]:
    """
    Rank genes groups by covariate

    Args:
        adata: AnnData object
        groupby: groupby column
        control_group: control group
        covariate: covariate column
        n_genes: number of genes
        rankby_abs: rank by absolute value
        key_added: key added
        return_dict: return dictionary
    """
    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        min_samples = 2
        valid_groups = (
            adata_cov.obs[groupby].value_counts()[lambda x: x >= min_samples].index
        )
        adata_cov = adata_cov[adata_cov.obs[groupby].isin(valid_groups)]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict
    return None
