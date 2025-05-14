from typing import Callable, Dict, List, Tuple
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import scipy
from sklearn.metrics import mean_squared_error as mse
from lightning import LightningModule
from collections import defaultdict
import anndata
from tqdm import tqdm

import gspp.constants as cs


# Pre-cache means of control, perts
# Global dictionary to store mean control values
CELL_TYPE_MEAN_CONTROL: Dict[str, np.ndarray] = {}
PERT_MEANS: Dict[str, pd.DataFrame] = {}


# TODO, the gloabl mean should be phased-out, but before doing so having 1:1 numbers
# from control matched vs global control is interesting; hence leaving for the moment
def compute_mean_control(adata, cell_type: str):
    """calculates and caches the mean control for each batch

    Args:
        adata: anndata.AnnData with obs values of 'condition' where 'ctrl' marks controls,
            'batch' for experimental batches, and 'cell_line' for the biological context (cell type, line, etc..)
        cell_type: string by which to subset biological context

    Mutates:
        CELL_TYPE_MEAN_CONTROL: Dict[str, pd.core.frame.DataFrame)]
            {cell_type: df of control means, batch X gene}

    Returns:
        None
    """
    if cell_type not in CELL_TYPE_MEAN_CONTROL:
        temp_adata = adata[adata.obs[cs.CELL_TYPE] == cell_type]
        # setup to group controls by batch
        ctrl_mask = temp_adata.obs[cs.CONDITION] == cs.CONTROL_LABEL
        tmpx = temp_adata[ctrl_mask].X
        if isinstance(tmpx, scipy.sparse._csr.csr_matrix):
            tmpx = tmpx.toarray()
        tmpx = pd.DataFrame(tmpx)
        ctrls = tmpx.mean(0)

        tmpx["group"] = temp_adata.obs.loc[ctrl_mask, cs.BATCH].values.astype(str)
        # take the control mean for each batch
        batch_ctrls = tmpx.groupby("group").apply(lambda x: x.iloc[:, :-1].mean(0))

        CELL_TYPE_MEAN_CONTROL[cell_type] = (ctrls, batch_ctrls)


def cache_perturbation_means(adata, cell_type):
    """setup df with index pert and mean expression for each pert
    pert names are in 'pert_raw' format, e.g. gene symbols and do not contail +ctrl, such that they can be easily
    matched to pert passed to metric functions

    Args:
        adata: anndata object, with obs columns 'condition' marking unique perturbations, 'control' (mask) and 'cell_type'
        cell_type: cell type to use from adata
    """

    if cell_type not in PERT_MEANS:
        temp_adata = adata[adata.obs[cs.CELL_TYPE] == cell_type].copy()
        temp_adata.obs["pert_raw"] = [
            "+".join([p for p in perts.split("+") if p != "ctrl"])
            for perts in temp_adata.obs.condition
        ]
        # work on a non-sparse copy of X
        newx = temp_adata.X.copy()
        if isinstance(newx, scipy.sparse._csr.csr_matrix):
            newx = newx.toarray()
        # we want the perturbation deltas vs the closest control, so we will center on batches (i.e. subtract mean)
        control_mask = temp_adata.obs[cs.CONTROL].astype(bool).values
        for batch in temp_adata.obs[cs.BATCH].unique():
            batch_mask = temp_adata.obs[cs.BATCH] == batch
            # mean center in place
            control_mean = newx[batch_mask & control_mask].mean(0)
            newx[batch_mask] -= control_mean
        temp_adata.X = newx  # set centered values as X

        # mean aggregate the perturbation deltas
        temp_adata = temp_adata[~control_mask]
        means_x = sc.get.aggregate(temp_adata, by="pert_raw", axis=0, func=["mean"])
        PERT_MEANS[cell_type] = pd.DataFrame(
            means_x.layers["mean"], index=means_x.obs["pert_raw"]
        )


def cache_de_idx(adata):
    """setup {pert: de_idx} cache if not present"""
    if "de_idx_cache" not in adata.uns:
        res = {}
        gmap = {gene: i for i, gene in enumerate(adata.var_names)}
        for pert in adata.uns["rank_genes_groups_cov_all"]:
            res[pert] = np.array(
                [
                    gmap[gene]
                    for gene in adata.uns["rank_genes_groups_cov_all"][pert]
                    if gene in gmap
                ]
            )
        res[cs.CONTROL_LABEL] = np.array([-1] * len(res[pert]))
        adata.uns["de_idx_cache"] = res


# Step 1: Metric Function Registry
METRIC_REGISTRY: Dict[
    str, Callable[[np.ndarray, np.ndarray, str, np.ndarray], float]
] = {}

TRAINING_METRIC_LIST: List[str] = []
TESTING_METRIC_LIST: List[str] = []

SLOW_METRIC_LIST: List[str] = []


def register_metric(
    name: str, training: bool = False, testing: bool = False, slow: bool = True
):
    """Decorator to register a metric function in the METRIC_REGISTRY."""

    def decorator(
        func: Callable[[np.ndarray, np.ndarray, str, np.ndarray], float],
        training=training,
        testing=testing,
    ):
        METRIC_REGISTRY[name] = func
        if training:
            TRAINING_METRIC_LIST.append(name)
        if testing:
            TESTING_METRIC_LIST.append(name)
        if slow:
            SLOW_METRIC_LIST.append(name)
        return func

    return decorator


@register_metric("pearson", training=False, slow=True)
def pearson(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    p = pearsonr(pred.mean(0), truth.mean(0))[0]
    if np.isnan(p):
        return 0
    return p


@register_metric("pearson_sample", slow=True)
def pearson_sample(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    p = np.mean(
        [pearsonr(pred[idx, :], truth[idx, :])[0] for idx in range(pred.shape[0])]
    )
    if np.isnan(p):
        return 0
    return p


@register_metric("r2", training=False, slow=True)
def r2(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    p = r2_score(pred.mean(0), truth.mean(0))
    if np.isnan(p):
        return 0
    return p


@register_metric("r2_sample", slow=True)
def r2_sample(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    p = np.mean([r2_score(pred[idx, :], truth[idx, :]) for idx in range(pred.shape[0])])
    if np.isnan(p):
        return 0
    return p


@register_metric("mse", training=False, slow=True)
def mse_metric(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    return mse(pred, truth)


@register_metric("pearson_delta", training=True, testing=True)
def pearson_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    val = pearsonr((pred - ctrl).mean(0), (truth - ctrl).mean(0))[0]
    if np.isnan(val):
        val = 0
    return val


@register_metric("spearman_delta", training=False, slow=True)
def spearman_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    val = spearmanr((pred - ctrl).mean(0), (truth - ctrl).mean(0))[0]
    if np.isnan(val):
        val = 0
    return val


@register_metric("pearson_delta_sample", slow=True)
def pearson_delta_sample(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    val = np.mean(
        [
            pearsonr((pred[idx, :] - ctrl[idx, :]), (truth[idx, :] - ctrl[idx, :]))[0]
            for idx in range(pred.shape[0])
        ]
    )
    if np.isnan(val):
        val = 0
    return val


@register_metric("r2_delta", slow=True)
def r2_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    val = r2_score((truth - ctrl).mean(0), (pred - ctrl).mean(0))
    if np.isnan(val):
        val = 0
    return val


@register_metric("r2_delta_sample", slow=True)
def r2_delta_sample(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    val = np.mean(
        [
            r2_score(truth[idx, :] - ctrl[idx, :], pred[idx, :] - ctrl[idx, :])
            for idx in range(pred.shape[0])
        ]
    )
    if np.isnan(val):
        val = 0
    return val


class BreakdownAssayedMetric:
    def __init__(self, metric_name: str, *args, **kwargs):
        self.metric_name = metric_name

    def mask(self, cell_type, adata, pert, dose: str, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        cell_type: str,
        ctrl: np.ndarray,
        adata: anndata.AnnData,
        pert: str,
        dose: str = "1+1",
    ) -> float:
        mask = self.mask(cell_type, adata, pert, dose)

        if mask is None:
            return np.nan

        subpred = pred[:, mask]
        subtruth = truth[:, mask]
        subctrl = ctrl[:, mask]
        return METRIC_REGISTRY[self.metric_name](
            subpred, subtruth, cell_type, subctrl, adata, pert, dose
        )


class PharosBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, kl: int, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)
        self.kl = kl

    def mask(self, cell_type, adata, pert, dose: str):
        return adata.uns["pharos"] == self.kl


# slight special handling for pharos breakdowns
pharoses = {
    f"pearson_delta_kl{kl}": PharosBreakdown("pearson_delta", kl) for kl in range(4)
}
for key, val in pharoses.items():
    register_metric(key, slow=True)(val)


class DEBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, num_de_genes: int, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)
        self.num_de_genes = num_de_genes

    def mask(self, cell_type, adata, pert, dose: str):
        cache_de_idx(adata)  # setup cache if not present

        perts = f"{cell_type}_{pert}_{dose}"

        if perts in adata.uns["de_idx_cache"]:
            return adata.uns["de_idx_cache"][perts][: self.num_de_genes]
        else:
            return None


# slight special handling for DE breakdowns
de_breakdowns = {
    f"pearson_delta_de{num_de_genes}": DEBreakdown("pearson_delta", num_de_genes)
    for num_de_genes in [20, 100]
}


for key, val in de_breakdowns.items():
    register_metric(key, slow=True)(val)


class RetrievalMetric:
    def __init__(self, subset=False, *args, **kwargs):
        self.subset = subset

    def __call__(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        cell_type: str,
        ctrl: np.ndarray,
        adata: anndata.AnnData,
        pert: str,
        dose: str = "1+1",
    ) -> float:
        """relative retrieval rank of true reference match for pert, to pred of pert, higher is better, uses pearson_delta"""
        pert = "+".join([p for p in pert.split("+") if p != "ctrl"])

        truth_deltas = PERT_MEANS[cell_type]
        if self.subset:
            state = np.random.get_state()
            np.random.seed(0)
            n = min(truth_deltas.shape[0], 100)
            samp = np.random.choice(
                list(range(len(truth_deltas.index))), n, replace=False
            )
            np.random.set_state(state)
            truth_index = truth_deltas.index[samp].tolist()
            if pert not in truth_index:
                truth_index.append(pert)
            truth_deltas = truth_deltas.loc[truth_index]

        sims = []
        assert pert in truth_deltas.index
        # calculate similarity of pred to everything in truth_means
        for item in truth_deltas.index:
            sims.append(
                (pearsonr((pred - ctrl).mean(0), truth_deltas.loc[item])[0], item)
            )

        sims = sorted(sims)  # sort by similarity
        rank = [sim[1] for sim in sims].index(pert)  # find rank of query pert
        return rank / len(sims)


register_metric("normed_retrieval", slow=True)(RetrievalMetric(subset=False))

register_metric("fast_retrieval", testing=True)(RetrievalMetric(subset=True))


class GraphConnectivityBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)

    def mask(self, cell_type, adata, pert, dose: str):
        pass


class GraphProximityBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)

    def mask(self, cell_type, adata, pert, dose: str):
        pass


def metrics_calculation(
    results: dict, metrics_list: List[str], adata, match_cntr: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """
    Generates generic metrics for a given results.
    Args:
            results (dict): {
                    pred: predictions,
                    truth: ground truth,
                    pred_cat: condition categories,
                    pred_de: differentially expressed genes,
                    truth_de: differentially expressed genes,
                    cell_type: cell types,
                    pert_cat: perturbation categories,
                }
            metrics_list (list of str): List of metric names to compute.
            adata: AnnData object containing the dataset.
            match_cntr: bool, whether to match control
    Returns:
            metrics (dict): {metric_name: metric_value}
            metrics_pert (dict): {perturbation: {metric_name: metric_value}}
    """
    metrics: Dict[str, List[float]] = {}
    metrics_pert: Dict[str, Dict[str, float]] = {}
    metrics = defaultdict(list)
    metrics_pert = defaultdict(lambda: defaultdict(list))

    ct_mask_dict: Dict[str, np.ndarray] = {
        cell_type: results[cs.RES_CELL_TYPE] == cell_type
        for cell_type in np.unique(results[cs.RES_CELL_TYPE])
    }

    for metric_name in metrics_list:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric_name}' is not registered.")
        metric_fn = METRIC_REGISTRY[metric_name]
        print(f"running {metric_name}")
        for pert in tqdm(np.unique(results[cs.RES_PERT_CAT])):
            for dose in np.unique(results[cs.RES_DOSE_CAT]):
                p_d_mask = np.logical_and(
                    results[cs.RES_PERT_CAT] == pert, results[cs.RES_DOSE_CAT] == dose
                )

                if np.sum(p_d_mask) == 0:
                    continue

                for cell_type in np.unique(results[cs.RES_CELL_TYPE]):

                    ct_mask = ct_mask_dict[cell_type]
                    p_d_ct_mask = p_d_mask & ct_mask

                    # If this cell type + perturbation combination has no data, skip
                    if np.sum(p_d_ct_mask) == 0:
                        continue

                    # Compute mean of control and each pert if not already cached
                    compute_mean_control(adata, cell_type)
                    cache_perturbation_means(adata, cell_type)
                    if match_cntr:
                        control = (
                            CELL_TYPE_MEAN_CONTROL[cell_type][1]
                            .loc[results[cs.RES_EXP_BATCH][p_d_ct_mask].astype(str), :]
                            .to_numpy()
                        )
                    else:
                        control = (
                            CELL_TYPE_MEAN_CONTROL[cell_type][0]
                            .to_numpy()
                            .reshape(1, -1)
                        )

                    assert (
                        control.shape[0] > 0
                    ), f"No control found for {cell_type} x {pert} x {dose}"

                    # Calculate primary metric
                    value = metric_fn(
                        results[cs.RES_PRED][p_d_ct_mask],
                        results[cs.RES_TRUTH][p_d_ct_mask],
                        cell_type,
                        control,
                        adata,
                        pert,
                        dose,
                    )
                    value = 0 if np.isnan(value) else value
                    metrics_pert[pert][metric_name].append(value)
                    metrics[metric_name].append(value)

            metrics_pert[pert][metric_name] = np.mean(metrics_pert[pert][metric_name])

        metrics[metric_name] = np.mean(metrics[metric_name])

    return metrics, metrics_pert


def log_metrics(
    metrics: dict, metrics_pert: dict, stage: str, subgroup: dict, lm: LightningModule
):
    """
    Logs the metrics in the metrics dictionary
    Args:
            metrics (dict): {metric_name: metric_value}
            metrics_pert (dict): {perturbation: {metric_name: metric_value}}
            stage (str): Stage of the experiment (train, val, test)
            subgroup (dict): Dictionary containing the subgroup analysis
            lm (LightningModule): Lightning Module object
    """
    for metric_name, metric_value in metrics.items():
        log_key = f"{stage}_{metric_name}"
        print(f"{log_key}: {metric_value}")
        lm.log(log_key, metric_value)

    if stage == cs.TEST:
        subgroup_analysis = {}
        for name in subgroup["test_subgroup"].keys():
            subgroup_analysis[name] = {}
            for m in list(list(metrics_pert.values())[0].keys()):
                subgroup_analysis[name][m] = []

        for name, pert_list in subgroup["test_subgroup"].items():
            for pert in pert_list:
                for m, res in metrics_pert[pert].items():
                    if pert in metrics_pert.keys():
                        subgroup_analysis[name][m].append(res)

        for name, result in subgroup_analysis.items():
            for m in result.keys():
                subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                lm.log("test_" + name + "_" + m, subgroup_analysis[name][m])


def compute_metrics(results, adata, metric_stage, match_cntr: bool = True):
    """
    Given results from a model run and the ground truth, and the stage as 'fast' or 'extended': compute metrics
    """
    if metric_stage == cs.FAST:
        metrics_list = TRAINING_METRIC_LIST
    elif metric_stage == cs.EXTENDED:
        metrics_list = TESTING_METRIC_LIST
    elif metric_stage == cs.SLOW:
        metrics_list = TESTING_METRIC_LIST + SLOW_METRIC_LIST
    else:
        raise ValueError(f"Invalid metric stage: {metric_stage}")

    metrics_list = list(set(metrics_list))

    metrics, metrics_pert = metrics_calculation(
        results, metrics_list, adata, match_cntr
    )
    return metrics, metrics_pert
