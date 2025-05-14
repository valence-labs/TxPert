from typing import Union, List, Dict

from tqdm import tqdm

from loguru import logger

import random
import torch
import numpy as np

from gspp.data.datamodule import XcelltypePerturbDataset
from gspp.data.datamodule import PertDataModule


class PertBaseline:
    def __init__(self):
        """
        Base class for perturbation prediction baselines
        """
        logger.info("Instatiating Baseline")

    def parse_inputs(self, datamodule: PertDataModule):
        """
        Parse the inputs for the prepare_baseline funciton from the datamodule
        """
        raise NotImplementedError

    def prepare_baseline(self, **kwargs):
        """
        Prepare the baseline
        """
        raise NotImplementedError

    def apply_baseline(self, test_ds: XcelltypePerturbDataset):
        """
        Apply the baseline to the test data
        """
        raise NotImplementedError


class MeanBaseline(PertBaseline):
    def __init__(
        self,
        global_mean: bool = False,
        global_double: bool = False,
    ):
        """
        Simple baseline that predicts for each test sample the mean of the test control samples PLUS the
        - Mean delta of the applied perturbations (rel. to the control; grouped by cell-type) in training data IF the perturbation was seen in training
        - Mean delta of all perturbations (rel. to the control; grouped by cell-type) in training data IF the perturbation was not seen in training

        Args:
            global_mean (bool, optional): Whether to use the global mean of the training data instead of local (perturbation-specific) deltas
            global_double (bool, optional): Whether to use the global mean of the training data (doubles) for double perturbations instead additive global deltas
        """
        self.global_mean = global_mean
        self.global_double = global_double

        logger.info("Instatiating General Baseline")

    def parse_inputs(
        self,
        datamodule: PertDataModule,
    ):
        """
        Parse the inputs for the prepare_baseline funciton from the datamodule
        """
        return {
            "pert2id": datamodule.pert2id,
            "test_cell_type": datamodule.test_cell_type, 
            "train_cell_types": datamodule.train_cell_types,
            "train_ds": datamodule.train_data,
            "test_ds": datamodule.test_data,
            "output_dim": datamodule.output_dim
        }

    def prepare_baseline(
        self,
        pert2id: Dict[str, int],
        test_cell_type: str,
        train_cell_types: List[str],
        train_ds: XcelltypePerturbDataset,
        test_ds: XcelltypePerturbDataset,
        output_dim: int,
    ):
        """
        Prepare the baseline by computing the deltas for each cell-type x batch x perturbation combination in the training data OR computing the global deltas per cell-type if the perturbation was not seen in training

        Args:
            pert2id (Dict[str, int]): A dictionary mapping perturbations to their IDs
            test_cell_type (str): The cell-type of the test data
            train_cell_types (List[str]): The cell-types in the training data
            train_ds (XcelltypePerturbDataset): The training dataset
            test_ds (XcelltypePerturbDataset): The test dataset
            output_dim (int): The output dimension of the model
        """
        logger.info("Preparing baseline")

        self.pert2id = pert2id
        self.output_dim = output_dim

        test_perts = [[p for p in perts if p != -1] for perts in test_ds.pert_conditions]
        test_perts = [tuple(p) for p in test_perts if p != ["ctrl"]]

        unique_test_perts = list(set(test_perts))

        train_perts = [[p for p in perts if p != -1] for perts in train_ds.pert_conditions]

        single_pert_indices = [idx for idx, perts in enumerate(train_perts) if len(perts) == 1 and perts != ["ctrl"]]
        double_pert_indices = [idx for idx, perts in enumerate(train_perts) if len(perts) == 2]

        train_perts = [tuple(p) for p in train_perts if p != ["ctrl"]]

        unique_train_perts = list(set(train_perts))

        seen_perts =  [p for p in unique_train_perts if p in unique_test_perts or (len(p) == 1 and any([p[0] in pert for pert in unique_test_perts]))]

        # Compute mean control of the test cell-types
        self.test_cntr_mean = {
            ct: test_ds.control_data[test_ds.control_cell_types == ct][:, :self.output_dim].mean(0)
            for ct in test_ds.control_cell_types.unique()
        }

        # The below list and dictionaries will contain the building blocks for the baseline
        self.global_delta = {}
        self.delta_dict = {cell_type: {} for cell_type in train_cell_types}

        # This is the training data (notably also containing controls that we need to filter out)
        train_perturbed = train_ds.pert_data[:, :self.output_dim]

        cntr_mean_dict = {}
        for ct in train_cell_types:
            cntr_mean_dict[ct] = {}

            for b in train_ds.control_ct_batch_indices[ct].keys():
                cntr_mean_dict[ct][b] = train_ds.control_data[train_ds.control_ct_batch_indices[ct][b],:].mean(0)

        deltas = []
        
        # Derive delta for each sample in the training data
        for idx, row in enumerate(train_perturbed):
            c = train_ds.pert_conditions.iloc[idx]
            ct = train_ds.treatment_cell_types.iloc[idx]
            b = train_ds.treatment_cell_batches.iloc[idx]
            if c == ["ctrl"]:
                deltas.append(torch.zeros_like(row))
            else:
                deltas.append(row - cntr_mean_dict[ct][b])

        deltas = torch.stack(deltas)
        
        # Compute deltas separately for each training cell-type
        for ct in train_cell_types:
            logger.info(f"Computing heuristic for {ct}")

            self.global_delta[ct] = {}

            # Identify control and cell-type indices and remove the control indices
            cntr_indices = np.where(train_ds.pert_conditions.isin([["ctrl"]]))[0]
            ct_indices = np.where(train_ds.all_cell_types == ct)[0]
            ct_indices = np.setdiff1d(ct_indices, cntr_indices)

            for order in ["single", "double"]:
                # intersection
                ct_order_indices = np.intersect1d(ct_indices, single_pert_indices if order == "single" else double_pert_indices)

                # Compute global deltas
                self.global_delta[ct][order] = (
                    deltas[ct_order_indices].mean(0), len(ct_order_indices)
                )

            # Now computing the same per perturbation (if seen in training)
            for p in tqdm(seen_perts):
                if len(p) == 1:
                    pert_indices = np.where(
                        [tuple([p[0], -1]) == tuple(el) or tuple([-1, p[0]]) == tuple(el) for el in train_ds.pert_conditions]
                    )[0]
                elif len(p) == 2:
                    pert_indices = np.where(
                        [p == tuple(el) for el in train_ds.pert_conditions]
                    )[0]
                else:
                    raise ValueError(
                        f"Perturbation {p} has more than 2 perturbations; not supported at the moment"
                    )

                # Identify indices per perturbation and cell-type
                intersection_p = np.intersect1d(ct_indices, pert_indices)
                p_sample_count = len(intersection_p)

                # Skip if perturbation is not in this cell-type
                if len(intersection_p) == 0:
                    continue

                # Compute mean perturbation delta
                self.delta_dict[ct][p] = (
                    deltas[intersection_p].mean(0), p_sample_count
                )

    def apply_baseline(self, test_ds: XcelltypePerturbDataset):
        """
        Apply the baseline to the test data

        Args:
            test_ds (XcelltypePerturbDataset): The test dataset
        """
        logger.info("Applying baseline")

        # This list contains everything we need "[cell_type]_[perts]_[dose]" (except for the control/perturbed gene expr.)
        conditions = test_ds.condition_names

        # The below lists will contain the information required in PertPredictor.general_validation_epoch
        pert_cats = []
        dose_cats = []
        cell_types = []
        preds = []
        truths = []
        batch_cats = []

        for idx, c in enumerate(tqdm(conditions)):
            ct, perts, dose = c.split("_")
            cell_types.append(ct)
            pert_cats.append(perts)
            batch_cats.append(test_ds.treatment_cell_batches.iloc[idx])
            dose_cats.append(dose)
            perts = [p for p in perts.split("+") if p != "ctrl"]
            pert_indices = [self.pert2id[p] for p in perts]

            double_deltas = []
            individual_deltas = {}

            # First, check if pert is a double that has been observed
            if len(pert_indices) == 2 and any(
                [tuple(pert_indices) in val.keys() for val in self.delta_dict.values()]
            ) and not self.global_mean:
                # We combine the knowledge from all training cell-types that are available (will be averaged out in the end)
                for val in self.delta_dict.values():
                    if tuple(pert_indices) in val.keys():
                        double_deltas.append(val[tuple(pert_indices)])

            # Next, check if we want to use a global mean based exclusively on doubles in the training data
            elif len(pert_indices) == 2 and self.global_mean and self.global_double:
                # We combine the knowledge from all training cell-types that are available (will be averaged out in the end)
                for val in self.global_delta.values():
                    double_deltas.append(val["double"])

            else:
                # Loop over indiviual perturbations that were applied in sequence
                for sub_idx, p_idx in enumerate(pert_indices):
                    individual_deltas[sub_idx] = []
                    # Use pert-specific delta if available in at least one training cell-type, otherwise revert to global delta
                    if any(
                        [tuple([p_idx]) in val.keys() for val in self.delta_dict.values()]
                    ) and not self.global_mean:
                        # We combine the knowledge from all training cell-types that are available (will be averaged out in the end)
                        for val in self.delta_dict.values():
                            if tuple([p_idx]) in val.keys():
                                individual_deltas[sub_idx].append(val[tuple([p_idx])])
                    
                    else:
                        # We combine the knowledge from all training cell-types that are available (will be averaged out in the end)
                        for val in self.global_delta.values():
                            individual_deltas[sub_idx].append(val["single"])

            assert len(double_deltas) > 0 or any([len(deltas) > 0 for deltas in individual_deltas.values()])

            # Adding perturbation effect to mean control; averaging out the deltas weighted by sample count per cell-type
            y = test_ds._get_control_data(idx)

            # Use double delta if available
            if len(double_deltas) > 0:
                y += sum([d[0] * d[1] for d in double_deltas]) / max(sum([d[1] for d in double_deltas]), 1)

            # Otherwise use individual deltas (starting with first part of pert)
            if 0 in individual_deltas.keys():
                y += sum([d[0] * d[1] for d in individual_deltas[0]]) / max(sum([d[1] for d in individual_deltas[0]]), 1)
            
            # If applicable, use second part of pert
            if 1 in individual_deltas.keys():
                y += sum([d[0] * d[1] for d in individual_deltas[1]]) / max(sum([d[1] for d in individual_deltas[1]]), 1)

            preds.append(y)

            truths.append(test_ds.pert_data[idx, :self.output_dim])

        results = {}
        results["pert_cat"] = np.array(pert_cats)
        results["dose_cat"] = np.array(dose_cats)
        results["cell_type"] = np.array(cell_types)
        results["pred"] = torch.stack(preds).cpu().numpy()
        results["truth"] = torch.stack(truths).cpu().numpy()
        results["experimental_batch"] = np.array(batch_cats)

        return results


class ExperimentalAccuracy(PertBaseline):
    def __init__(
        self,
        test_seen: Union[int, float] = 0.5,
        with_replacement: bool = False,
        aggregate_batches: bool = True,
    ):
        """
        Simple baseline that helps to estimate the experimental accuracy of a scRNA-seq assay/dataset using a subsample for each perturbation of the test data

        Args:
            test_seen (Union[int, float], optional): The number of samples seen per perturbation; fraction if float (0 < x < 1) and total number if int (x >= 1)
            with_replacement (bool, optional): Whether to sample with replacement
            aggregate_batches (bool, optional): Whether to aggregate across batch effects
        """
        logger.info("Instatiating Experimental Reproducibility")

        self.test_seen = test_seen
        self.with_replacement = with_replacement
        self.aggregate_batches = aggregate_batches

    def parse_inputs(
        self,
        datamodule: PertDataModule,
    ):
        """
        Parse the inputs for the prepare_baseline funciton from the datamodule
        """
        return {
            "test_cell_type": datamodule.test_cell_type,
            "train_cell_types": datamodule.train_cell_types,
            "test_ds": datamodule.test_data,
            "output_dim": datamodule.output_dim
        }

    def prepare_baseline(
        self,
        test_cell_type: str,
        train_cell_types: List[str],
        test_ds: XcelltypePerturbDataset,
        output_dim: int,
    ):
        """
        Prepare the baseline

        Args:
            test_cell_type (str): The cell-type of the test data
            train_cell_types (List[str]): The cell-types in the training data
            test_ds (XcelltypePerturbDataset): The test dataset
            output_dim (int): The output dimension of the model
        """
        logger.info("Preparing baseline")
        
        self.output_dim = output_dim

        # Determine set of test cell-type x perturbations combinations in test set
        test_perts = [[p for p in perts if p != -1] for perts in test_ds.pert_conditions]
        test_perts = [tuple(p) for p in test_perts]
    
        unique_test_perts = list(set(test_perts))
        unique_test_batches = list(set(test_ds.treatment_cell_batches))

        test_pert_list = [tuple([p for p in perts if p != -1]) for perts in test_ds.pert_conditions]

        self.pert2samples = {p: {} for p in unique_test_perts}

        # For each unique perturbation...
        for p in tqdm(unique_test_perts):
            # Find indices/rows associated to this perturbation in the test data
            p_indices = [idx for idx, el in enumerate(test_pert_list) if tuple(el) == tuple(p)]
            
            for b in unique_test_batches:
                b_indices = [idx for idx, el in enumerate(test_ds.treatment_cell_batches) if el == b]

                combined_indices = np.intersect1d(p_indices, b_indices).tolist()
                
                if len(combined_indices) == 0:
                    continue

                # Determine absolute number of samples
                sample_count = (
                    int(self.test_seen * len(combined_indices))
                    if self.test_seen < 1
                    else int(self.test_seen)
                )
                # Always use at least one sample
                sample_count = max(1, sample_count)

                # Randomly sample indices
                self.pert2samples[p][b] = random.sample(combined_indices, sample_count)

    def apply_baseline(self, test_ds: XcelltypePerturbDataset):
        """
        Apply the baseline to the test data

        Args:
            test_ds (XcelltypePerturbDataset): The test dataset
        """
        logger.info("Applying baseline")

        # This list contains everything we need "[cell_type]_[perts]_[dose]" (except for the control/perturbed gene expr.)
        conditions = test_ds.condition_names

        # The below lists will contain the information required in PertPredictor.general_validation_epoch
        pert_cats = []
        dose_cats = []
        cell_types = []
        preds = []
        truths = []
        batch_cats = []

        all_seen = {p: [] for p in self.pert2samples.keys()}
        
        for p, batch_indices in self.pert2samples.items():
            for indices in batch_indices.values():
                all_seen[p].extend(indices)

        # For each cell in the test set...
        for idx, c in enumerate(tqdm(conditions)):
            test_cell_type, perts_raw, dose = c.split("_")
            perts = [p for p in test_ds.pert_conditions.iloc[idx] if p != -1]
            b = test_ds.treatment_cell_batches.iloc[idx]

            # Get the subsampled indices associated to the present perturbation; match on batch if enough samples available and not aggregating across batches
            if not self.aggregate_batches and b in self.pert2samples[tuple(perts)].keys():
                sample_indices = self.pert2samples[tuple(perts)][b]
            else:
                # Aggregate across batches if not enough samples available
                sample_indices = all_seen[tuple(perts)]

            # Skip samples that were used for the prediction unless sampling with replacement
            if (idx in sample_indices) and not self.with_replacement:
                continue

            dose_cats.append(dose)
            batch_cats.append(b)
            
            cell_types.append(test_cell_type)
            pert_cats.append(perts_raw)

            # Compute the mean of the subsampled data
            preds.append(
                test_ds.pert_data[sample_indices, :self.output_dim].mean(0)
            )

            truths.append(test_ds.pert_data[idx, :self.output_dim])

        results = {}
        results["pert_cat"] = np.array(pert_cats)
        results["dose_cat"] = np.array(dose_cats)
        results["cell_type"] = np.array(cell_types)
        results["pred"] = torch.stack(preds).cpu().numpy()
        results["truth"] = torch.stack(truths).cpu().numpy()
        results["experimental_batch"] = np.array(batch_cats)

        return results
