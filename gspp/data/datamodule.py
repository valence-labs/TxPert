from typing import Union, Tuple

import os
import sys
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import lightning as L
from torch.utils.data import Dataset, DataLoader

import joblib
from scipy.sparse import csr_matrix
from loguru import logger

import gspp.constants as cs

from gspp.data.data_utils import (
    collate_fn,
    calculate_de_for_eval,
)

logger.remove()
logger.add(sys.stdout, level="INFO")


class PertDataModule(L.LightningDataModule):
    """
    Datamodule for the perturbation prediction task. This datamodule loads the perturbation data and control data.

    Args:
        batch_size (int): Batch size for the dataloader
        match_cntr (bool): Whether to match the control based on the batch
        avg_cntr (bool): Whether to average the control rather than random sampling
        embed_cntr (bool): Whether to embed control samples
        obsm_key (str): Key for the embedding in adata.obsm
        task_type (str): Pre-defined demo task type for the datamodule
        mode (str): Mode for the datamodule, either "baseline" or "inference"
    """

    def __init__(
        self,
        batch_size: int = 64,
        match_cntr: bool = False,
        avg_cntr: bool = False,
        embed_cntr: bool = True,
        obsm_key: str = cs.ObsmKey.SCGPT,
        task_type: str = "K562_single_cell_line",
        mode: str = "baseline",
    ):
        """
        Initialize the PertDataModule with various parameters.
        """
        super().__init__()
        self.batch_size = batch_size
        self.cache_dir = cs.CACHE_DIR

        self.obsm_key = obsm_key
        self.match_cntr = match_cntr
        self.avg_cntr = avg_cntr
        self.mol_fp_dim = None
        self.embed_cntr = embed_cntr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_prepared = False
        self._initialize_gene_sets()

        self.task_type = task_type
        self._validate_task_type()
        self.train_cell_types, self.val_cell_type, self.test_cell_type = (
            cs.DEMO_DATASET_CELL_TYPES[self.task_type]
        )
        self._validate_and_set_cell_types()
        self.cache_path = self.cache_dir / self.task_type

        self.mode = mode

    def _validate_task_type(self) -> None:
        """
        Validate the task type against the available task types.
        """
        if self.task_type not in cs.DEMO_DATASET_CELL_TYPES:
            raise ValueError(
                f"task_type must be one of {cs.DEMO_DATASET_CELL_TYPES.keys()}, got {self.task_type}"
            )

    def _validate_and_set_cell_types(self) -> None:
        """
        Validate and set the provided cell types against the public cell types.
        """
        if self.test_cell_type not in cs.PUBLIC_CELL_TYPES:
            raise ValueError(
                f"test_cell_type must be one of {cs.PUBLIC_CELL_TYPES}, got {self.test_cell_type}"
            )

        if isinstance(self.train_cell_types, str) and self.train_cell_types != "all":
            self.train_cell_types = [self.train_cell_types]

        if not (
            self.train_cell_types == "all"
            or all(ct in cs.PUBLIC_CELL_TYPES for ct in self.train_cell_types)
        ):
            raise ValueError(
                f"train_cell_types must be 'all' or a subset of {cs.PUBLIC_CELL_TYPES}, got {self.train_cell_types}"
            )

        temp_train_cell_types = set(cs.PUBLIC_CELL_TYPES)
        temp_train_cell_types.remove(self.test_cell_type)
        if self.val_cell_type is not None:
            temp_train_cell_types.remove(self.val_cell_type)

        self.train_cell_types = (
            [
                ct
                for ct in temp_train_cell_types
                if not ct.startswith(("mol-", "doubles-"))
                and ct not in ["K562_adamson", "K562_dixit"]
            ]
            if self.train_cell_types == "all"
            else self.train_cell_types
        )
        self.train_cell_types.sort(reverse=True)

        self.total_cell_types = set(self.train_cell_types)
        self.total_cell_types.add(self.test_cell_type)

    def _initialize_gene_sets(self) -> None:
        """
        Initialize gene sets for perturbation data.
        """
        id2gene = dict(
            zip(
                (_df := pd.read_csv(cs.DATA_DIR / "master_gene_set_sorted.csv"))[
                    "gene_id"
                ],
                _df["gene_name"],
            )
        )

        id2pert = pd.read_csv(
            cs.DATA_DIR / "gears_gene_set.csv",
            index_col=0,
        ).to_dict()["0"]

        if self.embed_cntr:
            id2pert.update({-1: cs.CONTROL_LABEL})

        self.id2pert = {k: v for k, v in id2pert.items()}
        self.pert2id = {v: k for k, v in id2pert.items()}

        self.id2gene = {k: v for k, v in id2gene.items()}
        self.gene2id = {v: k for k, v in id2gene.items()}

    def prepare_data(self) -> None:

        if self.data_prepared:
            return

        cache_file = f"{self.cache_path}/de_adata_test.h5ad"
        logger.info(f"Attempt loading data from {cache_file}")

        if os.path.exists(cache_file):
            logger.info(f"adata cached, loading from {cache_file}")
            self.adata = sc.read_h5ad(cache_file)

        else:
            raise ValueError(f"Unsupported dataset.")

        self._split_data()
        self._load_predefined_splits()

        self.data_prepared = True

    def _split_data(self) -> None:
        """
        Split the data into control and treatment datasets.
        """
        self._split_control_data()
        self._process_treatment_data()
        self._set_input_output_dimensions()
        self._load_pharos_data()

    def _split_control_data(self) -> None:
        """
        Split the data into control and treatment datasets.
        """

        if self.mode == "baseline":
            train_control_mask = (self.adata.obs[cs.CONTROL] == 1) & (
                self.adata.obs[cs.CELL_TYPE].isin(
                    self.train_cell_types + [self.test_cell_type] + [self.val_cell_type]
                )
            )

            self.control_data_train = self.adata[train_control_mask]

            self.control_data_val = (
                self.control_data_train
                if self.val_cell_type is None
                else self.adata[
                    (self.adata.obs[cs.CONTROL] == 1)
                    & (self.adata.obs[cs.CELL_TYPE] == self.val_cell_type)
                ]
            )
            self.control_train_condition_names = self.control_data_train.obs[
                cs.CONDITION_NAME
            ]
            self.control_val_condition_names = (
                None
                if self.val_cell_type is None
                else self.control_data_val.obs[cs.CONDITION_NAME]
            )

        self.control_data_test = self.adata[
            (self.adata.obs[cs.CONTROL] == 1)
            & (self.adata.obs[cs.CELL_TYPE] == self.test_cell_type)
        ]

        self.control_data = self.control_data_test

        self.control_test_condition_names = self.control_data_test.obs[
            cs.CONDITION_NAME
        ]

    def _process_treatment_data(self) -> None:
        """
        Process the treatment data by filtering and setting conditions.
        """

        self.treatment_data = self.adata[self.adata.obs[cs.CONTROL] == 0]

        temp_conditions = self.treatment_data.obs[cs.CONDITION].str.split("+").tolist()
        pert_idxs = [
            [self.pert2id[p] for p in item if p in self.pert2id]
            for item in temp_conditions
        ]

        filtered_idx = [
            all(p in self.pert2id for p in item) for item in temp_conditions
        ]

        self.condition_names = self.treatment_data.obs[cs.CONDITION_NAME][filtered_idx]
        filtered_conditions = self.treatment_data.obs[cs.CONDITION][
            ~np.array(filtered_idx)
        ].unique()
        self.treatment_data = self.treatment_data[filtered_idx]
        self.conditions = pd.Series(pert_idxs)[filtered_idx]
        self.conditions.index = self.treatment_data.obs.index

        logger.info(f"Filtered out the following perturbations: {filtered_conditions}")

    def _set_input_output_dimensions(self) -> None:
        """
        Set the input and output dimensions for the data.
        """
        # used to set dimensions (with our without cell type onehot encoding)
        self.cell_types = np.unique(self.adata.obs[cs.CELL_TYPE])
        self.number_of_cell_types = len(self.cell_types)

        dim_adjustment = 0

        self.adata_output_dim = self.adata.X.shape[1] - dim_adjustment

        if self.obsm_key != cs.ObsmKey.RAW:
            self.input_dim = self.adata.obsm[self.obsm_key].shape[1]
            self.output_dim = self.adata.obsm[self.obsm_key].shape[1] - dim_adjustment
        else:
            self.input_dim = self.adata.X.shape[1]
            self.output_dim = self.adata.X.shape[1] - dim_adjustment

        logger.info(f"adata_output_dim :  {self.adata_output_dim}")

        self.adata = self.adata[:, : self.adata_output_dim].copy()

    def _load_pharos_data(self) -> None:
        """
        Load Pharos data and update the AnnData object.
        """
        dat = pd.read_csv(cs.DATA_DIR / "pharos_mean_rank.csv")
        dat = dat.sort_values("mean_rank")
        thresholds = [dat.shape[0] // x for x in [8, 4, 2, 1]]
        res = {}
        for i, gene in enumerate(dat["Symbol"]):
            for j in range(len(thresholds) - 1, -1, -1):
                if thresholds[j] >= i:
                    res[gene] = j

        knowledge_levels = np.array([res.get(v, -1) for v in self.adata.var_names])
        self.adata.uns["pharos"] = knowledge_levels
        logger.info(
            "knowledge_level_counts:",
            pd.Series(self.adata.uns["pharos"]).value_counts(),
        )

    def setup(self, stage: Union[str, None] = None) -> None:
        """
        Setup the data module for different stages (fit, validate, test).

        Args:
            stage (str): The stage to setup (fit, validate, test).
        """
        logger.info(f"Setting up in stage {stage}")

        if stage == "fit" or stage is None:
            self._setup_fit()
        elif stage == "validate":
            self._setup_validate()
        elif stage == "test":
            self._setup_test()

    def _setup_fit(self) -> None:
        """
        Setup the data module for the fit stage.
        """

        self._create_datasets()

    def _load_predefined_splits(self) -> None:
        """
        Load predefined splits for the data.
        """
        print(f"Loading predefined splits from {self.cache_path}")
        if not os.path.exists(
            self.cache_path / f"splits/train_test_split.pkl"
        ) or not os.path.exists(self.cache_path / f"splits/subgroup.pkl"):
            print(self.cache_path / f"splits/train_test_split.pkl")
            raise ValueError("Predefined splits not found")
        self.condition_group = joblib.load(
            self.cache_path / f"splits/train_test_split.pkl"
        )
        self.subgroup = joblib.load(self.cache_path / f"splits/subgroup.pkl")
        logger.info("Using predefined splits")

    def _create_datasets(self) -> None:
        """
        Create datasets for training, validation, and testing.
        """

        self.train_cell_type_mask = self.treatment_data.obs[cs.CELL_TYPE].isin(
            self.train_cell_types
        )

        train_indices = (
            self.treatment_data.obs[cs.CONDITION]
            .isin(self.condition_group["train"])
            .values
        ) & self.train_cell_type_mask
        if train_indices.sum() == 0:
            raise ValueError("Training split is empty.")

        self.train_adata = self.treatment_data[train_indices]
        self.train_conditions = self.conditions[train_indices]
        self.train_condition_names = self.condition_names[train_indices]

        self.train_data = XcelltypePerturbDataset(
            self.train_adata,
            self.control_data_train,
            self.train_conditions,
            self.train_condition_names,
            self.device,
            train_cond_names=self.control_train_condition_names,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            obsm_key=self.obsm_key,
        )

        self.val_cell_type_mask = (
            (self.treatment_data.obs[cs.CELL_TYPE] == self.val_cell_type)
            if self.val_cell_type is not None
            else None
        )
        val_cell_type_mask = (
            self.train_cell_type_mask
            if self.val_cell_type is None
            else self.val_cell_type_mask
        )

        val_indices = (
            self.treatment_data.obs[cs.CONDITION]
            .isin(self.condition_group["val"])
            .values
        ) & val_cell_type_mask
        if val_indices.sum() == 0:
            raise ValueError("Validation split is empty.")

        self.val_adata = self.treatment_data[val_indices]
        self.val_conditions = self.conditions[val_indices]
        self.val_condition_names = self.condition_names[val_indices]

        self.val_data = XcelltypePerturbDataset(
            self.val_adata,
            self.control_data_val,
            self.val_conditions,
            self.val_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            obsm_key=self.obsm_key,
        )

        self.test_cell_type_mask = (
            self.treatment_data.obs[cs.CELL_TYPE] == self.test_cell_type
        )

        test_mask = (
            self.treatment_data.obs[cs.CONDITION]
            .isin(self.condition_group["test"])
            .values
        ) & self.test_cell_type_mask

        if test_mask.sum() == 0:
            raise ValueError("Test split is empty.")

        self.test_adata = self.treatment_data[test_mask]
        self.test_conditions = self.conditions[test_mask]
        self.test_condition_names = self.condition_names[test_mask]

        self.test_data = XcelltypePerturbDataset(
            self.test_adata,
            self.control_data_test,
            self.test_conditions,
            self.test_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            obsm_key=self.obsm_key,
        )

    def _setup_validate(self) -> None:
        """
        Setup the data module for the validation stage.
        """
        val_mask = (
            self.treatment_data.obs[cs.CONDITION]
            .isin(self.condition_group["val"])
            .values
        ) & self.val_cell_type_mask

        self.val_adata = self.treatment_data[val_mask]
        self.val_conditions = self.conditions[val_mask]
        self.val_condition_names = self.condition_names[val_mask]

        self.val_data = XcelltypePerturbDataset(
            self.val_adata,
            self.control_data_val,
            self.val_conditions,
            self.val_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            obsm_key=self.obsm_key,
        )

    def _setup_test(self) -> None:
        """
        Setup the data module for the test stage.
        """
        self.test_cell_type_mask = (
            self.treatment_data.obs[cs.CELL_TYPE] == self.test_cell_type
        )

        test_mask = (
            self.treatment_data.obs[cs.CONDITION]
            .isin(self.condition_group["test"])
            .values
        ) & self.test_cell_type_mask

        self.test_adata = self.treatment_data[test_mask]
        self.test_conditions = self.conditions[test_mask]
        self.test_condition_names = self.condition_names[test_mask]

        self.test_data = XcelltypePerturbDataset(
            self.test_adata,
            self.control_data_test,
            self.test_conditions,
            self.test_condition_names,
            self.device,
            match_cntr=self.match_cntr,
            avg_cntr=self.avg_cntr,
            obsm_key=self.obsm_key,
        )

    def _load_or_calculate_de_adata(
        self, cache_file: str, mask: pd.Series
    ) -> sc.AnnData:
        """
        Load or calculate differentially expressed genes for evaluation.

        Args:
            cache_file (str): Path to the cache file.
            indices (pd.Series): The indices to use for calculating DE genes.

        Returns:
            sc.AnnData: The differentially expressed genes AnnData object.
        """
        if os.path.exists(cache_file):
            logger.info(f"de_adata cached, loading from {cache_file}")
            de_adata = sc.read_h5ad(cache_file)
        else:
            logger.info(f"Calculating de_adata for {self.test_cell_type}...")
            de_adata = self.adata[mask]
            de_adata = calculate_de_for_eval(de_adata, molecular_perts=False)
            de_adata.write(cache_file)
            logger.info(f"de_adata saved to {cache_file}")

        return de_adata

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for training data.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for validation data.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for test data.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def get_all_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return all DataLoaders (train, validation, test).
        """
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()


class XcelltypePerturbDataset(Dataset):
    def __init__(
        self,
        adata: sc.AnnData,
        control_data: sc.AnnData,
        pert_conditions: pd.Series,
        condition_names: pd.Series,
        device: str = "cuda",
        train_cond_names: pd.Series = None,
        match_cntr: bool = False,
        avg_cntr: bool = False,
        obsm_key: str = cs.ObsmKey.SCGPT,
    ):
        """
        Dataset for perturbation data.

        Args:
            adata (AnnData): The perturbation data
            control_data (AnnData): The control data
            pert_conditions (pd.Series): The perturbation conditions
            condition_names (pd.Series): The condition names
            device (str): The device to use
            train_cond_names (pd.Series): The condition names for the control data
            match_cntr (bool): Whether to match the control based on the batch
            avg_cntr (bool): Whether to average the control rather than random sampling
            obsm_key (str): The key for the embedding in adata.obsm
        """
        self.obsm_key = obsm_key

        self.device = device
        self.match_cntr = match_cntr
        self.avg_cntr = avg_cntr

        self.control_data_raw_X = self._to_tensor(control_data.X)
        self.pert_data = self._to_tensor(adata.X)
        if self.obsm_key == cs.ObsmKey.RAW:
            self.control_data = self._to_tensor(control_data.X)
        else:
            self.control_data = self._to_tensor(control_data.obsm[obsm_key])

        self.pert_conditions = pert_conditions
        self.condition_names = condition_names

        self.index = adata.obs.index.to_series()
        self.control_index = control_data.obs.index.to_series()

        self.treatment_cell_types = adata.obs[cs.CELL_TYPE]
        self.treatment_cell_batches = adata.obs[cs.BATCH]

        self.control_cell_types = control_data.obs[cs.CELL_TYPE]
        self.control_cell_batches = control_data.obs[cs.BATCH]

        self._initialize_all_treatment_data()
        self._initialize_control_data(train_cond_names, adata, control_data)

    def _to_tensor(self, data: csr_matrix) -> torch.Tensor:
        """
        Convert sparse matrix to tensor.

        Args:
            data (csr_matrix): Sparse matrix data.

        Returns:
            torch.Tensor: Tensor data.
        """
        return torch.Tensor(data.toarray()).to(self.device)

    def _initialize_all_treatment_data(self) -> None:
        """
        Initialize all treatment data by combining control and treatment
        """

        self.all_cell_types = pd.concat(
            [self.treatment_cell_types, self.control_cell_types]
        )

        self.all_cell_batches = pd.concat(
            [self.treatment_cell_batches, self.control_cell_batches]
        )

    def _initialize_control_data(
        self, train_cond_names: pd.Series, adata: sc.AnnData, control_data: sc.AnnData
    ) -> None:
        """
        Initialize control data based on the matching mode.

        Args:
            train_cond_names (pd.Series): The condition names for the control data.
            adata (AnnData): The perturbation data.
            control_data (AnnData): The control data.
        """
        # Collect information for control matching
        self.control_ct_indices = {}  # For random matching on cell-type only
        self.control_ct_batch_indices = {}  # For random matching on cell-type & batch

        for ct in np.unique(self.all_cell_types).tolist():
            self.control_ct_batch_indices[ct] = {}

            ct_mask = adata.obs[cs.CELL_TYPE] == ct

            # For random matching on cell-type only if not using match_cntr:
            ct_mask_cntr = control_data.obs[cs.CELL_TYPE] == ct
            self.control_ct_indices[ct] = ct_mask_cntr.values.nonzero()[0]

            available_batches = list(
                set(
                    np.unique(adata[ct_mask].obs[cs.BATCH]).tolist()
                    + np.unique(control_data.obs[cs.BATCH]).tolist()
                )
            )

            # Collect indices of control data for each batch
            for batch in available_batches:
                ct_batch_mask_cntr = (control_data.obs[cs.CELL_TYPE] == ct) & (
                    control_data.obs[cs.BATCH] == batch
                )
                self.control_ct_batch_indices[ct][
                    batch
                ] = ct_batch_mask_cntr.values.nonzero()[0]

        # Extend the dataset with training data only if we don't use FM
        if train_cond_names is not None:
            self._extend_with_train_data(train_cond_names)

        self.control_cache = {}

    def _extend_with_train_data(self, train_cond_names: pd.Series) -> None:
        """
        Extend the dataset with training data.

        Args:
            train_cond_names (pd.Series): The condition names for the control data.
        """
        self.pert_data = torch.cat([self.pert_data, self.control_data_raw_X])
        self.pert_conditions = pd.concat(
            [
                self.pert_conditions,
                pd.Series([["ctrl"]] * self.control_data_raw_X.shape[0]),
            ]
        )
        self.condition_names = pd.concat([self.condition_names, train_cond_names])
        self.index = pd.concat([self.index, self.control_index])
        self.treatment_cell_types = self.all_cell_types
        self.treatment_cell_batches = self.all_cell_batches

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return self.pert_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        pd.Series,
        pd.Series,
        pd.Series,
        Union[str, None],
        pd.Series,
    ]:
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple: Perturbation data, control data, perturbation condition, condition name, index, cell type, experimental batch.
        """
        control = self._get_control_data(idx)
        pert_data = self.pert_data[idx]

        p_emb = None

        return (
            pert_data,
            control,
            (
                torch.cat([p_emb, self.dose2enc[self.treatment_doses.iloc[idx]]])
                if p_emb is not None
                else torch.zeros(2)
            ),
            self.pert_conditions.iloc[idx],
            self.condition_names.iloc[idx],
            "1+1",
            self.index.iloc[idx],
            self.treatment_cell_types.iloc[idx],
            self.treatment_cell_batches.iloc[idx],
        )

    def _get_control_data(self, idx: int) -> torch.Tensor:
        """
        Get control data based on the matching mode.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: Control data.
        """
        cell_type = self.treatment_cell_types.iloc[idx]
        batch = self.treatment_cell_batches.iloc[idx]
        if self.match_cntr:
            if self.avg_cntr:
                control = self.control_data[
                    self.control_ct_batch_indices[cell_type][batch], :
                ].mean(0)
            else:
                control = self.control_data[
                    np.random.choice(self.control_ct_batch_indices[cell_type][batch])
                ]
        else:
            if self.avg_cntr:
                if cell_type not in self.control_cache.keys():
                    self.control_cache[cell_type] = self.control_data[
                        self.control_ct_indices[cell_type], :
                    ].mean(0)

                control = self.control_cache[cell_type]
            else:
                control = self.control_data[
                    np.random.choice(self.control_ct_indices[cell_type])
                ]
        return control
