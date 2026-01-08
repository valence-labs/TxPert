from typing import Any, Dict

import torch

import numpy as np
import scanpy as sc
import pandas as pd

from lightning import LightningModule
from torch.utils.data import DataLoader

from gspp.data.graphmodule import GSPGraph
from gspp.models.txpert import TxPert
from gspp.models.baselines import MeanBaseline, ExperimentalAccuracy
from gspp.evaluation import evaluate
from gspp.metrics import log_metrics, compute_metrics

import gspp.constants as cs

import os
import pickle
from loguru import logger


MODEL_DICT = {
    "txpert": TxPert,
    "mean_baseline": MeanBaseline,
    "experimental_baseline": ExperimentalAccuracy,
}


class PertPredictor(LightningModule):
    """
    Predictor class that manages training, testing, logging and instantiate the model. To allow for different model types,
    the model is selected based on the `model_type` argument in the `model_args` dictionary and the loss function is defined
    in the model class rather than here.

    Args:
        input_dim (int): The input dimension of the data.
        output_dim (int): The output dimension of the data.
        adata_output_dim (int): The output dimension of the adata.
        model_args (Dict[str, Any]): The arguments for the model.
        graph (GSPGraph): Graph object containing the edge index and edge weight.
        pert_input_dim (int): The input dimension of the perturbation embeddings.
        no_pert (bool): Whether to use perturbations.
        lr (float): The learning rate.
        min_lr (float): Minimum learning rate for scheduler
        weight_decay (float): The weight decay.
        device (str): The device.
        run_val_on_train_data (bool): Whether to run validation on the train data.
        match_cntr_for_eval (bool): Whether to match controls for evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adata_output_dim: int,
        model_args: Dict[str, Any],
        graph: GSPGraph = None,
        pert_input_dim: int = None,
        no_pert: bool = False,
        lr: float = 0.001,
        min_lr: float = 0.0,
        lr_scheduler_args: Dict[str, Any] = {},
        weight_decay: float = 0.0,
        device: str = "cpu",
        run_val_on_train_data: bool = False,
        match_cntr_for_eval: bool = True,
    ):
        super(PertPredictor, self).__init__()

        self.no_pert = no_pert

        self.slow_benchmark = model_args.pop("slow_benchmark", False)
        self.run_val_on_train_data = run_val_on_train_data
        self.match_cntr_for_eval = match_cntr_for_eval

        self.lr = lr
        self.min_lr = min_lr
        self.lr_scheduler_args = lr_scheduler_args
        self.weight_decay = weight_decay

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adata_output_dim = adata_output_dim

        self.model_type = model_args.pop("model_type", "txpert")

        if "baseline" in self.model_type:
            self.model = MODEL_DICT[self.model_type](**model_args)
        else:
            self.latent_dim = model_args["latent_dim"]
            cntr_model_args = model_args.pop("cntr_model")
            pert_model_args = model_args.pop("pert_model")
            self.basal_mode = model_args.pop("basal_mode", "construct_control")

            self.model = MODEL_DICT[self.model_type](
                input_dim=input_dim,
                output_dim=output_dim,
                adata_output_dim=adata_output_dim,
                graph=graph,
                cntr_model_args=cntr_model_args,
                pert_model_args=pert_model_args,
                pert_input_dim=pert_input_dim,
                device=device,
                **model_args,
            ).to(device)

    def forward(self, cntr, pert_idxs, p_emb):
        cntr = cntr.to(self.device)
        p_emb = p_emb.to(self.device)

        return self.model.forward(cntr, pert_idxs, p_emb)

    def loss(self, prediction, intrinsic_mean, intrinsic_log_var, target):
        # Call loss function provided through the model
        loss, logs = self.model.loss(
            prediction, intrinsic_mean, intrinsic_log_var, target
        )

        self.log_dict(
            logs,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        prediction, intrinsic_mean, intrinsic_log_var = self.forward(
            batch.control, batch.pert_idxs, batch.p
        )

        loss = self.loss(
            prediction,
            intrinsic_mean,
            intrinsic_log_var,
            target=batch.x[:, : self.adata_output_dim],
        )

        return loss

    def sample_inference(self, cntr, pert_idxs, p_emb, cell_types):
        """
        Sample from the model for inference. The basal mode determines different ways how the control is sampled.
        """
        size = cntr.size(0)
        prediction, _, _ = self.forward(cntr, pert_idxs, p_emb)

        if self.no_pert:
            return cntr

        if self.basal_mode == "latent":
            zs = torch.randn(size, self.latent_dim, device=self.device)
            prediction = self.model.decoder(
                torch.cat(
                    [zs + self.model.pert_z, cntr[:, self.model.output_dim :]], dim=-1
                )
            )

        elif self.basal_mode == "rand_cntr":
            cntr_data = self.trainer.datamodule.control_data
            intrinsic_recon = torch.Tensor(
                cntr_data.X[np.random.randint(0, cntr_data.shape[0], size)].toarray()
            )
            prediction = self.model.decoder(self.model.pert_z) + intrinsic_recon

        elif self.basal_mode == "construct_control":
            prediction = prediction

        elif self.basal_mode == "matched_control":
            prediction = self.model.decoder(self.model.pert_z) + cntr

        else:
            prediction = None
            raise ValueError("Invalid basal mode")

        return prediction

    def general_validation_epoch(
        self, stage: str, loader: DataLoader, metric_stage: str = cs.FAST
    ):
        """
        Run a general validation epoch. This function is used to run the validation on the train, val, and test data.

        Args:
            stage (str): The stage of the experiment (train, val, test).
            loader (DataLoader): The data loader.
            metric_stage (str): The metric stage (fast, extended).
        """
        test_adata = self.trainer.datamodule.adata

        if "baseline" in self.model_type:
            results = self.model.apply_baseline(self.trainer.datamodule.test_data)
        else:
            results = evaluate(
                loader,
                self,
                self.device,
                test_adata,
                self.trainer.datamodule.id2pert,
            )

        metrics, test_pert_res = compute_metrics(
            results, test_adata, metric_stage, match_cntr=self.match_cntr_for_eval
        )

        log_metrics(
            metrics, test_pert_res, stage, self.trainer.datamodule.subgroup, self
        )

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        if self.run_val_on_train_data:
            self.general_validation_epoch(
                cs.TRAIN, self.trainer.datamodule.train_dataloader()
            )

        self.general_validation_epoch(cs.VAL, self.trainer.datamodule.val_dataloader())
        self.general_validation_epoch(
            cs.TEST, self.trainer.datamodule.test_dataloader()
        )

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        metric_set = cs.SLOW if self.slow_benchmark else cs.EXTENDED
        self.general_validation_epoch(
            cs.TEST, self.trainer.datamodule.test_dataloader(), metric_set
        )

    def configure_optimizers(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=self.lr, weight_decay=self.weight_decay
        )

        schedulers = []
        milestones = []

        warmup_epochs = self.lr_scheduler_args.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            schedulers.append(
                cs.SCHEDULER_DICT["linear"](
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
            )
            milestones.append(warmup_epochs)

        scheduler_type = self.lr_scheduler_args.get("type", None)
        if scheduler_type is not None:
            total_epochs = self.lr_scheduler_args["total_epochs"]
            schedulers.append(
                cs.SCHEDULER_DICT[scheduler_type](
                    optimizer, T_max=total_epochs - warmup_epochs, eta_min=self.min_lr
                )
            )

        scheduler = (
            cs.SCHEDULER_DICT["sequential"](
                optimizer, schedulers, milestones=milestones
            )
            if len(schedulers) > 0
            else None
        )

        return (
            optimizer
            if scheduler is None
            else {"optimizer": optimizer, "lr_scheduler": scheduler}
        )

    def predict_step(self, batch, batch_idx):
        """
        Perform prediction on a batch and return all relevant data for saving.
        """
        with torch.no_grad():
            # Get predictions
            if self.no_pert:
                predictions = batch.control[:, : self.adata_output_dim]
            else:
                predictions = self.sample_inference(
                    batch.control, batch.pert_idxs, batch.p, batch.cell_types
                )

            # Return dictionary with all data needed for saving
            return {
                'base_states': batch.control.cpu(),
                'outputs': predictions.cpu(),
                'perturbations': batch.p.cpu(),
                'ground_truths': batch.x[:, : self.adata_output_dim].cpu(),
                'pert_idxs': batch.pert_idxs,
                'pert_cond_names': batch.pert_cond_names,
                'cell_types': batch.cell_types,
                'experimental_batches': batch.experimental_batches,
            }

    def on_predict_epoch_end(self):
        """
        Collect all predictions and save them to files.
        """
        if not hasattr(self, '_predict_save_dir') or not hasattr(self, '_predict_split_name'):
            return

        save_dir = self._predict_save_dir
        split_name = self._predict_split_name

        # Get predictions from trainer's predict loop
        results = self.trainer.predict_loop.predictions

        # Collect all predictions from all batches
        #all_perturbations = []
        all_base_states = []
        all_outputs = []
        all_pert_idxs = []
        all_pert_cond_names = []
        all_cell_types = []
        all_experimental_batches = []
        all_ground_truths = []

        for batch_results in results:
            all_base_states.append(batch_results['base_states'])
            all_outputs.append(batch_results['outputs'])
            #all_perturbations.append(batch_results['perturbations'])
            all_ground_truths.append(batch_results['ground_truths'])
            all_pert_idxs.extend(batch_results['pert_idxs'])
            all_pert_cond_names.extend(batch_results['pert_cond_names'])
            all_cell_types.extend(batch_results['cell_types'])
            all_experimental_batches.extend(batch_results['experimental_batches'])

        # Concatenate tensors
        base_states = torch.cat(all_base_states).numpy()
        outputs = torch.cat(all_outputs).numpy()
        #perturbations = torch.cat(all_perturbations).numpy()
        ground_truths = torch.cat(all_ground_truths).numpy()

        # Prepare data dictionary
        results_data = {
            #'input_perturbation': perturbations,
            'base_state': base_states,
            'output': outputs,
            'ground_truth': ground_truths,
            'pert_idxs': all_pert_idxs,
            'pert_cond_names': all_pert_cond_names,
            'cell_types': all_cell_types,
            'experimental_batches': all_experimental_batches,
        }

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save as pickle file
        save_path = os.path.join(save_dir, f"{split_name}_results.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results_data, f)

        # Save as anndata objects
        obs = pd.DataFrame({
            "pert_cond_names": all_pert_cond_names,
            "cell_types": all_cell_types,
            "experimental_batches": all_experimental_batches,
            "control": False
        })

        # var from original anndata - use stored reference if datamodule not available
        adata = self.trainer.datamodule.adata
        
        var = adata.var[:self.adata_output_dim].copy()
        pred_adata = sc.AnnData(outputs, obs=obs, var=var, uns=adata.uns)
        ground_truth_adata = sc.AnnData(ground_truths, obs=obs, var=var, uns=adata.uns)

        pred_adata.write_h5ad(os.path.join(save_dir, f"{split_name}_predictions.h5ad"))
        ground_truth_adata.write_h5ad(os.path.join(save_dir, f"{split_name}_ground_truth.h5ad"))

        obs_ct = obs.copy()
        obs_ct["control"] = True
        control_adata = sc.AnnData(base_states, obs=obs_ct, var=var, uns=adata.uns)
        control_adata.write_h5ad(os.path.join(save_dir, f"{split_name}_controls.h5ad"))

        logger.info(f"Saved {split_name} results to {save_dir} ({len(all_pert_cond_names)} samples)")
