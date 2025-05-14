import sys
import hydra

from omegaconf import DictConfig, OmegaConf

import torch

from time import time

from loguru import logger

import traceback
from lightning import Trainer

from gspp.predictor import PertPredictor
from gspp.data.graphmodule import GSPGraph
from gspp.data.datamodule import PertDataModule

from gspp.utils import set_seed
from gspp.constants import CHECKPOINT_DIR
from gspp.utils import download_files_from_zenodo


# Main function to initialize Hydra and parse the configuration
@hydra.main(config_name="config-gat", config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    download_files_from_zenodo("15393145", "KcPONWTmFPfYGH2vSI8SLqeND0tu8GCRDh2cgXldyAk6GJsWvVkMBkA04JTN")
    try:
        t0 = time()
        infer(cfg)
        logger.info(f"Time taken: {time() - t0} seconds")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def infer(cfg: DictConfig):
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"

    set_seed(cfg.seed)

    data_args = OmegaConf.to_container(cfg.datamodule, resolve=True)

    datamodule = PertDataModule(**data_args)
    datamodule.prepare_data()

    # Instatiate graph(s) used by the GNN
    graph = GSPGraph(
        pert2id=datamodule.pert2id,
        gene2id=datamodule.gene2id,
        **cfg.graph,
    )

    trainer = Trainer(
        accelerator="gpu" if gpu_available else "cpu",
    )

    if cfg.mode == "baseline":
        datamodule.setup("fit")
        GSP_model = PertPredictor(
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            adata_output_dim=datamodule.adata_output_dim,
            device=device,
            match_cntr_for_eval=cfg.get("match_cntr_for_eval", True),
        ).to(device)

        GSP_model.model.prepare_baseline(**GSP_model.model.parse_inputs(datamodule))

    elif cfg.mode == "inference":
        GSP_model = PertPredictor.load_from_checkpoint(
            CHECKPOINT_DIR / cfg.checkpoint_name,
            input_dim=datamodule.input_dim,
            output_dim=datamodule.output_dim,
            adata_output_dim=datamodule.adata_output_dim,
            num_perts=len(datamodule.pert2id),
            graph=graph,
            pert_names=list(datamodule.pert2id.keys()),
            model_args=OmegaConf.to_container(cfg.model, resolve=True),
            device=device,
        ).to(device)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    # Testing
    trainer.test(GSP_model, datamodule)


if __name__ == "__main__":
    main()
