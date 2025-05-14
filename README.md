
# TxPert
A model for graph-supported perturbation prediction with transcriptomic data.

## Installation
### Install UV
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repo
```sh
git clone git@github.com:valence-labs/TxPert.git
cd TxPert
```
### Create and activate a Python 3.12 environment:
```sh
uv venv --python 3.12
source .venv/bin/activate
```
### Install dependencies:
Make sure you are on a system with compatible NVIDIA GPUs and CUDA drivers (typically a compute node).
If using environment modules, load CUDA 12.4:
```sh
module load CUDA/12.4.0
```
Install the project (this step will install all the dependencies)
```sh
# Assuming you are in the TxPert folder
uv pip install -e .
```
**Note:** The default `uv` cache dir is pointing to user's local home directory, which causes uv to fail hard-linking files (as partitions differ when using compute nodes). For enhanced performance, you can set the `UV_CACHE_DIR` variable to point to current folder (in the shell or in the slurm script). In interactive sessions, it would be something like:
```sh
 export UV_CACHE_DIR=<PROJECT-DIRT>/.cache
```

## Data Download
Data should be automatically downloaded when running the code. The appropriate data will be downloaded to the `cache` folder in the current working directory. 

## Data Description
### Single-Cell Line Cache
The single-cell line adata cache is derived from the raw K562 essential gene perturbation dataset provided by Replogle et al. (2022). To ensure comparability with prior work, we follow the same preprocessing pipeline used in GEARS (Roohani et al., 2024) and CausalBench (Chevalley et al., 2022). This preprocessing includes:
- Filtering to retain strong perturbation signals.

- Library size normalization (fixed to 4000 counts).

- log1p transformation.

- Highly variable gene selection (top 5000 genes).

The code for this process is provided by the GEARS authors and is available in this [notebook](https://github.com/yhr91/GEARS_misc/blob/main/data/preprocessing/Replogle_2022_preprocess.ipynb). Additional discussion can be found in this [Github issue](https://github.com/snap-stanford/GEARS/issues/28).

### Cross-Cell Line Cache
The cross-cell line adata cache includes perturbation data from four different cell lines:

- K562 and RPE1 from [Replogle et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35688146/)

- Jurkat and HepG2 from [Nadig et al. (2025)](https://www.nature.com/articles/s41588-025-02169-3)

- K562_adamson from [Adamson et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27984733/)

Each cell line is initially processed independently using the same procedure as for the single-cell line cache, except that highly variable gene (HVG) selection is deferred. After individual processing, the datasets are concatenated, and HVGs are selected based on the intersection of:

- HVGs from the test cell line, and

- All genes present in the training cell lines

This ensures a consistent and comparable gene feature space across datasets.


### Differential Expression Analysis
For each cell line, we also perform differential gene expression analysis. Perturbed cells are grouped by perturbation target, and unperturbed cells are used as the reference group. The results of this analysis are stored as metadata under the uns attribute of each adata object.



## Experiments
We provide datasets and code here to reproduce the experiments shown in the paper for two OOD tasks:
1. Transfer to **unseen perturbations** within cell type
2. Transfer to an **unseen cell line**

We focus on the **K562 cell type** here as well as models that **only rely on public data sources** for training.

### TxPert inference with a checkpoint
To run inference for prediction of perturbation effects on unseen perturbations with one of our pretrained models, use
```
python main.py --config-name=config-[model_type] 
```
Available models (`model_type`) are `gat`, `exphormer` and `exphormer-mg`. Note that we provide only a subset of models reported in the paper as most of our best models rely on proprietary knowledge graphs (KGs). In particular, the `exphormer-mg` reported here uses two KGs (STRINGdb and GO) as an example of multi-graph (MG) reasoning. However, the model reported in the paper uses two additional proprietary KGs and achieves higher performance.

To run the same experiment for transfer to an unseen cell type, use
```
python main.py --config-name=config-x-cell-gat
```

### General baseline
We provide a "general" baseline that combines the mean control expression in the test cell-type with the mean perturbation-specific delta observed across training cell-types. If the perturbation is not in the training set, the global delta is used instead. To apply this baseline for the prediction of unseen perturbation effects, run the following command:
```
python main.py --config-name=config-baseline
```

For the prediction of perturbation effects in unseen cell types, use the option `--config-name=config-x-cell-baseline` instead.

### Experimental reproducibility
A second baseline characterizing the experimental reproducibility of the experiment is available using the following command:
```
python main.py --config-name=config-baseline model.model_type=experimental_baseline +model.test_seen=0.5
```

This will estimate the perturbation-wise mean across samples from a random subsample (here using 50% of the samples per perturbation).
