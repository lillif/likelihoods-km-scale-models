# Likelihood-based model intercomparison of km-scale climate models

A record of the code and experiments for the paper:

> L. J. Freischem, T. Reichelt, R. Clark, P. Stier, H. M. Christensen (2026). A Generative Likelihood Framework for High-Resolution Climate Model Evaluation [*Manuscript submitted for publication*]


This repository was forked from [bayesiains/nsf](https://github.com/bayesiains/nsf) and the normalising flow functionality in `src/nsf/[nde, nn, optim, utils]` is the original implementation, while `src/nsf/[experiments, olr_data]` were updated and added to use custom datasets and use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable).


## Dependencies

See `environment.yml` for required Conda/pip packages, or use this to create a Conda environment with
all dependencies:
```bash
conda env create -f environment.yml
```

Tested with Python 3.12.3, PyTorch 2.5.1 and Lightning 2.6.1.

## Data

NextGEMS production simulations for ICON and IFS are archived by the German Climate Computing Center (DKRZ) and can be accessed via DKRZ's supercomputer Levante after registration at https://luv.dkrz.de/register.

GOES-16 OLR data was derived from Level 1b radiance measurements which were supplied by the National Oceanic and Atmospheric Administration (NOAA) and can be downloaded at https://console.cloud.google.com/marketplace/product/noaa-public/goes.

### Pre-patched datasets for model training

For efficient NSF model training, we pre-patched the GOES dataset. The expected folder structure is:

```
/path/to/datasets/
├── goes/
    └── patches64_stride32/
        ├── 20240101T000020_patch0094.npz
        ├── 20240101T000020_patch0095.npz
        └── ...
```

Before running experiments, set the following environment variables in `.env` (see `.env.example`):

| Variable | Description |
|---|---|
| `DATAROOT` | Root directory containing the pre-patched GOES training data |
| `LIKELIHOODS_SAVE_DIR` | Directory where computed likelihood CSV files are saved |

## Usage

The model is implemented in PyTorch Lightning. We use [hydra](https://hydra.cc) for model training and likelihood estimation, and logged model training using [wandb](https://wandb.ai).


### Model training

Example configuration files can be found in `configs/training/examples`. To train the model, fill in the corresponding data paths in the configuration files [`datamodule.yaml`] and wandb configurations [`wandb.yaml`], and set `config_path` in `train.py` to the relevant configuration path before running:

```
python train.py
```

### Likelihood estimation

Configuration files for likelihood estimation can be found in `configs/likelihoods`, which contains the following subfolders:

```
./configs/likelihoods/
├── analysis/     # analysis period, patching, and dataloader options
├── data/         # configurations for the km-scale model to be analysed
├── model/        # ML model settings for likelihood estimation
├── paths/        # base directory locations
└── config.yaml   # top-level config composing the above via Hydra defaults
```

#### Model likelihood estimation

To compute likelihoods for the ICON model in 2024 using a trained NSF model checkpoint run:

```
python compute_likelihoods.py data=icon_ngc4008 model=nsf_example paths=default
```

The config file `configs/likelihoods/model/nsf_example.yaml` shows the format required to use an existing NSF checkpoint. Set `model.checkpoint` to the path of the trained `.ckpt` file (or pass it as a Hydra override: `model.checkpoint=/path/to/checkpoint.ckpt`). Note that other filepaths need to be correctly set in `configs/likelihoods/paths` and `.env`.

#### GOES-16 likelihood estimation

Likelihoods for GOES-16 are computed with a separate script using `configs/likelihoods/goes_config.yaml` as the top-level config:

```
python compute_goes_likelihoods.py
```

`goes_config.yaml` uses the `DATAROOT` environment variable to point to the directory containing the pre-patched `.npz` files. Checkpoint and normalisation settings follow the same format as `configs/likelihoods/model/nsf_example.yaml`.

## Notebooks

The `notebooks/` folder contains Jupyter notebooks to reproduce the figures in the paper. They use pre-computed likelihood CSV files produced by `compute_likelihoods.py` and `compute_goes_likelihoods.py`. The figure notebooks expect an additional environment variable `RESULTS_DIR_64x64` pointing to a directory with the following files:

```
goes_train_likelihoods.csv  icon_train_likelihoods.csv  ifs_train_likelihoods.csv
goes_test_likelihoods.csv   icon_test_likelihoods.csv   ifs_test_likelihoods.csv
```

where train contains the first half and test the second half of each month which is the split used in the paper. Analysis utilities in `notebooks/utils/` can be adapted to other experiments.

```
notebooks/
├── Evaluation.ipynb                          # computes symmetric KL divergence between GOES, IFS, and ICON likelihood distributions
├── Fig3_likelihood_histograms.ipynb          # Figure 3: distributions of log-likelihoods per model
├── Fig4_spatial_biases.ipynb                 # Figure 4: spatial maps of likelihoods and KL divergence
└── Fig5_temporal_biases.ipynb                # Figure 5: diurnal (local solar time) bias analysis
```

## Citation

```bibtex
@software{freischem2025github,
  author  = {Lilli J. Freischem},
  title   = {Likelihood-based model intercomparison of km-scale climate models},
  year    = {2026},
  url     = {https://github.com/lillif/likelihoods-km-scale-models/},
}
```