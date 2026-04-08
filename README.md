# Likelihood-based model intercomparison of km-scale climate models


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

Before running experiments, the `DATAROOT` environment variable needs to be set in `.env` before running experiments (see `.env.example`).

## Usage

The model is implemented in PyTorch Lightning. We use [hydra](https://hydra.cc) for model training and likelihood estimation, and logged model training using [wandb](https://wandb.ai).


### Model training

Example configuration files can be found in `configs/training/examples`. To train the model, fill in the corresponding data paths in the configuration files, and set `config_path` in `train.py` to the configuration file path before running:

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

To compute likelihoods for the ICON model in 2024 using a trained NSF model checkpoint run:

```
python compute_likelihoods.py data=icon_ngc4008 model=nsf_example paths=default
```

The config file `configs/model/nsf_example.yaml` shows the format required to use an existing NSF checkpoint.
Note that filepaths need to be correctly set in `configs/path` and `.env`. 