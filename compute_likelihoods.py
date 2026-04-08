import os

import autoroot  # for imports from nsf repository
import hydra
import numpy as np
import nvtx  # for profiling annotations in pytorch lightning
import pandas as pd
import torch
import xarray as xr
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.intercomparison.dataset import ModelHealpixDataset
from src.likelihoods import get_log_probs
from src.nsf.olr_data import OlrTransform


def apply_postprocessing(ds: xr.Dataset, olr_key: str, postprocess_cfg: dict):
    """
    Apply a sequence of postprocessing functions to an xarray Dataset.
    """
    if postprocess_cfg is None:
        logger.debug("No post-processing steps specified, skipping post-processing.")
        return ds, olr_key

    for step_cfg in postprocess_cfg:
        logger.debug(f"Applying post-processing step: {step_cfg._target_}")
        fn = instantiate(step_cfg)
        ds, olr_key = fn(ds, olr_key)

    return ds, olr_key


@hydra.main(
    version_base=None, config_path="./configs/likelihoods", config_name="config"
)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    # -----------------------
    # time slice
    # -----------------------
    t = cfg.analysis.time
    step = t.step * cfg.data.get("time_step_multiplier", 1)
    time_slice = slice(t.start, t.end, step)

    intake_args = dict(cfg.data.get("intake_args", {}))

    # -----------------------
    # output paths
    # -----------------------
    save_dir = os.path.join(
        cfg.paths.mip_dir,
        cfg.data.model_key,
        f"likelihoods-{cfg.model.nsf_model_key}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------
    # model
    # -----------------------

    model_cfg = OmegaConf.load(cfg.model.model_config)
    model = hydra.utils.instantiate(model_cfg.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"Loading model checkpoint from {cfg.model.checkpoint} to device {device}..."
    )

    checkpoint = torch.load(
        cfg.model.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    # -----------------------
    # transforms
    # -----------------------
    transform = OlrTransform(
        histogram_matcher_path=cfg.data.histogram_matcher.path,
        normalise_method=cfg.model.normalisation.method,
        normalise_path=cfg.model.normalisation.path,
        fill_nan=True,
        load_coords=True,
    )

    # load precomputed patch centers
    patch_centers = np.load(cfg.analysis.patches.centers_path)

    dataset = ModelHealpixDataset(
        time_slice=time_slice,
        cat_keys=cfg.data.load.cat_keys,
        cat_url=cfg.data.load.cat_url,
        var=cfg.data.load.var,
        intake_args=intake_args,
        patch_centers=patch_centers,
        cell_var=cfg.data.load.cell_var,
        postprocess_cfg=cfg.data.get("postprocess", None),
        transform=transform,
        patch_size=cfg.analysis.patches.size,
        use_cache=cfg.analysis.get("use_dataset_cache", True),
    )

    logger.debug(f"ModelHealpixDataset xr.dataset: {dataset.ds}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.analysis.dataloader.batch_size,
        num_workers=cfg.analysis.dataloader.num_workers,
        shuffle=False,
        pin_memory=True,
        multiprocessing_context=cfg.analysis.dataloader.get(
            "multiprocessing_context", None
        ),
        prefetch_factor=cfg.analysis.dataloader.get("prefetch_factor", 2),
        persistent_workers=True if cfg.analysis.dataloader.num_workers > 0 else False,
    )

    logger.info("Starting batched likelihood computation...")

    # -----------------------
    # likelihood loop
    # -----------------------
    if "accumulate" in cfg.analysis:
        logger.info(f"Accumulation mode: {cfg.analysis.accumulate.mode}")
        save_dir = os.path.join(save_dir, f"accumulate_{cfg.analysis.accumulate.mode}")
        os.makedirs(save_dir, exist_ok=True)
        accumulate_mode = cfg.analysis.accumulate.mode
    else:
        accumulate_mode = "all"

    df = get_log_probs(
        dataloader,
        model,
        accumulate=accumulate_mode,
        save_dir=save_dir if "accumulate" in cfg.analysis else None,
    )

    out_filename = (
        f"likelihoods-{cfg.analysis.time.start}_{cfg.analysis.time.end}_step{step}.csv"
    )
    out_path = os.path.join(save_dir, out_filename)
    df.to_csv(out_path, index=False)

    logger.info(f"Saved likelihoods to {out_path}")


if __name__ == "__main__":
    main()
