import os

import autoroot  # for imports from src
import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.likelihoods import get_log_probs
from src.nsf.olr_data import OlrDataset, OlrTransform, get_list_olrfiles, get_split

@hydra.main(
    version_base=None, config_path="./configs/likelihoods", config_name="goes_config"
)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

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
        histogram_matcher_path=None, # goes: no histogram matching
        normalise_method=cfg.model.normalisation.method,
        normalise_path=cfg.model.normalisation.path,
        fill_nan=True,
        load_coords=True,
    )

    # -----------------------
    # dataset and dataloader
    # -----------------------
    all_files = get_list_olrfiles(cfg.data.data_dir, ext=".npz")
    logger.info(f"Found {len(all_files)} npz files in {cfg.data.data_dir}")

    files = get_split(all_files, dict(cfg.data.time))
    logger.info(f"Processing {len(files)} files after time filtering")

    dataset = OlrDataset(
        filepaths=files,
        transforms=transform,
        load_coords=True,
        load_time=True,
        npz_image_key=cfg.data.npz_image_key,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.dataloader.batch_size,
        num_workers=cfg.data.dataloader.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=cfg.data.dataloader.get("prefetch_factor", 2),
        persistent_workers=cfg.data.dataloader.num_workers > 0,
    )

    logger.info("Starting batched likelihood computation...")

    # -----------------------
    # likelihood loop
    # -----------------------
    accumulate_mode = cfg.accumulate.mode
    if accumulate_mode != "all":
        logger.info(f"Accumulation mode: {accumulate_mode}")
        save_dir = os.path.join(save_dir, f"accumulate_{accumulate_mode}")
        os.makedirs(save_dir, exist_ok=True)

    df = get_log_probs(
        dataloader,
        model,
        accumulate=accumulate_mode,
        save_dir=save_dir if accumulate_mode != "all" else None,
    )

    out_filename = "likelihoods-goes16.csv"
    out_path = os.path.join(save_dir, out_filename)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved likelihoods to {out_path}")


if __name__ == "__main__":
    main()
