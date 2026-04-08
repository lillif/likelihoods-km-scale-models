import os

import autoroot  # for imports from nsf repository
import numpy as np
import nvtx  # for profiling annotations in pytorch lightning
import pandas as pd
import torch
from loguru import logger


def _batch_to_numpy(x, lat, lon, time, log_prob):
    # ---- GPU reductions ----
    center_lat = lat.mean(dim=[1, 2, 3])
    center_lon = lon.mean(dim=[1, 2, 3])
    olr_mean = x.mean(dim=[1, 2, 3])
    olr_std = x.std(dim=[1, 2, 3])
    cloud_fraction = (x < 0.51).float().mean(dim=[1, 2, 3])
    cloud_fraction_dcc = (x < 0.35).float().mean(dim=[1, 2, 3])

    # ---- Move to CPU ONCE ----
    return {
        "time": time.numpy(),
        "lat": center_lat.cpu().numpy(),
        "lon": center_lon.cpu().numpy(),
        "likelihood": log_prob.cpu().numpy(),
        "olr_mean": olr_mean.cpu().numpy(),
        "olr_std": olr_std.cpu().numpy(),
        "cloud_fraction": cloud_fraction.cpu().numpy(),
        "cloud_fraction_dcc": cloud_fraction_dcc.cpu().numpy(),
    }


def get_log_probs(
    dataloader,
    model,
    accumulate: str = "all",  # all | month | day
    limit_to_first_n_batches=None,
    save_dir=None,
):
    model.eval()

    all_columns = {
        "year": [],
        "month": [],
        "day": [],
        "hour": [],
        "minute": [],
        "lat": [],
        "lon": [],
        "likelihood": [],
        "olr_mean": [],
        "olr_std": [],
        "cloud_fraction": [],
        "cloud_fraction_dcc": [],
    }

    if accumulate not in ["all", "month", "day"]:
        raise ValueError(
            f"Invalid value for accumulate: {accumulate}. Must be one of 'all', 'month', or 'day'."
        )
    if accumulate != "all":
        if save_dir is None:
            raise ValueError(
                f"save_dir must be specified when accumulate is set to '{accumulate}'."
            )
        current_key = None
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        logger.debug("starting likelihood computation loop over dataloader")
        for i, item in enumerate(dataloader):
            if accumulate == "month" and current_key is None:
                ## create key = (year, month):
                current_key = (item["time"][0, 0], item["time"][0, 1])
            elif accumulate == "day" and current_key is None:
                ## create key = (year, month, day):
                current_key = (
                    item["time"][0, 0],
                    item["time"][0, 1],
                    item["time"][0, 2],
                )
            elif accumulate in ["month", "day"]:
                acc_key = (
                    (item["time"][0, 0], item["time"][0, 1])
                    if accumulate == "month"
                    else (item["time"][0, 0], item["time"][0, 1], item["time"][0, 2])
                )
                if acc_key != current_key:
                    # ---- concatenate data for this month ----
                    for k in all_columns:
                        all_columns[k] = np.concatenate(all_columns[k])
                    acc_df = pd.DataFrame(all_columns)
                    acc_str = (
                        f"{current_key[0]:04d}-{current_key[1]:02d}"
                        if accumulate == "month"
                        else f"{current_key[0]:04d}-{current_key[1]:02d}-{current_key[2]:02d}"
                    )

                    logger.info(
                        f"{accumulate.capitalize()} change detected at batch {i}, finished {accumulate}: {acc_str}."
                    )

                    out_filename = f"likelihoods-{accumulate}-{acc_str}.csv"
                    out_path = os.path.join(save_dir, out_filename)
                    acc_df.to_csv(out_path, index=False)
                    current_key = acc_key
                    # reset accumulated data
                    for k in all_columns:
                        all_columns[k] = []
            with nvtx.annotate(f"Processing batch {i}", color="blue"):
                x = item["image"].to("cuda", non_blocking=True)

                lat = item["lat"]
                lon = item["lon"]
                time = item["time"]

                log_prob = model.flow.log_prob(x)

                batch_np = _batch_to_numpy(x, lat, lon, time, log_prob)

                # unpack time columns vectorized
                times = batch_np["time"]
                all_columns["year"].append(times[:, 0])
                all_columns["month"].append(times[:, 1])
                all_columns["day"].append(times[:, 2])
                all_columns["hour"].append(times[:, 3])
                all_columns["minute"].append(times[:, 4])

                for k in [
                    "lat",
                    "lon",
                    "likelihood",
                    "olr_mean",
                    "olr_std",
                    "cloud_fraction",
                    "cloud_fraction_dcc",
                ]:
                    all_columns[k].append(batch_np[k])

                if (
                    limit_to_first_n_batches is not None
                    and (i + 1) >= limit_to_first_n_batches
                ):
                    break

    # ---- Concatenate once at the end ----
    for k in all_columns:
        all_columns[k] = np.concatenate(all_columns[k])

    if save_dir is not None:
        if accumulate == "all":
            out_filename = f"likelihoods-all.csv"
            out_path = os.path.join(save_dir, out_filename)
            pd.DataFrame(all_columns).to_csv(out_path, index=False)
            logger.info(f"Saved likelihoods to {out_path}")
        else:
            logger.info(
                f"Finished processing all batches. Final {accumulate} will be saved."
            )
            acc_str = (
                f"{current_key[0]:04d}-{current_key[1]:02d}"
                if accumulate == "month"
                else f"{current_key[0]:04d}-{current_key[1]:02d}-{current_key[2]:02d}"
            )
            out_filename = f"likelihoods-{accumulate}-{acc_str}.csv"
            out_path = os.path.join(save_dir, out_filename)
            pd.DataFrame(all_columns).to_csv(out_path, index=False)
            logger.info(f"Saved likelihoods to {out_path}")

    return pd.DataFrame(all_columns)
