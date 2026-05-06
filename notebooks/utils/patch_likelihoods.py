import math
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from global_land_mask import globe

import autoroot # for imports from src
from src.analysis import add_land_ocean_flag

from loguru import logger

def load_likelihood_dfs(
    likelihoods_dir: str,
    datasets: list[str] = [
        "goes_train",
        "goes_test",
        "ifs_train",
        "ifs_test",
        "icon_train",
        "icon_test",
    ],
    add_land_ocean: bool = False,
    add_local_solar_time: bool = False,
    patch_size: int = 64,
) -> dict:
    likelihood_dfs = {}
    for dataset in datasets:
        try:
            likelihood_dfs[dataset] = pd.read_parquet(
                f"{likelihoods_dir}/{dataset}_likelihoods.parquet"
            )
        except Exception:
            try:
                likelihood_dfs[dataset] = pd.read_csv(
                    f"{likelihoods_dir}/{dataset}_likelihoods.csv"
                )
            except Exception as e:
                logger.error(f"Error loading likelihoods for {dataset}: {e}")
                continue

        likelihood_dfs[dataset]["bpd"] = likelihood_dfs[dataset]["likelihood"].apply(
            lambda x: x / (patch_size * patch_size)
        )
        if add_land_ocean:
            likelihood_dfs[dataset] = add_land_ocean_flag(likelihood_dfs[dataset])
        if add_local_solar_time:
            # approximate 'LST' column
            likelihood_dfs[dataset]["hour"] = likelihood_dfs[dataset]["hour"].astype(
                int
            )
            # LST = GMT + longitude/15, wrap around 24 hours
            likelihood_dfs[dataset]["hour_LST"] = (
                likelihood_dfs[dataset]["hour"] + likelihood_dfs[dataset]["lon"] / 15
            ) % 24
            # Optional: convert to integer hours for binning
            likelihood_dfs[dataset]["hour_LST"] = likelihood_dfs[dataset][
                "hour_LST"
            ].astype(int)
    return likelihood_dfs


def load_all_dataset_likelihoods(
    likelihood_dir: str,
    datasets: list[str] = [
        "goes_train",
        "goes_test",
        "ifs_train",
        "ifs_test",
        "icon_train",
        "icon_test",
    ],
    add_land_ocean: bool = False,
    add_local_solar_time: bool = False,
) -> dict:
    """
    Loads likelihoods csvs and then merges the dataframes for each dataset (e.g., merge 'train' and 'test' into one dataset).
    """
    all_dataset_likelihoods = {}
    dataset_keys = {d.split("_")[0] for d in datasets}
    for key in dataset_keys:
        relevant_datasets = [d for d in datasets if d.startswith(key)]
        dataset_dfs = load_likelihood_dfs(
            likelihood_dir,
            relevant_datasets,
            add_land_ocean=add_land_ocean,
            add_local_solar_time=add_local_solar_time,
        )
        all_dataset_likelihoods[key] = pd.concat(
            [dataset_dfs[d] for d in relevant_datasets], ignore_index=True
        )
    return all_dataset_likelihoods


def match_patch_index(row, df_patch_numbers, atol):
    mask = np.isclose(df_patch_numbers["lat"], row["lat"], atol=atol) & np.isclose(
        df_patch_numbers["lon"], row["lon"], atol=atol
    )
    match = df_patch_numbers.loc[mask, "patch_index"]
    return match.iloc[0] if not match.empty else np.nan


def minmax_by_patch_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        {
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].astype(int),
            "minute": df["minute"].astype(int),
        }
    )
    df["ts_str"] = df["timestamp"].dt.strftime("%Y%m%dT%H%M%S")

    grp_keys = ["patch_index", "hour"]
    # idx of min/max bpd per (patch_index, hour)
    idx_min = df.groupby(grp_keys)["bpd"].idxmin()
    idx_max = df.groupby(grp_keys)["bpd"].idxmax()

    min_rows = df.loc[idx_min, grp_keys + ["bpd", "ts_str"]].rename(
        columns={"bpd": "min_bpd", "ts_str": "min_bpd_time"}
    )

    max_rows = df.loc[idx_max, grp_keys + ["bpd", "ts_str"]].rename(
        columns={"bpd": "max_bpd", "ts_str": "max_bpd_time"}
    )

    out = (
        min_rows.merge(max_rows, on=grp_keys, how="outer")
        .loc[
            :,
            [
                "patch_index",
                "hour",
                "min_bpd",
                "max_bpd",
                "min_bpd_time",
                "max_bpd_time",
            ],
        ]
        .sort_values(["patch_index", "hour"])
        .reset_index(drop=True)
    )
    return out


def lon360_to_lon180(lon):
    """
    Convert longitude from [0, 360) to [-180, 180).
    """
    return ((lon + 180) % 360) - 180


def plot_min_max_patches(
    f_min_patch,
    f_max_patch,
    dataset_example,
    dataset,
    axs=None,
    fig=None,
    vmin=None,
    vmax=None,
):
    """
    Function to plot min and max likelihood patches side by side.
    """
    min_patch = np.load(f_min_patch)
    max_patch = np.load(f_max_patch)

    min_bpd = dataset_example["min_bpd"].values[0]
    max_bpd = dataset_example["max_bpd"].values[0]

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(5.5, 3))

    ax = axs[0]
    im = ax.imshow(min_patch["olr"], cmap="Blues", vmin=vmin, vmax=vmax)

    lat = min_patch["lat"].mean()
    lon = lon360_to_lon180(
        min_patch["lon"].mean()
    )  # convert to [-180, 180) for globe_is_land

    # Parse the input string
    dt = datetime.strptime(dataset_example["min_bpd_time"].values[0], "%Y%m%dT%H%M%S")
    dt_formatted = dt.strftime("%d-%m-%Y %H:%M") + "GMT"
    ax.set_title(
        f'{dt_formatted.split(" ")[0]} ({min_bpd:.2f} bits/dim)', fontsize=10.5
    )

    ax.set_yticks([0, 20, 40, 60])

    ax = axs[1]
    im = ax.imshow(max_patch["olr"], cmap="Blues", vmin=vmin, vmax=vmax)
    max_lat = max_patch["lat"].mean()
    max_lon = lon360_to_lon180(
        max_patch["lon"].mean()
    )  # convert to [-180, 180) for globe_is_land

    # sanity check
    assert np.isclose(lat, max_lat)
    assert np.isclose(lon, max_lon)

    dt = datetime.strptime(dataset_example["max_bpd_time"].values[0], "%Y%m%dT%H%M%S")
    dt_formatted = dt.strftime("%d-%m-%Y %H:%M") + "GMT"

    cbar_ax = fig.add_axes([0.985, 0.085, 0.02, 0.745])  # tweak as needed
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("OLR (W/m²)")
    ax.set_title(
        f'{dt_formatted.split(" ")[0]} ({max_bpd:.2f} bits/dim)', fontsize=10.5
    )

    ax.set_yticks([0, 20, 40, 60])
    n_or_s = "N" if lat >= 0 else "S"
    e_or_w = "E" if lon >= 0 else "W"

    suptitle = f"{dataset.split('_')[0].upper()} - {dt_formatted.split(' ')[1]} at {abs(lat):.1f}°{n_or_s}, {abs(lon):.1f}°{e_or_w}"

    plt.suptitle(suptitle, weight="bold")

    if globe.is_land(lat, lon):
        return "land"
    else:
        return "ocean"
