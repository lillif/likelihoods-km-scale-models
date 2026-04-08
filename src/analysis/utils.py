import math
import os
from datetime import datetime

import pandas as pd
from global_land_mask import globe
from loguru import logger


def get_patch_likelihood_distribution(df, column="likelihood"):
    # Group by unique (lon, lat) pairs
    grouped = df.groupby(["lon", "lat"])

    # Get likelihood distribution per group (as lists of values)
    grouped[column].apply(list)

    # Calculate the mean likelihood per group
    grouped[column].mean().reset_index()

    # If you want both distributions and mean together
    summary = (
        grouped[column]
        .agg(values=list, mean="mean")  # full distribution as list  # mean likelihood
        .reset_index()
    )
    return summary


def add_land_ocean_flag(df, lat_col="lat", lon_col="lon", flag_col="is_land"):
    # ensure longitude is in [-180, 180] (this is required by globe.is_land)
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    df[flag_col] = globe.is_land(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    return df


# split likelihoods by land/ocean flag
def split_likelihoods(df, like_col="likelihood", flag_col="is_land"):
    land = df.loc[df[flag_col], like_col].to_numpy()
    ocean = df.loc[~df[flag_col], like_col].to_numpy()
    return land, ocean


def get_dates_from_files(files: list[str], ext=".csv") -> list[datetime]:
    """
    Extract dates from a list of filenames.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """

    dates = [
        datetime.strptime(file.split("/")[-1].split("_")[0], f"%Y-%m-%dT%H{ext}")
        for file in files
    ]
    return dates


def get_split(files: list, split_dict: dict) -> tuple[list, list]:
    """
    Split files based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_dict (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    # Extract dates from filenames
    filenames = [file.split("/")[-1] for file in files]
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "files": files, "date": dates})

    # Check if years, months, and days are specified
    if "years" not in split_dict.keys() or split_dict["years"] is None:
        logger.info("No years specified for split. Using all years.")
        split_dict["years"] = df.date.dt.year.unique().tolist()
    if "months" not in split_dict.keys() or split_dict["months"] is None:
        logger.info("No months specified for split. Using all months.")
        split_dict["months"] = df.date.dt.month.unique().tolist()
    if "days" not in split_dict.keys() or split_dict["days"] is None:
        logger.info("No days specified for split. Using all days.")
        split_dict["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions specified split
    condition = (
        (df.date.dt.year.isin(split_dict["years"]))
        & (df.date.dt.month.isin(split_dict["months"]))
        & (df.date.dt.day.isin(split_dict["days"]))
    )

    # Extract filenames based on conditions
    split_files = df[condition].files.tolist()

    # Check if files are allocated properly
    if len(split_files) == 0:
        raise ValueError("No files found. Check split specification.")

    # Sort files
    split_files.sort()

    return split_files


def load_mip_likelihood_df(
    model_dir: str,
    add_land_ocean: bool = False,
    add_local_solar_time: bool = False,
    patch_size: tuple = (64, 64),
    split_dict: dict[str, list] = {},
) -> dict:
    """
    example split_dict :
        {
            'years': [2020],
            'months': [1,2],
            'days': [1,2,3,4,5]
        }
    (each key is optional, if not provided, all years/months/days will be loaded)
    """

    files = os.listdir(model_dir)

    if split_dict:
        files = get_split(files=files, split_dict=split_dict)

    _dfs = []
    for f in files:
        _dfs += [pd.read_csv(os.path.join(model_dir, f))]

    df = pd.concat(_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # calculate bits per dimension and add to df (change this in inference script to save bpd directly?)
    df["bpd"] = df["likelihood"].apply(
        lambda x: x / (math.log(2) * patch_size[0] * patch_size[1])
    )
    if add_land_ocean:
        df = add_land_ocean_flag(df)
    if add_local_solar_time:
        # approximate 'LST' column
        df["hour"] = df["hour"].astype(int)
        # LST = GMT + longitude/15, wrap around 24 hours
        df["hour_LST"] = ((df["hour"] + df["lon"] / 15) % 24).astype(int)
    return df
