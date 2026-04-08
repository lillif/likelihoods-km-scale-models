import autoroot  # for imports from scripts/ and src/
import healpy as hp
import intake
import numpy as np
import xarray as xr

from .utils import add_crs


def add_latlon(ds: xr.Dataset, nested: bool = True):
    # get pixel indices from ds
    cell = ds["cell"].values
    nside = np.sqrt(len(cell) / 12).astype(int)
    # convert pixel index to theta, phi
    theta, phi = hp.pix2ang(nside, cell, nest=nested)
    # convert to degrees
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)
    # add ass coordinates to ds
    ds = ds.assign_coords(
        lat=("cell", lat),
        lon=("cell", lon),
    )
    return ds


# unify load_ifs, load_icon and load_global_hackathon_dataset:
def load_olr_dataset(
    time_slice: slice,
    cat_keys: list[str],
    cat_url: str = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml",
    var: str = "rlut",
    intake_args: dict = {"zoom": 9, "time_freq": "PT1H"},
    cell_var: str = "cell",
):
    cat = intake.open_catalog(cat_url)
    for key in cat_keys:
        cat = cat[key]

    ds = cat(**intake_args).to_dask().sel(time=time_slice)

    # unify cell dimension name across model datasets
    if cell_var != "cell":
        ds = ds.rename_dims({cell_var: "cell"})

    if "lat" not in ds or "lon" not in ds:
        ds = ds.pipe(add_latlon)

    if "crs" not in ds:
        ds = add_crs(ds)
    ds = ds[["crs", var]]
    return ds, var
