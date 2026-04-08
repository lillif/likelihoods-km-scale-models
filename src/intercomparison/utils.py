from typing import Callable, Optional

import autoroot  # for imports from nsf repository
import healpy as hp
import numpy as np
import torch
import xarray as xr
from loguru import logger
from torch.utils.data import Dataset

from src.patcher.numpy_patcher import NumpyPatcherMultipleArrays
from src.utils.healpix import healpix_to_rotated_2d_matrix, nest_to_ring


def process_healpix_matrix(
    image_i: np.ndarray,
    lat_i: np.ndarray,
    lon_i: np.ndarray,
    lon_bounds: tuple[float, float],
):
    # crop to keep only region of interest using the longitude bounds set in self.lon_bounds
    mask = np.where(
        (~np.isnan(lon_i)) & (lon_i >= lon_bounds[0]) & (lon_i <= lon_bounds[1]),
        1,
        0,
    )

    # remove areas that are outside the longitude bounds (set to NaN)
    all_zero_rows = np.where(np.all(mask == 0, axis=1))[0]
    all_zero_columns = np.where(np.all(mask == 0, axis=0))[0]

    lon_i = np.delete(lon_i, all_zero_rows, axis=0)
    lon_i = np.delete(lon_i, all_zero_columns, axis=1)

    lat_i = np.delete(lat_i, all_zero_rows, axis=0)
    lat_i = np.delete(lat_i, all_zero_columns, axis=1)

    image_i = np.delete(image_i, all_zero_rows, axis=0)
    image_i = np.delete(image_i, all_zero_columns, axis=1)

    return image_i, lat_i, lon_i


class OlrDatasetFromMultipatches(Dataset):
    def __init__(
        self, multipatches: list[np.ndarray], transform: Callable, time: np.datetime64
    ):
        self.multipatches = multipatches
        self.transform = transform

        # create time array as expected by _batch_to_records
        t = time.astype("datetime64[ms]").astype(object)
        self.time = np.array([t.year, t.month, t.day, t.hour, t.minute, t.second])

    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, idx) -> np.ndarray:
        item = {}
        image, lat, lon = self.multipatches[idx]

        item["lat"] = lat[np.newaxis, ...]
        item["lon"] = lon[np.newaxis, ...]
        item["image"] = image[np.newaxis, ...]
        item["time"] = self.time

        if self.transform:
            item = self.transform(item)

        for key, x in item.items():
            if not isinstance(x, torch.Tensor) and key != "time":
                logger.error(
                    f"Transforms did not return a torch tensor for key {key}, but {type(x)}"
                )
        return item

    def __len__(self):
        return len(self.multipatches)


def get_multipatches_from_ds(
    ds_i: xr.Dataset,
    olr_key: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    patch_size: int,
    stride: int,
    patch_numbers: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    ds_i.time
    ds_i = nest_to_ring(ds_i, olr_key=olr_key)

    ds_matrix_i = healpix_to_rotated_2d_matrix(
        ds_i, olr_key=olr_key, latitudes=lat_bounds
    )

    image_i, lat_i, lon_i = process_healpix_matrix(
        ds_matrix_i[olr_key].values,
        ds_matrix_i.lat.values,
        ds_matrix_i.lon.values,
        lon_bounds=lon_bounds,
    )

    assert image_i.shape == (916, 916)  # check that shape matches GOES shape

    # some model datasets (e.g. ICON) have NaNs at lat=0, fix these
    lat_i[np.where(np.isnan(lat_i))] = 0

    multipatcher = NumpyPatcherMultipleArrays(
        [image_i, lat_i, lon_i], patch_size=patch_size, stride=stride
    )

    multipatches = np.array(list(multipatcher.get_patches()))

    if patch_numbers is not None:
        multipatches = multipatches[patch_numbers]

    return list(multipatches)


### IFS helper functions


class GetOLRFromTTR:
    def __init__(
        self, var: str = "ttr", olr_name: str = "rlut", seconds_per_timestep: int = 3600
    ):
        self.var = var
        self.olr_name = olr_name
        self.seconds_per_timestep = seconds_per_timestep

    def __call__(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        # Compute OLR from TTR
        da_olr = -ds[self.var] / self.seconds_per_timestep

        # Make a copy of the dataset so we don't modify in-place
        ds_new = ds.copy()

        # Drop the original TTR variable
        ds_new = ds_new.drop_vars(self.var)

        # Add the new OLR variable
        ds_new[self.olr_name] = da_olr

        return ds_new, self.olr_name


class InterpolateNestedHealpix:
    def __init__(self, zoom_fine: int, zoom_coarse: int, cell_dim: str = "cell"):
        self.zoom_fine = zoom_fine
        self.zoom_coarse = zoom_coarse
        self.cell_dim = cell_dim

    def __call__(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        da = ds[var]

        if self.zoom_coarse >= self.zoom_fine:
            raise ValueError("zoom_coarse must be < zoom_fine")

        nside_fine = 2**self.zoom_fine
        nside_coarse = 2**self.zoom_coarse

        npix_fine = hp.nside2npix(nside_fine)
        npix_coarse = hp.nside2npix(nside_coarse)

        if da.sizes[self.cell_dim] != npix_fine:
            raise ValueError(f"{self.cell_dim} dimension size does not match zoom_fine")

        ratio = nside_fine // nside_coarse
        n_children = ratio**2
        child_dim = f"{self.cell_dim}_child"

        # reshape to (coarse_cell, child)
        da_reshaped = da.assign_coords({self.cell_dim: np.arange(npix_fine)}).stack(
            __tmp=(self.cell_dim,)
        )
        da_reshaped = da_reshaped.data.reshape(
            (*da.shape[:-1], npix_coarse, n_children)
        )

        new_dims = list(da.dims[:-1]) + [self.cell_dim, child_dim]
        da_reshaped = xr.DataArray(
            da_reshaped,
            dims=new_dims,
            coords={
                **{d: da.coords[d] for d in da.dims if d != self.cell_dim},
                self.cell_dim: np.arange(npix_coarse),
                child_dim: np.arange(n_children),
            },
            attrs=da.attrs,
            name=da.name,
        )

        # average over child dimension to get coarse resolution
        da_coarse = da_reshaped.mean(dim=child_dim)

        return da_coarse.to_dataset(name=var), var


def add_crs(ds, nested: bool = True):
    ds.coords["crs"] = np.array([np.nan])
    ds.coords["crs"].attrs["grid_mapping_name"] = "healpix"
    nside = np.sqrt(len(ds.cell) / 12).astype(int)
    ds.coords["crs"].attrs["healpix_nside"] = str(nside)
    ds.coords["crs"].attrs["healpix_order"] = "nest" if nested else "ring"
    return ds


class AddCRS:
    def __init__(self, nested: bool = True):
        self.nested = nested

    def __call__(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        ds.coords["crs"] = np.array([np.nan])
        ds.coords["crs"].attrs["grid_mapping_name"] = "healpix"
        nside = np.sqrt(len(ds.cell) / 12).astype(int)
        ds.coords["crs"].attrs["healpix_nside"] = nside
        ds.coords["crs"].attrs["healpix_order"] = "nest" if self.nested else "ring"
        return ds, var


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


class AddLatLon:
    def __init__(self, nested: bool = True):
        self.nested = nested

    def __call__(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        cell = ds["cell"].values
        nside = np.sqrt(len(cell) / 12).astype(int)
        theta, phi = hp.pix2ang(nside, cell, nest=self.nested)
        lat = 90.0 - np.degrees(theta)
        lon = np.degrees(phi)
        ds = ds.assign_coords(
            lat=("cell", lat),
            lon=("cell", lon),
        )
        return ds, var


class TemporalResample:
    def __init__(self, target_time: str = "1h"):
        self.target_time = target_time

    def __call__(self, ds: xr.Dataset, var: str) -> xr.Dataset:
        ds = ds.resample(time=self.target_time).mean()
        return ds, var
