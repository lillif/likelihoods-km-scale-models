from datetime import datetime
from typing import Callable, Optional

import autoroot  # required for imports from src
import numpy as np
import nvtx  # for profiling annotations in pytorch lightning
import torch
import xarray as xr
from earth2grid.healpix import XY, PixelOrder, local2xy, reorder, ring2xy
from earth2grid.healpix_bare import ang2pix
from hydra.utils import instantiate
from loguru import logger
from torch.utils.data import Dataset

from .mip_dataset_loaders import load_olr_dataset


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


def extract_local_xy_patch(
    nside: int, lon0: float, lat0: float, pad: int
) -> torch.Tensor:
    """Extract a local patch of pixels around the specified point."""
    # TODO: might need to change to use corner rather than centre

    # Convert target point to pixel index
    i = ang2pix(nside, torch.tensor([lon0]), torch.tensor([lat0]), lonlat=True)
    origin = ring2xy(nside, i)
    x0 = origin % nside
    y0 = (origin % nside**2) // nside
    face0 = origin // (nside**2)

    # Create offset grid
    dx = torch.arange(-pad, pad)
    dy = torch.arange(-pad, pad)
    dy, dx = torch.meshgrid(dy, dx, indexing="ij")

    # Get local coordinates
    x, y, f = local2xy(nside, x0 + dx, y0 + dy, face0)

    # Convert to pixel indices
    pix = torch.where(f < 12, nside**2 * f + nside * y + x, -1)
    return pix


def extract_patch_pixels(patch_centers, nside, pad):
    zoom = int(np.log2(nside))
    n_pix = 12 * 4**zoom
    pixels_xy = torch.arange(0, n_pix)

    # for each XY index i, xy_to_nest[i] = the NEST index of that pixel
    xy_to_nest = reorder(
        pixels_xy,
        src_pixel_order=PixelOrder.NEST,
        dest_pixel_order=XY(),
    )

    patch_pixels = []
    for lon0, lat0 in patch_centers:
        xy_patch = extract_local_xy_patch(nside, lon0, lat0, pad)
        nest_patch = xy_to_nest[xy_patch]
        patch_pixels.append(nest_patch)
    return patch_pixels


class ModelHealpixDataset(Dataset):
    def __init__(
        self,
        time_slice: slice,
        cat_keys: list[str],
        cat_url: str,
        var: str,
        intake_args: dict,
        patch_centers: list[tuple[float, float]],
        cell_var: str = "cell",
        postprocess_cfg: dict = None,
        transform: Optional[Callable] = None,
        patch_size: int = 64,
        use_cache: bool = True,  # could remove this, should always be used!
    ):
        with nvtx.annotate("dataset init", color="cyan"):
            ds, olr_key = load_olr_dataset(
                time_slice=time_slice,
                cat_keys=cat_keys,
                cat_url=cat_url,
                var=var,
                intake_args=intake_args,
                cell_var=cell_var,
            )
            if postprocess_cfg is not None:
                ds, olr_key = apply_postprocessing(
                    ds=ds, olr_key=olr_key, postprocess_cfg=postprocess_cfg
                )

            self.ds = ds
            self.olr_key = olr_key
            self.transform = transform
            self.patch_centers = patch_centers
            self.n_times = len(self.ds.time)
            self.n_patches = len(self.patch_centers)
            self.pad = patch_size // 2

            # pre-compute patch pixel indices
            # use nested patch pixels to avoid reorder later
            self.patch_pixels = extract_patch_pixels(
                patch_centers=self.patch_centers,
                nside=int(self.ds.crs.healpix_nside),  # cast as int in case it is str
                pad=self.pad,
            )

            self.use_cache = use_cache
            if self.use_cache:
                logger.info("Using caching for OLR and time in ModelHealpixDataset.")
                self._cached_olr = None
                self._cached_time = None
                self._cached_idx = None

            ## lat is the same for all time steps, so load once here
            self.lat = torch.Tensor(self.ds.lat.values)
            self.lon = torch.Tensor(self.ds.lon.values)

    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def _unravel_index(self, idx):
        """
        Convert a flat index into time and patch indices.
        """
        time_idx = idx // self.n_patches
        ## TODO: to split times across train / val / test datasets:
        ## create list of time_idxes in dataloader, supply to dataset
        ## and use time_idx to get time_idxes[time_idx]
        ## which is used to index the xarray ds
        patch_idx = idx % self.n_patches
        return time_idx, patch_idx

    def _time_to_tensor(self, time_i):
        dt = time_i.astype("datetime64[ms]").astype(datetime)

        time_tensor = torch.tensor(
            [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second],
            dtype=torch.int16,
        )
        return time_tensor

    def __getitem__(self, idx):
        item = {}
        with nvtx.annotate("ds.isel", color="blue"):
            time_idx, patch_idx = self._unravel_index(idx)
            ds_i = self.ds.isel(time=time_idx)

        with nvtx.annotate("load olr", color="green"):
            if self.use_cache and self._cached_idx == time_idx:
                olr = self._cached_olr
                time_tensor = self._cached_time
            elif self.use_cache:
                olr = torch.Tensor(ds_i[self.olr_key].values)
                time_tensor = self._time_to_tensor(ds_i.time.values)
                self._cached_time = time_tensor
                self._cached_olr = olr
                self._cached_idx = time_idx
            else:
                olr = torch.Tensor(ds_i[self.olr_key].values)
                time_tensor = self._time_to_tensor(ds_i.time.values)

            """
            lat = torch.Tensor(ds_i["lat"].values)
            lon = torch.Tensor(ds_i["lon"].values)
            patch_pixels = self.patch_pixels[patch_idx]

            olr = torch.Tensor(ds_i[self.olr_key].isel(pixels=patch_pixels).values)
            lat = torch.Tensor(ds_i["lat"].isel(pixels=patch_pixels).values)
            lon = torch.Tensor(ds_i["lon"].isel(pixels=patch_pixels).values)
            """

        with nvtx.annotate("extract patch", color="orange"):
            # extract olr, lat and lon patch from pixel indices
            patch_pixels = self.patch_pixels[patch_idx]

            item["image"] = olr[patch_pixels][np.newaxis, ...]
            item["lat"] = self.lat[patch_pixels][np.newaxis, ...]
            item["lon"] = self.lon[patch_pixels][np.newaxis, ...]

            # time_i = ds_i.time.values
            item["time"] = time_tensor

        with nvtx.annotate("transform", color="purple"):
            if self.transform:
                item = self.transform(item)

        return item

    def __len__(self):
        return self.n_times * self.n_patches
