import healpy as hp
import numpy as np
import xarray as xr
from scipy.ndimage import rotate


## healpix helper functions
def get_nest(dx: xr.Dataset):
    return dx.crs.healpix_order == "nest"


def get_nside(dx: xr.Dataset):
    return dx.crs.healpix_nside


def attach_coords(ds: xr.Dataset):
    lons, lats = hp.pix2ang(
        get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
    )
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degree_north"}),
        lon=(("cell",), lons, {"units": "degree_east"}),
    )


def nest_to_ring(ds: xr.Dataset, olr_key: str):
    # convert to ring
    olr_ring = hp.reorder(ds[olr_key], n2r=True)
    lat_ring = hp.reorder(ds.lat, n2r=True)
    lon_ring = hp.reorder(ds.lon, n2r=True)

    # copy crs with updated healpix ordering
    crs_ring = ds.crs
    crs_ring.attrs["healpix_order"] = "ring"

    ds = xr.Dataset(
        {
            olr_key: (("cell"), olr_ring),
            "lat": (("cell"), lat_ring),
            "lon": (("cell"), lon_ring),
        },
        coords={"crs": crs_ring, "cell": np.arange(len(olr_ring))},
    )

    return ds


def healpix_to_rotated_2d_matrix(
    ds: xr.Dataset,
    latitudes: tuple[float, float] = (-25, 25),
    olr_key: str = "OLR",
):
    if get_nest(ds):
        # convert to ring
        ds = nest_to_ring(ds, olr_key)

    # extract tropical band
    ds_tropics = ds.where((ds.lat > latitudes[0]) & (ds.lat < latitudes[1]), drop=True)
    lat_rows = np.unique(ds_tropics.lat).size
    lon_rows = ds_tropics.lon.size // lat_rows

    # reshape to 2D matrix (each row is a different healpix ring)
    olr_2d = np.reshape(ds_tropics[olr_key].values, [lat_rows, lon_rows]).astype(
        np.float32
    )
    lat_2d = np.reshape(ds_tropics.lat.values, [lat_rows, lon_rows]).astype(np.float32)
    lon_2d = np.reshape(ds_tropics.lon.values, [lat_rows, lon_rows]).astype(np.float32)

    _columns = np.where(~np.isnan(olr_2d).all(axis=0))[0][[0, -1]]
    _columns = slice(_columns[0], _columns[1] + 1)

    rotated_olr = rotate(olr_2d[:, _columns], 45, order=1, mode="constant", cval=0)
    rotated_olr[rotated_olr == 0] = np.nan

    rotated_lat = rotate(lat_2d[:, _columns], 45, order=1, mode="constant", cval=0)
    rotated_lat[rotated_lat == 0] = np.nan
    rotated_lon = rotate(lon_2d[:, _columns], 45, order=1, mode="constant", cval=0)
    rotated_lon[rotated_lon == 0] = np.nan

    ds_rotated = xr.Dataset(
        {
            olr_key: (("x", "y"), rotated_olr),
            "lat": (("x", "y"), rotated_lat),
            "lon": (("x", "y"), rotated_lon),
        },
        coords={
            "x": np.arange(rotated_lat.shape[0]),
            "y": np.arange(rotated_lon.shape[1]),
        },
    )
    return ds_rotated
