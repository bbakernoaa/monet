"""Interpolation utility functions for MONET"""

import numpy as np
import xarray as xr
from .resample import resample

def latlon_xarray_to_CoordinateDefinition(longitude=None, latitude=None):
    """Deprecated: Create pyresample SwathDefinition from xarray object.

    This function was part of the pyresample dependency and is deprecated.
    """
    raise NotImplementedError("This function relies on pyresample which has been removed.")


def lonlat_to_xesmf(longitude=None, latitude=None):
    """Deprecated: Create an empty xarray.Dataset with longitude and latitude coordinates.

    This function was part of the xesmf dependency and is deprecated.
    """
    from numpy import asarray, meshgrid

    lat = asarray(latitude)
    lon = asarray(longitude)

    # Handle scalar values
    if lat.ndim == 0:
        lat = lat[None]
    if lon.ndim == 0:
        lon = lon[None]

    # If both are 1D, create a 2D meshgrid
    if lat.ndim == 1 and lon.ndim == 1:
        lon_2d, lat_2d = meshgrid(lon, lat)
    # If both are already 2D with same shape, use them directly
    elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
        lon_2d, lat_2d = lon, lat
    # If they have different shapes or dimensions, create meshgrid
    else:
        if lat.ndim > 1:
            lat = lat.flatten()
        if lon.ndim > 1:
            lon = lon.flatten()
        lon_2d, lat_2d = meshgrid(lon, lat)

    dset = xr.Dataset(coords={"lon": (["y", "x"], lon_2d), "lat": (["y", "x"], lat_2d)})
    return dset


def lonlat_to_swathdefinition(longitude=None, latitude=None):
    """Deprecated: Create a pyresample SwathDefinition from longitude and latitude arrays.
    """
    raise NotImplementedError("This function relies on pyresample which has been removed.")


def nearest_point_swathdefinition(longitude=None, latitude=None):
    """Deprecated: Create a SwathDefinition for a single point.
    """
    raise NotImplementedError("This function relies on pyresample which has been removed.")


def constant_1d_xesmf(longitude=None, latitude=None):
    """Create a dataset with a constant latitude along a longitude array.

    Parameters
    ----------
    longitude : array-like
        Array of longitude values.
    latitude : float or array-like
        Latitude value(s) to use as a constant.

    Returns
    -------
    xarray.Dataset
        A dataset with coordinates suitable for regridding, where longitude varies
        but latitude is constant.
    """
    from numpy import asarray

    lat = asarray(latitude)
    lon = asarray(longitude)
    if lat.ndim == 0:
        lat = lat[None]
    if lon.ndim == 0:
        lon = lon[None]
    s = lat.shape[0]
    dset = xr.Dataset(
        coords={"lon": (["x", "y"], lon.reshape(s, 1)), "lat": (["x", "y"], lat.reshape(s, 1))}
    )
    return dset


def constant_lat_swathdefition(longitude=None, latitude=None):
    """Deprecated: Create a SwathDefinition with constant latitude along a longitude array.
    """
    raise NotImplementedError("This function relies on pyresample which has been removed.")


def constant_lon_swathdefition(longitude=None, latitude=None):
    """Deprecated: Create a SwathDefinition with constant longitude along a latitude array.
    """
    raise NotImplementedError("This function relies on pyresample which has been removed.")

