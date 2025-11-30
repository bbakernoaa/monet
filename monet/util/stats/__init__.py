"""
Statistics submodule for MONET utility functions.
"""

import numpy as np
import xarray as xr
try:
    import dask.array as da
except ImportError:
    da = None

from monet_stats import *
from monet_stats.error_metrics import *
from monet_stats.relative_metrics import *
from monet_stats.contingency_metrics import *
from monet_stats.correlation_metrics import *
from monet_stats.spatial_ensemble_metrics import *
from monet_stats.utils_stats import *

# Overrides for compatibility with legacy MONET behavior or fixing inconsistencies in monet-stats

def MO(obs, mod, axis=None):
    """Mean Observations."""
    if isinstance(obs, xr.DataArray):
        return obs.mean(dim=axis)
    if hasattr(obs, 'mean'):
        return obs.mean(axis=axis)
    return np.mean(obs, axis=axis)

def MdnO(obs, mod, axis=None):
    """Median Observations."""
    if isinstance(obs, xr.DataArray):
        return obs.median(dim=axis)
    if da is not None and isinstance(obs, da.Array):
        if axis is None:
            axis = 0 # Default to 0 for dask if None, mimicking old behavior
        return da.median(obs, axis=axis)
    if hasattr(obs, 'median'):
        return obs.median(axis=axis)
    return np.median(obs, axis=axis)

def STDO(obs, mod, axis=None):
    """Standard Deviation of Observations."""
    if isinstance(obs, xr.DataArray):
        return obs.std(dim=axis)
    return np.std(obs, axis=axis)

def STDP(obs, mod, axis=None):
    """Standard Deviation of Predictions."""
    if isinstance(mod, xr.DataArray):
        return mod.std(dim=axis)
    return np.std(mod, axis=axis)

def MP(obs, mod, axis=None):
    """Mean Predictions."""
    # Ensure we use mod, not obs-mod
    if isinstance(mod, xr.DataArray):
        return mod.mean(dim=axis)
    if hasattr(mod, 'mean'):
        return mod.mean(axis=axis)
    return np.mean(mod, axis=axis)

def MdnP(obs, mod, axis=None):
    """Median Predictions."""
    if isinstance(mod, xr.DataArray):
        return mod.median(dim=axis)
    if da is not None and isinstance(mod, da.Array):
        if axis is None:
            axis = 0
        return da.median(mod, axis=axis)
    if hasattr(mod, 'median'):
        return mod.median(axis=axis)
    return np.median(mod, axis=axis)

def RM(obs, mod, axis=None):
    """Ratio of Means (mean(obs/mod))."""
    # Note: legacy code used mean(obs/mod).
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
         obs, mod = xr.align(obs, mod, join="inner")
         return (obs / mod).mean(dim=axis)
    return np.mean(obs / mod, axis=axis)

def RMdn(obs, mod, axis=None):
    """Ratio of Medians (median(obs/mod))."""
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
         obs, mod = xr.align(obs, mod, join="inner")
         return (obs / mod).median(dim=axis)
    # Handle dask
    ratio = obs / mod
    if da is not None and isinstance(ratio, da.Array):
        if axis is None:
            axis = 0
        return da.median(ratio, axis=axis)
    return np.median(ratio, axis=axis)

def MB(obs, mod, axis=None):
    """Mean Bias (Mod - Obs)."""
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return (mod - obs).mean(dim=axis)
    return np.mean(mod - obs, axis=axis)

def MdnB(obs, mod, axis=None):
    """Median Bias (Mod - Obs)."""
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return (mod - obs).median(dim=axis)
    diff = mod - obs
    if da is not None and isinstance(diff, da.Array):
        if axis is None:
            axis = 0
        return da.median(diff, axis=axis)
    return np.median(diff, axis=axis)

# Re-exporting everything including overrides
__all__ = [
    "HSS", "ETS", "CSI", "scores", "POD", "FAR", "FBI", "TSS",
    "R2", "RMSE", "WDRMSE_m", "WDRMSE", "RMSEs", "RMSEu", "d1", "E1", "IOA_m", "IOA", "WDIOA_m", "WDIOA", "AC", "WDAC", "taylor_skill", "KGE", "spearmanr", "kendalltau",
    "STDO", "STDP", "MNB", "MNE", "MdnNB", "MdnNE", "NMdnGE", "NO", "NOP", "NP", "MO", "MP", "MdnO", "MdnP", "RM", "RMdn", "MB", "MdnB", "WDMB_m", "WDMB", "WDMdnB",
    "NMB", "WDNMB_m", "NMB_ABS", "NMdnB", "FB", "ME", "MdnE", "WDME_m", "WDME", "WDMdnE", "NME_m", "NME_m_ABS", "NME", "NMdnE", "FE", "USUTPB", "USUTPE", "MNPB", "MdnNPB", "MNPE", "MdnNPE",
    "FSS", "EDS", "CRPS", "spread_error", "SAL",
    "matchedcompressed", "matchmasks", "circlebias_m", "circlebias",
]
