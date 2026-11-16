import importlib.machinery
import sys
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def mock_if_missing(module_names):
    for name in module_names:
        try:
            __import__(name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            if name not in sys.modules:
                m = MagicMock()
                # Use a real ModuleSpec to avoid xarray/importlib issues
                m.__spec__ = importlib.machinery.ModuleSpec(name, None)
                # Ensure mocked packages have __path__ and __version__ where needed
                if "." in name or name in ["cartopy", "monetio"]:
                    m.__path__ = []
                if name == "cartopy":
                    m.__version__ = "0.22.0"
                sys.modules[name] = m


@pytest.fixture(autouse=True)
def close_all_figures():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_map_data():
    """Small deterministic field for offline map and grid tests."""
    return xr.DataArray(
        270.0 + np.arange(120, dtype=float).reshape(10, 12) / 10.0,
        dims=("lat", "lon"),
        coords={"lat": np.linspace(30.0, 60.0, 10), "lon": np.linspace(-125.0, -65.0, 12)},
        name="air",
    )


@pytest.fixture
def sample_map_frame(sample_map_data):
    """Paired observations on the sample field's grid."""
    lon, lat = np.meshgrid(sample_map_data.lon, sample_map_data.lat)
    values = sample_map_data.values.ravel()
    return pd.DataFrame(
        {
            "latitude": lat.ravel(),
            "longitude": lon.ravel(),
            "CMAQ": values * 0.9,
            "Obs": values * 1.1,
            "datetime": pd.Timestamp("2013-01-01 01:00:00"),
        }
    )


# Only mock truly optional dependencies.
# Core dependencies like dask, axis, monet_stats, pytspack should NOT be mocked
# so that Vectorized Protocol compliance and proper test skipping can be verified.
mock_if_missing(
    [
        "monetio",
        "geopandas",
        "rasterio",
        "shapely",
    ]
)
