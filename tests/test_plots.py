import numpy as np
import pytest
import xarray as xr

from monet.plots.plots import _thin_data, wind_quiver, wind_barbs


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    return xr.DataArray(
        np.random.rand(100, 100),
        dims=("lat", "lon"),
        coords={"lat": np.arange(100), "lon": np.arange(100)},
    )


def test_thin_data(sample_da):
    """Test the _thin_data function."""
    da = sample_da
    thin = 10
    thinned_da = _thin_data(da, thin)

    assert thinned_da.shape == (10, 10)
    assert thinned_da.lat.size == 10
    assert thinned_da.lon.size == 10
    assert np.array_equal(thinned_da.lat, np.arange(0, 100, 10))
    assert np.array_equal(thinned_da.lon, np.arange(0, 100, 10))


def test_wind_quiver_smoke(sample_da):
    """Smoke test for the wind_quiver function."""
    u = sample_da
    v = sample_da
    fig, ax = wind_quiver(u, v)
    assert fig is not None
    assert ax is not None


def test_wind_barbs_smoke(sample_da):
    """Smoke test for the wind_barbs function."""
    u = sample_da
    v = sample_da
    fig, ax = wind_barbs(u, v)
    assert fig is not None
    assert ax is not None
from monet.plots.plots import _thin_data


@pytest.fixture
def wind_data():
    """Create sample wind component data."""
    lat = np.arange(40, 50, 0.5)
    lon = np.arange(-100, -90, 0.5)
    u = xr.DataArray(
        np.random.rand(len(lat), len(lon)),
        coords=[("lat", lat), ("lon", lon)],
    )
    v = xr.DataArray(
        np.random.rand(len(lat), len(lon)),
        coords=[("lat", lat), ("lon", lon)],
    )
    return u, v


def test_thin_data(wind_data):
    """Test the _thin_data helper function."""
    u, v = wind_data
    thin = 5

    u_thinned, v_thinned, lon2d, lat2d = _thin_data(u, v, thin=thin)

    # Check that the dimensions are thinned correctly
    expected_lat_len = len(np.arange(len(u.lat))[::thin])
    expected_lon_len = len(np.arange(len(u.lon))[::thin])
    assert u_thinned.sizes["lat"] == expected_lat_len
    assert u_thinned.sizes["lon"] == expected_lon_len
    assert v_thinned.sizes["lat"] == expected_lat_len
    assert v_thinned.sizes["lon"] == expected_lon_len

    # Check that the meshgrid dimensions are correct
    assert lon2d.shape == (expected_lat_len, expected_lon_len)
    assert lat2d.shape == (u_thinned.sizes["lat"], u_thinned.sizes["lon"])

    # Check that the first and last longitude values in the meshgrid match the thinned coordinates
    assert lon2d[0, 0] == u_thinned.lon.values[0]
    assert lon2d[0, -1] == u_thinned.lon.values[-1]

    # Check that the first and last latitude values in the meshgrid match the thinned coordinates
    assert lat2d[0, 0] == u_thinned.lat.values[0]
    assert lat2d[-1, 0] == u_thinned.lat.values[-1]
