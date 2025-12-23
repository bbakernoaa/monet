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
