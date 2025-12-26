import numpy as np
import pytest
import xarray as xr

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


def test_thin_data_non_standard_dims():
    """Test _thin_data with non-standard dimension names ('y', 'x')."""
    y = np.arange(100)
    x = np.arange(200)
    u = xr.DataArray(np.zeros((len(y), len(x))), dims=["y", "x"], coords={"y": y, "x": x})
    v = xr.DataArray(np.zeros((len(y), len(x))), dims=["y", "x"], coords={"y": y, "x": x})
    thin = 10

    u_thinned, v_thinned, x2d, y2d = _thin_data(u, v, thin=thin)

    # Check that the dimensions are thinned correctly
    expected_y_len = 10  # 100 / 10
    expected_x_len = 20  # 200 / 10
    assert u_thinned.sizes["y"] == expected_y_len
    assert u_thinned.sizes["x"] == expected_x_len
    assert v_thinned.sizes["y"] == expected_y_len
    assert v_thinned.sizes["x"] == expected_x_len

    # Check that the meshgrid dimensions are correct
    assert x2d.shape == (expected_y_len, expected_x_len)
    assert y2d.shape == (expected_y_len, expected_x_len)
