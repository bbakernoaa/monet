import numpy as np
import pytest
import xarray as xr

from monet.plots.plots import _thin_data


def test__thin_data_default_dims():
    """Test _thin_data with default 'lat' and 'lon' dimensions."""
    u = xr.DataArray(
        np.random.rand(100, 100),
        dims=["lat", "lon"],
        coords={"lat": np.arange(100), "lon": np.arange(100)},
    )
    v = u.copy()
    thin = 10
    u_thinned, v_thinned, lon2d, lat2d = _thin_data(u, v, thin=thin)

    # The expected shape after thinning by 10 is (10, 10)
    expected_shape = (10, 10)
    assert u_thinned.shape == expected_shape
    assert v_thinned.shape == expected_shape
    assert lon2d.shape == expected_shape
    assert lat2d.shape == expected_shape

    # Check that the values are correct
    assert u_thinned.values[0, 0] == u.values[0, 0]
    assert u_thinned.values[1, 1] == u.values[10, 10]


def test__thin_data_custom_dims():
    """Test _thin_data with custom 'y' and 'x' dimensions."""
    u = xr.DataArray(
        np.random.rand(50, 50),
        dims=["y", "x"],
        coords={"y": np.arange(50), "x": np.arange(50)},
    )
    v = u.copy()
    thin = 5
    u_thinned, v_thinned, x2d, y2d = _thin_data(u, v, thin=thin)

    # The expected shape after thinning by 5 is (10, 10)
    expected_shape = (10, 10)
    assert u_thinned.shape == expected_shape
    assert v_thinned.shape == expected_shape
    assert x2d.shape == expected_shape
    assert y2d.shape == expected_shape

    # Check that the coordinate names are preserved
    assert "y" in u_thinned.dims
    assert "x" in u_thinned.dims

    # Check that the values are correct
    assert u_thinned.values[0, 0] == u.values[0, 0]
    assert u_thinned.values[1, 1] == u.values[5, 5]


def test__thin_data_1d_input_raises_error():
    """Test that _thin_data raises a ValueError for 1D input."""
    u = xr.DataArray(np.random.rand(100), dims=["x"])
    v = u.copy()
    with pytest.raises(ValueError, match="Input DataArray `u` must have at least 2 dimensions."):
        _thin_data(u, v)
