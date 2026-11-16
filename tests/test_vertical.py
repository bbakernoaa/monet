import numpy as np
import xarray as xr

from monet.util.vertical import calc_fv3_height, calc_fv3_pressure


def test_calc_fv3_pressure_eager_lazy():
    # Setup data
    ak = np.array([10.0, 20.0, 30.0])
    bk = np.array([0.1, 0.2, 0.3])
    ps = np.array([[100.0, 200.0], [300.0, 400.0]])

    # Expected result
    # P = ak + bk * ps
    # Result should be (3, 2, 2)
    expected = ak[:, np.newaxis, np.newaxis] + bk[:, np.newaxis, np.newaxis] * ps

    # Eager test (mixed inputs, should return xarray)
    res_xr = calc_fv3_pressure(ak, bk, ps)
    assert isinstance(res_xr, xr.DataArray)
    assert np.allclose(res_xr.values, expected)
    assert "history" in res_xr.attrs

    # Lazy test (xarray Dask)
    ak_da = xr.DataArray(ak, dims=["z"])
    bk_da = xr.DataArray(bk, dims=["z"])
    ps_da = xr.DataArray(ps, dims=["y", "x"])
    ps_lazy = ps_da.chunk({"x": 1, "y": 1})
    res_lazy = calc_fv3_pressure(ak_da, bk_da, ps_lazy)

    # Check it is still lazy
    assert res_lazy.chunks is not None

    # Compare results
    assert np.allclose(res_lazy.compute().values, expected)
    assert "history" in res_lazy.attrs


def test_calc_fv3_height_eager_lazy():
    # Simple setup
    # 2 layers, 3 interfaces
    temp = np.full((2, 2, 2), 300.0)
    phalf = np.array([500.0, 800.0, 1000.0])[:, np.newaxis, np.newaxis] * np.ones((1, 2, 2))
    hsfc = np.zeros((2, 2))

    from monet.met_funcs import R_d, g

    dz1 = (R_d * 300.0 / g) * np.log(1000.0 / 800.0)
    dz0 = (R_d * 300.0 / g) * np.log(800.0 / 500.0)

    h2 = 0.0  # bottom interface
    h1 = h2 + dz1  # middle interface
    h0 = h1 + dz0  # top interface
    expected = np.array([h0, h1, h2])[:, np.newaxis, np.newaxis] * np.ones((1, 2, 2))

    # Eager test
    res_xr = calc_fv3_height(temp, phalf, hsfc)
    assert isinstance(res_xr, xr.DataArray)
    assert np.allclose(res_xr.values, expected)
    assert "history" in res_xr.attrs

    # Lazy test
    temp_da = xr.DataArray(temp, dims=["z", "y", "x"])
    phalf_da = xr.DataArray(phalf, dims=["z", "y", "x"])
    temp_lazy = temp_da.chunk({"y": 1})
    phalf_lazy = phalf_da.chunk({"y": 1})

    res_lazy = calc_fv3_height(temp_lazy, phalf_lazy, hsfc)

    # Check it is still lazy
    assert res_lazy.chunks is not None
    assert np.allclose(res_lazy.compute().values, expected)
    assert "history" in res_lazy.attrs


def test_height_default_surface_and_integer_pressure():
    from monet.util.constants import R_d, g

    actual = calc_fv3_height(np.array([300., 300.]), np.array([500, 800, 1000]))
    expected = (R_d * 300 / g) * np.log(np.array([2., 1.25, 1.]))
    np.testing.assert_allclose(actual, expected)


def test_height_rejects_invalid_column_when_computed():
    import pytest

    temp = xr.DataArray([[300., 300.], [300., 300.]], dims=('column', 'z'))
    pressure = xr.DataArray([[500., 800., 1000.], [500., 400., 1000.]], dims=('column', 'z'))
    for lazy in (False, True):
        p = pressure.chunk({'column': 1}) if lazy else pressure
        with pytest.raises(ValueError, match='pressure'):
            calc_fv3_height(temp, p).compute()


def test_height_fractional_surface_with_integer_pressure():
    from monet.util.constants import R_d, g

    actual = calc_fv3_height(np.array([300., 300.]), np.array([500, 800, 1000]), 0.25)
    expected = 0.25 + (R_d * 300 / g) * np.log(np.array([2., 1.25, 1.]))
    np.testing.assert_allclose(actual, expected)


def test_height_rejects_nonpositive_and_reversed_pressure():
    import pytest

    for pressure in ([0., 800., 1000.], [1000., 800., 500.]):
        for lazy in (False, True):
            p = xr.DataArray(pressure, dims='z')
            if lazy:
                p = p.chunk()
            with pytest.raises(ValueError, match='pressure'):
                calc_fv3_height(np.array([300., 300.]), p).compute()
