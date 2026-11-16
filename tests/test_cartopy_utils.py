import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from monet.plots import cartopy_utils


@pytest.mark.parametrize("plot_method", ["plot_quick_imshow", "plot_quick_map"])
def test_quick_plot_creates_geographic_axes_with_data(plot_method):
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes

    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [30, 40, 50], "lon": [-120, -110, -100, -90]},
    )
    fig, ax = getattr(cartopy_utils, plot_method)(data, colorbar=False)
    try:
        assert isinstance(ax, GeoAxes)
        assert ax.figure is fig
        assert ax.images or ax.collections
    finally:
        plt.close(fig)
