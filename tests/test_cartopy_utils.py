import matplotlib
import pytest

matplotlib.use("Agg")
from monet.plots import cartopy_utils


@pytest.mark.parametrize("plot_method", ["plot_quick_imshow", "plot_quick_map"])
def test_quick_plot_creates_geographic_axes_with_data(plot_method, sample_map_data):
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes

    fig, ax = getattr(cartopy_utils, plot_method)(sample_map_data, colorbar=False)
    try:
        assert isinstance(ax, GeoAxes)
        assert ax.figure is fig
        assert ax.images or ax.collections
    finally:
        plt.close(fig)
