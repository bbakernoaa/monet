import matplotlib.pyplot as plt
import pytest
from packaging.version import Version

try:
    import cartopy
    import cartopy.crs as ccrs

    cartopy_version = Version(cartopy.__version__)
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

import monet  # noqa: F401
from monet.plots import plots as p
from monet.plots.mapgen import draw_map


@pytest.mark.parametrize("which", ["imshow", "map", "contourf"])
@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_quick_with_cartopy_ax(which, sample_map_data):
    if not CARTOPY_AVAILABLE:
        pytest.skip("Cartopy is not installed")

    proj = tran = ccrs.PlateCarree()
    _fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
    getattr(sample_map_data.monet, f"quick_{which}")(ax=ax, transform=tran)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_draw_map_counties():
    _ = draw_map(counties=True, extent=[-110.5, -101, 36, 42])


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_spatial_plot(sample_map_data):
    fig, ax = p.spatial_plot(sample_map_data)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_spatial_bias_scatter(sample_map_frame):
    # Convert DataFrame to an xarray Dataset and rename columns for the new API
    ds = sample_map_frame.rename(columns={"CMAQ": "model", "Obs": "obs"}).to_xarray()
    ds = ds.set_coords(["latitude", "longitude"])
    fig, ax = p.spatial_bias_scatter(ds)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_spatial_imshow(sample_map_data):
    fig, ax = p.spatial_imshow(sample_map_data)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_spatial_contourf(sample_map_data):
    fig, ax = p.spatial_contourf(sample_map_data, cmap="viridis", levels=5)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_wind_quiver(sample_map_data):
    u, v = sample_map_data, sample_map_data
    fig, ax = p.wind_quiver(u, v)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_wind_barbs(sample_map_data):
    u, v = sample_map_data, sample_map_data
    fig, ax = p.wind_barbs(u, v)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
