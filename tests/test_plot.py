import matplotlib.pyplot as plt
import pytest
import xarray as xr
from packaging.version import Version

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cartopy_version = Version(cartopy.__version__)
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

import monet  # noqa: F401
from monet.plots.mapgen import draw_map

da = xr.tutorial.load_dataset("air_temperature").air.isel(time=1)


@pytest.mark.parametrize("which", ["imshow", "map", "contourf"])
@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_quick_with_cartopy_ax(which):
    if not CARTOPY_AVAILABLE:
        pytest.skip("Cartopy is not installed")

    proj = tran = ccrs.PlateCarree()
    _, ax = plt.subplots(subplot_kw=dict(projection=proj))
    getattr(da.monet, f"quick_{which}")(ax=ax, transform=tran)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_draw_map_counties():
    _ = draw_map(counties=True, extent=[-110.5, -101, 36, 42])


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_draw_map_styling_kwargs(mocker):
    """Test that styling kwargs are passed to cartopy features."""
    mock_ax = mocker.Mock()
    mocker.patch("matplotlib.pyplot.subplots", return_value=(mocker.Mock(), mock_ax))

    draw_map(
        coastlines=True,
        countries=True,
        states=True,
        counties=True,
        coastlines_linewidth=0.5,
        countries_linewidth=1.0,
        states_linewidth=1.5,
        counties_linewidth=2.0,
        states_edgecolor="red",
        counties_edgecolor="blue",
    )

    mock_ax.coastlines.assert_called_once_with("10m", linewidth=0.5)

    calls = mock_ax.add_feature.call_args_list

    countries_call = None
    states_call = None
    counties_call = None

    for call in calls:
        feature = call.args[0]
        if feature == cfeature.BORDERS:
            countries_call = call
        elif isinstance(feature, cfeature.NaturalEarthFeature):
            if feature.name == "admin_1_states_provinces_lines":
                states_call = call
            elif feature.name == "admin_2_counties":
                counties_call = call

    assert countries_call is not None
    assert states_call is not None
    assert counties_call is not None

    assert countries_call.kwargs["linewidth"] == 1.0
    assert states_call.kwargs["linewidth"] == 1.5
    assert counties_call.kwargs["linewidth"] == 2.0

    states_feature = states_call.args[0]
    counties_feature = counties_call.args[0]

    assert states_feature.kwargs["edgecolor"] == "red"
    assert counties_feature.kwargs["edgecolor"] == "blue"


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed")
def test_draw_map_default_linewidth(mocker):
    """Test that the default linewidth is used when specific ones are not."""
    mock_ax = mocker.Mock()
    mocker.patch("matplotlib.pyplot.subplots", return_value=(mocker.Mock(), mock_ax))

    draw_map(
        coastlines=True,
        countries=True,
        states=True,
        counties=True,
        linewidth=0.7,
    )

    mock_ax.coastlines.assert_called_once_with("10m", linewidth=0.7)

    calls = mock_ax.add_feature.call_args_list
    for call in calls:
        assert call.kwargs["linewidth"] == 0.7


if __name__ == "__main__":
    test_quick_with_cartopy_ax("map")
    test_draw_map_counties()
    plt.show()
