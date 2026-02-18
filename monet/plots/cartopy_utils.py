"""Cartopy-based plotting utilities for MONET."""

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
except ImportError:
    ccrs = None
    GeoAxes = None


def _get_plot_xy(da):
    """Detect latitude and longitude coordinate names for xarray plotting."""
    from ..accessors.base import BaseAccessor

    lat_name, lon_name = BaseAccessor._detect_latlon_names(da)
    return lon_name, lat_name


def plot_quick_imshow(
    da,
    map_kws=None,
    projection=None,
    colorbar=True,
    figsize=None,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    dpi=150,
    xlabel=None,
    ylabel=None,
    title=None,
    cbar_label=None,
    cbar_inset=False,
    xticks=None,
    yticks=None,
    annotations=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Create a imshow plot of the data on a map using Cartopy.

    Parameters
    ----------
    da : xarray.DataArray
        The data to plot.
    map_kws : dict, optional
        Dictionary of keyword arguments for map features.
    projection : cartopy.crs.Projection, optional
        Cartopy projection to use. Defaults to PlateCarree.
    colorbar : bool, default: True
        Whether to add a colorbar.
    figsize : tuple, optional
        Figure size.
    cmap : str or Colormap, optional
        Colormap to use.
    vmin, vmax : float, optional
        Color limits.
    norm : Normalize, optional
        Matplotlib normalization.
    dpi : int, optional
        Dots per inch for export.
    xlabel, ylabel, title : str, optional
        Axis labels and plot title.
    cbar_label : str, optional
        Label for the colorbar.
    cbar_inset : bool, default: False
        Place colorbar as an inset (right) if True.
    xticks, yticks : list, optional
        Custom tick locations.
    annotations : list of dict, optional
        List of annotation dicts.
    export_path : str, optional
        Path to export the figure.
    export_formats : list, optional
        List of formats to export.
    **kwargs : dict
        Additional keyword arguments for imshow.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    plot_args = dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    plot_args.update({k: v for k, v in kwargs.items() if k not in ["ax", "transform", "x", "y"]})

    # Detect coordinates if not provided
    x_name, y_name = _get_plot_xy(da)
    if "x" not in kwargs and x_name:
        plot_args["x"] = x_name
    if "y" not in kwargs and y_name:
        plot_args["y"] = y_name

    mesh = da.plot.imshow(ax=ax, transform=ccrs.PlateCarree(), **plot_args)

    # Map features
    if GeoAxes is not None and isinstance(ax, GeoAxes):
        coast_kws = map_kws.get("coastlines", {})
        ax.coastlines(**coast_kws)
        grid_kws = map_kws.get(
            "gridlines",
            {
                "draw_labels": True,
                "linewidth": 0.5,
                "color": "gray",
                "alpha": 0.5,
                "linestyle": "--",
            },
        )
        gl = ax.gridlines(**grid_kws)
        if hasattr(gl, "top_labels"):
            gl.top_labels = False
        if hasattr(gl, "right_labels"):
            gl.right_labels = False
        for feature_name in ["land", "ocean", "borders", "lakes", "rivers", "states"]:
            if feature_name in map_kws:
                import cartopy.feature as cfeature

                feat = getattr(cfeature, feature_name.upper(), None)
                if feat is not None:
                    ax.add_feature(feat(), **map_kws[feature_name])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if annotations:
        for ann in annotations:
            ax.annotate(**ann)
    if colorbar:
        if cbar_inset:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="lower left",
                bbox_to_anchor=(1.05, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
        else:
            cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, aspect=30)
        if cbar_label:
            cbar.set_label(cbar_label)
    fig.tight_layout()
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_quick_map(
    da,
    map_kws=None,
    projection=None,
    colorbar=True,
    figsize=None,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    dpi=150,
    xlabel=None,
    ylabel=None,
    title=None,
    cbar_label=None,
    cbar_inset=False,
    xticks=None,
    yticks=None,
    annotations=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Create a map plot of the data using Cartopy and xarray's default plot method.
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    plot_args = dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    plot_args.update({k: v for k, v in kwargs.items() if k not in ["ax", "transform", "x", "y"]})

    x_name, y_name = _get_plot_xy(da)
    if "x" not in kwargs and x_name:
        plot_args["x"] = x_name
    if "y" not in kwargs and y_name:
        plot_args["y"] = y_name

    mesh = da.plot(ax=ax, transform=ccrs.PlateCarree(), **plot_args)

    if GeoAxes is not None and isinstance(ax, GeoAxes):
        coast_kws = map_kws.get("coastlines", {})
        ax.coastlines(**coast_kws)
        grid_kws = map_kws.get(
            "gridlines",
            {
                "draw_labels": True,
                "linewidth": 0.5,
                "color": "gray",
                "alpha": 0.5,
                "linestyle": "--",
            },
        )
        gl = ax.gridlines(**grid_kws)
        if hasattr(gl, "top_labels"):
            gl.top_labels = False
        if hasattr(gl, "right_labels"):
            gl.right_labels = False
        for feature_name in ["land", "ocean", "borders", "lakes", "rivers", "states"]:
            if feature_name in map_kws:
                import cartopy.feature as cfeature

                feat = getattr(cfeature, feature_name.upper(), None)
                if feat is not None:
                    ax.add_feature(feat(), **map_kws[feature_name])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if annotations:
        for ann in annotations:
            ax.annotate(**ann)
    if colorbar:
        if cbar_inset:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="lower left",
                bbox_to_anchor=(1.05, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
        else:
            cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, aspect=30)
        if cbar_label:
            cbar.set_label(cbar_label)
    fig.tight_layout()
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_quick_contourf(
    da,
    map_kws=None,
    projection=None,
    colorbar=True,
    figsize=None,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    dpi=150,
    xlabel=None,
    ylabel=None,
    title=None,
    cbar_label=None,
    cbar_inset=False,
    xticks=None,
    yticks=None,
    annotations=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Create a filled contour plot of the data on a map using Cartopy.
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    plot_args = dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    plot_args.update({k: v for k, v in kwargs.items() if k not in ["ax", "transform", "x", "y"]})

    x_name, y_name = _get_plot_xy(da)
    if "x" not in kwargs and x_name:
        plot_args["x"] = x_name
    if "y" not in kwargs and y_name:
        plot_args["y"] = y_name

    mesh = da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), **plot_args)

    if GeoAxes is not None and isinstance(ax, GeoAxes):
        coast_kws = map_kws.get("coastlines", {})
        ax.coastlines(**coast_kws)
        grid_kws = map_kws.get(
            "gridlines",
            {
                "draw_labels": True,
                "linewidth": 0.5,
                "color": "gray",
                "alpha": 0.5,
                "linestyle": "--",
            },
        )
        gl = ax.gridlines(**grid_kws)
        if hasattr(gl, "top_labels"):
            gl.top_labels = False
        if hasattr(gl, "right_labels"):
            gl.right_labels = False
        for feature_name in ["land", "ocean", "borders", "lakes", "rivers", "states"]:
            if feature_name in map_kws:
                import cartopy.feature as cfeature

                feat = getattr(cfeature, feature_name.upper(), None)
                if feat is not None:
                    ax.add_feature(feat(), **map_kws[feature_name])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if annotations:
        for ann in annotations:
            ax.annotate(**ann)
    if colorbar:
        if cbar_inset:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="lower left",
                bbox_to_anchor=(1.05, 0.1, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
        else:
            cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, aspect=30)
        if cbar_label:
            cbar.set_label(cbar_label)
    fig.tight_layout()
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, ax


def facet_time_map(
    da,
    time_dim="time",
    ncols=3,
    map_kws=None,
    projection=None,
    colorbar=True,
    figsize=None,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    dpi=150,
    xlabel=None,
    ylabel=None,
    suptitle=None,
    cbar_label=None,
    xticks=None,
    yticks=None,
    annotations=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Create a facet grid of map plots for each time slice in a DataArray using Cartopy.
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    times = da[time_dim].values
    nt = len(times)
    ncols = min(ncols, nt)
    nrows = int(np.ceil(nt / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).flatten()
    plot_args = dict(cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    plot_args.update({k: v for k, v in kwargs.items() if k not in ["ax", "transform", "x", "y"]})

    x_name, y_name = _get_plot_xy(da)
    if "x" not in kwargs and x_name:
        plot_args["x"] = x_name
    if "y" not in kwargs and y_name:
        plot_args["y"] = y_name

    mesh = None
    for i, t in enumerate(times):
        ax = axes[i]
        dat = da.sel({time_dim: t})
        mesh = dat.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, **plot_args)
        if GeoAxes is not None and isinstance(ax, GeoAxes):
            coast_kws = map_kws.get("coastlines", {})
            ax.coastlines(**coast_kws)
            grid_kws = map_kws.get(
                "gridlines",
                {
                    "draw_labels": False,
                    "linewidth": 0.5,
                    "color": "gray",
                    "alpha": 0.5,
                    "linestyle": "--",
                },
            )
            ax.gridlines(**grid_kws)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title(str(np.datetime_as_string(t)))
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if annotations and i < len(annotations):
            ax.annotate(**annotations[i])
    for j in range(nt, len(axes)):
        fig.delaxes(axes[j])
    if colorbar and mesh is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            axes[-1],
            width="5%",
            height="80%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0.1, 1, 1),
            bbox_transform=axes[-1].transAxes,
            borderpad=0,
        )
        fig.colorbar(mesh, cax=cax, orientation="vertical", label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, axes


def plot_points_map(
    df,
    lon_col="longitude",
    lat_col="latitude",
    projection=None,
    color="C0",
    marker="o",
    size=40,
    edgecolor="k",
    alpha=0.8,
    map_kws=None,
    figsize=(8, 6),
    dpi=150,
    title=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Plot points from a DataFrame on a Cartopy map.
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    if GeoAxes is not None and isinstance(ax, GeoAxes):
        coast_kws = map_kws.get("coastlines", {})
        ax.coastlines(**coast_kws)
        grid_kws = map_kws.get(
            "gridlines",
            {
                "draw_labels": True,
                "linewidth": 0.5,
                "color": "gray",
                "alpha": 0.5,
                "linestyle": "--",
            },
        )
        gl = ax.gridlines(**grid_kws)
        if hasattr(gl, "top_labels"):
            gl.top_labels = False
        if hasattr(gl, "right_labels"):
            gl.right_labels = False
        for feature_name in ["land", "ocean", "borders", "lakes", "rivers", "states"]:
            if feature_name in map_kws:
                import cartopy.feature as cfeature

                feat = getattr(cfeature, feature_name.upper(), None)
                if feat is not None:
                    ax.add_feature(feat(), **map_kws[feature_name])
    ax.scatter(
        df[lon_col],
        df[lat_col],
        color=color,
        marker=marker,
        s=size,
        edgecolor=edgecolor,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_lines_map(
    df,
    lon_col="longitude",
    lat_col="latitude",
    group_col=None,
    projection=None,
    color="C0",
    linewidth=2,
    alpha=0.8,
    map_kws=None,
    figsize=(8, 6),
    dpi=150,
    title=None,
    export_path=None,
    export_formats=None,
    **kwargs,
):
    """
    Plot lines from a DataFrame on a Cartopy map.
    """
    if ccrs is None:
        raise ImportError("Cartopy is required for mapping utilities.")
    if projection is None:
        projection = ccrs.PlateCarree()
    if map_kws is None:
        map_kws = {}
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=figsize, dpi=dpi)
    if GeoAxes is not None and isinstance(ax, GeoAxes):
        coast_kws = map_kws.get("coastlines", {})
        ax.coastlines(**coast_kws)
        grid_kws = map_kws.get(
            "gridlines",
            {
                "draw_labels": True,
                "linewidth": 0.5,
                "color": "gray",
                "alpha": 0.5,
                "linestyle": "--",
            },
        )
        gl = ax.gridlines(**grid_kws)
        if hasattr(gl, "top_labels"):
            gl.top_labels = False
        if hasattr(gl, "right_labels"):
            gl.right_labels = False
        for feature_name in ["land", "ocean", "borders", "lakes", "rivers", "states"]:
            if feature_name in map_kws:
                import cartopy.feature as cfeature

                feat = getattr(cfeature, feature_name.upper(), None)
                if feat is not None:
                    ax.add_feature(feat(), **map_kws[feature_name])
    if group_col:
        for _, group in df.groupby(group_col):
            ax.plot(
                group[lon_col],
                group[lat_col],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                **kwargs,
            )
    else:
        ax.plot(
            df[lon_col],
            df[lat_col],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if export_path:
        if export_formats is None:
            export_formats = ["png"]
        for fmt in export_formats:
            fig.savefig(f"{export_path}.{fmt}", dpi=dpi, bbox_inches="tight")
    return fig, ax
