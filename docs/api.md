# API Reference

This page provides an overview of the MONET API.

## Top-level Functions

::: monet.monet_accessor._dataset_to_monet
    options:
      show_root_heading: true

::: monet.monet_accessor._rename_latlon
    options:
      show_root_heading: true

## Xarray Accessors

### DataArray Accessor

::: monet.monet_accessor.MONETAccessor
    options:
      show_root_heading: true
      members:
        - wrap_longitudes
        - tidy
        - is_land
        - is_ocean
        - cftime_to_datetime64
        - structure_for_monet
        - stratify
        - window
        - interp_constant_lat
        - interp_constant_lon
        - nearest_ij
        - nearest_latlon
        - quick_imshow
        - quick_map
        - quick_contourf
        - remap_nearest
        - remap_xesmf
        - combine_point

### Dataset Accessor

::: monet.monet_accessor.MONETAccessorDataset
    options:
      show_root_heading: true
      members:
        - is_land
        - is_ocean
        - cftime_to_datetime64
        - remap_xesmf
        - remap_nearest
        - remap_nearest_unstructured
        - nearest_ij
        - nearest_latlon
        - interp_constant_lat
        - interp_constant_lon
        - stratify
        - window
        - combine_point
        - wrap_longitudes
        - tidy

## Pandas Accessor

### DataFrame Accessor

::: monet.monet_accessor.MONETAccessorPandas
    options:
      show_root_heading: true
      members:
        - center
        - to_ascii2nc_df
        - to_ascii2nc_list
        - rename_for_monet
        - get_sparse_SwathDefinition
        - remap_nearest
        - cftime_to_datetime64

## Plotting Utilities

::: monet.plots.savefig
    options:
      show_root_heading: true

::: monet.plots.sp_scatter_bias
    options:
      show_root_heading: true

::: monet.plots.mapgen.draw_map
    options:
      show_root_heading: true

## Meteorological Functions

::: monet.met_funcs
    options:
      show_root_heading: true

## Utilities

::: monet.util.stats
    options:
      show_root_heading: true

::: monet.util.tools
    options:
      show_root_heading: true

::: monet.util.combinetool
    options:
      show_root_heading: true
