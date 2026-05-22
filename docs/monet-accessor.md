# MONET Xarray Accessor

MONET can add georeferencing tools to xarray's data structures through their [accessor mechanism](https://docs.xarray.dev/en/stable/internals/extending-xarray.html). These tools can be accessed via a special `.monet` attribute, available for both `xarray.DataArray` and `xarray.Dataset` objects after a simple `import monet` in your code.

## Initializing the Accessor

All you have to do is import monet and xarray.

```python
import monet
import xarray as xr

# Example with a CMAQ file (assuming you have one)
# from monet.models import cmaq
# ds = cmaq.open_dataset('aqm.t12z.aconc.ncf')
# ds.O3[0,0,:,:].monet.quick_map()
```

## Interpolation Methods

The MONET accessor provides several useful interpolation routines including:

*   Getting the nearest point to a given latitude and longitude.
*   Interpolating to a constant latitude or longitude.
*   Interpolating to vertical levels.
*   Remapping entire 2D `xarray.DataArray` or `xarray.Dataset`.

### Find Nearest Point

To find the nearest latitude/longitude point you just need to use the `nearest_latlon` method.

```python
import xarray as xr
import numpy as np
import monet

# Create a mock dataset
ds = xr.Dataset(
    {"O3": (("time", "y", "x"), np.random.rand(10, 20, 30))},
    coords={
        "time": np.arange(10),
        "latitude": (("y", "x"), np.meshgrid(np.linspace(20, 50, 30), np.linspace(30, 45, 20))[1]),
        "longitude": (("y", "x"), np.meshgrid(np.linspace(20, 50, 30), np.linspace(30, 45, 20))[0]),
    }
)

# Find nearest point
subset = ds.monet.nearest_latlon(lat=35.5, lon=40.2)
print(subset)
```

Notice that the dimensions are reduced to the single nearest grid point.
If you wanted to only find the nearest location for a single variable you can use the accessor on the `xarray.DataArray`.

```python
ds.O3.monet.nearest_latlon(lat=35.5, lon=40.2)
```
