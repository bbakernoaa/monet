# Tutorial

This tutorial provides a basic introduction to using MONET for analyzing geospatial data.

## Getting Started with the Accessor

MONET's power lies in its accessors for Xarray and Pandas. Let's look at a simple example using Xarray.

```python
import xarray as xr
import numpy as np
import monet
import matplotlib.pyplot as plt

# 1. Create mock gridded data
lons = np.linspace(-125, -65, 100)
lats = np.linspace(25, 50, 50)
lon2d, lat2d = np.meshgrid(lons, lats)

data = np.sin(np.deg2rad(lat2d)) * np.cos(np.deg2rad(lon2d))

da = xr.DataArray(
    data,
    coords={"latitude": (("y", "x"), lat2d), "longitude": (("y", "x"), lon2d)},
    dims=("y", "x"),
    name="sample_data"
)

# 2. Use the MONET accessor to plot
ax = da.monet.quick_map()
plt.title("MONET Quick Map Example")
plt.show()

# 3. Find the nearest point to a location
# Let's find data near Washington, D.C. (38.9N, 77.0W)
dc_data = da.monet.nearest_latlon(lat=38.9, lon=-77.0)
print(f"Data at DC: {dc_data.values}")
```

## Pairing Model Data with Observations

One of the primary use cases for MONET is "pairing" model output (usually gridded) with point observations (usually in a DataFrame).

```python
import pandas as pd

# Create some mock observation points
obs_df = pd.DataFrame({
    'latitude': [34.05, 40.71, 37.77],
    'longitude': [-118.24, -74.00, -122.41],
    'obs_value': [1.2, 0.8, 1.5],
    'time': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01'])
})

# Use the combine_point tool (requires pyresample or xesmf)
# paired_df = da.monet.combine_point(obs_df)
# print(paired_df)
```

For more advanced examples, please refer to the [MONETIO documentation](https://monetio.readthedocs.io/en/stable/tutorial.html) which covers loading real-world datasets.
