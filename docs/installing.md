# Installation

## Required Dependencies

MONET requires **Python 3.6 or later**.

*   [numpy](https://numpy.org) (1.11 or later)
*   [pandas](https://pandas.pydata.org/docs/) (1.0 or later)
*   [xarray](https://docs.xarray.dev) (0.10 or later)
*   [dask](https://docs.dask.org)
*   [netcdf4](https://unidata.github.io/netcdf4-python/)
*   [matplotlib](https://matplotlib.org/)
*   [seaborn](https://seaborn.pydata.org/)
*   [cartopy](https://scitools.org.uk/cartopy/docs/latest/)
*   [pyresample](https://pyresample.readthedocs.io/)
*   [scipy](https://scipy.org/)

### Optional Dependencies

*   [xesmf](https://xesmf.readthedocs.io/) (1.9.0 or later): For advanced regridding (not available on Windows).
*   [python-stratify](https://github.com/SciTools/python-stratify): For vertical interpolation.

## Instructions

MONET itself is a pure Python package, but some of its dependencies may not be. The simplest way to install MONET is to install it from the conda-forge channel:

```bash
conda install -c conda-forge monet
```

This will install all of the dependencies needed by MONET and MONET itself.

To install MONET from source code, you can use `pip`:

```bash
pip install git+https://github.com/noaa-oar-arl/MONET.git
```

Or you can manually download it from GitHub and install it:

```bash
git clone https://github.com/noaa-oar-arl/MONET.git
cd MONET
pip install .
```
