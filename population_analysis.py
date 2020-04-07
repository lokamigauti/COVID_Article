import xarray as xr
import matplotlib.pyplot as plt
import dask
from affine import Affine
import numpy as np
popfile = '/home/gab/Downloads/ppp_2020_1km_Aggregated.tif'
da = xr.open_rasterio(popfile)
da = da.chunk(dict(x=10, y=10))
transform = Affine.from_gdal(*da.attrs['transform'])
nx, ny = da.sizes['x'], da.sizes['y']
x, y = dask.array.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform
vls = da.isel(band=0, y=slice(0,1000), x=slice(0, 1000)).values

pop_da.sel(band=1).plot()
plt.show()