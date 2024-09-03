import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray, rioxarray, rasterio
import geopandas as gpd
import oggm
from oggm import utils
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans

utils.get_rgi_dir(version='62')  # setup oggm version
utils.get_rgi_intersects_dir(version='62')
oggm_rgi_shp = utils.get_rgi_region_file(f"{19:02d}", version='62')
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
gl_geoms = oggm_rgi_glaciers['geometry']
gl_geoms_ext_gs = gl_geoms.exterior  # Geoseries
gl_geoms_ext_gdf = gpd.GeoDataFrame(geometry=gl_geoms_ext_gs, crs="EPSG:4326")  # Geodataframe
gl_geoms_ext_gdf3031 = gl_geoms_ext_gdf.to_crs("EPSG:3031")

RACMO_PATH = "/media/maffe/nvme/racmo/antarctica_racmo2.3p2/2km/"
racmo_file = "smb_rec.1979-2021.RACMO2.3p2_ANT27_ERA5-3h.AIS.2km.YY.nc"
racmo = rioxarray.open_rasterio(f'{RACMO_PATH}{racmo_file}')
racmo.rio.write_crs("EPSG:3031", inplace=True)
racmo = racmo.where(racmo != 0.0)
racmo.attrs['_FillValue'] = np.nan

# Make a 1979-2021 mean
racmo = racmo.mean(dim='time', keep_attrs=True)

# Smooth with a gaussian filter of 1 pixel (=2 km)
gauss_kernel = Gaussian2DKernel(1)
racmo_smoothed_np = convolve(racmo.values, gauss_kernel, boundary='fill', fill_value=np.nan)

# Create a new xarray with the smoothed smb that I will save to disk
racmo_smoothed = xarray.DataArray(racmo_smoothed_np, dims=racmo.dims, coords=racmo.coords, attrs=None, name="smb_smoothed")
racmo_smoothed.attrs['_FillValue'] = np.nan
racmo_smoothed.rio.write_crs("EPSG:3031", inplace=True)

plot = True
if plot:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    im1 = racmo.plot(ax=ax1, cmap='hsv', vmin=-5000, vmax=5000, add_colorbar=True)
    im2 = racmo_smoothed.plot(ax=ax2, cmap='hsv', vmin=-5000, vmax=5000, add_colorbar=True)
    for ax in (ax1, ax2): gl_geoms_ext_gdf3031.plot(ax=ax, color='k')
    gl_geoms_ext_gdf.plot(ax=ax3,color='k')
    plt.show()

save = False
if save:
    file_out = "smb_antarctica_mean_1979_2021_RACMO23p2_gf.nc"
    racmo_smoothed.to_netcdf(f"{RACMO_PATH}{file_out}")
    print(f"{file_out} saved. Exit.")

