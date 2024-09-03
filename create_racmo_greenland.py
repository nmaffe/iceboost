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
oggm_rgi_shp = utils.get_rgi_region_file(f"{5:02d}", version='62')
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
gl_geoms = oggm_rgi_glaciers['geometry']
gl_geoms_ext_gs = gl_geoms.exterior  # Geoseries
gl_geoms_ext_gdf = gpd.GeoDataFrame(geometry=gl_geoms_ext_gs, crs="EPSG:4326")  # Geodataframe
gl_geoms_ext_gdf = gl_geoms_ext_gdf.to_crs("EPSG:3413")


RACMO_PATH = "/media/maffe/nvme/racmo/greenland_racmo2.3p2/"
racmo_file = "SMB_rec_RACMO2.3p2_1km_1961-1990.nc"
racmo = rioxarray.open_rasterio(f'{RACMO_PATH}{racmo_file}', decode_times=False)
racmo = racmo.drop_vars(['LON', 'LAT'])
racmo.rio.write_crs("EPSG:3413", inplace=True)
transform = racmo['SMB_rec'].rio.transform()
racmo['SMB_rec'] = racmo['SMB_rec'].where(racmo['SMB_rec'] != 0.0)
racmo.attrs['_FillValue'] = np.nan

# Make a 1961-1990 mean
racmo = racmo.mean(dim='time', keep_attrs=True)

# Smooth with a gaussian filter of 1 pixel (=1 km)
gauss_kernel = Gaussian2DKernel(1)
racmo_smoothed_np = convolve(racmo['SMB_rec'].values, gauss_kernel, boundary='fill', fill_value=np.nan)

# Add to existing dataset (this is just to plot)
racmo['SMB_rec_gf'] = (('y', 'x', 'band1'), racmo_smoothed_np[..., np.newaxis])

# Create a new xarray with the smoothed smb that I will save to disk
racmo_smoothed = xarray.DataArray(racmo_smoothed_np, dims=['y', 'x'], coords={'x': racmo.coords['x'], 'y': racmo.coords['y']},
                                  attrs=None, name="smb_smoothed")
racmo_smoothed.attrs['_FillValue'] = np.nan
racmo_smoothed.rio.write_crs("EPSG:3413", inplace=True)
racmo_smoothed = racmo_smoothed.rio.write_transform(transform) # important

plot = True
if plot:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    racmo['SMB_rec'].plot(ax=ax1, cmap='hsv')
    racmo['SMB_rec_gf'].plot(ax=ax2, cmap='hsv')
    racmo_smoothed.plot(ax=ax3, cmap='hsv')
    for ax in (ax1, ax2, ax3): gl_geoms_ext_gdf.plot(ax=ax, color='k')
    plt.show()

save = False
if save:
    file_out = "smb_greenland_mean_1961_1990_RACMO23p2_gf.nc"
    racmo_smoothed.to_netcdf(f"{RACMO_PATH}{file_out}")
    print(f"{file_out} saved. Exit.")

