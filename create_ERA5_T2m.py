import os, sys
from glob import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray, rioxarray, rasterio
import geopandas as gpd
from rasterio.enums import Resampling

parser = argparse.ArgumentParser()
parser.add_argument('--ERA5Land_folder', type=str,default="/media/maffe/nvme/ERA5/ERA5-land-T2m/2000-2010/")
parser.add_argument('--ERA5_folder', type=str,default="/media/maffe/nvme/ERA5/ERA5-T2m/2000-2010/")
parser.add_argument('--ERA5_outfolder', type=str,default="/media/maffe/nvme/ERA5/")
args = parser.parse_args()

file_era5land = 'era5land_t2m.nc'
file_era5 = 'era5_t2m.nc'

tile_era5land = rioxarray.open_rasterio(f"{args.ERA5Land_folder}{file_era5land}", masked=False)
tile_era5 = rioxarray.open_rasterio(f"{args.ERA5_folder}{file_era5}", masked=False)

tile_era5land.rio.write_crs("EPSG:4326", inplace=True)
tile_era5.rio.write_crs("EPSG:4326", inplace=True)

tile_era5 = tile_era5.mean(dim='time', keep_attrs=True)
tile_era5 = tile_era5.where(tile_era5 != tile_era5.rio.nodata)
tile_era5.rio.write_nodata(np.nan, inplace=True)
tile_era5 = tile_era5 * tile_era5.attrs['scale_factor'] + tile_era5.attrs['add_offset']

tile_era5land = tile_era5land.mean(dim='time', keep_attrs=True)
tile_era5land = tile_era5land.where(tile_era5land != tile_era5land.rio.nodata)
tile_era5land.rio.write_nodata(np.nan, inplace=True)
tile_era5land = tile_era5land * tile_era5land.attrs['scale_factor'] + tile_era5land.attrs['add_offset']

print(f"Era5 resolution: {tile_era5.rio.resolution()}")
print(f"Era5-Land resolution: {tile_era5land.rio.resolution()}")
print(f"Before resampling era5 {tile_era5.shape} era5land {tile_era5land.shape}")

# Resample the low-resolution array to the high-resolution shape
tile_era5_HR = tile_era5.rio.reproject(
    tile_era5.rio.crs,
    shape=(tile_era5land.shape[0], tile_era5land.shape[1]),
    resampling=Resampling.bilinear
)

print(f"After resampling era5 {tile_era5_HR.shape} era5land {tile_era5land.shape}")

# Ensure Coordinates Alignment
tile_era5_HR = tile_era5_HR.assign_coords(
    x=tile_era5land.x,
    y=tile_era5land.y
)

# Align era5 array with era5land
tile_era5land, tile_era5_HR = xarray.align(tile_era5land, tile_era5_HR, join='exact')

# Calculate the final merged array that uses era5land and if null era5
era5land_era5 = xarray.where(tile_era5land.notnull(), tile_era5land, tile_era5_HR)
era5land_era5 = era5land_era5.squeeze()
era5land_era5.rio.write_nodata(np.nan, inplace=True)
era5land_era5.rio.write_crs("EPSG:4326", inplace=True)

# Shift longitude coordinates from 0..360 to -180..+180
era5land_era5 = era5land_era5.assign_coords(x=((era5land_era5['x'] + 180) % 360 - 180)).sortby('x')
era5land_era5 = era5land_era5.rio.write_transform(era5land_era5.rio.transform())

save = False
if save:
    file_out = "era5land_era5.nc"
    era5land_era5.to_netcdf(f"{args.ERA5_outfolder}{file_out}")
    print(f"{file_out} saved. Exit.")

plot = False
if plot:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    im1 = tile_era5_HR.plot(ax=ax1, cmap='jet')
    im2 = tile_era5land.plot(ax=ax2, cmap='jet')
    im3 = era5land_era5.plot(ax=ax3, cmap='jet')
    plt.show()

run_test = True
if run_test:
    file_in_for_test = 'era5land_era5.nc'
    tile_in_for_test = rioxarray.open_rasterio(f"{args.ERA5_outfolder}{file_in_for_test}", masked=False)
    fig, ax = plt.subplots()
    tile_in_for_test.plot(ax=ax, cmap='jet')
    plt.show()
