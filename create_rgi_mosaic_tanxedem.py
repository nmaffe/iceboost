import gc
import sys, time
from tqdm import tqdm
import os, glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
import rasterio
from rioxarray.merge import merge_arrays
import matplotlib
import matplotlib.pyplot as plt


def get_NS(lat):
    if lat >= 0:
        return 'N'
    else:
        return 'S'


def get_EW(lon):
    if lon >= 0:
        return 'E'
    else:
        return 'W'


def min_closest_multiple(num, res):
    if not isinstance(num, int): num = int(num)
    if res == 1:
        return num
    elif res == 2:
        return num - (num % 2)  # Rounds down to the nearest even multiple of num
    elif res == 4:
        return num - (num % 4)  # Rounds down to the nearest multiple of 4


def max_closest_multiple(num, res):
    if not isinstance(num, int): num = int(num)
    if res == 1:
        return num
    elif res == 2:
        return num + (2 - num % 2)
    elif res == 4:
        return num + (4 - num % 4)

def getTDXlonres(lat):
    """Product Tile Extent
    Between 0° - 60° North/South latitude have a file extent of 1° in latitude and 1° in longitude direction.
    Between 60° - 80° North/South latitudes a product has an extent of 1° x 2°,
    between 80° - 90° North/South latitudes a product tile has an extent of 1° x 4°.
    See https: // geoservice.dlr.de / web / dataguide / tdm30 /
    Note that slight North-South difference. """
    if lat >= 80 or lat <= -81:
        reslon = 4
    elif 60 <= lat <= 79 or -80 <= lat <= -61:
        reslon = 2
    else:
        reslon = 1
    return reslon

def get_codes(miny, minx, maxy, maxx):
    """In: box. Out: tile codes to be merged"""
    tile_minx, tile_miny = int(np.floor(minx)), int(np.floor(miny))
    tile_maxx, tile_maxy = int(np.ceil(maxx)), int(np.ceil(maxy))
    #print(f"miny {miny} minx {minx} maxy {maxy} maxx {maxx}")
    #print(f"Lat min: {tile_miny} max: {tile_maxy}")
    #print(f"Lon min: {tile_minx} max: {tile_maxx}")

    # Create a dataframe of lat lon tiles
    # Lat times always have 1 degree res
    lats_interval = np.arange(tile_miny, tile_maxy, dtype=int)
    df_latlon = pd.DataFrame({'lat': lats_interval, 'lon': None})
    for index, row in df_latlon.iterrows():
        lat = row['lat']
        res_lon = getTDXlonres(lat)
        lons_interval = np.arange(min_closest_multiple(tile_minx, res_lon),
                                  max_closest_multiple(tile_maxx, res_lon),
                                  getTDXlonres(lat), dtype=int).tolist()
        df_latlon.at[index, 'lon'] = list(lons_interval)
    #print(f"Combination of lat lon tiles {df_latlon}")

    possible_codes = [f"{get_NS(lat)}{abs(lat):02d}{get_EW(lon)}{abs(lon):03d}"
                          for lat, lon_list in zip(df_latlon['lat'], df_latlon['lon'])
                          for lon in lon_list]
    #print(f"Possible code combinations: {possible_codes}")

    return possible_codes

def fetch_dem(folder_mosaic=None, rgi=None):
    if os.path.exists(folder_mosaic + f'mosaic_RGI_{rgi}.tif'):
        dem_rgi = rioxarray.open_rasterio(folder_mosaic + f'mosaic_RGI_{rgi}.tif')
    else:  # We have to create the mosaic
        print(f"Mosaic {rgi} not present. Let's create it on the fly.")
        dem_rgi = create_mosaic_rgi_tandemx(rgi=rgi, path_rgi_tiles=folder_mosaic, save=0)
    return dem_rgi

def create_glacier_tile_dem_mosaic(minx, miny, maxx, maxy, rgi, path_tandemx):
    """Find either the tile or the tandemx tiles that contain the glacier
    See https://geoservice.dlr.de/web/dataguide/tdm30/"""
    t0_mosaic_tiles = time.time()
    if isinstance(rgi, int):
        rgi = f"{rgi:02d}"

    folder_rgi_tiles = f"{path_tandemx}RGI_{rgi}/"

    # Get the codes of the tiles that contain the glacier
    codes_tiles_for_mosaic = get_codes(miny, minx, maxy, maxx)
    #print(codes_tiles_for_mosaic)

    # Look for the actual existing files from the possible codes
    matching_files = []
    for code in codes_tiles_for_mosaic:
        matching_files.extend(glob.glob(f"{folder_rgi_tiles}TDM1_EDEM_10_*{code}*_V01_C/EDEM/*_W84.tif", recursive=False))
    #print(matching_files)

    # Create Mosaic
    src_files_to_mosaic = []
    for i, file in enumerate(matching_files):
        #print(i, file)
        src = rioxarray.open_rasterio(file, cache=True)
        src.rio.write_crs("EPSG:4326", inplace=True)
        src = src.where(src != src.rio.nodata)  # replace nodata (-32767).0 with nans.
        src.rio.write_nodata(np.nan, inplace=True)  # set nodata as nan
        #fig, ax1 = plt.subplots()
        #im1 = src.plot(ax=ax1, cmap='terrain')
        #plt.show()
        src_files_to_mosaic.append(src)

    res_out = min(np.abs(src.rio.resolution()))
    assert res_out == 1. / 3600, "Unexpected resolution for merging tiles."
    # todo: verificare che risoluzione devo usare. A questo proposito vedere Pixel Spacing at https://geoservice.dlr.de/web/dataguide/tdm30/
    # Imposing the res_out in merge_arrays is time consuming. Do I need it ? res=(res_out, res_out)
    # I sjould see that res along lat is constant, that in lon varies.
    # If I leave it unspecified it will take the res of the first arryay
    mosaic_tiles = merge_arrays(src_files_to_mosaic, nodata=np.nan)

    try:
        focus = mosaic_tiles.rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy)
    except:
        raise ValueError(f"Problems creation of clipping the focus around the glacier tiles")

    #fig, ax1 = plt.subplots()
    #im1 = focus.plot(ax=ax1, cmap='terrain')
    #plt.show()

    t1_mosaic_tiles = time.time()
    #print(f"Mosaic of tiles done in {t1_mosaic_tiles-t0_mosaic_tiles}")
    return focus

def create_mosaic_rgi_tandemx(rgi=None, path_rgi_tiles=None, save=0):

    if isinstance(rgi, int):
        rgi = f"{rgi:02d}"

    tqdm.write(f"Begin creation of mosaic for region {rgi}...")
    folder_rgi = f"{path_rgi_tiles}RGI_{rgi}"

    src_files_to_mosaic = []
    list_rgi_w84tiles = glob.glob(f"{folder_rgi}/*/EDEM/*_W84.tif", recursive = False)
    tqdm.write(f"In rgi {rgi} we have {len(list_rgi_w84tiles)} tiles to be merged")

    for i, filename in tqdm(enumerate(list_rgi_w84tiles), total=len(list_rgi_w84tiles), desc=f"rgi {rgi} importing tiles", leave=False):

        #tqdm.write(f"rgi:{rgi}, import tile {i+1}/{len(list_rgi_w84tiles)}")
        src = rioxarray.open_rasterio(list_rgi_w84tiles[i], cache=False)
        src.rio.write_crs("EPSG:4326", inplace=True)
        src = src.where(src != src.rio.nodata) # replace nodata (-32767).0 with nans.
        src.rio.write_nodata(np.nan, inplace=True)  # set nodata as nan
        src_files_to_mosaic.append(src)

    # See https://geoservice.dlr.de/web/dataguide/tdm30/pdfs/TD_GS_PS_0215_TanDEM_X_30m_Edited_DEM_Product_Description_v1_1.pdf
    # Since lon resolution changes, I'd like to force the mosaic to have a arcsec resolution in both lat, lon.
    # The lon res can be e.g. 5/3600 and I'd like that to be 1/3600. Probably rioxarray merging operates some interpolation.
    # We extract here 1/3600 resolution. And specify such value when merging below.
    res_out = min(np.abs(src.rio.resolution()))
    assert res_out == 1./3600, "Unexpected resolution for merging tiles."


    mosaic_rgi = merge_arrays(src_files_to_mosaic, res=(res_out, res_out), nodata=np.nan)
    tqdm.write(f"Mosaic for rgi {rgi} created.")
    #print('Resolution mosaic:', mosaic_rgi.rio.resolution())

    plot_mosaic = False
    if plot_mosaic:
        mosaic_clipped = mosaic_rgi.rio.clip_box(
            minx=-95,
            miny=80,
            maxx=-90,
            maxy=80.5,
        )
        fig, ax1 = plt.subplots()
        im1 = mosaic_clipped.plot(ax=ax1, cmap='terrain')
        plt.show()


    # Save
    if save:
        tqdm.write(f"Saving mosaic ...")
        mosaic_rgi.rio.to_raster(f"{path_rgi_tiles}mosaic_RGI_{rgi}.tif") # I could compress='deflate'/...
        tqdm.write(f"mosaic_RGI_{rgi}.tif saved")

    return mosaic_rgi


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create DEM mosaic from TandemX-EDEM tiles')
    parser.add_argument("--input", type=str, default="/media/maffe/nvme/Tandem-X-EDEM/",
                        help="folder path to the TandemX-EDEM tiles")
    parser.add_argument("--region", type=int, default=None, help="RGI region in x format")
    parser.add_argument("--save", type=int, default=0, help="Save mosaic: 0/1")

    args = parser.parse_args()

    mosaic_rgi = create_mosaic_rgi_tandemx(rgi=args.region, path_rgi_tiles=args.input, save=args.save)

