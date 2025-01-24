import time
import pandas as pd
from glob import glob
import random
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
import xrspatial.slope
import argparse

from rioxarray import merge
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import oggm
from oggm import utils
import geopandas as gpd
from tqdm import tqdm
from scipy import spatial
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from sklearn.neighbors import KDTree
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, nearest_points
from pyproj import Transformer, Geod
from joblib import Parallel, delayed
from fetch_glacier_metadata import get_rgi_products
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import *
from imputation_policies import smb_elev_functs, smb_elev_functs_hugo
import misc as misc

parser = argparse.ArgumentParser()
parser.add_argument('--OGGM_folder', type=str,default="/home/maffe/OGGM", help="Path to OGGM main folder")
args = parser.parse_args()

geod = Geod(ellps="WGS84")

rgi_62_glaciers_list = []
rgi_7_glaciers_list = []

for rgi in list(range(1, 20)):

    rgi_62_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
    rgi_62_glaciers = gpd.read_file(rgi_62_shp, engine='pyogrio')
    rgi_62_glaciers.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

    rgi_7_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='70G')
    rgi_7_glaciers = gpd.read_file(rgi_7_shp, engine='pyogrio')

    rgi_62_glaciers_list.append(rgi_62_glaciers)
    rgi_7_glaciers_list.append(rgi_7_glaciers)

all_rgi_62_glaciers = gpd.GeoDataFrame(pd.concat(rgi_62_glaciers_list, ignore_index=True))
all_rgi_7_glaciers = gpd.GeoDataFrame(pd.concat(rgi_7_glaciers_list, ignore_index=True))

# Optional: Reset the index
all_rgi_62_glaciers.reset_index(drop=True, inplace=True)
all_rgi_7_glaciers.reset_index(drop=True, inplace=True)

all_rgi_62_glaciers['O1Region'] = all_rgi_62_glaciers['O1Region'].astype(int)
all_rgi_7_glaciers['o1region'] = all_rgi_7_glaciers['o1region'].astype(int)

print(f"RGI 62: {len(all_rgi_62_glaciers)}")
print(f"RGI 7: {len(all_rgi_7_glaciers)}")


# Empy dataframe with IDs of RGI 7 as index
link_ids_rgi_7_62 = pd.DataFrame(columns=['rgi_id_7', 'rgi_id_62'])

print('-'*100)
for rgi in list(range(1, 20)):
#for rgi in [5,]:

    rgi_62_glaciers = all_rgi_62_glaciers.loc[all_rgi_62_glaciers['O1Region'] == rgi]
    rgi_7_glaciers = all_rgi_7_glaciers.loc[all_rgi_7_glaciers['o1region'] == rgi]

    # loop over glaciers in rgi 62
    for idx7, row7 in tqdm(rgi_7_glaciers.iterrows(), total=len(rgi_7_glaciers), desc=f"Glaciers in rgi {rgi}"):
        geom7 = row7['geometry']
        rgi_id_7 = row7['rgi_id']
        #gl_geom_ext = Polygon(geom62.exterior)
        #glacier_centroid = gl_geom_ext.centroid
        #cenLon, cenLat = glacier_centroid.x, glacier_centroid.y
        #tqdm.write(rgi_id_62)

        # Check for overlaps with rgi_62 glaciers
        intersects = rgi_62_glaciers[rgi_62_glaciers['geometry'].intersects(geom7)].copy()

        if not intersects.empty:

            # Calculate intersection areas
            intersection_areas = []

            for geom in intersects['geometry']:
                # Calculate the intersection geometry with geom7
                intersection_geom = geom.intersection(geom7)
                # Calculate area using geometry_area_perimeter
                area, _ = geod.geometry_area_perimeter(intersection_geom)
                intersection_areas.append(abs(area) * 1e-6)  # Convert to km2


            # Assign calculated areas to a new column
            intersects.loc[:, 'intersection_area'] = intersection_areas

            # Find the geometry with the maximum intersection area
            max_idx = intersects['intersection_area'].idxmax()
            rgi_id_62_max_overlap = intersects.loc[max_idx, 'rgi_id']

            #if rgi_id_62_max_overlap == 'RGI60-11.01450':
            #    print(rgi_id_7, rgi_id_62_max_overlap)

            # Create a DataFrame for the new data
            df_ids_62_overlapping_wth_id_7 = pd.DataFrame({
                'rgi_id_7': [rgi_id_7],
                'rgi_id_62': [rgi_id_62_max_overlap]
            })

            # Concatenate the new DataFrame with the existing one
            link_ids_rgi_7_62 = pd.concat([link_ids_rgi_7_62, df_ids_62_overlapping_wth_id_7], ignore_index=True)


        else:
            # No overlaps found, can handle if necessary
            pass

save = True
if save:
    fileout = '/media/maffe/nvme/link_ids_rgi_7_62/link_ids_rgi_7_62.csv'
    link_ids_rgi_7_62.to_csv(fileout, index=False)
    print('saved ', fileout)

print(f"Final dataset: {len(link_ids_rgi_7_62)} lines")

plot = False
if plot:
    ids_7_check = link_ids_rgi_7_62.loc[link_ids_rgi_7_62['rgi_id_62'] == 'RGI60-05.10315', 'rgi_id_7'].values
    geom_aletsch_rgi6 = all_rgi_62_glaciers.loc[all_rgi_62_glaciers['rgi_id'] == 'RGI60-05.10315', 'geometry']
    geom_to_aletsch_rgi7 = [all_rgi_7_glaciers.loc[all_rgi_7_glaciers['rgi_id'] == id, 'geometry'] for id in ids_7_check]
    num_geometries = len(geom_to_aletsch_rgi7)
    colors = plt.cm.viridis(np.linspace(0, 1, num_geometries))

    fig, ax = plt.subplots()
    geom_aletsch_rgi6.plot(ax=ax, facecolor='none', edgecolor='red')
    for g_7, color in zip(geom_to_aletsch_rgi7, colors):
        g_7.plot(ax=ax, facecolor='none', edgecolor=color)
    plt.show()


