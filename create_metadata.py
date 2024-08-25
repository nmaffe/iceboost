import time
import pandas as pd
from glob import glob
import random
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
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
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import from_lat_lon_to_utm_and_epsg, gaussian_filter_with_nans, haversine, lmax_imputer
from imputation_policies import smb_elev_functs, smb_elev_functs_hugo

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
"""
This program creates a dataframe of metadata for the points in glathida.
Time all rgis: 293m

1. add_rgi. Time: 1min. TESTED.
2. add_RGIId_and_OGGM_stats. TESTED. All rgis: 15 min 
3. add_slopes_elevation. TESTED. All rgis: 70 min
    - No nan can be produced here. 
4. add_millan_vx_vy_ith. TESTED. All rgis: 8 min
    - Points inside the glacier but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
5. add_dist_from_boder_using_geometries. TESTED. All rgis: 1h40m
    - Note: if a point is inside a nunatak the distance will be set to nan.
6. add_farinotti_ith. TESTED. All rgis: 1h15m
    - Points inside the glacier but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
7. add_dist_from_water. 5 mins
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path_ttt_csv', type=str,default="/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT.csv",
                    help="Path to TTT.csv file")
parser.add_argument('--path_ttt_rgi_csv', type=str,default="/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_rgi.csv",
                    help="Path to TTT_rgi.csv file")
parser.add_argument('--path_O1Regions_shp', type=str,default="/home/maffe/OGGM/rgi/RGIV62/00_rgi62_regions/00_rgi62_O1Regions.shp",
                    help="Path to OGGM's 00_rgi62_O1Regions.shp shapefiles of all 19 RGI regions")
parser.add_argument('--mosaic', type=str,default="/media/maffe/nvme/Tandem-X-EDEM/",
                    help="Path to DEM mosaics")
parser.add_argument('--oggm', type=str,default="/home/maffe/OGGM/", help="Path to OGGM folder")
parser.add_argument('--millan_velocity_folder', type=str,default="/media/maffe/nvme/Millan/velocity/",
                    help="Path to Millan velocity data")
parser.add_argument('--millan_icethickness_folder', type=str,default="/media/maffe/nvme/Millan/thickness/",
                    help="Path to Millan ice thickness data")
parser.add_argument('--NSIDC_icethickness_folder_Greenland', type=str,default="/media/maffe/nvme/BedMachine_v5/",
                    help="Path to BedMachine v5 Greenland")
parser.add_argument('--NSIDC_velocity_folder_Antarctica', type=str,default="/media/maffe/nvme/Antarctica_NSIDC/velocity/NSIDC-0754/",
                    help="Path to AnIS velocity data")
parser.add_argument('--NSIDC_icethickness_folder_Antarctica', type=str,default="/media/maffe/nvme/Antarctica_NSIDC/thickness/NSIDC-0756/",
                    help="Path to AnIS velocity data")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/media/maffe/nvme/Farinotti/composite_thickness_RGI60-all_regions/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--OGGM_folder', type=str,default="/home/maffe/OGGM", help="Path to OGGM main folder")
parser.add_argument('--RACMO_folder', type=str,default="/media/maffe/nvme/racmo", help="Path to RACMO main folder")
parser.add_argument('--path_ERA5_t2m_folder', type=str,default="/media/maffe/nvme/ERA5/", help="Path to ERA5 folder")
parser.add_argument('--GSHHG_folder', type=str,default="/media/maffe/nvme/gshhg/", help="Path to GSHHG folder")
parser.add_argument('--save', type=int, default=0, help="Save final dataset or not.")
parser.add_argument('--save_outname', type=str,
            default="/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/metadata35",
            help="Saved dataframe name.")


# Setup oggm
utils.get_rgi_dir(version='62')
utils.get_rgi_intersects_dir(version='62')

args = parser.parse_args()

""" Add rgi values """
def add_rgi(glathida, path_O1_shp):
    print(f'Adding RGI method...')

    if ('RGI' in list(glathida)):
        print('Variable RGI already in dataframe.')
        return glathida

    world = gpd.read_file(path_O1_shp)
    glathida['RGI'] = [np.nan]*len(glathida)
    lats = glathida['POINT_LAT']
    lons = glathida['POINT_LON']
    points = [Point(ilon, ilat) for (ilon, ilat) in zip(lons, lats)]

    # Define the regions
    region1a = world.loc[0]['geometry']
    region1b = world.loc[1]['geometry']
    region1 = shapely.ops.unary_union([region1a, region1b]) #class 'shapely.geometry.polygon.Polygon
    region2 = world.loc[2]['geometry']
    region3 = world.loc[3]['geometry']
    region4 = world.loc[4]['geometry']
    region5 = world.loc[5]['geometry']
    region6 = world.loc[6]['geometry']
    region7 = world.loc[7]['geometry']
    region8 = world.loc[8]['geometry']
    region9 = world.loc[9]['geometry']
    region10a = world.loc[10]['geometry']
    region10b = world.loc[11]['geometry']
    region10 = shapely.ops.unary_union([region10a, region10b]) # shapely.geometry.multipolygon.MultiPolygon
    region11 = world.loc[12]['geometry']
    region12 = world.loc[13]['geometry']
    region13 = world.loc[14]['geometry']
    region14 = world.loc[15]['geometry']
    region15 = world.loc[16]['geometry']
    region16 = world.loc[17]['geometry']
    region17 = world.loc[18]['geometry']
    region18 = world.loc[19]['geometry']
    region19 = world.loc[20]['geometry']

    all_regions = [region1, region2, region3, region4, region5, region6, region7, region8, region9, region10,
                   region11, region12, region13, region14, region15, region16, region17, region18, region19]

    # loop over the region geometries
    for n, region in tqdm(enumerate(all_regions), total=len(all_regions), leave=True):

        # mask True/False to decide whether the points are inside the region geometry
        mask_points_in_region = region.contains(points)

        # select only those points inside the glacier
        df_poins_in_region = glathida[mask_points_in_region]

        # add to dataframe
        glathida.loc[df_poins_in_region.index, 'RGI'] = n+1

    print(glathida['RGI'].value_counts())
    print(glathida['RGI'].count()) # does not count nans

    ifplot = False
    if ifplot:
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        num_colors = len(all_regions)
        bounds = np.linspace(1, num_colors, num_colors)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # rgi boundaries colors
        colors = cmap(norm(bounds))

        fig, ax1 = plt.subplots()
        ax1.scatter(lons, lats, s=1, c=glathida['RGI'], cmap=cmap, norm=norm)

        for n, rgi_geom in enumerate(all_regions):
            if (rgi_geom.geom_type == 'Polygon'):
                ax1.plot(*rgi_geom.exterior.xy, c=colors[n])
            elif (rgi_geom.geom_type == 'MultiPolygon'):
                for geom in rgi_geom.geoms:
                    ax1.plot(*geom.exterior.xy, c=colors[n])
            else: raise ValueError("Geom type not recognized. Check.")
        plt.show()

    return glathida

""" Add Slopes and Elevation """
def add_slopes_elevation(glathida, path_mosaic):
    print('Running slope and elevation method...')
    ts = time.time()

    if (any(ele in list(glathida) for ele in ['elevation', 'slope50'])):
        print('Variables slope_lat, slope_lon or elevation already in dataframe.')
        return glathida

    glathida['elevation'] = [np.nan] * len(glathida)
    glathida['slope50'] = [np.nan] * len(glathida)
    glathida['slope75'] = [np.nan] * len(glathida)
    glathida['slope100'] = [np.nan] * len(glathida)
    glathida['slope125'] = [np.nan] * len(glathida)
    glathida['slope150'] = [np.nan] * len(glathida)
    glathida['slope300'] = [np.nan] * len(glathida)
    glathida['slope450'] = [np.nan] * len(glathida)
    glathida['slopegfa'] = [np.nan] * len(glathida)
    #glathida['slope_lat'] = [np.nan] * len(glathida)
    #glathida['slope_lon'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf50'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf50'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf75'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf75'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf100'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf100'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf125'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf125'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf150'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf150'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf300'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf300'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gf450'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gf450'] = [np.nan] * len(glathida)
    #glathida['slope_lat_gfa'] = [np.nan] * len(glathida)
    #glathida['slope_lon_gfa'] = [np.nan] * len(glathida)
    glathida['curv_50'] = [np.nan] * len(glathida)
    glathida['curv_300'] = [np.nan] * len(glathida)
    glathida['curv_gfa'] = [np.nan] * len(glathida)
    glathida['aspect_50'] = [np.nan] * len(glathida)
    glathida['aspect_300'] = [np.nan] * len(glathida)
    glathida['aspect_gfa'] = [np.nan] * len(glathida)
    datax = []  # just to analyse the results
    datay = []  # just to analyse the results

    regions = list(range(1, 20))
    for rgi in regions:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi)==0:
            continue

        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()

        for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), desc=f"rgi {rgi} Glathida ID", leave=True):

            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]  # collapse glathida_rgi to specific id
            glathida_id = glathida_id.copy()

            glathida_id['northings'] = np.nan
            glathida_id['eastings'] = np.nan
            glathida_id['epsg'] = np.nan
            for idx in glathida_id.index:
                lat = glathida_id.at[idx, 'POINT_LAT']
                lon = glathida_id.at[idx, 'POINT_LON']
                e, n, _, _, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
                glathida_id.at[idx, 'northings'] = n
                glathida_id.at[idx, 'eastings'] = e
                glathida_id.at[idx, 'epsg'] = int(epsg)

            # get unique epsgs
            unique_epsgs = glathida_id['epsg'].unique().astype(int).tolist()

            for epsg_unique in unique_epsgs:
                glathida_id_epsg = glathida_id.loc[glathida_id['epsg'] == epsg_unique]
                indexes_all_epsg = glathida_id_epsg.index.tolist()

                lats = np.array(glathida_id_epsg['POINT_LAT'])
                lons = np.array(glathida_id_epsg['POINT_LON'])
                northings = np.array(glathida_id_epsg['northings'])
                eastings = np.array(glathida_id_epsg['eastings'])

                swlat = np.amin(lats)
                swlon = np.amin(lons)
                nelat = np.amax(lats)
                nelon = np.amax(lons)

                deltalat = np.abs(swlat - nelat)
                deltalon = np.abs(swlon - nelon)
                delta = max(deltalat, deltalon, 0.1)
                northings_xar = xarray.DataArray(northings)
                eastings_xar = xarray.DataArray(eastings)


                # clip
                try:
                    focus = create_glacier_tile_dem_mosaic(minx=swlon - delta,
                                                            miny=swlat - delta,
                                                            maxx=nelon + delta,
                                                            maxy=nelat + delta,
                                                            rgi=rgi, path_tandemx=path_mosaic)
                except:
                    raise ValueError(f"Problems in method add_slopes_elevation for rgi {rgi} glacier_id: {id_rgi}, "
                                     f"glacier box {swlon - delta} {swlat - delta} {nelon + delta} {nelat + delta}")

                focus = focus.squeeze()

                # Reproject to utm (projection distortions along boundaries converted to nans)
                focus_utm = focus.rio.reproject(epsg_unique, resampling=rasterio.enums.Resampling.bilinear, nodata=-9999)
                focus_utm = focus_utm.where(focus_utm != -9999, np.nan)

                # clip the utm with a buffer of 2 km in both dimentions
                focus_utm_clipped = focus_utm.rio.clip_box(
                    minx=min(eastings)-2000,
                    miny=min(northings)-2000,
                    maxx=max(eastings)+2000,
                    maxy=max(northings)+2000)

                # Get the resolution in meters of the utm focus (resolutions in x and y should be the same!?)
                res_utm_metres = focus_utm_clipped.rio.resolution()[0]

                # Calculate sigma in meters for adaptive gaussian fiter
                sigma_af_min, sigma_af_max = 100.0, 2000.0
                try:
                    # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                    area_id = glathida_id_epsg['Area'].min()  # km2
                    lmax_id = glathida_id_epsg['Lmax'].max()  # m
                    a = 1e6*area_id/(np.pi*0.5*lmax_id)
                    sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                    #print(area_id, lmax_id, a, value)
                except Exception as e:
                    sigma_af = sigma_af_min
                # Ensure that our value correctly in range [50.0, 2000.0]
                assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"
                #print(f"Adaptive gaussian filter with sigma = {value} meters.")

                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                num_px_sigma_50 = max(1, round(50/res_utm_metres))
                num_px_sigma_75 = max(1, round(75/res_utm_metres))
                num_px_sigma_100 = max(1, round(100/res_utm_metres))
                num_px_sigma_125 = max(1, round(125 / res_utm_metres))
                num_px_sigma_150 = max(1, round(150/res_utm_metres))
                num_px_sigma_300 = max(1, round(300/res_utm_metres))
                num_px_sigma_450 = max(1, round(450/res_utm_metres))
                num_px_sigma_af = max(1, round(sigma_af / res_utm_metres))

                kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
                kernel75 = Gaussian2DKernel(num_px_sigma_75, x_size=4 * num_px_sigma_75 + 1, y_size=4 * num_px_sigma_75 + 1)
                kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
                kernel125 = Gaussian2DKernel(num_px_sigma_125, x_size=4 * num_px_sigma_125 + 1, y_size=4 * num_px_sigma_125 + 1)
                kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
                kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
                kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
                kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

                # New way, first slope, and then smooth it
                #dz_dlat_xar, dz_dlon_xar = focus_utm_clipped.differentiate(coord='y'), focus_utm_clipped.differentiate(coord='x')
                #slope = focus_utm_clipped.copy(deep=True, data=(dz_dlat_xar ** 2 + dz_dlon_xar ** 2) ** 0.5)

                #slope_50 = convolve_fft(slope.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_75 = convolve_fft(slope.values, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_100 = convolve_fft(slope.values, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_125 = convolve_fft(slope.values, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_150 = convolve_fft(slope.values, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_300 = convolve_fft(slope.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_450 = convolve_fft(slope.values, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                #slope_af = convolve_fft(slope.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

                """astropy"""
                preserve_nans = True
                focus_filter_50_utm = convolve_fft(focus_utm_clipped.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_75_utm = convolve_fft(focus_utm_clipped.values, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_100_utm = convolve_fft(focus_utm_clipped.values, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_125_utm = convolve_fft(focus_utm_clipped.values, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_150_utm = convolve_fft(focus_utm_clipped.values, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_300_utm = convolve_fft(focus_utm_clipped.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_450_utm = convolve_fft(focus_utm_clipped.values, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                focus_filter_af_utm = convolve_fft(focus_utm_clipped.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

                # create xarray object of filtered dem
                focus_filter_xarray_50_utm = focus_utm_clipped.copy(data=focus_filter_50_utm)
                focus_filter_xarray_75_utm = focus_utm_clipped.copy(data=focus_filter_75_utm)
                focus_filter_xarray_100_utm = focus_utm_clipped.copy(data=focus_filter_100_utm)
                focus_filter_xarray_125_utm = focus_utm_clipped.copy(data=focus_filter_125_utm)
                focus_filter_xarray_150_utm = focus_utm_clipped.copy(data=focus_filter_150_utm)
                focus_filter_xarray_300_utm = focus_utm_clipped.copy(data=focus_filter_300_utm)
                focus_filter_xarray_450_utm = focus_utm_clipped.copy(data=focus_filter_450_utm)
                focus_filter_xarray_af_utm = focus_utm_clipped.copy(data=focus_filter_af_utm)

                # calculate slopes for restricted dem
                # using numpy.gradient dz_dlat, dz_dlon = np.gradient(focus_utm_clipped.values, -res_utm_metres, res_utm_metres)  # [m/m]
                dz_dlat_xar, dz_dlon_xar = focus_utm_clipped.differentiate(coord='y'), focus_utm_clipped.differentiate(coord='x')
                dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
                dz_dlat_filter_xar_75, dz_dlon_filter_xar_75 = focus_filter_xarray_75_utm.differentiate(coord='y'), focus_filter_xarray_75_utm.differentiate(coord='x')
                dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
                dz_dlat_filter_xar_125, dz_dlon_filter_xar_125 = focus_filter_xarray_125_utm.differentiate(coord='y'), focus_filter_xarray_125_utm.differentiate(coord='x')
                dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
                dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')
                dz_dlat_filter_xar_450, dz_dlon_filter_xar_450  = focus_filter_xarray_450_utm.differentiate(coord='y'), focus_filter_xarray_450_utm.differentiate(coord='x')
                dz_dlat_filter_xar_af, dz_dlon_filter_xar_af  = focus_filter_xarray_af_utm.differentiate(coord='y'), focus_filter_xarray_af_utm.differentiate(coord='x')

                slope_50_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_50 ** 2 + dz_dlon_filter_xar_50 ** 2) ** 0.5)
                slope_75_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_75 ** 2 + dz_dlon_filter_xar_75 ** 2) ** 0.5)
                slope_100_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_100 ** 2 + dz_dlon_filter_xar_100 ** 2) ** 0.5)
                slope_125_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_125 ** 2 + dz_dlon_filter_xar_125 ** 2) ** 0.5)
                slope_150_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_150 ** 2 + dz_dlon_filter_xar_150 ** 2) ** 0.5)
                slope_300_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_300 ** 2 + dz_dlon_filter_xar_300 ** 2) ** 0.5)
                slope_450_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_450 ** 2 + dz_dlon_filter_xar_450 ** 2) ** 0.5)
                slope_af_xar = focus_utm_clipped.copy(data=(dz_dlat_filter_xar_af ** 2 + dz_dlon_filter_xar_af ** 2) ** 0.5)

                # Calculate curvature and aspect using xrspatial
                # Units of the curvature output (1/100) of a z-unit. Units of aspect are between [0, 360]
                # Note that xrspatial using a standard 3x3 grid around pixel to calculate stuff
                # Note that xrspatial produces nans at boundaries, but that should not be a problem for interpolation.
                curv_50 = xrspatial.curvature(focus_filter_xarray_50_utm)
                curv_300 = xrspatial.curvature(focus_filter_xarray_300_utm)
                curv_af = xrspatial.curvature(focus_filter_xarray_af_utm)
                aspect_50 = xrspatial.aspect(focus_filter_xarray_50_utm)
                aspect_300 = xrspatial.aspect(focus_filter_xarray_300_utm)
                aspect_af = xrspatial.aspect(focus_filter_xarray_af_utm)


                # interpolate slope and dem
                elevation_data = focus_utm_clipped.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_50_data = slope_50_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_75_data = slope_75_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_100_data = slope_100_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_125_data = slope_125_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_150_data = slope_150_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_300_data = slope_300_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_450_data = slope_450_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_af_data = slope_af_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data

                #slope_lat_data = dz_dlat_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data = dz_dlon_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_50 = dz_dlat_filter_xar_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_50 = dz_dlon_filter_xar_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_75 = dz_dlat_filter_xar_75.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_75 = dz_dlon_filter_xar_75.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_100 = dz_dlat_filter_xar_100.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_100 = dz_dlon_filter_xar_100.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_125 = dz_dlat_filter_xar_125.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_125 = dz_dlon_filter_xar_125.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_150 = dz_dlat_filter_xar_150.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_150 = dz_dlon_filter_xar_150.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_300 = dz_dlat_filter_xar_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_300 = dz_dlon_filter_xar_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_450 = dz_dlat_filter_xar_450.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_450 = dz_dlon_filter_xar_450.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lat_data_filter_af = dz_dlat_filter_xar_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                #slope_lon_data_filter_af = dz_dlon_filter_xar_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_50 = curv_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_300 = curv_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_af = curv_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_50 = aspect_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_300 = aspect_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_af = aspect_af.interp(y=northings_xar, x=eastings_xar, method='linear').data

                # check if any nan in the interpolate data
                contains_nan = any(np.isnan(arr).any() for arr in [slope_50_data, slope_75_data, slope_100_data,
                                                                   slope_125_data, slope_150_data, slope_300_data,
                                                                   slope_450_data, slope_af_data,
                                                                   curv_data_50, curv_data_300, curv_data_af,
                                                                   aspect_data_50, aspect_data_300, aspect_data_af])

                #contains_nan = any(np.isnan(arr).any() for arr in [slope_lon_data, slope_lat_data,
                #                                                   slope_lon_data_filter_50, slope_lat_data_filter_50,
                #                                                   slope_lon_data_filter_75, slope_lat_data_filter_75,
                #                                                   slope_lon_data_filter_100, slope_lat_data_filter_100,
                #                                                   slope_lon_data_filter_125, slope_lat_data_filter_125,
                #                                                   slope_lon_data_filter_150, slope_lat_data_filter_150,
                #                                                   slope_lon_data_filter_300, slope_lat_data_filter_300,
                #                                                   slope_lon_data_filter_450, slope_lat_data_filter_450,
                #                                                   slope_lon_data_filter_af, slope_lat_data_filter_af,
                #                                                   curv_data_50, curv_data_300, curv_data_af,
                #                                                   aspect_data_50, aspect_data_300, aspect_data_af])
                if contains_nan:
                    raise ValueError(f"Nan detected in elevation/slope calc. Check")

                # other checks
                assert slope_50_data.shape == slope_150_data.shape == elevation_data.shape, "Different shapes, something wrong!"
                assert len(slope_50_data) == len(indexes_all_epsg), "Different shapes, something wrong!"
                #assert slope_lat_data.shape == slope_lon_data.shape == elevation_data.shape, "Different shapes, something wrong!"
                #assert slope_lat_data_filter_150.shape == slope_lon_data_filter_150.shape == elevation_data.shape, "Different shapes, something wrong!"
                #assert len(slope_lat_data) == len(indexes_all_epsg), "Different shapes, something wrong!"
                assert curv_data_50.shape == curv_data_300.shape == curv_data_af.shape, "Different shapes, something wrong!"
                assert aspect_data_50.shape == aspect_data_300.shape == aspect_data_af.shape, "Different shapes, something wrong!"

                # write to dataframe
                glathida.loc[indexes_all_epsg, 'elevation'] = elevation_data
                glathida.loc[indexes_all_epsg, 'slope50'] = slope_50_data
                glathida.loc[indexes_all_epsg, 'slope75'] = slope_75_data
                glathida.loc[indexes_all_epsg, 'slope100'] = slope_100_data
                glathida.loc[indexes_all_epsg, 'slope125'] = slope_125_data
                glathida.loc[indexes_all_epsg, 'slope150'] = slope_150_data
                glathida.loc[indexes_all_epsg, 'slope300'] = slope_300_data
                glathida.loc[indexes_all_epsg, 'slope450'] = slope_450_data
                glathida.loc[indexes_all_epsg, 'slopegfa'] = slope_af_data
                #glathida.loc[indexes_all_epsg, 'slope_lat'] = slope_lat_data
                #glathida.loc[indexes_all_epsg, 'slope_lon'] = slope_lon_data
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf50'] = slope_lat_data_filter_50
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf50'] = slope_lon_data_filter_50
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf75'] = slope_lat_data_filter_75
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf75'] = slope_lon_data_filter_75
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf100'] = slope_lat_data_filter_100
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf100'] = slope_lon_data_filter_100
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf125'] = slope_lat_data_filter_125
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf125'] = slope_lon_data_filter_125
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf150'] = slope_lat_data_filter_150
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf150'] = slope_lon_data_filter_150
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf300'] = slope_lat_data_filter_300
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf300'] = slope_lon_data_filter_300
                #glathida.loc[indexes_all_epsg, 'slope_lat_gf450'] = slope_lat_data_filter_450
                #glathida.loc[indexes_all_epsg, 'slope_lon_gf450'] = slope_lon_data_filter_450
                #glathida.loc[indexes_all_epsg, 'slope_lat_gfa'] = slope_lat_data_filter_af
                #glathida.loc[indexes_all_epsg, 'slope_lon_gfa'] = slope_lon_data_filter_af
                glathida.loc[indexes_all_epsg, 'curv_50'] = curv_data_50
                glathida.loc[indexes_all_epsg, 'curv_300'] = curv_data_300
                glathida.loc[indexes_all_epsg, 'curv_gfa'] = curv_data_af
                glathida.loc[indexes_all_epsg, 'aspect_50'] = aspect_data_50
                glathida.loc[indexes_all_epsg, 'aspect_300'] = aspect_data_300
                glathida.loc[indexes_all_epsg, 'aspect_gfa'] = aspect_data_af

                plot_curvature = False
                if plot_curvature:
                    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
                    im1 = focus_filter_xarray_50_utm.plot(ax=ax1, cmap='viridis')
                    im2 = focus_filter_xarray_300_utm.plot(ax=ax2, cmap='viridis')
                    im3 = curv_50.plot(ax=ax3, cmap='viridis')
                    im4 = curv_300.plot(ax=ax4, cmap='viridis')
                    im5 = aspect_50.plot(ax=ax5, cmap='viridis')
                    im6 = aspect_300.plot(ax=ax6, cmap='viridis')
                    plt.show()


                plot_utm = False
                if plot_utm:
                    fig, ((ax0, ax1, ax2, ax3), (ax01, ax4, ax5, ax6)) = plt.subplots(2, 4, figsize=(8, 5))

                    im0 =focus.plot(ax=ax0, cmap='viridis', )

                    im01 =focus_utm.plot(ax=ax01, cmap='viridis', )
                    s01 = ax01.scatter(x=eastings, y=northings, s=15, c='k')

                    im1 = focus_utm_clipped.plot(ax=ax1, cmap='viridis', vmin=focus_utm_clipped.min(),vmax=focus_utm_clipped.max())
                    s1 = ax1.scatter(x=eastings, y=northings, s=15, c=elevation_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=focus_utm_clipped.min(), vmax=focus_utm_clipped.max(), zorder=1)

                    im2 = dz_dlat_xar.plot(ax=ax2, cmap='viridis', vmin=dz_dlat_xar.min(), vmax=dz_dlat_xar.max())
                    s2 = ax2.scatter(x=eastings, y=northings, s=15, c=slope_lat_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=dz_dlat_xar.min(), vmax=dz_dlat_xar.max(), zorder=1)

                    im3 = dz_dlon_xar.plot(ax=ax3, cmap='viridis', vmin=dz_dlon_xar.min(), vmax=dz_dlon_xar.max())
                    s3 = ax3.scatter(x=eastings, y=northings, s=15, c=slope_lon_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=dz_dlon_xar.min(), vmax=dz_dlon_xar.max(), zorder=1)

                    im4 = focus_filter_xarray_300_utm.plot(ax=ax4, cmap='viridis', )
                    s4 = ax4.scatter(x=eastings, y=northings, s=15, c='k')

                    im5 = dz_dlat_filter_xar_300.plot(ax=ax5, cmap='viridis', vmin=dz_dlat_filter_xar_300.min(), vmax=dz_dlat_filter_xar_300.max())
                    s5 = ax5.scatter(x=eastings, y=northings, s=15, c=slope_lat_data_filter_300, ec=(0, 0, 0, 0.1),
                                     cmap='viridis', vmin=dz_dlat_filter_xar_300.min(), vmax=dz_dlat_filter_xar_300.max(), zorder=1)
                    im6 = dz_dlon_filter_xar_300.plot(ax=ax6, cmap='viridis', vmin=dz_dlon_filter_xar_300.min(), vmax=dz_dlon_filter_xar_300.max())
                    s6 = ax6.scatter(x=eastings, y=northings, s=15, c=slope_lon_data_filter_300, ec=(0, 0, 0, 0.1),
                                     cmap='viridis', vmin=dz_dlon_filter_xar_300.min(), vmax=dz_dlon_filter_xar_300.max(), zorder=1)

                    plt.show()

    print(f"Slopes and elevation done in {(time.time()-ts)/60} min")
    return glathida

"""Add surface mass balance"""
def add_smb(glathida, path_RACMO_folder):
    if ('smb' in list(glathida)):
        print('Variable smb already in dataframe.')
        return glathida

    glathida['smb'] = [np.nan] * len(glathida)

    # Greenland and Antarctica smb using racmo
    for rgi in [5, 19,]:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        indexes_rgi = glathida_rgi.index.tolist()

        if len(glathida_rgi) == 0:
            continue

        # Import our racmo smoothed (and time averaged)
        if rgi==5:
            racmo_file = "/greenland_racmo2.3p2/smb_greenland_mean_1961_1990_RACMO23p2_gf.nc"
        elif rgi==19:
            racmo_file = "/antarctica_racmo2.3p2/2km/smb_antarctica_mean_1979_2021_RACMO23p2_gf.nc"
        else: raise ValueError('rgi value for RACMO smb calculation not recognized')

        # Units should be in both regions mm w.e./yr = kg/m2yr
        racmo = rioxarray.open_rasterio(f'{path_RACMO_folder}{racmo_file}')

        # Get rgi measurement coordinates
        lats = glathida_rgi['POINT_LAT']
        lons = glathida_rgi['POINT_LON']
        eastings, northings = (Transformer.from_crs("EPSG:4326", racmo.rio.crs)
                                     .transform(glathida_rgi['POINT_LAT'], glathida_rgi['POINT_LON']))

        # Convert coordinates to racmo projection EPSG:3413 (racmo Greenland) or EPSG:3031 (racmo Antarctica)
        eastings_ar = xarray.DataArray(eastings)
        northings_ar = xarray.DataArray(northings)

        # Interpolate racmo onto the points
        smb_data = racmo.interp(y=northings_ar, x=eastings_ar, method='linear').data.squeeze()

        # Push to dataframe
        glathida.loc[indexes_rgi, 'smb'] = smb_data
        #print(rgi, np.nanmean(smb_data))

        plot_smb = False
        if plot_smb:
            vmin, vmax = -700, 4000#racmo.min(), racmo.max()
            fig, (ax1, ax2) = plt.subplots(1,2)
            racmo.plot(ax=ax1, cmap='hsv', vmin=vmin, vmax=vmax)
            ax1.scatter(x=eastings, y=northings, c='k', s=20)
            racmo.plot(ax=ax2, cmap='hsv', vmin=vmin, vmax=vmax)
            ax2.scatter(x=eastings, y=northings, c=smb_data, ec=(0, 0, 0, 0.3), cmap='hsv', vmin=vmin, vmax=vmax, s=20)
            plt.show()

    # For regions outside Greenland and Antarctica I use another method
    regions = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]
    for rgi in regions:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        if len(glathida_rgi) == 0:
            continue

        indexes_rgi = glathida_rgi.index.tolist()

        lats = glathida_rgi['POINT_LAT']
        lons = glathida_rgi['POINT_LON']
        elevation_data = glathida_rgi['elevation']

        # MERRA-GRACE smb elevation relation (BAD IDEA)
        #smb_data = []
        #for (lat, lon, elev) in zip(lats, lons, elevation_data):
            #smb = smb_elev_functs(rgi, elev, lat, lon) # kg/m2s
            #smb *= 31536000 # kg/m2yr
            #print(rgi, lat, lon, smb)
            #smb_data.append(smb)
        #smb_data = np.array(smb_data)

        # Surface mass balance loop with Hugonnet-elevation relation
        m_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'm']
        q_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'q']
        smb_data_hugo = m_hugo * elevation_data + q_hugo  # m w.e./yr = (1000kg/m2yr)
        smb_data_hugo *= 1.e3  # mm w.e./yr = kg/m2yr
        smb_data = np.array(smb_data_hugo)

        glathida.loc[indexes_rgi, 'smb'] = smb_data
        #print(rgi, m_hugo, q_hugo, np.mean(smb_data))

    return glathida

"""Add Millan's velocity vx, vy, ith"""
def add_millan_vx_vy_ith(glathida, path_millan_velocity, path_millan_icethickness):

    print('Adding Millan velocity and ice thickness method...')
    tm = time.time()

    if (any(ele in list(glathida) for ele in ['ith_m', 'v50', 'v100'])):
        print('Variable already in dataframe.')
        return glathida

    glathida['ith_m'] = [np.nan] * len(glathida)
    glathida['v50'] = [np.nan] * len(glathida)
    glathida['v100'] = [np.nan] * len(glathida)
    glathida['v150'] = [np.nan] * len(glathida)
    glathida['v300'] = [np.nan] * len(glathida)
    glathida['v450'] = [np.nan] * len(glathida)
    glathida['vgfa'] = [np.nan] * len(glathida)
    #glathida['vx'] = [np.nan] * len(glathida)
    #glathida['vy'] = [np.nan] * len(glathida)
    #glathida['vx_gf50'] = [np.nan] * len(glathida)
    #glathida['vx_gf100'] = [np.nan] * len(glathida)
    #glathida['vx_gf150'] = [np.nan] * len(glathida)
    #glathida['vx_gf300'] = [np.nan] * len(glathida)
    #glathida['vx_gf450'] = [np.nan] * len(glathida)
    #glathida['vx_gfa'] = [np.nan] * len(glathida)
    #glathida['vy_gf50'] = [np.nan] * len(glathida)
    #glathida['vy_gf100'] = [np.nan] * len(glathida)
    #glathida['vy_gf150'] = [np.nan] * len(glathida)
    #glathida['vy_gf300'] = [np.nan] * len(glathida)
    #glathida['vy_gf450'] = [np.nan] * len(glathida)
    #glathida['vy_gfa'] = [np.nan] * len(glathida)
    #glathida['dvx_dx'] = [np.nan] * len(glathida)
    #glathida['dvx_dy'] = [np.nan] * len(glathida)
    #glathida['dvy_dx'] = [np.nan] * len(glathida)
    #glathida['dvy_dy'] = [np.nan] * len(glathida)

    # I believe in rgi 19 I do not contemplate Millan tiles since GlaThiDa GT data are not present outside the ice sheet.
    for rgi in [19,]:
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]  # glathida to specific rgi

        if len(glathida_rgi) == 0:
            continue

        tqdm.write(f'rgi: {rgi}, Total points: {len(glathida_rgi)}')

        # get NSIDC
        file_vel_NSIDC = f"{args.NSIDC_velocity_folder_Antarctica}antarctic_ice_vel_phase_map_v01.nc"

        mosaic_vel_NSIDC = rioxarray.open_rasterio(file_vel_NSIDC, masked=False)
        vx_NSIDC = mosaic_vel_NSIDC.VX
        vy_NSIDC = mosaic_vel_NSIDC.VY

        assert vx_NSIDC.rio.bounds() == vy_NSIDC.rio.bounds()
        assert vx_NSIDC.rio.crs == vy_NSIDC.rio.crs

        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()  # unique IDs

        for i, id_rgi in tqdm(enumerate(ids_rgi), total=len(ids_rgi), desc=f"rgi {rgi} Glathida ID", leave=True):

            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]  # collapse glathida_rgi to specific id
            indexes_id = glathida_id.index.tolist()

            eastings_id, northings_id = (Transformer.from_crs("EPSG:4326", vx_NSIDC.rio.crs)
                                         .transform(glathida_id['POINT_LAT'], glathida_id['POINT_LON']))

            eastings_rgi_id_ar = xarray.DataArray(eastings_id)
            northings_rgi_id_ar = xarray.DataArray(northings_id)

            minE, maxE = min(eastings_id), max(eastings_id)
            minN, maxN = min(northings_id), max(northings_id)
            #print(i, len(glathida_id), minE, maxE, minN, maxN, vx_NSIDC.rio.bounds())

            ris_metre_nsidc = vx_NSIDC.rio.resolution()[0] #450m
            eps = 15000
            try:
                vx_NSIDC_focus = vx_NSIDC.rio.clip_box(minx=minE - eps, miny=minN - eps, maxx=maxE + eps, maxy=maxN + eps)
                vy_NSIDC_focus = vy_NSIDC.rio.clip_box(minx=minE - eps, miny=minN - eps, maxx=maxE + eps, maxy=maxN + eps)
            except:
                tqdm.write(f'{i} No NSIDC data for rgi {rgi} GlaThiDa_ID {id_rgi}')
                continue

            # Condition 1. Either v is .rio.nodata or it is zero or it is nan
            cond0 = np.all(vx_NSIDC_focus.values == 0)
            condnodata = np.all(np.abs(vx_NSIDC_focus.values - vx_NSIDC_focus.rio.nodata) < 1.e-6)
            condnan = np.all(np.isnan(vx_NSIDC_focus.values))
            all_zero_or_nodata = cond0 or condnodata or condnan

            if all_zero_or_nodata:
                tqdm.write(f'{i} Cond 1 triggered - No NSIDC vel data for rgi {rgi} GlaThiDa_ID {id_rgi}')
                continue

            # Condition no. 2. A fast and quick interpolation to see if points intercepts a valid raster region
            vals_fast_interp = vx_NSIDC_focus.interp(y=xarray.DataArray(northings_id),
                                               x=xarray.DataArray(eastings_id),
                                               method='nearest').data

            cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                      np.all(np.abs(vals_fast_interp - vx_NSIDC_focus.rio.nodata) < 1.e-6))

            if cond_valid_fast_interp:
                #fig, ax = plt.subplots()
                #vx_NSIDC_focus.values[vx_NSIDC_focus.values == vx_NSIDC_focus.rio.nodata] = np.nan
                #im = vx_NSIDC_focus.plot(ax=ax, cmap='binary', zorder=0)
                #ax.scatter(x=eastings_id, y=northings_id, s=10, c='r', zorder=1)
                #plt.show()
                tqdm.write(f'{i} Cond 2 triggered - No NSIDC vel data for rgi {rgi} GlaThiDa_ID {id_rgi} around'
                           f' {glathida_id['POINT_LAT'].mean():.2f} lat {glathida_id['POINT_LON'].mean():.2f} lon')
                continue

            # At this stage we should have a good interpolation
            #vx_NSIDC_focus.values[vx_NSIDC_focus.values == vx_NSIDC_focus.rio.nodata] = np.nan
            #vy_NSIDC_focus.values[vy_NSIDC_focus.values == vy_NSIDC_focus.rio.nodata] = np.nan
            vx_NSIDC_focus.values = np.where((vx_NSIDC_focus.values == vx_NSIDC_focus.rio.nodata) | np.isinf(vx_NSIDC_focus.values),
                                      np.nan, vx_NSIDC_focus.values)
            vy_NSIDC_focus.values = np.where((vy_NSIDC_focus.values == vy_NSIDC_focus.rio.nodata) | np.isinf(vy_NSIDC_focus.values),
                                      np.nan, vy_NSIDC_focus.values)
            vx_NSIDC_focus.rio.write_nodata(np.nan, inplace=True)
            vy_NSIDC_focus.rio.write_nodata(np.nan, inplace=True)

            assert vx_NSIDC_focus.rio.bounds() == vy_NSIDC_focus.rio.bounds(), "NSIDC vx, vy bounds not the same"

            # Note: for rgi 19 we do not interpolate NSIDC to remove nans.
            tile_vx = vx_NSIDC_focus.squeeze()
            tile_vy = vy_NSIDC_focus.squeeze()

            # Calculate sigma in meters for adaptive gaussian fiter
            sigma_af_min, sigma_af_max = 100.0, 2000.0
            try:
                area_id = glathida_rgi.loc[indexes_id, 'Area'].min()
                lmax_id = glathida_rgi.loc[indexes_id, 'Lmax'].max()
                # print('area', area_id, 'lmax', lmax_id)
                # print(lats_rgi_k_id.min(), lats_rgi_k_id.max(), lons_rgi_k_id.min(), lons_rgi_k_id.max(), area_id)
                # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                # area_id = glathida_rgi_tile_id['Area'].min()  # km2
                # lmax_id = glathida_rgi_tile_id['Lmax'].max()  # m
                a = 1e6 * area_id / (np.pi * 0.5 * lmax_id)
                sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                #print(area_id, lmax_id, a)
            except Exception as e:
                sigma_af = sigma_af_min
            assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"

            # Calculate how many pixels I need for a resolution of xx
            # Since NDIDC has res of 450 m, num pixels will can be very small.
            num_px_sigma_50 = max(1, round(50 / ris_metre_nsidc))
            num_px_sigma_100 = max(1, round(100 / ris_metre_nsidc))
            num_px_sigma_150 = max(1, round(150 / ris_metre_nsidc))
            num_px_sigma_300 = max(1, round(300 / ris_metre_nsidc))
            num_px_sigma_450 = max(1, round(450 / ris_metre_nsidc))
            num_px_sigma_af = max(1, round(sigma_af / ris_metre_nsidc))

            kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
            kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
            kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
            kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
            kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
            kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

            tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)

            # A check to see if velocity modules is as expected
            assert float(tile_v.sum()) > 0, "tile v is not as expected."

            '''astropy'''
            preserve_nans = True
            focus_filter_v50 = convolve_fft(tile_v.values, kernel50, nan_treatment='interpolate',
                                                    preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v100 = convolve_fft(tile_v.values, kernel100, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v150 = convolve_fft(tile_v.values, kernel150, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v300 = convolve_fft(tile_v.values, kernel300, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v450 = convolve_fft(tile_v.values, kernel450, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_af = convolve_fft(tile_v.values, kernelaf, nan_treatment='interpolate',
                                           preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

            #focus_filter_vx_50 = convolve_fft(tile_vx.values.squeeze(), kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vx_100 = convolve_fft(tile_vx.values.squeeze(), kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vx_150 = convolve_fft(tile_vx.values.squeeze(), kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vx_300 = convolve_fft(tile_vx.values.squeeze(), kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vx_450 = convolve_fft(tile_vx.values.squeeze(), kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vx_af = convolve_fft(tile_vx.values.squeeze(), kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

            #focus_filter_vy_50 = convolve_fft(tile_vy.values.squeeze(), kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vy_100 = convolve_fft(tile_vy.values.squeeze(), kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vy_150 = convolve_fft(tile_vy.values.squeeze(), kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vy_300 = convolve_fft(tile_vy.values.squeeze(), kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vy_450 = convolve_fft(tile_vy.values.squeeze(), kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            #focus_filter_vy_af = convolve_fft(tile_vy.values.squeeze(), kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

            '''old method with scipy'''
            '''
            # Apply filter to velocities
            focus_filter_vx_50 = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_50, trunc=3.0)
            focus_filter_vx_100 = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_100, trunc=3.0)
            focus_filter_vx_150 = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_150, trunc=3.0)
            focus_filter_vx_300 = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_300, trunc=3.0)
            focus_filter_vx_450 = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_450, trunc=3.0)
            focus_filter_vx_af = gaussian_filter_with_nans(U=tile_vx.values, sigma=num_px_sigma_af, trunc=3.0)
            focus_filter_vy_50 = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_50, trunc=3.0)
            focus_filter_vy_100 = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_100, trunc=3.0)
            focus_filter_vy_150 = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_150, trunc=3.0)
            focus_filter_vy_300 = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_300, trunc=3.0)
            focus_filter_vy_450 = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_450, trunc=3.0)
            focus_filter_vy_af = gaussian_filter_with_nans(U=tile_vy.values, sigma=num_px_sigma_af, trunc=3.0)

            # Mask back the filtered arrays
            focus_filter_vx_50 = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_50)
            focus_filter_vx_100 = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_100)
            focus_filter_vx_150 = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_150)
            focus_filter_vx_300 = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_300)
            focus_filter_vx_450 = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_450)
            focus_filter_vx_af = np.where(np.isnan(tile_vx.values), np.nan, focus_filter_vx_af)
            focus_filter_vy_50 = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_50)
            focus_filter_vy_100 = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_100)
            focus_filter_vy_150 = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_150)
            focus_filter_vy_300 = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_300)
            focus_filter_vy_450 = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_450)
            focus_filter_vy_af = np.where(np.isnan(tile_vy.values), np.nan, focus_filter_vy_af)
            '''

            # create xarrays of filtered velocities
            focus_filter_v50_ar = tile_v.copy(deep=True, data=focus_filter_v50)
            focus_filter_v100_ar = tile_v.copy(deep=True, data=focus_filter_v100)
            focus_filter_v150_ar = tile_v.copy(deep=True, data=focus_filter_v150)
            focus_filter_v300_ar = tile_v.copy(deep=True, data=focus_filter_v300)
            focus_filter_v450_ar = tile_v.copy(deep=True, data=focus_filter_v450)
            focus_filter_vfa_ar = tile_v.copy(deep=True, data=focus_filter_af)

            #focus_filter_vx_50_ar = tile_vx.copy(deep=True, data=focus_filter_vx_50)
            #focus_filter_vx_100_ar = tile_vx.copy(deep=True, data=focus_filter_vx_100)
            #focus_filter_vx_150_ar = tile_vx.copy(deep=True, data=focus_filter_vx_150)
            #focus_filter_vx_300_ar = tile_vx.copy(deep=True, data=focus_filter_vx_300)
            #focus_filter_vx_450_ar = tile_vx.copy(deep=True, data=focus_filter_vx_450)
            #focus_filter_vx_af_ar = tile_vx.copy(deep=True, data=focus_filter_vx_af)
            #focus_filter_vy_50_ar = tile_vy.copy(deep=True, data=focus_filter_vy_50)
            #focus_filter_vy_100_ar = tile_vy.copy(deep=True, data=focus_filter_vy_100)
            #focus_filter_vy_150_ar = tile_vy.copy(deep=True, data=focus_filter_vy_150)
            #focus_filter_vy_300_ar = tile_vy.copy(deep=True, data=focus_filter_vy_300)
            #focus_filter_vy_450_ar = tile_vy.copy(deep=True, data=focus_filter_vy_450)
            #focus_filter_vy_af_ar = tile_vy.copy(deep=True, data=focus_filter_vy_af)

            # Calculate the velocity gradients
            #dvx_dx_ar, dvx_dy_ar = focus_filter_vx_300_ar.differentiate(
            #    coord='x'), focus_filter_vx_300_ar.differentiate(coord='y')
            #dvy_dx_ar, dvy_dy_ar = focus_filter_vy_300_ar.differentiate(
            #    coord='x'), focus_filter_vy_300_ar.differentiate(coord='y')

            # Interpolate (note: nans can be produced near boundaries)
            v_data = tile_v.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method="nearest").data
            v_filter_50_data = focus_filter_v50_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            v_filter_100_data = focus_filter_v100_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            v_filter_150_data = focus_filter_v150_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            v_filter_300_data = focus_filter_v300_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            v_filter_450_data = focus_filter_v450_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data

            '''
            vx_data = tile_vx.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method="nearest").data
            vy_data = tile_vy.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method="nearest").data
            vx_filter_50_data = focus_filter_vx_50_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                             method='nearest').data
            vx_filter_100_data = focus_filter_vx_100_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vx_filter_150_data = focus_filter_vx_150_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vx_filter_300_data = focus_filter_vx_300_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vx_filter_450_data = focus_filter_vx_450_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vx_filter_af_data = focus_filter_vx_af_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                             method='nearest').data
            vy_filter_50_data = focus_filter_vy_50_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                             method='nearest').data
            vy_filter_100_data = focus_filter_vy_100_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vy_filter_150_data = focus_filter_vy_150_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vy_filter_300_data = focus_filter_vy_300_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vy_filter_450_data = focus_filter_vy_450_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                               method='nearest').data
            vy_filter_af_data = focus_filter_vy_af_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar,
                                                             method='nearest').data

            dvx_dx_data = dvx_dx_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            dvx_dy_data = dvx_dy_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            dvy_dx_data = dvy_dx_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            dvy_dy_data = dvy_dy_ar.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method='nearest').data
            '''

            # some checks
            assert v_data.shape == v_filter_50_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_100_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_150_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_300_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_450_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_af_data.shape, "NSIDC interp something wrong!"


            # Fill dataframe with NSIDC velocities
            # Note this vectors may contain nans from interpolation close to margin/nunatak
            glathida.loc[indexes_id, 'v50'] = v_filter_50_data
            glathida.loc[indexes_id, 'v100'] = v_filter_100_data
            glathida.loc[indexes_id, 'v150'] = v_filter_150_data
            glathida.loc[indexes_id, 'v300'] = v_filter_300_data
            glathida.loc[indexes_id, 'v450'] = v_filter_450_data
            glathida.loc[indexes_id, 'vgfa'] = v_filter_af_data
            #glathida.loc[indexes_id, 'vx'] = vx_data
            #glathida.loc[indexes_id, 'vy'] = vy_data
            #glathida.loc[indexes_id, 'vx_gf50'] = vx_filter_50_data
            #glathida.loc[indexes_id, 'vx_gf100'] = vx_filter_100_data
            #glathida.loc[indexes_id, 'vx_gf150'] = vx_filter_150_data
            #glathida.loc[indexes_id, 'vx_gf300'] = vx_filter_300_data
            #glathida.loc[indexes_id, 'vx_gf450'] = vx_filter_450_data
            #glathida.loc[indexes_id, 'vx_gfa'] = vx_filter_af_data
            #glathida.loc[indexes_id, 'vy_gf50'] = vy_filter_50_data
            #glathida.loc[indexes_id, 'vy_gf100'] = vy_filter_100_data
            #glathida.loc[indexes_id, 'vy_gf150'] = vy_filter_150_data
            #glathida.loc[indexes_id, 'vy_gf300'] = vy_filter_300_data
            #glathida.loc[indexes_id, 'vy_gf450'] = vy_filter_450_data
            #glathida.loc[indexes_id, 'vy_gfa'] = vy_filter_af_data
            #glathida.loc[indexes_id, 'dvx_dx'] = dvx_dx_data
            #glathida.loc[indexes_id, 'dvx_dy'] = dvx_dy_data
            #glathida.loc[indexes_id, 'dvy_dx'] = dvy_dx_data
            #glathida.loc[indexes_id, 'dvy_dy'] = dvy_dy_data

        "Now we can interpolate BedMachine to fill ith_m for rgi 19"
        tqdm.write(f"Begin ice thickness for rgi {rgi}")

        # get NSIDC
        file_ith_NSIDC = f"{args.NSIDC_icethickness_folder_Antarctica}BedMachineAntarctica-v3.nc"

        mosaic_ith_NSIDC = rioxarray.open_rasterio(file_ith_NSIDC, masked=False)

        # Nb: ith_NSIDC nodata is 9.96921e+36. Also it contains zeros. All will to be converted to nans
        ith_NSIDC = mosaic_ith_NSIDC.thickness

        for i, id_rgi in tqdm(enumerate(ids_rgi), total=len(ids_rgi), desc=f"rgi {rgi} Glathida ID", leave=True):


            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]
            indexes_id = glathida_id.index.tolist()

            eastings_id, northings_id = (Transformer.from_crs("EPSG:4326", vx_NSIDC.rio.crs)
                                         .transform(glathida_id['POINT_LAT'], glathida_id['POINT_LON']))

            eastings_rgi_id_ar = xarray.DataArray(eastings_id)
            northings_rgi_id_ar = xarray.DataArray(northings_id)

            minE, maxE = min(eastings_id), max(eastings_id)
            minN, maxN = min(northings_id), max(northings_id)

            ris_metre_nsidc = ith_NSIDC.rio.resolution()[0]  # 500m
            eps = 15000

            try:
                ith_NSIDC_focus = ith_NSIDC.rio.clip_box(minx=minE - eps, miny=minN - eps, maxx=maxE + eps,
                                                       maxy=maxN + eps)
            except:
                tqdm.write(f'{i} No ith NSIDC data for rgi {rgi} GlaThiDa_ID {id_rgi}')
                continue

            # Condition 1. Either v is .rio.nodata or it is zero or it is nan
            cond0 = np.all(ith_NSIDC_focus.values == 0)
            condnodata = np.all(np.abs(ith_NSIDC_focus.values - ith_NSIDC_focus.rio.nodata) < 1.e-6)
            condnan = np.all(np.isnan(ith_NSIDC_focus.values))
            all_zero_or_nodata = cond0 or condnodata or condnan

            if all_zero_or_nodata:
                tqdm.write(f'{i} Cond 1 triggered - No NSIDC ith data for rgi {rgi} GlaThiDa_ID {id_rgi}')
                continue

            # Condition no. 2. A fast and quick interpolation to see if points intercepts a valid raster region
            vals_fast_interp = ith_NSIDC_focus.interp(y=xarray.DataArray(northings_id),
                                                     x=xarray.DataArray(eastings_id),
                                                     method='nearest').data

            cond_valid_fast_interp = (np.all(vals_fast_interp == 0) or np.isnan(vals_fast_interp).all() or
                                      np.all(np.abs(vals_fast_interp - ith_NSIDC_focus.rio.nodata) < 1.e-6))

            if cond_valid_fast_interp:
                tqdm.write(f'{i} Cond 2 triggered - No NSIDC ith data for rgi {rgi} GlaThiDa_ID {id_rgi}')
                continue


            #fig, ax = plt.subplots()
            #ith_NSIDC_focus.values[ith_NSIDC_focus.values == ith_NSIDC_focus.rio.nodata] = np.nan
            #ith_NSIDC_focus.values[ith_NSIDC_focus.values == 0.0] = np.nan
            #im = ith_NSIDC_focus.plot(ax=ax, cmap='viridis', zorder=0)
            #ax.scatter(x=eastings_id, y=northings_id, s=10, c='r', zorder=1)
            #plt.show()

            # At this stage we should have a good interpolation
            #ith_NSIDC_focus.values[ith_NSIDC_focus.values == ith_NSIDC_focus.rio.nodata] = np.nan
            ith_NSIDC_focus.values = np.where(
                (ith_NSIDC_focus.values == ith_NSIDC_focus.rio.nodata) | np.isinf(ith_NSIDC_focus.values),
                np.nan, ith_NSIDC_focus.values)
            ith_NSIDC_focus.values[ith_NSIDC_focus.values == 0.0] = np.nan
            ith_NSIDC_focus.rio.write_nodata(np.nan, inplace=True)

            # Note: for rgi 19 we do not interpolate NSIDC to remove nans.
            tile_ith = ith_NSIDC_focus.squeeze()

            # Interpolate (note: nans can be produced).
            ith_data = tile_ith.interp(y=northings_rgi_id_ar, x=eastings_rgi_id_ar, method="nearest").data

            #fig, ax = plt.subplots()
            #im = ith_NSIDC_focus.plot(ax=ax, cmap='viridis', zorder=0)
            #ax.scatter(x=eastings_id, y=northings_id, s=20, c=ith_data, ec='r', vmin=ith_NSIDC_focus.min(), vmax=ith_NSIDC_focus.max(), zorder=1)
            #plt.show()

            # Fill dataframe with NSIDC ith
            # Note this vectors may contain nans from interpolation close to margin/nunatak
            glathida.loc[indexes_id, 'ith_m'] = ith_data

        # How many nans we have produced from the interpolation
        glathida_rgi_ = glathida.loc[glathida['RGI'] == rgi]
        tqdm.write(
            f"\t From rgi {rgi} the no. nans in vx/vy/ith/vx300/vy300/etc: {np.sum(np.isnan(glathida_rgi_['ith_m']))}/{np.sum(np.isnan(glathida_rgi_['v50']))}"
            f"/{np.sum(np.isnan(glathida_rgi_['v100']))}/{np.sum(np.isnan(glathida_rgi_['v150']))}/{np.sum(np.isnan(glathida_rgi_['v300']))}/"
            f"{np.sum(np.isnan(glathida_rgi_['v450']))}/{np.sum(np.isnan(glathida_rgi_['vgfa']))}")


    for rgi in [5,]:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi) == 0:
            continue

        tqdm.write(f'rgi: {rgi}, Total points: {len(glathida_rgi)}')

        # get Millan files
        file_vx = f"{args.millan_velocity_folder}RGI-{rgi}/greenland_vel_mosaic250_vx_v1.tif"
        file_vy = f"{args.millan_velocity_folder}RGI-{rgi}/greenland_vel_mosaic250_vy_v1.tif"
        # files_ith = sorted(glob(f"{args.millan_icethickness_folder}RGI-{rgi}/THICKNESS_RGI-5*")) # we dont use millan
        file_ith_bedmacv5 = f"{args.NSIDC_icethickness_folder_Greenland}BedMachineGreenland-v5.nc"

        ''' BEDMACHINEV5 ITH '''
        tile_ith_bedmacv5 = rioxarray.open_rasterio(file_ith_bedmacv5, masked=False)
        tile_ith = tile_ith_bedmacv5['thickness']
        tile_ith = tile_ith.rio.write_crs("EPSG:3413")

        tile_ith.values[(tile_ith.values == tile_ith.rio.nodata) | (tile_ith.values == 0.0)] = np.nan
        tile_ith.rio.write_nodata(np.nan, inplace=True)
        tile_ith = tile_ith.squeeze()

        eastings, northings = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(glathida_rgi['POINT_LAT'],
                                                                                            glathida_rgi['POINT_LON'])
        eastings_ar = xarray.DataArray(eastings)
        northings_ar = xarray.DataArray(northings)

        # Interpolate BedMachinev5
        ith_data = tile_ith.interp(y=northings_ar, x=eastings_ar, method="nearest").data

        # fig, ax = plt.subplots()
        # tile_ith.rio.clip_box(minx=400000, miny=-2.6e6, maxx=800000, maxy=-1.8e6).plot(ax=ax)
        # ax.scatter(x=eastings_ar[~np.isnan(ith_data)], y=northings_ar[~np.isnan(ith_data)], c='b')
        # ax.scatter(x=eastings_ar[np.isnan(ith_data)], y=northings_ar[np.isnan(ith_data)], c='r', s=3)
        # plt.show()

        # Fill dataframe
        glathida.loc[glathida_rgi.index, 'ith_m'] = ith_data # roughly 35k are nans (of 557k)


        ''' NSIDC VELOCITY '''
        tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
        tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
        assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), 'Different bounds found.'
        assert tile_vx.rio.crs == tile_vy.rio.crs, 'Different crs found.'
        assert tile_vx.shape == tile_vy.shape, 'Different shape.'

        tile_vx.values[tile_vx.values==tile_vx.rio.nodata] = np.nan
        tile_vy.values[tile_vy.values==tile_vy.rio.nodata] = np.nan
        tile_vx.rio.write_nodata(np.nan, inplace=True)
        tile_vy.rio.write_nodata(np.nan, inplace=True)

        tile_vx = tile_vx.squeeze()
        tile_vy = tile_vy.squeeze()

        ris_metre_nsidc = tile_vx.rio.resolution()[0]  # 250m

        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()

        for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), desc=f"rgi {rgi} Glathida ID", leave=True):
            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]  # collapse glathida_rgi to specific id
            glathida_id = glathida_id.copy()

            indexes_id = glathida_id.index.tolist()

            lats = np.array(glathida_id['POINT_LAT'])
            lons = np.array(glathida_id['POINT_LON'])

            eastings_id, northings_id = (Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(lats, lons))

            eastings_id_ar = xarray.DataArray(eastings_id)
            northings_id_ar = xarray.DataArray(northings_id)

            minE, maxE = min(eastings_id), max(eastings_id)
            minN, maxN = min(northings_id), max(northings_id)
            #print(id, minE, maxE, minN, maxN)

            epsNSIDC = 5000

            try:
                tile_vx_id = tile_vx.rio.clip_box(minx=minE - epsNSIDC, miny=minN - epsNSIDC, maxx=maxE + epsNSIDC,
                                               maxy=maxN + epsNSIDC)
                tile_vy_id = tile_vy.rio.clip_box(minx=minE - epsNSIDC, miny=minN - epsNSIDC, maxx=maxE + epsNSIDC,
                                               maxy=maxN + epsNSIDC)
            except:
                print(f'no NSIDC velocity data for id {id_rgi}. Go to next id.')
                continue

            # Calculate sigma in meters for adaptive gaussian fiter
            sigma_af_min, sigma_af_max = 100.0, 2000.0
            try:
                area_id = glathida_rgi.loc[indexes_id, 'Area'].min()
                lmax_id = glathida_rgi.loc[indexes_id, 'Lmax'].max()
                a = 1e6 * area_id / (np.pi * 0.5 * lmax_id)
                sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
            except Exception as e:
                sigma_af = sigma_af_min
            assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"

            # Calculate how many pixels I need for a resolution of xx
            # Since NDIDC has res of 250 m, num pixels will can be very small.
            num_px_sigma_50 = max(1, round(50 / ris_metre_nsidc))
            num_px_sigma_100 = max(1, round(100 / ris_metre_nsidc))
            num_px_sigma_150 = max(1, round(150 / ris_metre_nsidc))
            num_px_sigma_300 = max(1, round(300 / ris_metre_nsidc))
            num_px_sigma_450 = max(1, round(450 / ris_metre_nsidc))
            num_px_sigma_af = max(1, round(sigma_af / ris_metre_nsidc))

            kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
            kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1,
                                         y_size=4 * num_px_sigma_100 + 1)
            kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1,
                                         y_size=4 * num_px_sigma_150 + 1)
            kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1,
                                         y_size=4 * num_px_sigma_300 + 1)
            kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1,
                                         y_size=4 * num_px_sigma_450 + 1)
            kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)


            tile_v_id = tile_vx_id.copy(deep=True, data=(tile_vx_id ** 2 + tile_vy_id ** 2) ** 0.5)

            # A check to see if velocity modules is as expected
            assert float(tile_v_id.sum()) > 0, "tile v is not as expected."

            '''astropy'''
            preserve_nans = True
            focus_filter_v50 = convolve_fft(tile_v_id.values, kernel50, nan_treatment='interpolate',
                                            preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v100 = convolve_fft(tile_v_id.values, kernel100, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v150 = convolve_fft(tile_v_id.values, kernel150, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v300 = convolve_fft(tile_v_id.values, kernel300, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_v450 = convolve_fft(tile_v_id.values, kernel450, nan_treatment='interpolate',
                                             preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
            focus_filter_af = convolve_fft(tile_v_id.values, kernelaf, nan_treatment='interpolate',
                                           preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

            # create xarrays of filtered velocities
            focus_filter_v50_ar = tile_v_id.copy(deep=True, data=focus_filter_v50)
            focus_filter_v100_ar = tile_v_id.copy(deep=True, data=focus_filter_v100)
            focus_filter_v150_ar = tile_v_id.copy(deep=True, data=focus_filter_v150)
            focus_filter_v300_ar = tile_v_id.copy(deep=True, data=focus_filter_v300)
            focus_filter_v450_ar = tile_v_id.copy(deep=True, data=focus_filter_v450)
            focus_filter_vfa_ar = tile_v_id.copy(deep=True, data=focus_filter_af)

            # Interpolate (note: nans can be produced near boundaries)
            v_data = tile_v_id.interp(y=northings_id_ar, x=eastings_id_ar, method="nearest").data
            v_filter_50_data = focus_filter_v50_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data
            v_filter_100_data = focus_filter_v100_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data
            v_filter_150_data = focus_filter_v150_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data
            v_filter_300_data = focus_filter_v300_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data
            v_filter_450_data = focus_filter_v450_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data
            v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_id_ar, x=eastings_id_ar, method='nearest').data

            #fig, ax = plt.subplots()
            #focus_filter_v50_ar.plot(ax=ax)
            #ax.scatter(x=eastings_id, y=northings_id, c=v_filter_50_data, s=30, vmin=focus_filter_v50_ar.min(), vmax=focus_filter_v50_ar.max())
            #plt.show()

            # some checks
            assert v_data.shape == v_filter_50_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_100_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_150_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_300_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_450_data.shape, "NSIDC interp something wrong!"
            assert v_data.shape == v_filter_af_data.shape, "NSIDC interp something wrong!"

            # Fill dataframe with NSIDC velocities
            glathida.loc[indexes_id, 'v50'] = v_filter_50_data
            glathida.loc[indexes_id, 'v100'] = v_filter_100_data
            glathida.loc[indexes_id, 'v150'] = v_filter_150_data
            glathida.loc[indexes_id, 'v300'] = v_filter_300_data
            glathida.loc[indexes_id, 'v450'] = v_filter_450_data
            glathida.loc[indexes_id, 'vgfa'] = v_filter_af_data

        """
        # ----------------------------------------------------------------------------------------
        # OLD GROUP METHOD THAT USES MILLAN ITH TILES (AND NSIDC VELOCITIES)
        # I need a dataframe for Millan with same indexes and lats lons
        df_pointsM = glathida_rgi[['POINT_LAT', 'POINT_LON']].copy()
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_ith})

        # Fill the dataframe for occupancy
        tocc0 = time.time()
        for i, file_ith in enumerate(files_ith):
            #print(i, file_ith)
            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

            eastings, northings = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(df_pointsM['POINT_LAT'],
                                                                                               df_pointsM['POINT_LON'])

            df_pointsM['eastings'] = eastings
            df_pointsM['northings'] = northings

            # Get the points inside the tile
            left, bottom, right, top = tile_ith.rio.bounds()

            within_bounds_mask = (
                    (df_pointsM['eastings'] >= left) &
                    (df_pointsM['eastings'] <= right) &
                    (df_pointsM['northings'] >= bottom) &
                    (df_pointsM['northings'] <= top))

            df_pointsM.loc[within_bounds_mask, file_ith] = 1

        df_pointsM.drop(columns=['eastings', 'northings'], inplace=True)
        ncols = df_pointsM.shape[1]
        print(f"Created dataframe of occupancies for all points in {time.time() - tocc0} s.")

        # Grouping by ith occupancy. Each group will have an occupancy value
        df_pointsM['ntiles_ith'] = df_pointsM.iloc[:, 2:].sum(axis=1)
        print(df_pointsM['ntiles_ith'].value_counts())
        groups_rgi5 = df_pointsM.groupby('ntiles_ith')  # Groups.
        df_pointsM.drop(columns=['ntiles_ith'], inplace=True)  # Remove this column that we used to create groups
        print(f"Num groups in Millan: {groups_rgi5.ngroups}")


        # Loop over k groups
        for k, (g_value, df_rgi_k) in enumerate(groups_rgi5):
            print(f"Group {k + 1}/{groups_rgi5.ngroups} with {len(df_rgi_k)} measurements")

            indexes_rgi_k = df_rgi_k.index

            # Insert this column at the beginning since we need it (at the beginning)
            # As a result df_rgi_k columns will be: |GlaThiDa_ID|POINT_LAT|POINT_LON|<ithtiles>|,
            # there are 3 columns at the beginning and then the tiles
            df_rgi_k.insert(0, 'GlaThiDa_ID', glathida_rgi.loc[indexes_rgi_k, 'GlaThiDa_ID'])

            # Get unique IDs of ids_rgi_k
            ids_rgi_k = df_rgi_k['GlaThiDa_ID'].unique().tolist()

            # loop over the unique IDs of group k
            for id_rgi_k in tqdm(ids_rgi_k, total=len(ids_rgi_k), desc=f"rgi {rgi} group {k+1}/{groups_rgi5.ngroups} Glathida ID",
                               leave=True):

                # FORCE AN ID FOR DEBUGGING
                # id_rgi_k = 2752

                # Get dataframe for group k and id
                df_rgi_k_id = df_rgi_k.loc[df_rgi_k['GlaThiDa_ID'] == id_rgi_k]
                indexes_rgi_k_id = df_rgi_k_id.index.tolist()

                # Get the unique valid tiles for each id
                unique_ith_tiles_k_id = df_rgi_k_id.iloc[:, 3:].columns[df_rgi_k_id.iloc[:, 3:].sum() != 0].tolist()

                lats_rgi_k_id = np.array(df_rgi_k_id['POINT_LAT'])
                lons_rgi_k_id = np.array(df_rgi_k_id['POINT_LON'])

                valid_ith_tile_rgi_k_id = None

                # Loop over tiles for group k and id
                for t, file_ith in enumerate(unique_ith_tiles_k_id):
                    #print(f"Tile {t}, {file_ith}")
                    tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

                    if tile_ith.rio.nodata is None: tile_ith.rio.write_nodata(np.nan, inplace=True)

                    assert tile_ith.rio.crs == "EPSG:3413", "projection not expected for Greenland Millan tiles."

                    eastings_rgi_k_id, northings_rgi_k_id = (Transformer.from_crs("EPSG:4326",tile_ith.rio.crs)
                                                            .transform(lats_rgi_k_id, lons_rgi_k_id))
                    minE, maxE = min(eastings_rgi_k_id), max(eastings_rgi_k_id)
                    minN, maxN = min(northings_rgi_k_id), max(northings_rgi_k_id)
                    # print(f"Boundaries measurements: {minE, minN, maxE, maxN}")
                    # print(f"tile {t} bounds {tile_ith.rio.bounds()}")

                    #fig, ax = plt.subplots()
                    #tile_ith.plot(ax=ax, cmap='viridis')
                    #ax.scatter(x=eastings_rgi_k_id, y=northings_rgi_k_id, c='k')
                    #plt.show()

                    epsM = 500
                    try:
                        tile_ith = tile_ith.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)

                        # Condition 1. Either ith is .rio.nodata or it is zero or it is nan
                        cond0 = np.all(tile_ith.values == 0)
                        condnodata = np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6)
                        condnan = np.all(np.isnan(tile_ith.values))
                        #print(cond0, condnodata, condnan)
                        #print(np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6))
                        #print(tile_ith.rio.nodata)
                        #all_zero_or_nodata = np.all(
                        #    np.logical_or(tile_ith.values == 0, tile_ith.values == tile_ith.rio.nodata))
                        all_zero_or_nodata = cond0 or condnodata or condnan

                        #print(f"Tile {t} condition {all_zero_or_nodata} {np.sum(tile_ith.values)} {tile_ith.rio.nodata}")
                        #input('wait')
                        #fig, ax = plt.subplots()
                        #tile_ith.plot(ax=ax, cmap='viridis')
                        #plt.show()

                        if all_zero_or_nodata:
                            # The tile t is not valid. Go to next tile
                            print('The tile t is not valid. Go to next tile')
                            continue

                        # Condition no. 2. A fast and quick interpolation to see if points intercepts a valid raster region
                        vals_fast_interp = tile_ith.interp(y=xarray.DataArray(northings_rgi_k_id),
                                                           x=xarray.DataArray(eastings_rgi_k_id),
                                                           method='nearest').data

                        cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                            np.all(np.abs(vals_fast_interp - tile_ith.rio.nodata) < 1.e-6))

                        if cond_valid_fast_interp:
                            continue

                        valid_ith_tile_rgi_k_id = tile_ith

                    except:
                        # The tile t could not include id_rgi_k data, go to next tile
                        tqdm.write(f'No millan data for rgi {rgi} group {k} GlaThiDa_ID {id_rgi_k} tile {t}')
                        continue

                if valid_ith_tile_rgi_k_id is None:
                    print(f"Impossible to get valid tile for group {k} ID {id_rgi_k}, no. meas {len(df_rgi_k_id)}.")

                else:
                    # We should have found the valid tile if we have reached this point

                    tile_ith = valid_ith_tile_rgi_k_id

                    # Mask nodata and inf values with np.nan
                    tile_ith.values = np.where((tile_ith.values == tile_ith.rio.nodata) | np.isinf(tile_ith.values),
                                               np.nan, tile_ith.values)

                    tile_ith.rio.write_nodata(np.nan, inplace=True)

                    # Note: for rgi 5 we do not interpolate to remove nans.
                    tile_ith = tile_ith.squeeze()

                    eastings_rgi_k_id_ar = xarray.DataArray(eastings_rgi_k_id)
                    northings_rgi_k_id_ar = xarray.DataArray(northings_rgi_k_id)

                    # Interpolate (note: nans can be produced near boundaries).
                    ith_data = tile_ith.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data

                    #fig, ax = plt.subplots()
                    #tile_ith.plot(ax=ax, cmap='viridis', vmin=tile_ith.min(), vmax=tile_ith.max())
                    #ax.scatter(x=eastings_rgi_k_id_ar, y=northings_rgi_k_id_ar, s=20, ec='k', c=ith_data, vmin=tile_ith.min(), vmax=tile_ith.max())
                    #plt.show()

                    # Fill dataframe with ith_m
                    glathida.loc[indexes_rgi_k_id, 'ith_m'] = ith_data


                '''At this point I am ready to interpolate the NSIDC velocity'''
                tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
                tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
                assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), 'Different bounds found.'
                assert tile_vx.rio.crs == tile_vy.rio.crs, 'Different crs found.'

                for tile in [tile_vx, tile_vy]:
                    if tile.rio.nodata is None:
                        tile.rio.write_nodata(np.nan, inplace=True)

                eastings_rgi_k_id, northings_rgi_k_id = (Transformer.from_crs("EPSG:4326", tile_ith.rio.crs)
                                                         .transform(lats_rgi_k_id, lons_rgi_k_id))

                eastings_rgi_k_id_ar = xarray.DataArray(eastings_rgi_k_id)
                northings_rgi_k_id_ar = xarray.DataArray(northings_rgi_k_id)

                minE, maxE = min(eastings_rgi_k_id), max(eastings_rgi_k_id)
                minN, maxN = min(northings_rgi_k_id), max(northings_rgi_k_id)

                epsNSIDC = 500
                tile_vx = tile_vx.rio.clip_box(minx=minE - epsNSIDC, miny=minN - epsNSIDC, maxx=maxE + epsNSIDC, maxy=maxN + epsNSIDC)
                tile_vy = tile_vy.rio.clip_box(minx=minE - epsNSIDC, miny=minN - epsNSIDC, maxx=maxE + epsNSIDC, maxy=maxN + epsNSIDC)

                # Condition for NSIDC v
                tile_vx_is_all_nodata = np.all(tile_vx.values == tile_vx.rio.nodata)

                # If we have some NSIDC data
                if not tile_vx_is_all_nodata:
                    tile_vx.values = np.where((tile_vx.values == tile_vx.rio.nodata) | np.isinf(tile_vx.values),
                                               np.nan, tile_vx.values)
                    tile_vy.values = np.where((tile_vy.values == tile_vy.rio.nodata) | np.isinf(tile_vy.values),
                                               np.nan, tile_vy.values)
                    #tile_vx.values[tile_vx.values == tile_vx.rio.nodata] = np.nan
                    #tile_vy.values[tile_vy.values == tile_vy.rio.nodata] = np.nan
                    tile_vx.rio.write_nodata(np.nan, inplace=True)
                    tile_vy.rio.write_nodata(np.nan, inplace=True)

                    assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, "NSIDC tiles vx, vy with different epsg."
                    assert tile_vx.rio.resolution() == tile_vy.rio.resolution(), "NSIDC vx, vy have different resolution."
                    assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), "NSIDC vx, vy bounds not the same"

                    # Note: for rgi 5 we do not interpolate NSIDC to remove nans.
                    tile_vx = tile_vx.squeeze()
                    tile_vy = tile_vy.squeeze()

                    ris_metre_nsidc = tile_vx.rio.resolution()[0]  # 250m

                    # Calculate sigma in meters for adaptive gaussian fiter
                    sigma_af_min, sigma_af_max = 100.0, 2000.0
                    try:
                        area_id = glathida_rgi.loc[indexes_rgi_k_id, 'Area'].min()
                        lmax_id = glathida_rgi.loc[indexes_rgi_k_id, 'Lmax'].max()
                        #print('area', area_id, 'lmax', lmax_id)
                        # print(lats_rgi_k_id.min(), lats_rgi_k_id.max(), lons_rgi_k_id.min(), lons_rgi_k_id.max(), area_id)
                        # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                        # area_id = glathida_rgi_tile_id['Area'].min()  # km2
                        # lmax_id = glathida_rgi_tile_id['Lmax'].max()  # m
                        a = 1e6 * area_id / (np.pi * 0.5 * lmax_id)
                        sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                        # print(area_id, lmax_id, a, value)
                    except Exception as e:
                        sigma_af = sigma_af_min
                    # Ensure that our value correctly in range [50.0, 2000.0]
                    assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"
                    # print(f"Adaptive gaussian filter with sigma = {value} meters.")

                    # Calculate how many pixels I need for a resolution of xx
                    # Since NDIDC has res of 250 m, num pixels will can be very small.
                    num_px_sigma_50 = max(1, round(50 / ris_metre_nsidc))
                    num_px_sigma_100 = max(1, round(100 / ris_metre_nsidc))
                    num_px_sigma_150 = max(1, round(150 / ris_metre_nsidc))
                    num_px_sigma_300 = max(1, round(300 / ris_metre_nsidc))
                    num_px_sigma_450 = max(1, round(450 / ris_metre_nsidc))
                    num_px_sigma_af = max(1, round(sigma_af / ris_metre_nsidc))

                    kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
                    kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
                    kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
                    kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
                    kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
                    kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

                    tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)

                    # A check to see if velocity modules is as expected
                    assert float(tile_v.sum()) > 0, "tile v is not as expected."

                    '''astropy'''
                    preserve_nans = True
                    focus_filter_v50 = convolve_fft(tile_v.values, kernel50, nan_treatment='interpolate',
                                                    preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    focus_filter_v100 = convolve_fft(tile_v.values, kernel100, nan_treatment='interpolate',
                                                     preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    focus_filter_v150 = convolve_fft(tile_v.values, kernel150, nan_treatment='interpolate',
                                                     preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    focus_filter_v300 = convolve_fft(tile_v.values, kernel300, nan_treatment='interpolate',
                                                     preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    focus_filter_v450 = convolve_fft(tile_v.values, kernel450, nan_treatment='interpolate',
                                                     preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    focus_filter_af = convolve_fft(tile_v.values, kernelaf, nan_treatment='interpolate',
                                                   preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

                    #focus_filter_vx_50 = convolve_fft(tile_vx.values.squeeze(), kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vx_100 = convolve_fft(tile_vx.values.squeeze(), kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vx_150 = convolve_fft(tile_vx.values.squeeze(), kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vx_300 = convolve_fft(tile_vx.values.squeeze(), kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vx_450 = convolve_fft(tile_vx.values.squeeze(), kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vx_af = convolve_fft(tile_vx.values.squeeze(), kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

                    #focus_filter_vy_50 = convolve_fft(tile_vy.values.squeeze(), kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vy_100 = convolve_fft(tile_vy.values.squeeze(), kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vy_150 = convolve_fft(tile_vy.values.squeeze(), kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vy_300 = convolve_fft(tile_vy.values.squeeze(), kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vy_450 = convolve_fft(tile_vy.values.squeeze(), kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                    #focus_filter_vy_af = convolve_fft(tile_vy.values.squeeze(), kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)


                    # create xarrays of filtered velocities
                    focus_filter_v50_ar = tile_v.copy(deep=True, data=focus_filter_v50)
                    focus_filter_v100_ar = tile_v.copy(deep=True, data=focus_filter_v100)
                    focus_filter_v150_ar = tile_v.copy(deep=True, data=focus_filter_v150)
                    focus_filter_v300_ar = tile_v.copy(deep=True, data=focus_filter_v300)
                    focus_filter_v450_ar = tile_v.copy(deep=True, data=focus_filter_v450)
                    focus_filter_vfa_ar = tile_v.copy(deep=True, data=focus_filter_af)

                    #focus_filter_vx_50_ar = tile_vx.copy(deep=True, data=focus_filter_vx_50)
                    #focus_filter_vx_100_ar = tile_vx.copy(deep=True, data=focus_filter_vx_100)
                    #focus_filter_vx_150_ar = tile_vx.copy(deep=True, data=focus_filter_vx_150)
                    #focus_filter_vx_300_ar = tile_vx.copy(deep=True, data=focus_filter_vx_300)
                    #focus_filter_vx_450_ar = tile_vx.copy(deep=True, data=focus_filter_vx_450)
                    #focus_filter_vx_af_ar = tile_vx.copy(deep=True, data=focus_filter_vx_af)
                    #focus_filter_vy_50_ar = tile_vy.copy(deep=True, data=focus_filter_vy_50)
                    #focus_filter_vy_100_ar = tile_vy.copy(deep=True, data=focus_filter_vy_100)
                    #focus_filter_vy_150_ar = tile_vy.copy(deep=True, data=focus_filter_vy_150)
                    #focus_filter_vy_300_ar = tile_vy.copy(deep=True, data=focus_filter_vy_300)
                    #focus_filter_vy_450_ar = tile_vy.copy(deep=True, data=focus_filter_vy_450)
                    #focus_filter_vy_af_ar = tile_vy.copy(deep=True, data=focus_filter_vy_af)

                    # Calculate the velocity gradients
                    #dvx_dx_ar, dvx_dy_ar = focus_filter_vx_300_ar.differentiate(
                    #    coord='x'), focus_filter_vx_300_ar.differentiate(coord='y')
                    #dvy_dx_ar, dvy_dy_ar = focus_filter_vy_300_ar.differentiate(
                    #    coord='x'), focus_filter_vy_300_ar.differentiate(coord='y')

                    # Interpolate (note: nans can be produced near boundaries)
                    v_data = tile_v.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data
                    v_filter_50_data = focus_filter_v50_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                  method='nearest').data
                    v_filter_100_data = focus_filter_v100_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                    method='nearest').data
                    v_filter_150_data = focus_filter_v150_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                    method='nearest').data
                    v_filter_300_data = focus_filter_v300_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                    method='nearest').data
                    v_filter_450_data = focus_filter_v450_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                    method='nearest').data
                    v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                  method='nearest').data
                    '''
                    vx_data = tile_vx.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data
                    vy_data = tile_vy.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data
                    vx_filter_50_data = focus_filter_vx_50_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                     method='nearest').data
                    vx_filter_100_data = focus_filter_vx_100_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vx_filter_150_data = focus_filter_vx_150_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vx_filter_300_data = focus_filter_vx_300_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vx_filter_450_data = focus_filter_vx_450_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vx_filter_af_data = focus_filter_vx_af_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                     method='nearest').data
                    vy_filter_50_data = focus_filter_vy_50_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                     method='nearest').data
                    vy_filter_100_data = focus_filter_vy_100_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vy_filter_150_data = focus_filter_vy_150_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vy_filter_300_data = focus_filter_vy_300_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vy_filter_450_data = focus_filter_vy_450_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                       method='nearest').data
                    vy_filter_af_data = focus_filter_vy_af_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                     method='nearest').data

                    dvx_dx_data = dvx_dx_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method='nearest').data
                    dvx_dy_data = dvx_dy_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method='nearest').data
                    dvy_dx_data = dvy_dx_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method='nearest').data
                    dvy_dy_data = dvy_dy_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method='nearest').data
                    '''

                    # some checks
                    assert v_data.shape == v_filter_50_data.shape, "NSIDC interp something wrong!"
                    assert v_data.shape == v_filter_100_data.shape, "NSIDC interp something wrong!"
                    assert v_data.shape == v_filter_150_data.shape, "NSIDC interp something wrong!"
                    assert v_data.shape == v_filter_300_data.shape, "NSIDC interp something wrong!"
                    assert v_data.shape == v_filter_450_data.shape, "NSIDC interp something wrong!"
                    assert v_data.shape == v_filter_af_data.shape, "NSIDC interp something wrong!"


                    # Fill dataframe with NSIDC velocities
                    glathida.loc[indexes_rgi_k_id, 'v50'] = v_filter_50_data
                    glathida.loc[indexes_rgi_k_id, 'v100'] = v_filter_100_data
                    glathida.loc[indexes_rgi_k_id, 'v150'] = v_filter_150_data
                    glathida.loc[indexes_rgi_k_id, 'v300'] = v_filter_300_data
                    glathida.loc[indexes_rgi_k_id, 'v450'] = v_filter_450_data
                    glathida.loc[indexes_rgi_k_id, 'vgfa'] = v_filter_af_data
                    #glathida.loc[indexes_rgi_k_id, 'vx'] = vx_data
                    #glathida.loc[indexes_rgi_k_id, 'vy'] = vy_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gf50'] = vx_filter_50_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gf100'] = vx_filter_100_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gf150'] = vx_filter_150_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gf300'] = vx_filter_300_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gf450'] = vx_filter_450_data
                    #glathida.loc[indexes_rgi_k_id, 'vx_gfa'] = vx_filter_af_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gf50'] = vy_filter_50_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gf100'] = vy_filter_100_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gf150'] = vy_filter_150_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gf300'] = vy_filter_300_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gf450'] = vy_filter_450_data
                    #glathida.loc[indexes_rgi_k_id, 'vy_gfa'] = vy_filter_af_data
                    #glathida.loc[indexes_rgi_k_id, 'dvx_dx'] = dvx_dx_data
                    #glathida.loc[indexes_rgi_k_id, 'dvx_dy'] = dvx_dy_data
                    #glathida.loc[indexes_rgi_k_id, 'dvy_dx'] = dvy_dx_data
                    #glathida.loc[indexes_rgi_k_id, 'dvy_dy'] = dvy_dy_data
        """

        # How many nans we have produced from the interpolation
        glathida_rgi_ = glathida.loc[glathida['RGI'] == rgi]
        tqdm.write(
            f"\t From rgi {rgi} the no. nans in ith/v50/v100/v150/etc: {np.sum(np.isnan(glathida_rgi_['ith_m']))}/{np.sum(np.isnan(glathida_rgi_['v50']))}"
            f"/{np.sum(np.isnan(glathida_rgi_['v100']))}/{np.sum(np.isnan(glathida_rgi_['v150']))}/{np.sum(np.isnan(glathida_rgi_['v300']))}/"
            f"{np.sum(np.isnan(glathida_rgi_['v450']))}/{np.sum(np.isnan(glathida_rgi_['vgfa']))}")


    for rgi in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    #for rgi in [2,]:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        if len(glathida_rgi) == 0:
            continue

        tqdm.write(f'rgi: {rgi}, Total points: {len(glathida_rgi)}')

        # get Millan files
        if rgi in (1, 2):
            files_vx = sorted(glob(f"{args.millan_velocity_folder}RGI-1-2/VX_RGI-1-2*"))
            files_vy = sorted(glob(f"{args.millan_velocity_folder}RGI-1-2/VY_RGI-1-2*"))
            files_ith = sorted(glob(f"{args.millan_icethickness_folder}RGI-1-2/THICKNESS_RGI-1-2*"))
        elif rgi in (13, 14, 15):
            files_vx = sorted(glob(f"{args.millan_velocity_folder}RGI-13-15/VX_RGI-13-15*.tif"))
            files_vy = sorted(glob(f"{args.millan_velocity_folder}RGI-13-15/VY_RGI-13-15*.tif"))
            files_ith = sorted(glob(f"{args.millan_icethickness_folder}RGI-13-15/THICKNESS_RGI-13-15*.tif"))
        else:
            files_vx = sorted(glob(f"{path_millan_velocity}RGI-{rgi}/VX_RGI-{rgi}*.tif"))
            files_vy = sorted(glob(f"{path_millan_velocity}RGI-{rgi}/VY_RGI-{rgi}*.tif"))
            files_ith = sorted(glob(f"{path_millan_icethickness}RGI-{rgi}/THICKNESS_RGI-{rgi}*.tif"))

        n_rgi_tiles = len(files_vx)

        # I need a dataframe for Millan with same indexes and lats lons
        df_pointsM = glathida_rgi[['POINT_LAT', 'POINT_LON']].copy()
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_vx})
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_ith})

        # Fill the dataframe for occupancy
        for i, (file_vx, file_vy, file_ith) in enumerate(zip(files_vx, files_vy, files_ith)):
            tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
            tile_vy = rioxarray.open_rasterio(file_vy, masked=False)  # may relax this
            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

            if not tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds():
                tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)

            assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds(), 'Different bounds found.'
            assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, 'Different crs found.'

            eastings, northings = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(df_pointsM['POINT_LAT'],
                                                                                               df_pointsM['POINT_LON'])
            df_pointsM['eastings'] = eastings
            df_pointsM['northings'] = northings

            # Get the points inside the tile
            left, bottom, right, top = tile_vx.rio.bounds()

            within_bounds_mask = (
                    (df_pointsM['eastings'] >= left) &
                    (df_pointsM['eastings'] <= right) &
                    (df_pointsM['northings'] >= bottom) &
                    (df_pointsM['northings'] <= top))

            df_pointsM.loc[within_bounds_mask, file_vx] = 1
            df_pointsM.loc[within_bounds_mask, file_ith] = 1

        df_pointsM.drop(columns=['eastings', 'northings'], inplace=True)
        ncols = df_pointsM.shape[1] # 2 + num tiles vx + num tiles ith
        print(f"Millan method: created dataframe of occupancies for all points of shape {df_pointsM.shape}")

        # Sanity check that all points are contained in the same way in vx and ith tiles
        n_tiles_occupancy_vx = df_pointsM.iloc[:, 2:2 + (ncols - 2) // 2].sum().sum()
        n_tiles_occupancy_ith = df_pointsM.iloc[:, 2 + (ncols - 2) // 2:].sum().sum()
        assert n_tiles_occupancy_vx == n_tiles_occupancy_ith, "Mismatch between vx and ith coverage."

        # Grouping by vx occupancy
        df_pointsM['ntiles_vx'] = df_pointsM.iloc[:, 2:2 + (ncols - 2) // 2].sum(axis=1)
        groups_rgi = df_pointsM.groupby('ntiles_vx')  # Groups. Each group will have an occupancy value
        df_pointsM.drop(columns=['ntiles_vx'], inplace=True)  # Remove this column that we used to create groups

        print(f"Num groups in Millan: {groups_rgi.ngroups}")

        # Loop over k groups
        for k, (g_value, df_rgi_k) in enumerate(groups_rgi):

            # These are the valid tiles in group k
            # but these are useless i think, as we want to loop over the tiles for each id
            unique_vx_tiles_k = df_rgi_k.iloc[:, 2:2 + (ncols - 2) // 2].columns[df_rgi_k.iloc[:, 2:2 + (ncols - 2) // 2].sum() != 0].tolist()
            unique_ith_tiles_k = df_rgi_k.iloc[:, 2 + (ncols - 2) // 2:].columns[df_rgi_k.iloc[:, 2 + (ncols - 2) // 2:].sum() != 0].tolist()
            print(f"Group {k + 1}/{groups_rgi.ngroups} with {len(df_rgi_k)} measurements")

            indexes_rgi_k = df_rgi_k.index

            # Insert this column at the beginning since we need it (at the beginning)
            # As a result df_rgi_k columns will be: |GlaThiDa_ID|POINT_LAT|POINT_LON|<vtiles>|<ithtiles>|,
            # there are 3 columns at the beginning and then the tiles
            df_rgi_k.insert(0, 'GlaThiDa_ID', glathida_rgi.loc[indexes_rgi_k, 'GlaThiDa_ID'])
            #df_rgi_k['GlaThiDa_ID'] = glathida_rgi.loc[indexes_rgi_k, 'GlaThiDa_ID']

            # Get unique IDs of ids_rgi_k
            ids_rgi_k = df_rgi_k['GlaThiDa_ID'].unique().tolist()


            # loop over the unique IDs of group k
            for id_rgi_k in tqdm(ids_rgi_k, total=len(ids_rgi_k), desc=f"rgi {rgi} group {k+1}/{groups_rgi.ngroups} Glathida ID",
                               leave=True):


                # Get dataframe for group k and id
                df_rgi_k_id = df_rgi_k.loc[df_rgi_k['GlaThiDa_ID'] == id_rgi_k]
                indexes_rgi_k_id = df_rgi_k_id.index.tolist()

                # Get the unique valid tiles for each id
                unique_vx_tiles_k_id =  df_rgi_k_id.iloc[:, 3:3 + (ncols - 2) // 2].columns[df_rgi_k_id.iloc[:, 3:3 + (ncols - 2) // 2].sum() != 0].tolist()
                unique_ith_tiles_k_id =  df_rgi_k_id.iloc[:, 3 + (ncols - 2) // 2:].columns[df_rgi_k_id.iloc[:, 3 + (ncols - 2) // 2:].sum() != 0].tolist()
                #print(unique_vx_tiles_k_id, unique_ith_tiles_k_id)

                lats_rgi_k_id = np.array(df_rgi_k_id['POINT_LAT'])
                lons_rgi_k_id = np.array(df_rgi_k_id['POINT_LON'])

                valid_vx_tile_rgi_k_id = None
                valid_vy_tile_rgi_k_id = None
                valid_ith_tile_rgi_k_id = None

                # Loop over tiles for group k and id
                for t, (file_vx, file_ith) in enumerate(zip(unique_vx_tiles_k_id, unique_ith_tiles_k_id)):
                    file_vy = file_vx.replace('VX', 'VY')
                    tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
                    tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
                    tile_ith = rioxarray.open_rasterio(file_ith, masked=False)


                    # Sometimes the attribute no data is not defined in Millans tiles
                    for tile in [tile_vx, tile_vy, tile_ith]:
                        if tile.rio.nodata is None:
                            tile.rio.write_nodata(np.nan, inplace=True)

                    if not tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds():
                        tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)

                    assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, 'Different crs found.'
                    assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds(), 'Different bounds found.'

                    eastings_rgi_k_id, northings_rgi_k_id = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(lats_rgi_k_id, lons_rgi_k_id)
                    minE, maxE = min(eastings_rgi_k_id), max(eastings_rgi_k_id)
                    minN, maxN = min(northings_rgi_k_id), max(northings_rgi_k_id)
                    #print(f"Boundaries measurements: {minE, minN, maxE, maxN}")
                    #print(f"tile {t} bounds {tile_vx.rio.bounds()}")

                    epsM = 500
                    try:
                        tile_vx = tile_vx.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
                        tile_vy = tile_vy.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
                        tile_ith = tile_ith.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
                        #print('si', 'ID', id_rgi_k, 'tile', t, lats_rgi_k_id.min(), lats_rgi_k_id.max(),
                        #      lons_rgi_k_id.min(), lons_rgi_k_id.max())
                        #fig, ax = plt.subplots()
                        #tile_vx.plot(ax=ax, cmap="viridis")
                        #ax.scatter(x=eastings_rgi_k_id, y=northings_rgi_k_id, s=10)
                        #plt.show()

                        # Either ith is .rio.nodata or it is zero
                        # Note these conditions are on ith_m. Misteriously it may happen that the
                        # v is empty but ith is not, e.g. rgi=2 id_rgi_k = 2084. I dont know how this can be possible
                        cond0 = np.all(tile_ith.values == 0)
                        condnodata = np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6)
                        condnan = np.all(np.isnan(tile_ith.values))
                        all_zero_or_nodata = cond0 or condnodata or condnan

                        if all_zero_or_nodata:
                            # The tile t is not valid. Go to next tile
                            continue
                        else:
                            #print(t, file_vx, file_ith)
                            valid_vx_tile_rgi_k_id = tile_vx
                            valid_vy_tile_rgi_k_id = tile_vy
                            valid_ith_tile_rgi_k_id = tile_ith

                    except:
                        # The tile t could not include id_rgi_k data, go to next tile
                        tqdm.write(f'No millan data for rgi {rgi} group {k} GlaThiDa_ID {id_rgi_k} tile {t}')
                        continue

                if valid_ith_tile_rgi_k_id is None:
                    # These are very few points in rgi 4
                    print(f"Impossible to get valid tile for group {k} ID {id_rgi_k}, no. meas {len(df_rgi_k_id)}. Go to next ID")
                    #print(lats_rgi_k_id.min(), lats_rgi_k_id.max(), lons_rgi_k_id.min(), lons_rgi_k_id.max())
                    continue


                # We should have found the valid tile if we have reached this point
                tile_vx = valid_vx_tile_rgi_k_id
                tile_vy = valid_vy_tile_rgi_k_id
                tile_ith = valid_ith_tile_rgi_k_id

                #print(minE, maxE, minN, maxN)
                #fig, ax = plt.subplots()
                #tile_ith.plot(ax=ax, cmap='viridis')
                #ax.scatter(x=eastings_rgi_k_id, y=northings_rgi_k_id, s=10)
                #plt.show()

                tile_vx.values = np.where(tile_vx.values == tile_vx.rio.nodata, np.nan, tile_vx.values)
                tile_vy.values = np.where(tile_vy.values == tile_vy.rio.nodata, np.nan, tile_vy.values)
                tile_ith.values = np.where(tile_ith.values == tile_ith.rio.nodata, np.nan, tile_ith.values)

                tile_vx.rio.write_nodata(np.nan, inplace=True)
                tile_vy.rio.write_nodata(np.nan, inplace=True)
                tile_ith.rio.write_nodata(np.nan, inplace=True)

                assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, "Tiles vx, vy, ith with different epsg."
                assert tile_vx.rio.resolution() == tile_vy.rio.resolution() == tile_ith.rio.resolution(), \
                    "Tiles vx, vy, ith have different resolution."

                if not tile_vx.rio.bounds() == tile_ith.rio.bounds():
                    tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)

                assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds(), "All tiles bounds not the same"

                ris_metre_millan = tile_vx.rio.resolution()[0]

                # Calculate sigma in meters for adaptive gaussian fiter
                sigma_af_min, sigma_af_max = 100.0, 2000.0
                try:
                    area_id = glathida_rgi.loc[indexes_rgi_k_id, 'Area'].min()
                    lmax_id = glathida_rgi.loc[indexes_rgi_k_id, 'Lmax'].max()
                    #print(lats_rgi_k_id.min(), lats_rgi_k_id.max(), lons_rgi_k_id.min(), lons_rgi_k_id.max(), area_id)
                    # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                    #area_id = glathida_rgi_tile_id['Area'].min()  # km2
                    #lmax_id = glathida_rgi_tile_id['Lmax'].max()  # m
                    a = 1e6 * area_id / (np.pi * 0.5 * lmax_id)
                    sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                    # print(area_id, lmax_id, a, value)
                except Exception as e:
                    sigma_af = sigma_af_min
                # Ensure that our value correctly in range [50.0, 2000.0]
                assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"
                # print(f"Adaptive gaussian filter with sigma = {value} meters.")
                # print(sigma_af)

                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
                num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
                num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
                num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6
                num_px_sigma_450 = max(1, round(450 / ris_metre_millan))
                num_px_sigma_af = max(1, round(sigma_af / ris_metre_millan))

                kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
                kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
                kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
                kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
                kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
                kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

                #fig, ax = plt.subplots()
                #tile_vy.plot(ax=ax, cmap='viridis')
                #ax.scatter(x=eastings_rgi_k_id, y=northings_rgi_k_id, s=10)
                #plt.show()

                # Points for interpolatetion
                eastings_rgi_k_id_ar = xarray.DataArray(eastings_rgi_k_id)
                northings_rgi_k_id_ar = xarray.DataArray(northings_rgi_k_id)

                # Deal with ith first
                focus_ith = tile_ith.squeeze()
                ith_data = focus_ith.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data

                # Fill dataframe with ith_m
                if not np.all(np.isnan(ith_data)):
                    glathida.loc[indexes_rgi_k_id, 'ith_m'] = ith_data

                    # If we have successfully filled ith_m we can proceed with velocity
                    tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)
                    tile_v = tile_v.squeeze()

                    #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                    #tile_vx.plot(ax=ax1)
                    #tile_ith.plot(ax=ax2)
                    #tile_v.plot(ax=ax3)
                    #plt.show()

                    # If velocity exist
                    if float(tile_v.sum()) > 0:

                        preserve_nans = True
                        # What may happen is that the tile actually contains some data (maybe from neighboring glaciers)
                        # but so little that interpolation fails. Or it can be also that the current glacier is not even
                        # covered by the tile
                        focus_filter_v50 = convolve_fft(tile_v.values, kernel50, nan_treatment='interpolate',
                                                        preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                        focus_filter_v100 = convolve_fft(tile_v.values, kernel100, nan_treatment='interpolate',
                                                         preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                        focus_filter_v150 = convolve_fft(tile_v.values, kernel150, nan_treatment='interpolate',
                                                         preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                        focus_filter_v300 = convolve_fft(tile_v.values, kernel300, nan_treatment='interpolate',
                                                         preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                        focus_filter_v450 = convolve_fft(tile_v.values, kernel450, nan_treatment='interpolate',
                                                         preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
                        focus_filter_af = convolve_fft(tile_v.values, kernelaf, nan_treatment='interpolate',
                                                       preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

                        # create xarrays of filtered velocities
                        focus_filter_v50_ar = tile_v.copy(deep=True, data=focus_filter_v50)
                        focus_filter_v100_ar = tile_v.copy(deep=True, data=focus_filter_v100)
                        focus_filter_v150_ar = tile_v.copy(deep=True, data=focus_filter_v150)
                        focus_filter_v300_ar = tile_v.copy(deep=True, data=focus_filter_v300)
                        focus_filter_v450_ar = tile_v.copy(deep=True, data=focus_filter_v450)
                        focus_filter_vfa_ar = tile_v.copy(deep=True, data=focus_filter_af)

                        # Interpolate
                        v_data = tile_v.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar, method="nearest").data
                        v_filter_50_data = focus_filter_v50_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                      method='nearest').data
                        v_filter_100_data = focus_filter_v100_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                        method='nearest').data
                        v_filter_150_data = focus_filter_v150_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                        method='nearest').data
                        v_filter_300_data = focus_filter_v300_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                        method='nearest').data
                        v_filter_450_data = focus_filter_v450_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                        method='nearest').data
                        v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_rgi_k_id_ar, x=eastings_rgi_k_id_ar,
                                                                      method='nearest').data


                        # some checks
                        assert ith_data.shape == v_data.shape == v_filter_50_data.shape, "Millan interp something wrong!"
                        assert v_filter_50_data.shape == v_filter_100_data.shape, "Millan interp something wrong!"
                        assert v_filter_50_data.shape == v_filter_150_data.shape, "Millan interp something wrong!"
                        assert v_filter_50_data.shape == v_filter_300_data.shape, "Millan interp something wrong!"
                        assert v_filter_50_data.shape == v_filter_450_data.shape, "Millan interp something wrong!"
                        assert v_filter_50_data.shape == v_filter_af_data.shape, "Millan interp something wrong!"

                        # Fill dataframe
                        glathida.loc[indexes_rgi_k_id, 'v50'] = v_filter_50_data
                        glathida.loc[indexes_rgi_k_id, 'v100'] = v_filter_100_data
                        glathida.loc[indexes_rgi_k_id, 'v150'] = v_filter_150_data
                        glathida.loc[indexes_rgi_k_id, 'v300'] = v_filter_300_data
                        glathida.loc[indexes_rgi_k_id, 'v450'] = v_filter_450_data
                        glathida.loc[indexes_rgi_k_id, 'vgfa'] = v_filter_af_data


                # debug
                #if np.isnan(ith_data).sum()>0:
                    #df_rgi_k.loc[df_rgi_k['GlaThiDa_ID'] == id_rgi_k]
                    #rgiid = df_rgi_k['RGIId']
                    #print(len(df_rgi_k_id), df_rgi_k_id)
                    #print(id_rgi_k, np.isnan(vx_data).sum(), np.isnan(ith_data).sum(),
                    #     min(lons_rgi_k_id), max(lons_rgi_k_id), min(lats_rgi_k_id), max(lats_rgi_k_id))

                # Plot
                ifplot_millan = False
                if ifplot_millan and np.any(np.isnan(vx_data)):
                    ifplot_millan = True
                else:
                    ifplot_millan = False
                if ifplot_millan:
                    lons_crs, lats_crs = eastings_rgi_k_id, northings_rgi_k_id
                    #lons_crs, lats_crs = lons_crs.to_numpy(), lats_crs.to_numpy()

                    fig, axes = plt.subplots(3, 3, figsize=(10, 5))
                    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

                    im1 = focus_vx.plot(ax=ax1, cmap='viridis', vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx))

                    s1 = ax1.scatter(x=lons_crs, y=lats_crs, s=15, c=vx_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx), zorder=1)

                    s1_1 = ax1.scatter(x=lons_crs[np.argwhere(np.isnan(vx_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vx_data))],
                                       s=15, c='magenta', zorder=1)

                    im2 = focus_vy.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy))
                    s2 = ax2.scatter(x=lons_crs, y=lats_crs, s=15, c=vy_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy), zorder=1)
                    s2_1 = ax2.scatter(x=lons_crs[np.argwhere(np.isnan(vy_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vy_data))],
                                       s=15, c='magenta', zorder=1)

                    im3 = focus_ith.plot(ax=ax3, cmap='viridis', vmin=0, vmax=np.nanmax(ith_data))
                    s3 = ax3.scatter(x=lons_crs, y=lats_crs, s=15, c=ith_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=0, vmax=np.nanmax(ith_data), zorder=1)
                    s3_1 = ax3.scatter(x=lons_crs[np.argwhere(np.isnan(ith_data))],
                                       y=lats_crs[np.argwhere(np.isnan(ith_data))],
                                       s=15, c='magenta', zorder=1)

                    im4 = focus_filter_vx_150_ar.plot(ax=ax4, cmap='viridis', vmin=np.nanmin(focus_vx),
                                                      vmax=np.nanmax(focus_vx))
                    s4 = ax4.scatter(x=lons_crs, y=lats_crs, s=15, c=vx_filter_150_data, ec=(0, 0, 0, 0.3),
                                     cmap='viridis',
                                     vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx), zorder=1)
                    s4_1 = ax4.scatter(x=lons_crs[np.argwhere(np.isnan(vx_filter_150_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vx_filter_150_data))], s=15, c='magenta',
                                       zorder=1)

                    im5 = focus_filter_vy_150_ar.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(focus_vy),
                                                      vmax=np.nanmax(focus_vy))
                    s5 = ax5.scatter(x=lons_crs, y=lats_crs, s=15, c=vy_filter_150_data, ec=(0, 0, 0, 0.3),
                                     cmap='viridis',
                                     vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy), zorder=1)
                    s5_1 = ax5.scatter(x=lons_crs[np.argwhere(np.isnan(vy_filter_150_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vy_filter_150_data))], s=15,
                                       c='magenta', zorder=1)

                    ax6.axis("off")
                    ax9.axis("off")

                    im7 = dvx_dx_ar.plot(ax=ax7, cmap='viridis', )
                    im8 = dvy_dy_ar.plot(ax=ax8, cmap='viridis', )

                    ax1.title.set_text('vx')
                    ax2.title.set_text('vy')
                    ax3.title.set_text('ice thickness')
                    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
                        ax.set(xlabel='', ylabel='')
                    plt.tight_layout()
                    plt.show()


        # How many nans we have produced from the interpolation
        glathida_rgi_ = glathida.loc[glathida['RGI'] == rgi]
        tqdm.write(
            f"\t From rgi {rgi} the no. nans in ith/v50/v100/v150/etc: {np.sum(np.isnan(glathida_rgi_['ith_m']))}/{np.sum(np.isnan(glathida_rgi_['v50']))}"
            f"/{np.sum(np.isnan(glathida_rgi_['v100']))}/{np.sum(np.isnan(glathida_rgi_['v150']))}/{np.sum(np.isnan(glathida_rgi_['v300']))}/"
            f"{np.sum(np.isnan(glathida_rgi_['v450']))}/{np.sum(np.isnan(glathida_rgi_['vgfa']))}")

    print(f"Millan done in {(time.time()-tm)/60} min.")

    return glathida

"""Add distance from border using glacier geometries"""
def add_dist_from_boder_using_geometries(glathida):
    print("Adding distance to border using a geometrical approach...")

    if ('dist_from_border_km_geom' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['dist_from_border_km_geom'] = [np.nan] * len(glathida)

    def add_new_neighbors(neigbords, df):
        """ I give a list of neighbors and I should return a new list with added neighbors"""
        for id in neigbords:
            neighbors_wrt_id = df[df['RGIId_1'] == id]['RGIId_2'].unique()
            neigbords = np.append(neigbords, neighbors_wrt_id)
        neigbords = np.unique(neigbords)
        return neigbords

    def find_cluster_RGIIds(id, df):
        neighbors0 = np.array([id])
        len0 = len(neighbors0)
        neighbors1 = add_new_neighbors(neighbors0, df)
        len1 = len(neighbors1)
        while (len1 > len0):
            len0 = len1
            neighbors1 = add_new_neighbors(neighbors1, df)
            len1 = len(neighbors1)
        if (len(neighbors1)) ==1: return None
        else: return neighbors1

    regions = list(range(1, 20))

    # loop over regions
    for rgi in tqdm(regions, total=len(regions), desc='Distances in RGI',  leave=True):
        t_ = time.time()

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi)==0:
            continue
        #tqdm.write(f"Region RGI: {rgi}, {len(glathida_rgi['RGIId'].dropna())} points")

        rgi_ids = glathida_rgi['RGIId'].dropna().unique().tolist()
        # print(f"We have {len(rgi_ids)} valid glaciers and {len(glathida_rgi)} rows "
        #      f"of which {glathida_rgi['RGIId'].isna().sum()} points without a glacier id (hence nan)"
        #      f"and {glathida_rgi['RGIId'].notna().sum()} points with valid glacier id")

        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')  # get rgi region shp
        oggm_rgi_intersects_shp = utils.get_rgi_intersects_region_file(f"{rgi:02d}", version='62')

        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')  # get rgi dataset of glaciers
        oggm_rgi_intersects = gpd.read_file(oggm_rgi_intersects_shp, engine='pyogrio')  # get rgi dataset of glaciers intersects

        # loop over glaciers
        # Note: It is important to note that since rgi_ids do not contain nans, looping over it automatically
        # selects only the points inside glaciers (and not those outside)
        for rgi_id in tqdm(rgi_ids, total=len(rgi_ids), desc=f"Glaciers in rgi {rgi}", leave=False, position=0):

            multipolygon = False

            # Get glacier glathida and oggm datasets
            try:
                glathida_id = glathida_rgi.loc[glathida_rgi['RGIId'] == rgi_id] # glathida dataset

                gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == rgi_id] # oggm dataset
                gl_geom = gl_df['geometry'].item()  # glacier geometry Polygon
                gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
                gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors]  # list of nunataks Polygons
                #print(f"Glacier {rgi_id} found and its {len(glathida_id)} points contained.")
                assert len(gl_df) == 1, "Check this please."
                # Get the UTM EPSG code from glacier center coordinates
                cenLon, cenLat = gl_df['CenLon'].item(), gl_df['CenLat'].item()
                _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)

            except Exception as e:
                print(f"Error {e} with glacier {rgi_id}. It was not found so it be skipped.")
                continue

            # intersects of glacier (need only for plotting purposes)
            gl_intersects = oggm.utils.get_rgi_intersects_entities([rgi_id], version='62')

            # Calculate intersects of all glaciers in the cluster
            #todo: replace find_cluster_RGIIds with networkx graph, more precise. See fetch metadata
            list_cluster_RGIIds = find_cluster_RGIIds(rgi_id, oggm_rgi_intersects)
            #print(f"List of glacier cluster: {list_cluster_RGIIds}")

            if list_cluster_RGIIds is not None:
                # (need only for plotting purposes)
                cluster_intersects = oggm.utils.get_rgi_intersects_entities(list_cluster_RGIIds, version='62')
            else: cluster_intersects = None

            # Now calculate the geometries
            # todo: remove this if-else
            if list_cluster_RGIIds is None:  # Case 1: isolated glacier
                #print(f"Isolated glacier")
                exterior_ring = gl_geom.exterior  # shapely.geometry.polygon.LinearRing
                interior_rings = gl_geom.interiors  # shapely.geometry.polygon.InteriorRingSequence of polygon.LinearRing
                geoseries_geometries_4326 = gpd.GeoSeries([exterior_ring] + list(interior_rings), crs="EPSG:4326")
                geoseries_geometries_epsg = geoseries_geometries_4326.to_crs(epsg=glacier_epsg)

            elif list_cluster_RGIIds is not None:  # Case 2: cluster of glaciers with ice divides
                #print(f"Cluster of glaciers with ice divides.")
                cluster_geometry_list = []
                for gl_neighbor_id in list_cluster_RGIIds:
                    gl_neighbor_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_neighbor_id]
                    gl_neighbor_geom = gl_neighbor_df['geometry'].item()
                    cluster_geometry_list.append(gl_neighbor_geom)

                # Combine into a series of all glaciers in the cluster
                cluster_geometry_4326 = gpd.GeoSeries(cluster_geometry_list, crs="EPSG:4326")

                # Now remove all ice divides
                # todo: replace .unary_union with .union_all(method='coverage')
                cluster_geometry_no_divides_4326 = gpd.GeoSeries(cluster_geometry_4326.unary_union, crs="EPSG:4326")
                cluster_geometry_no_divides_epsg = cluster_geometry_no_divides_4326.to_crs(epsg=glacier_epsg)
                if cluster_geometry_no_divides_epsg.item().geom_type == 'Polygon':
                    cluster_exterior_ring = [cluster_geometry_no_divides_epsg.item().exterior]  # shapely.geometry.polygon.LinearRing
                    cluster_interior_rings = list(cluster_geometry_no_divides_epsg.item().interiors) # shapely.geometry.polygon.LinearRing
                elif cluster_geometry_no_divides_epsg.item().geom_type == 'MultiPolygon':
                    polygons = list(cluster_geometry_no_divides_epsg.item().geoms)
                    cluster_exterior_ring = [polygon.exterior for polygon in polygons] # list of shapely.geometry.polygon.LinearRing
                    num_multipoly = len(cluster_exterior_ring)
                    cluster_interior_ringSequences = [polygon.interiors for polygon in polygons] # list of shapely.geometry.polygon.InteriorRingSequence
                    cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence] # list of shapely.geometry.polygon.LinearRing
                    multipolygon = True
                else: raise ValueError("Unexpected geometry type. Please check.")

                # Create a geoseries of all external and internal geometries (NB I set here the glacier_epsg)
                geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings).set_crs(epsg=glacier_epsg)

            # Get all points and create Geopandas geoseries and convert to glacier center UTM
            # Note: a delicate issue is that technically each point may have its own UTM zone.
            # To keep things consistent I decide to project all of them to the glacier center UTM.
            lats = glathida_id['POINT_LAT'].tolist()
            lons = glathida_id['POINT_LON'].tolist()
            list_points = [Point(lon, lat) for (lon, lat) in zip(lons, lats)]
            geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
            geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=glacier_epsg)

            # List of distances for glacier_id
            glacier_id_dist = []

            # Decide which method to use (default should be method_KDTree_spatial_index)
            method_geopandas_spatial_index = False
            method_KDTree_spatial_index = True
            method_geopandas_distances = False

            if method_geopandas_spatial_index:
                # Create spatial index for the geometries
                sindex_id = geoseries_geometries_epsg.sindex

            if method_KDTree_spatial_index:
                # 1. Extract all coordinates from the GeoSeries geometries for the current glacier
                #if (rgi in (5, 19) and len(geoseries_geometries_epsg) > 1):
                    # In rgi 5 and 19 given that we have an ice sheet we remove cluster external geometry from calculation
                    # geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in geoseries_geometries_epsg[1:].geometry])
                #else:
                geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in geoseries_geometries_epsg.geometry])

                # 2. instantiate kdtree
                kdtree = KDTree(geoms_coords_array)

            for i, (idx, lon, lat) in tqdm(enumerate(zip(glathida_id.index, lons, lats)), total=len(lons), desc='Points', leave=False, position=0):

                # Make check 0.
                make_check0 = False
                if make_check0 and i==0:
                    lat_check = glathida_id.loc[idx, 'POINT_LAT']
                    lon_check = glathida_id.loc[idx, 'POINT_LON']
                    #print(lon_check, lat_check, lon, lat)

                # Make check 1.
                make_check1 = True
                if make_check1:
                    is_inside = gl_geom_ext.contains(Point(lon, lat))
                    assert is_inside is True, f"The point is expected to be inside but is outside glacier."

                make_check2 = False
                # Note that the UTM zone of one point may not be the UTM zone of the glacier center
                # I think projecting the point using the UTM zone of the glacier center is fine.
                if make_check2:
                    easting, nothing, zonenum, zonelett, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
                    if epsg != glacier_epsg:
                        print(f"Note differet UTM zones. Point espg {epsg} and glacier center epsg {glacier_epsg}.")

                # Decide whether point is inside a nunatak. If yes set the distance to nan
                is_nunatak = any(nunatak.contains(Point(lon, lat)) for nunatak in gl_geom_nunataks_list)
                if is_nunatak is True:
                    min_dist = np.nan

                # if not
                else:
                    # get shapely Point
                    point_epsg = geoseries_points_epsg.iloc[i]

                    # Method 1 with geopandas spatial index (fast)
                    if method_geopandas_spatial_index:

                        # Find the index of the nearest geometry
                        nearest_idx = sindex_id.nearest(point_epsg.bounds)
                        # Get the nearest geometry (NB may consists of more than one geometry)
                        nearest_geometries = geoseries_geometries_epsg.iloc[nearest_idx]
                        # Calculate the distance between the closest geometry and the point
                        min_distances = nearest_geometries.distance(point_epsg)
                        # Find the index of the row with the minimum distance
                        min_idx = min_distances.idxmin()

                        # Get the minimum distance and corresponding geometry
                        min_dist_spatial_index = min_distances.loc[min_idx]
                        nearest_geometry = nearest_geometries.loc[min_idx]
                        #print(min_distances)
                        #print(min_dist_spatial_index)
                        #print(nearest_geometries)

                        # Find the nearest point on the boundary of the polygon
                        get_closest_point = True
                        if get_closest_point:
                            nearest_point_on_boundary, nearest_point_point = nearest_points(nearest_geometry, point_epsg)
                            # Calculate the minimum distance again, just to verify
                            #min_distance_check = nearest_point_on_boundary.distance(point_epsg)
                            #print(min_distance_check)

                    # Method 2 with KDTree
                    if method_KDTree_spatial_index:

                        point_array = np.array(point_epsg.coords)  # (1,2)

                        # Perform nearest neighbor search for each point and calculate minimum distances
                        # both distances and indices have shape (1, len(geoseries_geometries_epsg))
                        distances, indices = kdtree.query(point_array, k=len(geoseries_geometries_epsg))
                        min_dist_KDTree = distances[0,0] # np.min(distances, axis=1)) or equivalently distances[:,0]
                        closest_point_index = indices[0, 0]
                        closest_point = Point(geoms_coords_array[closest_point_index])
                        #tqdm.write(f"{min_dist_KDTree}")
                        #tqdm.write(f"{closest_point, nearest_point_on_boundary}")

                    # Method 3 (exact but slow): Calculate the distances between such point and all glacier geometries.
                    if method_geopandas_distances:
                        min_distances_point_geometries = geoseries_geometries_epsg.distance(point_epsg)
                        min_dist_geopandas_distances = np.min(min_distances_point_geometries)  # unit UTM: m

                        # To debug we want to check what point corresponds to the minimum distance.
                        debug_distance = True
                        if debug_distance:
                            min_distance_index = min_distances_point_geometries.idxmin()
                            nearest_line = geoseries_geometries_epsg.loc[min_distance_index]
                            nearest_point_on_line = nearest_line.interpolate(nearest_line.project(point_epsg))


                # Fill distance list for glacier id of point
                glacier_id_dist.append(min_dist_KDTree/1000.)

                # For debugging
                if method_KDTree_spatial_index and method_geopandas_spatial_index and method_geopandas_distances:
                    if min_dist_KDTree>300 and abs(min_dist_KDTree-min_dist_geopandas_distances)/min_dist_geopandas_distances >.5:
                        plot_calculate_distance = True
                        print(min_dist_KDTree, min_dist_geopandas_distances, min_dist_spatial_index)
                    else: plot_calculate_distance = False
                else: plot_calculate_distance = False

                # Plot
                #r = random.uniform(0,1)
                #if (plot_calculate_distance and list_cluster_RGIIds is not None and r<1.0):
                if plot_calculate_distance:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.plot(*gl_geom_ext.exterior.xy, lw=1, c='magenta', zorder=4)
                    for interior in gl_geom.interiors:
                        ax1.plot(*interior.xy, lw=1, c='blue')

                    # Plot boundaries (only external periphery) of all glaciers in the cluster
                    if list_cluster_RGIIds is not None:
                        for gl_neighbor_id in list_cluster_RGIIds:
                            gl_neighbor_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_neighbor_id]
                            gl_neighbor_geom = gl_neighbor_df['geometry'].item()  # glacier geometry Polygon
                            ax1.plot(*gl_neighbor_geom.exterior.xy, lw=1, c='orange', zorder=0)

                    # Plot intersections of central glacier with its neighbors
                    for k, intersect in enumerate(gl_intersects['geometry']):  # Linestring gl_intersects
                        ax1.plot(*intersect.xy, lw=1, color='k')

                    # Plot intersections of all glaciers in the cluster
                    if cluster_intersects is not None:
                        for k, intersect in enumerate(cluster_intersects['geometry']):
                            ax1.plot(*intersect.xy, lw=1, color='k')  # np.random.rand(3)

                        # Plot cluster ice divides removed
                        if multipolygon:
                            polygons = list(cluster_geometry_no_divides_4326.item().geoms)
                            cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
                            cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
                            cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
                            for exterior in cluster_exterior_ring:
                                ax1.plot(*exterior.xy , lw=1, c='red', zorder=3)
                            for interior in cluster_interior_rings:
                                ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                        else:
                            ax1.plot(*cluster_geometry_no_divides_4326.item().exterior.xy, lw=1, c='red', zorder=3)
                            for interior in cluster_geometry_no_divides_4326.item().interiors:
                                ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                    if is_nunatak: ax1.scatter(lon, lat, s=50, lw=2, c='b')
                    else: ax1.scatter(lon, lat, s=50, lw=2, c='r', ec='r')

                    if multipolygon:
                        for i_poly in range(num_multipoly):
                            ax2.plot(*geoseries_geometries_epsg.loc[i_poly].xy, lw=1, c='red')  # first num_multipoly are outside borders
                        for inter in geoseries_geometries_epsg.loc[num_multipoly:]:  # all interiors if present
                            ax2.plot(*inter.xy, lw=1, c='blue')

                    else:
                        ax2.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='red')  # first entry is outside border
                        for inter in geoseries_geometries_epsg.loc[1:]:  # all interiors if present
                            ax2.plot(*inter.xy, lw=1, c='blue')

                    if is_nunatak: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='b', zorder=5)
                    else: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='r', ec='r', zorder=5)

                    if method_geopandas_distances and debug_distance:
                        ax2.scatter(*nearest_point_on_line.xy, s=50, lw=2, c='g', zorder=5)

                    if method_geopandas_spatial_index and get_closest_point:
                        ax2.scatter(x=nearest_point_on_boundary.x, y=nearest_point_on_boundary.y, s=40, lw=2, c='y', zorder=5)

                    if method_KDTree_spatial_index: ax2.scatter(x=closest_point.x, y=closest_point.y, s=30, lw=2, c='k', zorder=5)

                    ax1.set_title('EPSG 4326')
                    ax2.set_title(f'EPSG {glacier_epsg}')
                    plt.show()


            assert len(glacier_id_dist)==len(glathida_id.index), "Number of measurements does not match index length"
            #tqdm.write(f"Finished glacier {rgi_id} with {len(glacier_id_dist)} measurements")

            # Fill dataframe with distances from glacier rgi_id
            glathida.loc[glathida_id.index, 'dist_from_border_km_geom'] = glacier_id_dist

        tqdm.write(f"Finished region {rgi} in {(time.time()-t_)/60} mins.")

    return glathida

"""Add distance from water"""
def add_dist_from_water(glathida, path_GSHHG_folder):
    print("Adding distance to water...")

    if ('dist_from_ocean' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    # Import coastlines (we have merged L1 with L6)
    gdf16 = gpd.read_file(f'{path_GSHHG_folder}GSHHS_f_L1_L6.shp', engine='pyogrio')

    glathida['dist_from_ocean'] = [np.nan] * len(glathida)

    regions = list(range(1, 20))
    #regions = [19,]

    # loop over regions
    for rgi in tqdm(regions, total=len(regions), desc='Distances from ocean in RGI', leave=True):

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi) == 0:
            continue

        rgi_ids = glathida_rgi['RGIId'].dropna().unique().tolist()

        # loop over glaciers
        for rgi_id in tqdm(rgi_ids, total=len(rgi_ids), desc=f"Glaciers in rgi {rgi}", leave=False, position=0):

            glathida_id = glathida_rgi.loc[glathida_rgi['RGIId'] == rgi_id] # glathida dataset

            # get lons and lats, get epsg of center of data
            lats, lons = glathida_id['POINT_LAT'], glathida_id['POINT_LON']
            lats_mean, lons_mean = glathida_id['POINT_LAT'].mean(), glathida_id['POINT_LON'].mean()
            lats_min, lons_min = glathida_id['POINT_LAT'].min(), glathida_id['POINT_LON'].min()
            lats_max, lons_max = glathida_id['POINT_LAT'].max(), glathida_id['POINT_LON'].max()
            _, _, _, _, id_epsg = from_lat_lon_to_utm_and_epsg(lats_mean, lons_mean)
            #print(lats_mean, lons_mean, id_epsg)


            list_points = [Point(lon, lat) for (lon, lat) in zip(lons, lats)]
            geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
            geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=id_epsg)

            buffer = 1
            box_geoms = gdf16.cx[lons_min-buffer:lons_max+buffer, lats_min-buffer:lats_max+buffer]

            if len(box_geoms) == 0:
                # We are in the island case in Antarctica, e.g. -73.10288797227037 -105.166778923743
                # It may happen that no geometries gshhg are intercepted
                # In this case we fill dataframe with dist_from_border_km_geom
                glathida.loc[glathida_id.index, 'dist_from_ocean'] = glathida.loc[glathida_id.index, 'dist_from_border_km_geom']
                continue

            # else we have found at least one gshhg geometry
            box_geoms_epsg = box_geoms.to_crs(id_epsg)
            #print(f"No. geometries in box: {len(box_geoms)}")
            #print(f"Num points: ", len(geoseries_points_4326))

            # Extract all coordinates of GeoSeries points and geometries
            points_coords_array = np.column_stack((geoseries_points_epsg.geometry.x, geoseries_points_epsg.geometry.y))  # (10000,2)
            geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in box_geoms_epsg.geometry.exterior])

            #fig, ax = plt.subplots()
            #box_geoms_epsg.plot(ax=ax, linestyle='-', linewidth=1, facecolor='none', edgecolor='red')
            #geoseries_points_epsg.plot(ax=ax, c='k', markersize=2)
            #plt.show()

            # Reprojecting very big geometries cause distortion. Let's remove these points.
            valid_coords_mask = (
                    (geoms_coords_array[:, 0] >= -1e7) & (geoms_coords_array[:, 0] <= 1e7) &
                    (geoms_coords_array[:, 1] >= -1e7) & (geoms_coords_array[:, 1] <= 1e7)
            )
            valid_coords = geoms_coords_array[valid_coords_mask]

            kdtree = KDTree(valid_coords)

            distances, indices = kdtree.query(points_coords_array, k=len(box_geoms))
            min_distances = np.min(distances, axis=1)
            min_distances /= 1000.


            # Fill dataframe
            glathida.loc[glathida_id.index, 'dist_from_ocean'] = min_distances

    return glathida

"""Add RGIId and other OGGM stats like glacier area"""
def add_RGIId_and_OGGM_stats(glathida, path_OGGM_folder):
    # Note: points that are outside any glaciers will have nan to RGIId (and the other features)

    print("Adding OGGM's stats method and Hugonnet dmdtda")
    if (any(ele in list(glathida) for ele in ['RGIId', 'Area'])):
        print('Variables RGIId/Area etc already in dataframe.')
        return glathida

    glathida['RGIId'] = [np.nan] * len(glathida)
    glathida['RGIId'] = glathida['RGIId'].astype(object)
    glathida['Area'] = [np.nan] * len(glathida)
    glathida['Area_icefree'] = [np.nan] * len(glathida)
    glathida['Perimeter'] = [np.nan] * len(glathida)
    glathida['Zmin'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmax'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmed'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Slope'] = [np.nan] * len(glathida)
    glathida['Lmax'] = [np.nan] * len(glathida) # -9 missing values, see https://essd.copernicus.org/articles/14/3889/2022/essd-14-3889-2022.pdf
    glathida['Form'] = [np.nan] * len(glathida) # 9 Not assigned
    glathida['TermType'] = [np.nan] * len(glathida) # 9 Not assigned
    glathida['Aspect'] = [np.nan] * len(glathida) # -9 bad values
    glathida['dmdtda_hugo'] = [np.nan] * len(glathida)

    # Define this function for the parallelization
    def check_contains(point, geometry):
        return geometry.contains(point)

    # Setup Hugonnet product
    mbdf = utils.get_geodetic_mb_dataframe()
    mbdf = mbdf.loc[mbdf['period'] == '2000-01-01_2020-01-01']

    regions = list(range(1, 20))#[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]#[1,3,4,7,8,11,18]

    for rgi in regions:
        # get OGGM's dataframe of rgi glaciers
        oggm_rgi_shp = glob(f'{path_OGGM_folder}/rgi/RGIV62/{rgi:02d}*/{rgi:02d}*.shp')[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')

        # get Hugonnet dmdtda of rgi glaciers
        mbdf_rgi = mbdf.loc[mbdf['reg'] == rgi]

        # Glathida
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi)==0:
            print(f"rgi {rgi} there are no points. We go to next region")
            continue

        # loop sui ghiacciai di oggm
        for i, ind in tqdm(enumerate(oggm_rgi_glaciers.index), total=len(oggm_rgi_glaciers), desc=f"glaciers in rgi {rgi}", leave=True, position=0):

            # Oggm variables
            glacier_geometry = oggm_rgi_glaciers.at[ind, 'geometry']
            glacier_RGIId = oggm_rgi_glaciers.at[ind, 'RGIId']
            #glacier_area = oggm_rgi_glaciers.at[ind, 'Area']
            glacier_zmin = oggm_rgi_glaciers.at[ind, 'Zmin']
            glacier_zmax = oggm_rgi_glaciers.at[ind, 'Zmax']
            glacier_zmed = oggm_rgi_glaciers.at[ind, 'Zmed']
            glacier_slope = oggm_rgi_glaciers.at[ind, 'Slope']
            glacier_lmax = oggm_rgi_glaciers.at[ind, 'Lmax']
            glacier_form = oggm_rgi_glaciers.at[ind, 'Form']
            glacier_termtype = oggm_rgi_glaciers.at[ind, 'TermType']
            glacier_aspect = oggm_rgi_glaciers.at[ind, 'Aspect']
            glacier_cenLon = oggm_rgi_glaciers.at[ind, 'CenLon']
            glacier_cenLat = oggm_rgi_glaciers.at[ind, 'CenLat']

            # Calculate area and perimeter. Area in km2, perimeter in m.
            glacier_geometry_ext = Polygon(glacier_geometry.exterior)

            glacier_area, perimeter_ice = Geod(ellps="WGS84").geometry_area_perimeter(glacier_geometry)
            area_ice_and_noince, perimeter_ice_and_noice = Geod(ellps="WGS84").geometry_area_perimeter(glacier_geometry_ext)

            glacier_area = abs(glacier_area) * 1e-6  # km^2
            area_ice_and_noince = abs(area_ice_and_noince) * 1e-6  # km^2

            # Calculate area of nunataks in percentage to the total area
            area_noice = 1 - glacier_area / area_ice_and_noince
            #print(f'{glacier_area} {perimeter_ice} {area_noice}')

            # Hugonnet mass balance
            try:
                glacier_dmdtda = mbdf_rgi.at[glacier_RGIId, 'dmdtda']
            except: # impute the mean
                glacier_dmdtda = mbdf_rgi['dmdtda'].median()

            # calculate the area using pyproj and shapely for comparison with oggm
            # area_pyproj = abs(Geod(ellps="WGS84").geometry_area_perimeter(shapely.wkt.loads(str(glacier_geometry)))[0])*1.e-6 #km2

            llx, lly, urx, ury = glacier_geometry.bounds

            # dataframe of only points inside the raster bounds
            df_points_in_bound = glathida_rgi.loc[(glathida_rgi['POINT_LON'] >= llx) &
                                                  (glathida_rgi['POINT_LON'] <= urx) &
                                                  (glathida_rgi['POINT_LAT'] >= lly) &
                                                  (glathida_rgi['POINT_LAT'] <= ury)]

            # if no points inside the bound no reason to go further
            if (len(df_points_in_bound)==0): continue

            lats_in_bound = df_points_in_bound['POINT_LAT']
            lons_in_bound = df_points_in_bound['POINT_LON']
            points_in_bound = [Point(ilon, ilat) for (ilon, ilat) in zip(lons_in_bound, lats_in_bound)]
            #print(f'RGIId {glacier_RGIId} No. point in bound: {len(df_points_in_bound)}')

            # mask True/False to decide whether the points are inside the glacier geometry
            # NOTE: PARALLELIZATION HERE
            mask_points_in_glacier = Parallel(n_jobs=-1)(delayed(check_contains)(point, glacier_geometry) for point in points_in_bound)
            mask_points_in_glacier = np.array(mask_points_in_glacier)
            #mask_points_in_glacier_0 = glacier_geometry.contains(points_in_bound)
            #print(np.array_equal(mask_points_in_glacier_0, mask_points_in_glacier))

            # select only those points inside the glacier
            df_poins_in_glacier = df_points_in_bound[mask_points_in_glacier]
            # if no points inside the glacier move on
            if len(df_poins_in_glacier) == 0: continue

            #print(f'RGIId {glacier_RGIId} No. point in glacier: {len(df_poins_in_glacier)}')
            # at this point we should have found the points inside the glacier

            ifplot = False
            if ifplot:
                lats_in_glacier = df_poins_in_glacier['POINT_LAT']
                lons_in_glacier = df_poins_in_glacier['POINT_LON']
                fig, ax1 = plt.subplots()
                s1 = ax1.scatter(x=lons_in_bound, y=lats_in_bound, s=5, c='k')
                s2 = ax1.scatter(x=lons_in_glacier, y=lats_in_glacier, s=5, c='magenta', zorder=1)
                ax1.plot(*glacier_geometry.exterior.xy, c='magenta')
                plt.show()

            # some checks before returning values to dataframe
            assert glacier_zmin != -999, "Zmin should not be -999"
            assert glacier_zmax != -999, "Zmax should not be -999"
            #assert glacier_zmed != -999, "Zmed should not be -999"
            #assert glacier_lmax != -9, "Lmax should not be -9"
            assert glacier_form != 9, "Form should not be 9 (not assigned)"
            assert glacier_termtype != 9, "TermType should not be 9 (not assigned)"
            #assert glacier_aspect != -9, "Aspect should not be -9"

            # Data imputation for glacier_lmax (found needed 6 times in rgi 19)
            if glacier_lmax == -9:
                gl_geom_ext = Polygon(glacier_geometry.exterior)
                gl_geom_ext_gdf = gpd.GeoDataFrame(geometry=[gl_geom_ext], crs="EPSG:4326")
                _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(glacier_cenLat, glacier_cenLon)
                glacier_lmax = lmax_imputer(gl_geom_ext_gdf, glacier_epsg)
                #print(glacier_RGIId, glacier_cenLat, glacier_cenLon, glacier_lmax)

            # Data imputation for zmed (found needed for Antarctica, rgi 19)
            if glacier_zmed == -999:
                glacier_zmed = 0.5*(glacier_zmin+glacier_zmax)

            # Data imputation for aspect (found needed for Greenland)
            if glacier_aspect == -9: glacier_aspect = 0

            assert not np.any(np.isnan(np.array([glacier_area, glacier_zmin, glacier_zmax,
                                                     glacier_zmed, glacier_slope, glacier_lmax,
                                                 glacier_form, glacier_termtype, glacier_aspect]))), \
                "Found nan in some variables in method add_RGIId_and_OGGM_stats. Check why this value appeared."

            # Add to dataframe
            glathida.loc[df_poins_in_glacier.index, 'RGIId'] = glacier_RGIId
            glathida.loc[df_poins_in_glacier.index, 'Area'] = glacier_area
            glathida.loc[df_poins_in_glacier.index, 'Area_icefree'] = area_noice
            glathida.loc[df_poins_in_glacier.index, 'Perimeter'] = perimeter_ice
            glathida.loc[df_poins_in_glacier.index, 'Zmin'] = glacier_zmin
            glathida.loc[df_poins_in_glacier.index, 'Zmax'] = glacier_zmax
            glathida.loc[df_poins_in_glacier.index, 'Zmed'] = glacier_zmed
            glathida.loc[df_poins_in_glacier.index, 'Slope'] = glacier_slope
            glathida.loc[df_poins_in_glacier.index, 'Lmax'] = glacier_lmax
            glathida.loc[df_poins_in_glacier.index, 'Form'] = glacier_form
            glathida.loc[df_poins_in_glacier.index, 'TermType'] = glacier_termtype
            glathida.loc[df_poins_in_glacier.index, 'Aspect'] = glacier_aspect
            glathida.loc[df_poins_in_glacier.index, 'dmdtda_hugo'] = glacier_dmdtda

    return glathida

"""Add Farinotti's ith"""
def add_farinotti_ith(glathida, path_farinotti_icethickness):
    print("Adding Farinotti ice thickness...")

    if ('ith_f' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['ith_f'] = [np.nan] * len(glathida)

    regions = list(range(1, 20))

    for rgi in tqdm(regions, total=len(regions), leave=True):

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi)==0:
            continue

        # get dataframe of rgi glaciers from oggm
        oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{rgi:02d}*/{rgi:02d}*.shp")[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
        oggm_rgi_glaciers_geoms = oggm_rgi_glaciers['geometry']
        tqdm.write(f"rgi: {rgi}. Imported OGGMs {oggm_rgi_shp} dataframe of {len(oggm_rgi_glaciers_geoms)} glaciers")

        lats = np.array(glathida_rgi['POINT_LAT'])
        lons = np.array(glathida_rgi['POINT_LON'])
        lats_xar = xarray.DataArray(lats)
        lons_xar = xarray.DataArray(lons)

        tqdm.write(f'rgi: {rgi}. Glathida: {len(lats)} points')

        # Import farinotti ice thickness files
        files_names_farinotti = sorted(glob(f"{path_farinotti_icethickness}RGI60-{rgi:02d}/*"))
        list_glaciers_farinotti_4326 = []

        for n, tiffile in tqdm(enumerate(files_names_farinotti), total=len(files_names_farinotti),
                               desc=f"rgi {rgi}", leave=True, position=0):

            glacier_name = tiffile.split('/')[-1].replace('_thickness.tif', '')
            try:
                # Farinotti has solutions for glaciers that no longer exist in rgi/oggm (especially some in rgi 4)
                # See page 28 of https://www.glims.org/RGI/00_rgi60_TechnicalNote.pdf
                glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name, 'geometry'].item()
            except ValueError:
                tqdm.write(f"{glacier_name} not present in OGGM's RGI v62.")
                continue

            file_glacier_farinotti = rioxarray.open_rasterio(tiffile, masked=False)
            file_glacier_farinotti = file_glacier_farinotti.where(file_glacier_farinotti != 0.0) # set to nan outside glacier
            file_glacier_farinotti.rio.write_nodata(np.nan, inplace=True)

            file_glacier_farinotti_4326 = file_glacier_farinotti.rio.reproject("EPSG:4326", resampling=rasterio.enums.Resampling.bilinear)
            file_glacier_farinotti_4326.rio.write_nodata(np.nan, inplace=True)
            bounds_4326 = file_glacier_farinotti_4326.rio.bounds()

            # dataframe of only points inside the raster bounds
            df_points_in_bound = glathida_rgi.loc[(glathida_rgi['POINT_LON']>=bounds_4326[0]) &
                                        (glathida_rgi['POINT_LON']<=bounds_4326[2]) &
                                        (glathida_rgi['POINT_LAT']>=bounds_4326[1]) &
                                        (glathida_rgi['POINT_LAT']<=bounds_4326[3])]

            # if no points inside the bound no reason to go further
            if len(df_points_in_bound) == 0: continue

            # we want to mosaic only those raster that do contain glathida data inside the bound
            list_glaciers_farinotti_4326.append(file_glacier_farinotti_4326)

            lats_in_bound = df_points_in_bound['POINT_LAT']
            lons_in_bound = df_points_in_bound['POINT_LON']
            points_in_bound = [Point(ilon, ilat) for (ilon, ilat) in zip(lons_in_bound, lats_in_bound)]

            # mask True/False to decide whether the points are inside the glacier geometry
            # this command may be slow if many points in bound
            mask_points_in_glacier = glacier_geometry.contains(points_in_bound)

            # select only those points inside the glacier
            df_poins_in_glacier = df_points_in_bound[mask_points_in_glacier]
            lats_in_glacier = df_poins_in_glacier['POINT_LAT']
            lons_in_glacier = df_poins_in_glacier['POINT_LON']

            # if no points inside the glacier move on
            if len(df_poins_in_glacier) == 0: continue

            # if some points inside the glacier lets interpolate
            ithf_data_glacier = file_glacier_farinotti_4326.interp(y=xarray.DataArray(lats_in_glacier),
                                            x=xarray.DataArray(lons_in_glacier), method="nearest").data.squeeze()

            # plot
            ifplot = False
            p = random.randrange(0, 100, 1)
            if (ifplot and p > 0): ifplot = True
            if ifplot:
                fig, ax1 = plt.subplots()
                im1 = file_glacier_farinotti_4326.plot(ax=ax1, vmin=np.nanmin(ithf_data_glacier), vmax=np.nanmax(ithf_data_glacier), cmap='plasma')
                s1 = ax1.scatter(x=lons_in_bound, y=lats_in_bound, s=50, c='none', ec=(0, 1, 0, 1))
                s2 = ax1.scatter(x=lons_in_glacier, y=lats_in_glacier, s=50, c=ithf_data_glacier,
                                 vmin=np.nanmin(ithf_data_glacier), vmax=np.nanmax(ithf_data_glacier),
                                 ec='magenta', cmap='plasma', zorder=1)

                for n, gl in enumerate(oggm_rgi_glaciers_geoms):
                    ax1.plot(*gl.exterior.xy, c='g')
                ax1.plot(*glacier_geometry.exterior.xy, c='magenta')
                plt.show()


            # add to dataframe
            glathida.loc[df_poins_in_glacier.index, 'ith_f'] = ithf_data_glacier


        # mosaic, interpolate the mosaic and add data to dataframe
        # mosaic_ithf_4326 = merge.merge_arrays(list_glaciers_farinotti_4326, method='max')
        #ithf_data_reproj = mosaic_ithf_4326.interp(y=lats_xar, x=lons_xar, method="linear").data.squeeze()
        #glathida.loc[glathida_rgi.index, 'ith_f2'] = ithf_data_reproj

        # plot mosaic and glaciers
        plot_mosaic = False
        if plot_mosaic:
            mosaic_ithf_4326 = merge.merge_arrays(list_glaciers_farinotti_4326, method='max')
            fig, ax1 = plt.subplots()
            im1 = mosaic_ithf_4326.plot(ax=ax1, cmap='plasma')
            for n, gl in enumerate(oggm_rgi_glaciers_geoms):
                ax1.plot(*gl.exterior.xy, c='g')
            plt.show()

        # compare interpolating the mosaic vs interpolating the single rasters
        # for i in range(len(glathida_rgi)):
        #    print(glathida['ith_f'][i], '\t', glathida['ith_f2'][i])
        #print(glathida['ith_f'].describe())
        #print(glathida['ith_f2'].describe())
        #fig, ax1 = plt.subplots()
        #h1 = ax1.hist(glathida['ith_f'], bins=np.arange(0, 1300, 2), color='r', alpha=.3)
        #h2 = ax1.hist(glathida['ith_f2'], bins=np.arange(0, 1300, 2), color='b', alpha=.3)
        #plt.show()

    return glathida

"""Add T2m"""
def add_t2m(glathida, path_ERA5_t2m_folder):
    if ('t2m' in list(glathida)):
        print('Variable t2m already in dataframe.')
        return glathida

    glathida['t2m'] = [np.nan] * len(glathida)

    tile_era5_t2m = rioxarray.open_rasterio(f"{path_ERA5_t2m_folder}era5land_era5.nc", masked=False)
    tile_era5_t2m = tile_era5_t2m.squeeze()
    #fig, ax = plt.subplots()
    #im = tile_era5_t2m.plot(ax=ax, cmap='jet')
    #plt.show()

    regions = list(range(1, 20))
    for rgi in tqdm(regions, total=len(regions), leave=True):

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        if len(glathida_rgi) == 0:
            continue

        lats = glathida_rgi['POINT_LAT']
        lons = glathida_rgi['POINT_LON']

        lats_ar = xarray.DataArray(lats)
        lons_ar = xarray.DataArray(lons)

        # Interpolate (on a lat-lon grid.. ok no perfect but whatever)
        t2m_data = tile_era5_t2m.interp(y=lats_ar, x=lons_ar, method='linear').data

        plot_rgi_t2m_interp = False
        if plot_rgi_t2m_interp:
            fig, ax = plt.subplots()
            tile_era5_t2m_for_plot = tile_era5_t2m.rio.clip_box(minx=min(lons) - 2, miny=min(lats) - 2,
                                                                maxx=max(lons) + 2,
                                                                maxy=max(lats) + 2)
            im = tile_era5_t2m_for_plot.plot(ax=ax, cmap='jet', vmin=t2m_data.min(), vmax=t2m_data.max())
            #im = tile_era5_t2m.plot(ax=ax, cmap='jet', vmin=t2m_data.min(), vmax=t2m_data.max())
            s = ax.scatter(x=lons, y=lats, s=50, ec='k', c=t2m_data, cmap='jet', vmin=t2m_data.min(), vmax=t2m_data.max())
            plt.show()

        # Fill dataframe
        glathida.loc[glathida_rgi.index, 't2m'] = t2m_data


    return glathida



if __name__ == '__main__':

    print(f'Begin Metadata dataset creation !')
    t0 = time.time()

    #glathida = pd.read_csv(args.path_ttt_csv, low_memory=False)
    #glathida = add_rgi(glathida, args.path_O1Regions_shp)
    #glathida = add_RGIId_and_OGGM_stats(glathida, args.OGGM_folder)
    #glathida.to_csv("/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/metadata_rgi_oggm.csv", index=False)
    #glathida = add_slopes_elevation(glathida, args.mosaic)
    #glathida = add_smb(glathida, args.RACMO_folder)
    #glathida = add_millan_vx_vy_ith(glathida, args.millan_velocity_folder, args.millan_icethickness_folder)
    #glathida = add_dist_from_boder_using_geometries(glathida)
    #glathida = add_dist_from_water(glathida, args.GSHHG_folder)
    #glathida = add_farinotti_ith(glathida, args.farinotti_icethickness_folder)
    #glathida = add_t2m(glathida, args.path_ERA5_t2m_folder)

    glathida = pd.read_csv(args.path_ttt_rgi_csv.replace('TTT_rgi.csv', 'metadata34.csv'), low_memory=False)
    glathida = add_dist_from_water(glathida, args.GSHHG_folder)
    #glathida = add_smb(glathida, args.RACMO_folder)
    #glathida = add_farinotti_ith(glathida, args.farinotti_icethickness_folder)
    #glathida = add_RGIId_and_OGGM_stats(glathida, args.OGGM_folder)
    #glathida = add_dist_from_boder_using_geometries(glathida)
    #glathida = add_slopes_elevation(glathida, args.mosaic)
    #glathida = add_millan_vx_vy_ith(glathida, args.millan_velocity_folder, args.millan_icethickness_folder)
    #glathida = add_t2m(glathida, args.path_ERA5_t2m_folder)

    if args.save:
        glathida.to_csv(f'{args.save_outname}.csv', index=False)
        print(f"Metadata dataset saved: {args.save_outname}")
    print(f'Finished in {(time.time()-t0)/60} minutes. Bye bye.')
