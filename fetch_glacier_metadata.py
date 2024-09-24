import os, sys, time
from glob import glob
import argparse
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
import pandas as pd
import scipy
from scipy.interpolate import griddata
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import sklearn.neighbors
import pykdtree.kdtree
from sklearn.impute import KNNImputer, SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
from rioxarray import merge
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon, LineString, MultiLineString, box
from shapely.errors import GEOSException
from pyproj import Proj, Transformer, Geod
import utm
from joblib import Parallel, delayed
from functools import partial

from create_rgi_mosaic_tanxedem import fetch_dem, create_glacier_tile_dem_mosaic
from utils_metadata import *
from imputation_policies import smb_elev_functs, smb_elev_functs_hugo, velocity_median_rgi

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
"""
This program generates glacier metadata at some random locations inside the glacier geometry. 

Input: glacier name (RGIId), how many points you want to generate. 
Output: pandas dataframe with features calculated for each generated point. 

Note: the points are generated inside the glacier but outside nunataks (there is a check for this)

Note: Millan and Farinotti products needs to be interpolated. Interpolation close to the borders may result in nans. 
The interpolation method="nearest" yields much less nans close to borders if compared to linear
interpolation and therefore is preferred. 

"""

def populate_glacier_with_metadata(glacier_name,
                                   config = None,
                                   rgi_products=None,
                                   coastlines_dataframe = None,
                                   seed=None,
                                   verbose=True):


    print(f"******* FETCHING FEATURES FOR GLACIER {glacier_name} *******") if verbose else None
    tin=time.time()

    # unpack config
    n_points_regression = config.n_points_regression
    k_max_geoms = config.kdtree_dist_max_k_geometries
    graph_max_layer_depth = config.graph_max_layer_depth

    rgi = int(glacier_name[6:8]) # get rgi from the glacier code

    # Get rgi products
    oggm_rgi_glaciers, oggm_rgi_intersects, rgi_graph, mbdf_rgi = rgi_products

    try:
        # Get glacier dataset
        gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name]
        gl_geom = gl_df['geometry'].item()  # glacier geometry Polygon
        gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
        gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors]  # list of nunataks Polygons
        assert len(gl_df) == 1, "Check this please."
        # print(gl_df.T)
    except Exception as e:
        print(f"Error. {glacier_name} not present in OGGM's RGI v62.")
        return None

    # center of glacier and glacier epsg
    glacier_centroid = gl_geom_ext.centroid
    cenLon, cenLat = glacier_centroid.x, glacier_centroid.y
    _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)
    print(f"Glacier {glacier_name} found. Lat: {cenLat}, Lon: {cenLon}") if verbose else None

    llx, lly, urx, ury = gl_geom.bounds # geometry bounds

    # Geodataframes of external boundary and all internal nunataks
    gl_geom_nunataks_gdf = gpd.GeoDataFrame(geometry=gl_geom_nunataks_list, crs="EPSG:4326")
    gl_geom_ext_gdf = gpd.GeoDataFrame(geometry=[gl_geom_ext], crs="EPSG:4326")

    tgeometries = time.time() - tin

    # Generate points (no points can be generated inside nunataks)
    tp0 = time.time()
    points = {'lons': [], 'lats': [], 'nunataks': []}
    if seed is not None: np.random.seed(seed)

    while (len(points['lons']) < n_points_regression):
        batch_size = min(n_points_regression, n_points_regression - len(points['lons']))  # Adjust batch size as needed
        r_lons = np.random.uniform(llx, urx, batch_size)
        r_lats = np.random.uniform(lly, ury, batch_size)
        #points_batch = list(map(Point, r_lons, r_lats))
        #points_batch_gdf = gpd.GeoDataFrame(geometry=points_batch, crs="EPSG:4326")
        points_batch_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(r_lons, r_lats), crs="EPSG:4326")

        # A bit faster
        # 1) Select only those points generated inside the external polygon
        #points_in_glacier_gdf = gpd.sjoin(points_batch_gdf, gl_geom_ext_gdf, how="inner", predicate="within").drop(columns=['index_right'])
        # 2) Exclude points that are inside any internal polygons
        #points_in_internal_polygons_gdf = gpd.sjoin(points_in_glacier_gdf, gl_geom_nunataks_gdf, how="left", predicate="within")
        #points_not_in_nunataks_gdf = points_in_internal_polygons_gdf[points_in_internal_polygons_gdf.index_right.isna()]

        # A bit slower
        # 1) First we select only those points generated inside the glacier
        points_yes_no_ext_gdf = gpd.sjoin(points_batch_gdf, gl_geom_ext_gdf, how="left", predicate="within")
        points_in_glacier_gdf = points_yes_no_ext_gdf[~points_yes_no_ext_gdf.index_right.isna()].drop(columns=['index_right'])
        indexes_of_points_inside = points_in_glacier_gdf.index
        # 2) Then we get rid of all those generated inside nunataks
        points_yes_no_nunataks_gdf = gpd.sjoin(points_batch_gdf.loc[indexes_of_points_inside], gl_geom_nunataks_gdf, how="left", predicate="within")
        points_not_in_nunataks_gdf = points_yes_no_nunataks_gdf[points_yes_no_nunataks_gdf.index_right.isna()].drop(columns=['index_right'])

        points['lons'].extend(points_not_in_nunataks_gdf['geometry'].x.tolist())
        points['lats'].extend(points_not_in_nunataks_gdf['geometry'].y.tolist())
        points['nunataks'].extend([0.0]*len(points_not_in_nunataks_gdf))


        plot_gen_points = False
        if plot_gen_points:
            points_in_nunataks_gdf = points_yes_no_nunataks_gdf[~points_yes_no_nunataks_gdf.index_right.isna()].drop(
                columns=['index_right'])
            fig, ax = plt.subplots()
            ax.plot(*gl_geom.exterior.xy, color='blue')
            gl_geom_nunataks_gdf.plot(ax=ax, color='orange', alpha=0.5)
            #points_in_glacier_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
            points_not_in_nunataks_gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=1, zorder=2)
            points_in_nunataks_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
            plt.show()

    print(f"We have generated {len(points['lats'])} pointsin {time.time()-tp0:.3f}") if verbose else None
    # Feature dataframe
    points_df = pd.DataFrame(columns=['lons', 'lats', 'nunataks'])
    # Fill lats, lons and nunataks
    points_df['lats'] = points['lats']
    points_df['lons'] = points['lons']
    points_df['nunataks'] = points['nunataks']
    if (points_df['nunataks'].sum() != 0):
        print(f"The generation pipeline has produced n. {points_df['nunataks'].sum()} points inside nunataks")
        raise ValueError

    tp1 = time.time()
    tgenpoints = tp1-tp0

    # Fill these features
    points_df['RGI'] = rgi
    #points_df['Area'] = gl_df['Area'].item()
    #points_df['Zmin'] = gl_df['Zmin'].item() # we use tandemx for this
    #points_df['Zmax'] = gl_df['Zmax'].item() # we use tandemx for this
    #points_df['Zmed'] = gl_df['Zmed'].item() # we use tandemx for this
    points_df['Slope'] = gl_df['Slope'].item()
    points_df['Lmax'] = gl_df['Lmax'].item() #todo: to be replaced with lmax_with_covex_hull !
    #points_df['Form'] = gl_df['Form'].item() # not used anymore
    points_df['TermType'] = gl_df['TermType'].item()
    points_df['Aspect'] = gl_df['Aspect'].item()

    # Area in km2, perimeter in m
    # Note that the area in OGGM equals to area_ice.
    glacier_area, perimeter_ice = Geod(ellps="WGS84").geometry_area_perimeter(gl_geom)
    area_ice_and_noince, perimeter_ice_and_noice = Geod(ellps="WGS84").geometry_area_perimeter(gl_geom_ext)

    glacier_area = abs(glacier_area) * 1e-6                 # km^2
    area_ice_and_noince = abs(area_ice_and_noince) * 1e-6   # km^2

    # Calculate area of nunataks in percentage to the total area
    area_noice = 1 - glacier_area / area_ice_and_noince

    points_df['Area'] = glacier_area            # km^2
    points_df['Perimeter'] = perimeter_ice      # m
    points_df['Area_icefree'] = area_noice      # unitless

    # Data imputation for lmax (needed for sure for rgi 1 mostly)
    if gl_df['Lmax'].item() == -9:
        #lmax = lmax_imputer(gl_geom_ext_gdf, glacier_epsg) # slower
        lmax = lmax_with_covex_hull(gl_geom_ext_gdf, glacier_epsg) # faster
        points_df['Lmax'] = lmax

    # Data imputation for the aspect (found needed for Greenland)
    if gl_df['Aspect'].item() == -9: points_df['Aspect'] = 0

    # Calculate the adaptive filter size based on the Area value
    sigma_af_min, sigma_af_max = 100.0, 2000.0
    try:
        area_gl = points_df['Area'][0]
        lmax_gl = points_df['Lmax'][0]
        a = 1e6 * area_gl / (np.pi * 0.5 * lmax_gl)
        sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
    except Exception as e:
        sigma_af = sigma_af_min
    # Ensure that our value correctly in range
    assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"

    """ Calculate Millan vx, vy, v """
    print(f"Calculating vx, vy, v, ith_m...") if verbose else None
    tmillan1 = time.time()

    cols_millan = ['ith_m', 'v50', 'v100', 'v150', 'v300', 'v450', 'vgfa']

    for col in cols_millan: points_df[col] = np.nan

    def fetch_millan_data_An(points_df):

        files_vx = sorted(glob(f"{config.millan_velocity_dir}RGI-19/VX_RGI-19*"))
        files_ith = sorted(glob(f"{config.millan_icethickness_dir}RGI-19/THICKNESS_RGI-19*"))
        print(f"Glacier {glacier_name} found. Lat: {cenLat}, Lon: {cenLon}") if verbose else None

        # Check if glacier is inside the 5 Millan ith tiles
        # This loop is bulletproof except for RGI60-19.00889 found inside the tile but probably will be interpolated as nan
        inside_millan = False
        for i, file_ith in enumerate(files_ith):
            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)
            left, bottom, right, top = tile_ith.rio.bounds()
            e, n = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(cenLat, cenLon)
            if left < e < right and bottom < n < top:
                inside_millan = True
                print(f"Found glacier in Millan tile: {file_ith}")
                break

        if inside_millan:
            print("Interpolating Millan Antarctica") if verbose else None

            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

            eastings, northings = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(points_df['lats'],
                                                                                                points_df['lons'])

            cond0 = np.all(tile_ith.values == 0)
            condnodata = np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6)
            condnan = np.all(np.isnan(tile_ith.values))
            all_zero_or_nodata = cond0 or condnodata or condnan
            print(f"Cond1: {all_zero_or_nodata}") if verbose else None

            eastings_ar = xarray.DataArray(eastings)
            northings_ar = xarray.DataArray(northings)

            vals_fast_interp = tile_ith.interp(y=northings_ar, x=eastings_ar, method='nearest').data

            cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                      np.all(np.abs(vals_fast_interp - tile_ith.rio.nodata) < 1.e-6))
            print(f"Cond2: {cond_valid_fast_interp}") if verbose else None

            if all_zero_or_nodata==False and cond_valid_fast_interp==False:

                # If we reached this point we should have the valid tile to interpolate
                tile_ith.values = np.where((tile_ith.values == tile_ith.rio.nodata) | np.isinf(tile_ith.values),
                                           np.nan, tile_ith.values)

                tile_ith.rio.write_nodata(np.nan, inplace=True)

                tile_ith = tile_ith.squeeze()

                # Interpolate
                ith_data = tile_ith.interp(y=northings_ar, x=eastings_ar, method="nearest").data

                # Fill dataframe with Millan ith
                points_df['ith_m'] = ith_data

                print("Millan ith interpolated.") if verbose else None
                #fig, ax = plt.subplots()
                #tile_ith.plot(ax=ax, cmap='viridis')
                #ax.scatter(x=eastings, y=northings, s=10, c=ith_data)
                #plt.show()
            else:
                print('No Millan ith interpolation possible.') if verbose else None

            """Now interpolate Millan velocity"""
            # In rgi 19 there is no need to make all the group occupancy stuff since tiles do not overlap.

            # Fetch corresponding vx, vy files
            file_ith_nopath = file_ith.rsplit('/', 1)[1]
            file_ith_nopath_nodate = file_ith_nopath.rsplit('_', 1)[0]
            code_19_dot_x = file_ith_nopath_nodate.rsplit('-', 1)[1]
            #print(file_ith_nopath, file_ith_nopath_nodate, code_19_dot_x)

            file_vx = glob(f"{config.millan_velocity_dir}RGI-19/VX_RGI-{code_19_dot_x}*")
            file_vy = glob(f"{config.millan_velocity_dir}RGI-19/VY_RGI-{code_19_dot_x}*")

            if file_vx and file_vy:
                print('Found Millan velocity tiles.') if verbose else None
                file_vx, file_vy = file_vx[0], file_vy[0]
                print(file_vx) if verbose else None
                print(file_vy) if verbose else None

                tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
                tile_vy = rioxarray.open_rasterio(file_vy, masked=False)

                if not tile_vx.rio.bounds() == tile_vy.rio.bounds():
                    input('wait - reindex necessary')

                assert tile_vx.rio.crs == tile_vy.rio.crs, 'Different crs found.'
                assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), 'Different bounds found.'
                assert tile_vx.rio.resolution() == tile_vy.rio.resolution(), "Different resolutions found."

                ris_metre_millan = tile_vx.rio.resolution()[0]  # 50m

                minE, maxE = min(eastings), max(eastings)
                minN, maxN = min(northings), max(northings)

                epsM = 500
                tile_vx = tile_vx.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
                tile_vy = tile_vy.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)

                tile_vx.values = np.where((tile_vx.values == tile_vx.rio.nodata) | np.isinf(tile_vx.values),
                                          np.nan, tile_vx.values)
                tile_vy.values = np.where((tile_vy.values == tile_vy.rio.nodata) | np.isinf(tile_vy.values),
                                          np.nan, tile_vy.values)

                tile_vx.rio.write_nodata(np.nan, inplace=True)
                tile_vy.rio.write_nodata(np.nan, inplace=True)

                num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
                num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
                num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
                num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6
                num_px_sigma_450 = max(1, round(450 / ris_metre_millan))  # 9
                num_px_sigma_af = max(1, round(sigma_af / ris_metre_millan))

                kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
                kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
                kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
                kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
                kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
                kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

                tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)
                tile_v = tile_v.squeeze()

                #fig, ax = plt.subplots()
                #tile_v.plot(ax=ax)
                #plt.show()

                # A check to see if velocity modules is as expected
                assert float(tile_v.sum()) > 0, "tile v is not as expected."

                """astropy"""
                preserve_nans = False
                try:
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

                    focus_ith = tile_ith.squeeze()
                except Exception as generic_error:
                    print("Impossible to smooth Millan tile with astropy.") if verbose else None
                    return points_df


                # create xarrays of filtered velocities
                focus_filter_v50_ar = tile_v.copy(deep=True, data=focus_filter_v50)
                focus_filter_v100_ar = tile_v.copy(deep=True, data=focus_filter_v100)
                focus_filter_v150_ar = tile_v.copy(deep=True, data=focus_filter_v150)
                focus_filter_v300_ar = tile_v.copy(deep=True, data=focus_filter_v300)
                focus_filter_v450_ar = tile_v.copy(deep=True, data=focus_filter_v450)
                focus_filter_vfa_ar = tile_v.copy(deep=True, data=focus_filter_af)

                #fig, ax = plt.subplots()
                #focus_filter_v50_ar.plot(ax=ax)
                #plt.show()

                # Interpolate
                v_data = tile_v.interp(y=northings_ar, x=eastings_ar, method="nearest").data
                v_filter_50_data = focus_filter_v50_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_100_data = focus_filter_v100_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_150_data = focus_filter_v150_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_300_data = focus_filter_v300_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_450_data = focus_filter_v450_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data

                # Fill dataframe
                points_df['v50'] = v_filter_50_data
                points_df['v100'] = v_filter_100_data
                points_df['v150'] = v_filter_150_data
                points_df['v300'] = v_filter_300_data
                points_df['v450'] = v_filter_450_data
                points_df['vgfa'] = v_filter_af_data

            else:
                print('No Millan velocity tiles found. Likely 19.5 tile') if verbose else None

            if verbose:
                print(f"From Millan vx, vy, ith interpolations we have generated no. nans:")
                print(", ".join([f"{col}: {points_df[col].isna().sum()}" for col in cols_millan]))
            if points_df['ith_m'].isna().all():
                print(f"No Millan ith data can be found for rgi {rgi} glacier {glacier_name} at {cenLat} lat {cenLon} lon.") if verbose else None

            return points_df

        else:
            print("Interpolating NSIDC velocity and BedMachine Antarctica") if verbose else None

            file_ith_NSIDC = f"{config.NSIDC_icethickness_Antarctica_dir}BedMachineAntarctica-v3.nc"
            file_vel_NSIDC = f"{config.NSIDC_velocity_Antarctica_dir}antarctic_ice_vel_phase_map_v01.nc"

            v_NSIDC = rioxarray.open_rasterio(file_vel_NSIDC, masked=False)
            vx_NSIDC = v_NSIDC.VX
            vy_NSIDC = v_NSIDC.VY

            ith_NSIDC = rioxarray.open_rasterio(file_ith_NSIDC, masked=False)
            ith_NSIDC = ith_NSIDC.thickness

            assert vx_NSIDC.rio.crs == vy_NSIDC.rio.crs == ith_NSIDC.rio.crs, 'Different crs found in Antarctica NSIDC products.'
            #print(ith_NSIDC.rio.crs, ith_NSIDC.rio.nodata, ith_NSIDC.rio.bounds(), ith_NSIDC.rio.resolution())
            #print(vx.rio.crs, vx.rio.nodata, vx.rio.bounds(), vx.rio.resolution())

            eastings, northings = Transformer.from_crs("EPSG:4326", vx_NSIDC.rio.crs).transform(points_df['lats'],
                                                                                                points_df['lons'])
            eastings_ar = xarray.DataArray(eastings)
            northings_ar = xarray.DataArray(northings)

            minE, maxE = min(eastings), max(eastings)
            minN, maxN = min(northings), max(northings)

            epsM = 15000
            try:
                ith_NSIDC = ith_NSIDC.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
            except:
                print('No NSIDC BedMachine ice thickness tiles around the points found.') if verbose else None

            #fig, ax = plt.subplots()
            #ith_NSIDC.plot(ax=ax, cmap='viridis')
            #ax.scatter(x=eastings, y=northings, s=5, c='r')
            #plt.show()

            cond0 = np.all(ith_NSIDC.values == 0)
            condnodata = np.all(np.abs(ith_NSIDC.values - ith_NSIDC.rio.nodata) < 1.e-6)
            condnan = np.all(np.isnan(ith_NSIDC.values))
            all_zero_or_nodata = cond0 or condnodata or condnan
            print(f"Cond1 ice thickness: {all_zero_or_nodata}") if verbose else None

            vals_fast_interp = ith_NSIDC.interp(y=northings_ar, x=eastings_ar, method='nearest').data

            cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                      np.all(np.abs(vals_fast_interp - ith_NSIDC.rio.nodata) < 1.e-6))
            print(f"Cond2 ice thickness: {cond_valid_fast_interp}") if verbose else None

            if all_zero_or_nodata==False and cond_valid_fast_interp==False:

                # If we reached this point we should have the valid tile to interpolate
                ith_NSIDC.values = np.where((ith_NSIDC.values == ith_NSIDC.rio.nodata) | np.isinf(ith_NSIDC.values),
                                           np.nan, ith_NSIDC.values)
                ith_NSIDC.values[ith_NSIDC.values == 0.0] = np.nan

                ith_NSIDC.rio.write_nodata(np.nan, inplace=True)

                ith_NSIDC = ith_NSIDC.squeeze()

                # Interpolate
                ith_data = ith_NSIDC.interp(y=northings_ar, x=eastings_ar, method="nearest").data

                # Fill dataframe with NSIDC BedMachine ith
                points_df['ith_m'] = ith_data
                print("NSIDC BedMachine ith interpolated.") if verbose else None

                #fig, ax = plt.subplots()
                #ith_NSIDC.plot(ax=ax, cmap='viridis')
                #ax.scatter(x=eastings, y=northings, s=5, c='r')
                #plt.show()

            else:
                print('No NSIDC BedMachine ice thickness interpolation possible.') if verbose else None

            """Now interpolate NSIDC velocity"""
            eps = 15000
            try:
                vx_NSIDC_focus = vx_NSIDC.rio.clip_box(minx=minE - eps, miny=minN - eps, maxx=maxE + eps, maxy=maxN + eps)
                vy_NSIDC_focus = vy_NSIDC.rio.clip_box(minx=minE - eps, miny=minN - eps, maxx=maxE + eps, maxy=maxN + eps)
            except:
                print('No NSIDC velocity tiles around the points found') if verbose else None
                return points_df

            # Condition 1. Either v is .rio.nodata or it is zero or it is nan
            cond0 = np.all(vx_NSIDC_focus.values == 0)
            condnodata = np.all(np.abs(vx_NSIDC_focus.values - vx_NSIDC_focus.rio.nodata) < 1.e-6)
            condnan = np.all(np.isnan(vx_NSIDC_focus.values))
            all_zero_or_nodata = cond0 or condnodata or condnan
            print(f"Cond1 velocity: {all_zero_or_nodata}") if verbose else None

            vals_fast_interp = vx_NSIDC_focus.interp(y=northings_ar, x=eastings_ar, method='nearest').data

            cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                      np.all(np.abs(vals_fast_interp - vx_NSIDC_focus.rio.nodata) < 1.e-6))

            print(f"Cond2 velocity: {cond_valid_fast_interp}") if verbose else None

            if all_zero_or_nodata == False and cond_valid_fast_interp == False:

                vx_NSIDC_focus.values = np.where(
                    (vx_NSIDC_focus.values == vx_NSIDC_focus.rio.nodata) | np.isinf(vx_NSIDC_focus.values),
                    np.nan, vx_NSIDC_focus.values)
                vy_NSIDC_focus.values = np.where(
                    (vy_NSIDC_focus.values == vy_NSIDC_focus.rio.nodata) | np.isinf(vy_NSIDC_focus.values),
                    np.nan, vy_NSIDC_focus.values)
                vx_NSIDC_focus.rio.write_nodata(np.nan, inplace=True)
                vy_NSIDC_focus.rio.write_nodata(np.nan, inplace=True)

                assert vx_NSIDC_focus.rio.bounds() == vy_NSIDC_focus.rio.bounds(), "NSIDC vx, vy bounds not the same"

                # Note: for rgi 19 we do not interpolate NSIDC to remove nans.
                tile_vx = vx_NSIDC_focus.squeeze()
                tile_vy = vy_NSIDC_focus.squeeze()

                ris_metre_nsidc = vx_NSIDC.rio.resolution()[0]  # 450m

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
                tile_v = tile_v.squeeze()

                # A check to see if velocity modules is as expected
                assert float(tile_v.sum()) > 0, "tile v is not as expected."

                """astropy"""
                preserve_nans = False
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
                v_data = tile_v.interp(y=northings_ar, x=eastings_ar, method="nearest").data
                v_filter_50_data = focus_filter_v50_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_100_data = focus_filter_v100_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_150_data = focus_filter_v150_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_300_data = focus_filter_v300_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_450_data = focus_filter_v450_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data
                v_filter_af_data = focus_filter_vfa_ar.interp(y=northings_ar, x=eastings_ar, method='nearest').data


                # some checks
                assert v_data.shape == v_filter_50_data.shape, "NSIDC interp something wrong!"
                assert v_filter_50_data.shape == v_filter_100_data.shape, "NSIDC interp something wrong!"
                assert v_filter_100_data.shape == v_filter_150_data.shape, "NSIDC interp something wrong!"
                assert v_filter_150_data.shape == v_filter_300_data.shape, "NSIDC interp something wrong!"
                assert v_filter_300_data.shape == v_filter_450_data.shape, "NSIDC interp something wrong!"
                assert v_filter_450_data.shape == v_filter_af_data.shape, "NSIDC interp something wrong!"

                # Fill dataframe with Millan velocities
                points_df['v50'] = v_filter_50_data
                points_df['v100'] = v_filter_100_data
                points_df['v150'] = v_filter_150_data
                points_df['v300'] = v_filter_300_data
                points_df['v450'] = v_filter_450_data
                points_df['vgfa'] = v_filter_af_data


                print("NSIDC velocity interpolated.") if verbose else None

                #fig, ax = plt.subplots()
                #im = tile_vx.plot(ax=ax, cmap='binary', zorder=0, vmin=tile_vx.min(), vmax=tile_vx.max())
                #ax.scatter(x=eastings, y=northings, s=10, c=vx_filter_50_data, zorder=1, vmin=tile_vx.min(), vmax=tile_vx.max(), cmap='binary')
                #plt.show()

            else:
                print('No NSIDC velocity interpolation possible.') if verbose else None

            return points_df

    def fetch_millan_data_Gr(points_df):
        # Note: Millan has no velocity. Velocity needs to be extracted from NSICD.
        # Millan has only ith for ice caps. I can decide to use Millan ith or BedMachinev5 (has all Millan data inside)
        # The fact is that BedMachine appears to downgrade the resolution of Millan (see e.g. RGI60-05.15702).
        # Therefore I use Millan and, if not enough data at interpolation stage, rollback to BedMachine

        file_vx = f"{config.NSIDC_velocity_Greenland_dir}greenland_vel_mosaic250_vx_v1.tif"
        file_vy = f"{config.NSIDC_velocity_Greenland_dir}greenland_vel_mosaic250_vy_v1.tif"
        files_ith = sorted(glob(f"{config.millan_icethickness_dir}RGI-5/THICKNESS_RGI-5*"))
        file_ith_bedmacv5 = f"{config.NSIDC_icethickness_Greenland_dir}BedMachineGreenland-v5.nc"

        # Interpolate Millan
        # I need a dataframe for Millan with same indexes and lats lons
        df_pointsM = points_df[['lats', 'lons']].copy()
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_ith})

        # Fill the dataframe for occupancy
        tocc0 = time.time()
        for i, file_ith in enumerate(files_ith):

            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

            eastings, northings = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(df_pointsM['lats'],
                                                                                               df_pointsM['lons'])
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
        print(f"Created dataframe of occupancies for all points in {time.time() - tocc0} s.") if verbose else None

        # Grouping by ith occupancy. Each group will have an occupancy value
        df_pointsM['ntiles_ith'] = df_pointsM.iloc[:, 2:].sum(axis=1)
        print(df_pointsM['ntiles_ith'].value_counts()) if verbose else None
        groups = df_pointsM.groupby('ntiles_ith')  # Groups.
        df_pointsM.drop(columns=['ntiles_ith'], inplace=True)  # Remove this column that we used to create groups
        print(f"Num groups in Millan: {groups.ngroups}") if verbose else None

        for g_value, df_group in groups:

            unique_ith_tiles = df_group.iloc[:, 2:].columns[df_group.iloc[:, 2:].sum() != 0].tolist()

            group_lats, group_lons = df_group['lats'].values, df_group['lons'].values

            for file_ith in unique_ith_tiles:

                tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

                group_eastings, group_northings = (Transformer.from_crs("EPSG:4326", "EPSG:3413")
                                                   .transform(group_lats,group_lons))

                minE, maxE = min(group_eastings), max(group_eastings)
                minN, maxN = min(group_northings), max(group_northings)

                epsM = 500
                tile_ith = tile_ith.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)

                # Condition no. 1. Check if ith tile is only either nodata or zero
                # This condition is so soft. Glaciers may be still be present in the box. We need condition no. 2 as well
                #tile_ith_is_all_zero_or_nodata = np.all(
                #    np.logical_or(tile_ith.values == 0, tile_ith.values == tile_ith.rio.nodata))
                cond0 = np.all(tile_ith.values == 0)
                condnodata = np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6)
                condnan = np.all(np.isnan(tile_ith.values))
                all_zero_or_nodata = cond0 or condnodata or condnan
                #print(f"Cond1: {all_zero_or_nodata}")

                if all_zero_or_nodata:
                    continue

                # Condition no. 2. A fast and quick interpolation to see if points intercepts a valid raster region
                group_eastings_ar = xarray.DataArray(group_eastings)
                group_northings_ar = xarray.DataArray(group_northings)

                vals_fast_interp = tile_ith.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data

                cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                          np.all(np.abs(vals_fast_interp - tile_ith.rio.nodata) < 1.e-6))
                #print(f"Cond2: {cond_valid_fast_interp}")

                if cond_valid_fast_interp:
                    continue

                # If we reached this point we should have the valid tile to interpolate
                tile_ith.values = np.where((tile_ith.values == tile_ith.rio.nodata) | np.isinf(tile_ith.values),
                                           np.nan, tile_ith.values)

                tile_ith.rio.write_nodata(np.nan, inplace=True)

                # Note: for rgi 5 we do not interpolate to remove nans.
                tile_ith = tile_ith.squeeze()

                # Interpolate (note: nans can be produced near boundaries). This should be removed at the end.
                ith_data = tile_ith.interp(y=group_northings_ar, x=group_eastings_ar, method="nearest").data

                #fig, ax = plt.subplots()
                #tile_ith.plot(ax=ax, vmin=tile_ith.min(), vmax=tile_ith.max())
                #s = ax.scatter(x=group_eastings, y=group_northings, c=ith_data, ec=None, s=3, vmin=tile_ith.min(),
                #           vmax=tile_ith.max())
                #cbar = plt.colorbar(s)
                #plt.show()

                #fig, ax = plt.subplots()
                #s = ax.scatter(x=points_df['lons'], y=points_df['lats'], c=ith_data, s=3)
                #plt.colorbar(s)
                #plt.show()

                # Fill dataframe with ith_m
                #points_df.loc[df_group.index, 'ith_m'] = ith_data
                mask_valid_ith_m = ~np.isnan(ith_data)
                points_df.loc[df_group.index[mask_valid_ith_m], 'ith_m'] = ith_data[mask_valid_ith_m]

                #break # Since interpolation should have only happened for the only right tile no need to evaluate others


        # Check if Millan ith interpolation is satisfactory. If not, try BedMachine v5
        millan_ith_nan_count_perc = np.isnan(points_df['ith_m']).sum() / len(points_df['ith_m'])
        #print(millan_ith_nan_count_perc)
        if millan_ith_nan_count_perc > .5:
            print(f'Millan ith has too many nans for {glacier_name}. Will try to use BedMachine tiles') if verbose else None

            # Interpolate BedMachinev5 ice field
            tile_ith_bedmacv5 = rioxarray.open_rasterio(file_ith_bedmacv5, masked=False)

            tile_ith = tile_ith_bedmacv5['thickness'] # get the ith field. Note that source is also interesting
            tile_ith = tile_ith.rio.write_crs("EPSG:3413") # I know bedmachine projection is EPSG:3413

            # I know bedmachine projection is EPSG:3413
            eastings, northings = Transformer.from_crs("EPSG:4326", tile_ith.rio.crs).transform(points_df['lats'],
                                                                                                points_df['lons'])
            minE, maxE = min(eastings), max(eastings)
            minN, maxN = min(northings), max(northings)

            epsM = 7000
            tile_ith = tile_ith.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM, maxy=maxN + epsM)
            tile_ith.values[(tile_ith.values == tile_ith.rio.nodata) | (tile_ith.values == 0.0)] = np.nan
            tile_ith.rio.write_nodata(np.nan, inplace=True)

            tile_ith = tile_ith.squeeze()

            eastings_ar = xarray.DataArray(eastings)
            northings_ar = xarray.DataArray(northings)

            ith_data = tile_ith.interp(y=northings_ar, x=eastings_ar, method="nearest").data

            # Fill dataframe with ith_m
            points_df['ith_m'] = ith_data

            #fig, ax = plt.subplots()
            #tile_ith.plot(ax=ax, vmin=tile_ith.min(), vmax=tile_ith.max())
            #ax.scatter(x=eastings, y=northings, c=ith_data, ec='r', s=3, vmin=tile_ith.min(), vmax=tile_ith.max())
            #plt.show()


        """At this point I am ready to interpolate the NSIDC velocity"""
        tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
        tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
        assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), 'Different bounds found.'
        assert tile_vx.rio.crs == tile_vy.rio.crs, 'Different crs found.'
        #print(tile_vx.rio.nodata, tile_vy.rio.nodata)

        all_eastings, all_northings = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(points_df['lats'],
                                                                                                       points_df['lons'])
        all_eastings_ar = xarray.DataArray(all_eastings)
        all_northings_ar = xarray.DataArray(all_northings)

        minE, maxE = min(all_eastings), max(all_eastings)
        minN, maxN = min(all_northings), max(all_northings)

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

            ris_metre_nsidc = tile_vx.rio.resolution()[0] # 250m

            # Calculate how many pixels I need for a resolution of xx
            # Since NDIDC has res of 250 m, num pixels will be very small, 1-3.
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

            # Very important
            # tile_v = (tile_vx**2 + tile_vy**2)**0.5
            tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)
            tile_v = tile_v.squeeze()

            # A check to see if velocity modules is as expected
            assert float(tile_v.sum()) > 0, "tile v is not as expected."

            """astropy"""
            preserve_nans = False
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

            #fig, ax = plt.subplots()
            #focus_filter_v300_ar.plot(ax=ax)
            #plt.show()


            # Interpolate
            v_data = tile_v.interp(y=all_northings_ar, x=all_eastings_ar, method="nearest").data
            v_filter_50_data = focus_filter_v50_ar.interp(y=all_northings_ar, x=all_eastings_ar, method='nearest').data
            v_filter_100_data = focus_filter_v100_ar.interp(y=all_northings_ar, x=all_eastings_ar,method='nearest').data
            v_filter_150_data = focus_filter_v150_ar.interp(y=all_northings_ar, x=all_eastings_ar,method='nearest').data
            v_filter_300_data = focus_filter_v300_ar.interp(y=all_northings_ar, x=all_eastings_ar,method='nearest').data
            v_filter_450_data = focus_filter_v450_ar.interp(y=all_northings_ar, x=all_eastings_ar,method='nearest').data
            v_filter_af_data = focus_filter_vfa_ar.interp(y=all_northings_ar, x=all_eastings_ar,method='nearest').data


            # Fill dataframe with NSIDC velocities
            points_df['v50'] = v_filter_50_data
            points_df['v100'] = v_filter_100_data
            points_df['v150'] = v_filter_150_data
            points_df['v300'] = v_filter_300_data
            points_df['v450'] = v_filter_450_data
            points_df['vgfa'] = v_filter_af_data

        plot_nsidc_green_4paper = False
        if plot_nsidc_green_4paper:
            from matplotlib.gridspec import GridSpec
            # Note this works for a greenland glacier with EPSG:3413
            #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5.7,3))
            fig = plt.figure(figsize=(9,5))
            gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

            # Create the axes
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            cax = fig.add_subplot(gs[2])

            nuns = gl_geom_nunataks_gdf.to_crs(crs="EPSG:3413")
            ext = gl_geom_ext_gdf.to_crs(crs="EPSG:3413")
            ext.plot(ax=ax1, edgecolor='k', facecolor='none', linewidth=1, zorder=2)
            nuns.plot(ax=ax1, edgecolor='k', facecolor='none', linewidth=1, zorder=2)
            norm = LogNorm(vmin=1, vmax=4000)
            im = focus_filter_v50_ar.plot(ax=ax1, cmap='jet', alpha=1, norm=norm, add_colorbar=False)

            s = ax2.scatter(x=all_eastings_ar, y=all_northings_ar, c=points_df['v50'],
                           cmap='jet', s=5, ec=None, norm=norm, alpha=1, zorder=1)
            ext.plot(ax=ax2, edgecolor='k', facecolor='none', linewidth=1, zorder=2)
            nuns.plot(ax=ax2, edgecolor='k', facecolor='none', linewidth=1, zorder=2)
            #cbar1 = plt.colorbar(s, ax=cax, fraction=0.062, pad=0.04)
            #cbar2 = plt.colorbar(s, ax=ax2, fraction=0.062, pad=0.04)
            #cbar1.set_label('Ice surface velocity (m/yr)')
            #cbar2.set_label('Ice surface velocity (m/yr)')
            cbar1 = plt.colorbar(s, cax=cax)  # ax=ax1)
            cbar1.ax.tick_params(labelsize=16)
            cbar1.set_label('Ice surface velocity (m/yr)', labelpad=15, rotation=90, fontsize=16)

            ax1.set_xlim(ax2.get_xlim())
            ax1.set_ylim(ax2.get_ylim())
            ax1.set_title('')
            ax1.set_xlabel('Eastings (m)', fontsize=16)
            ax1.set_ylabel('Northings (m)', fontsize=16)
            ax2.set_title('')
            ax2.set_xlabel('Eastings (m)', fontsize=16)
            ax2.set_ylabel('Northings (m)', fontsize=16)
            ax1.tick_params(axis='both', labelsize=16)
            ax2.tick_params(axis='both', labelsize=16)

            plt.subplots_adjust(wspace=0.01)
            plt.tight_layout()
            plt.show()

        plot_nsidc_green = False
        if plot_nsidc_green:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            s1 = ax1.scatter(x=points_df['lons'], y=points_df['lats'], c=points_df['v50'], norm=LogNorm(), s=1)
            cbar1 = plt.colorbar(s1)
            s2 = ax2.scatter(x=points_df['lons'], y=points_df['lats'], c=points_df['ith_m'], s=1)
            cbar2 = plt.colorbar(s2)
            plt.show()

        if verbose:
            print(f"From Millan/NSIDC vx, vy, ith interpolations we have generated no. nans:")
            print(", ".join([f"{col}: {points_df[col].isna().sum()}" for col in cols_millan]))

        if points_df['ith_m'].isna().all():
            print(f"No Millan ith data can be found for rgi {rgi} glacier {glacier_name} at {cenLat} lat {cenLon} lon.") if verbose else None

        return points_df

    def fetch_millan_data(points_df, rgi):

        # get Millan files
        if rgi in (1, 2):
            files_vx = sorted(glob(f"{config.millan_velocity_dir}RGI-1-2/VX_RGI-1-2*"))
            files_vy = sorted(glob(f"{config.millan_velocity_dir}RGI-1-2/VY_RGI-1-2*"))
            files_ith = sorted(glob(f"{config.millan_icethickness_dir}RGI-1-2/THICKNESS_RGI-1-2*"))
        elif rgi in (13, 14, 15):
            files_vx = sorted(glob(f"{config.millan_velocity_dir}RGI-13-15/VX_RGI-13-15*"))
            files_vy = sorted(glob(f"{config.millan_velocity_dir}RGI-13-15/VY_RGI-13-15*"))
            files_ith = sorted(glob(f"{config.millan_icethickness_dir}RGI-13-15/THICKNESS_RGI-13-15*"))
        else:
            files_vx = sorted(glob(f"{config.millan_velocity_dir}RGI-{rgi}/VX_RGI-{rgi}*"))
            files_vy = sorted(glob(f"{config.millan_velocity_dir}RGI-{rgi}/VY_RGI-{rgi}*"))
            files_ith = sorted(glob(f"{config.millan_icethickness_dir}RGI-{rgi}/THICKNESS_RGI-{rgi}*"))

        # I need a dataframe for Millan with same indexes and lats lons
        df_pointsM = points_df[['lats', 'lons']].copy()
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_vx})
        df_pointsM = df_pointsM.assign(**{col: pd.Series() for col in files_ith})

        # Fill the dataframe for occupancy
        tocc0 = time.time()
        for i, (file_vx, file_vy, file_ith) in enumerate(zip(files_vx, files_vy, files_ith)):
            tile_vx = rioxarray.open_rasterio(file_vx, cache=True, masked=False)
            tile_vy = rioxarray.open_rasterio(file_vy, cache=True, masked=False) # may relax this
            tile_ith = rioxarray.open_rasterio(file_ith, cache=True, masked=False)

            assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), 'Different velocity bounds found.'

            if not tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds():

                tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)
                # for rgi1-2 and 17 I save the reindexed ith tiles since reindexing is time consuming
                # tile_ith.rio.to_raster(file_ith)

            assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds(), 'Different bounds found.'
            assert tile_vx.rio.crs == tile_vy.rio.crs == tile_vy.rio.crs, 'Different crs found.'

            eastings, northings = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(df_pointsM['lats'],
                                                                                               df_pointsM['lons'])
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
        print(f"Created dataframe of occupancies for all points in {time.time()-tocc0} s.") if verbose else None
        ncols = df_pointsM.shape[1]
        #print(df_pointsM[0:5].T)
        # Sanity check that all points are contained in the same way in vx and ith tiles
        n_tiles_occupancy_vx = df_pointsM.iloc[:, 2:2+(ncols-2)//2].sum().sum()
        n_tiles_occupancy_ith = df_pointsM.iloc[:, 2+(ncols-2)//2:].sum().sum()
        assert n_tiles_occupancy_vx == n_tiles_occupancy_ith, "Mismatch between vx and ith coverage."

        # Grouping by vx occupancy. Each group will have an occupancy value
        df_pointsM['ntiles_vx'] = df_pointsM.iloc[:, 2:2+(ncols-2)//2].sum(axis=1)
        print(df_pointsM['ntiles_vx'].value_counts()) if verbose else None
        groups = df_pointsM.groupby('ntiles_vx')  # Groups.
        df_pointsM.drop(columns=['ntiles_vx'], inplace=True)  # Remove this column that we used to create groups
        print(f"Num groups in Millan: {groups.ngroups}") if verbose else None

        for g_value, df_group in groups:
            print(f"Group: {g_value} {len(df_group)} points") if verbose else None
            unique_vx_tiles = df_group.iloc[:, 2:2 + (ncols - 2) // 2].columns[df_group.iloc[:, 2:2 + (ncols - 2) // 2].sum() != 0].tolist()
            unique_ith_tiles = df_group.iloc[:, 2 + (ncols - 2) // 2:].columns[df_group.iloc[:, 2 + (ncols - 2) // 2:].sum() != 0].tolist()

            group_lats, group_lons = df_group['lats'].values, df_group['lons'].values
            #print(unique_vx_tiles)

            for file_vx, file_ith in zip(unique_vx_tiles, unique_ith_tiles):
                #print(file_vx)

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

                group_eastings, group_northings = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs).transform(group_lats,
                                                                                                               group_lons)
                minE, maxE = min(group_eastings), max(group_eastings)
                minN, maxN = min(group_northings), max(group_northings)

                epsM = 500
                tile_vx = tile_vx.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM,  maxy=maxN + epsM)
                tile_vy = tile_vy.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM,  maxy=maxN + epsM)
                tile_ith = tile_ith.rio.clip_box(minx=minE - epsM, miny=minN - epsM, maxx=maxE + epsM,  maxy=maxN + epsM)

                # Condition no. 1. Check if ith tile is only either nodata or zero
                # This condition is so soft. Glaciers may be still be present in the box. We need condition no. 2 as well
                tile_ith_is_all_zero_or_nodata = np.all(
                    np.logical_or(tile_ith.values == 0, tile_ith.values == tile_ith.rio.nodata))
                cond0 = np.all(tile_ith.values == 0)
                condnodata = np.all(np.abs(tile_ith.values - tile_ith.rio.nodata) < 1.e-6)
                condnan = np.all(np.isnan(tile_ith.values))
                all_zero_or_nodata = cond0 or condnodata or condnan
                #print(f"Cond1: {tile_ith_is_all_zero_or_nodata} {cond0,condnodata,condnan,all_zero_or_nodata}")

                if all_zero_or_nodata:
                    continue

                # Condition no. 2. A fast and quick interpolation to see if points intercepts a valid raster region
                group_eastings_ar = xarray.DataArray(group_eastings)
                group_northings_ar = xarray.DataArray(group_northings)

                vals_fast_interp = tile_ith.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                #cond_valid_fast_interp = np.sum(vals_fast_interp) == tile_ith.rio.nodata
                cond_valid_fast_interp = (np.isnan(vals_fast_interp).all() or
                                          np.all(np.abs(vals_fast_interp - tile_ith.rio.nodata) < 1.e-6))
                #print(f"Cond fast interp: {cond_valid_fast_interp}")

                if cond_valid_fast_interp:
                    continue

                # If we reached this point we should have some valid data to interpolate
                tile_vx.values = np.where((tile_vx.values == tile_vx.rio.nodata) | np.isinf(tile_vx.values),
                                          np.nan, tile_vx.values)
                tile_vy.values = np.where((tile_vy.values == tile_vy.rio.nodata) | np.isinf(tile_vy.values),
                                          np.nan, tile_vy.values)
                tile_ith.values = np.where((tile_ith.values == tile_ith.rio.nodata) | np.isinf(tile_ith.values),
                                           np.nan, tile_ith.values)

                #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                #tile_ith.plot(ax=ax1)
                #tile_vx.plot(ax=ax2)
                #tile_vy.plot(ax=ax3)
                #ax1.scatter(x=group_eastings_ar, y=group_northings_ar, s=1)
                #plt.show()

                tile_vx.rio.write_nodata(np.nan, inplace=True)
                tile_vy.rio.write_nodata(np.nan, inplace=True)
                tile_ith.rio.write_nodata(np.nan, inplace=True)

                assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, "Tiles vx, vy, ith with different epsg."
                assert tile_vx.rio.resolution() == tile_vy.rio.resolution() == tile_ith.rio.resolution(), \
                    "Tiles vx, vy, ith have different resolution."

                if not tile_vx.rio.bounds() == tile_ith.rio.bounds():
                    tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)

                assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_ith.rio.bounds(), "All tiles bounds not the same"

                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                ris_metre_millan = tile_vx.rio.resolution()[0]

                num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
                num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
                num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
                num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6
                num_px_sigma_450 = max(1, round(450 / ris_metre_millan)) # 9
                num_px_sigma_af = max(1, round(sigma_af / ris_metre_millan))

                kernel50 = Gaussian2DKernel(num_px_sigma_50, x_size=4 * num_px_sigma_50 + 1, y_size=4 * num_px_sigma_50 + 1)
                kernel100 = Gaussian2DKernel(num_px_sigma_100, x_size=4 * num_px_sigma_100 + 1, y_size=4 * num_px_sigma_100 + 1)
                kernel150 = Gaussian2DKernel(num_px_sigma_150, x_size=4 * num_px_sigma_150 + 1, y_size=4 * num_px_sigma_150 + 1)
                kernel300 = Gaussian2DKernel(num_px_sigma_300, x_size=4 * num_px_sigma_300 + 1, y_size=4 * num_px_sigma_300 + 1)
                kernel450 = Gaussian2DKernel(num_px_sigma_450, x_size=4 * num_px_sigma_450 + 1, y_size=4 * num_px_sigma_450 + 1)
                kernelaf = Gaussian2DKernel(num_px_sigma_af, x_size=4 * num_px_sigma_af + 1, y_size=4 * num_px_sigma_af + 1)

                # Deal with ith first
                focus_ith = tile_ith.squeeze()
                ith_data = focus_ith.interp(y=group_northings_ar, x=group_eastings_ar, method="nearest").data

                # Fill dataframe with ith_m
                if not np.all(np.isnan(ith_data)):
                    #print(np.sum(~np.isnan(ith_data)))

                    # points_df.loc[df_group.index, 'ith_m'] = ith_data

                    # Valid data can be in two different tiles.
                    # The mechanics is to investigate all vlid tiles, and fill dataframe only where data is non nan
                    # Create a mask for non-NaN values in ith_data
                    mask_valid_ith_m = ~np.isnan(ith_data)
                    points_df.loc[df_group.index[mask_valid_ith_m], 'ith_m'] = ith_data[mask_valid_ith_m]

                    #fig, ax = plt.subplots()
                    #ax.scatter(x=group_eastings_ar, y=group_northings_ar, s=2, c=ith_data)
                    #plt.show()

                    # If we have successfully filled ith_m we can proceed with velocity
                    tile_v = tile_vx.copy(deep=True, data=(tile_vx ** 2 + tile_vy ** 2) ** 0.5)
                    tile_v = tile_v.squeeze()

                    # Investigate the angle. We use arctan2
                    #theta = np.arctan2(tile_vy.values, tile_vx.values) * 180 / np.pi
                    #theta_ar = tile_vx.copy(deep=True, data=theta)

                    # Investigate velocity divergence
                    #divv_ar = tile_vx.differentiate(coord='x') + tile_vy.differentiate(coord='y')
                    #divv_ar.values = convolve_fft(divv_ar.values.squeeze(), kernel450, nan_treatment='interpolate',
                    #                            preserve_nan=True, boundary='fill', fill_value=np.nan).reshape(divv_ar.values.shape)

                    #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                    #theta_ar.plot(ax=ax1)
                    #tile_v.plot(ax=ax2)
                    #divv_ar.plot(ax=ax3)
                    #plt.show()

                    # If velocity exist
                    if float(tile_v.sum()) > 0 :

                        # Note: if the kernel is too small for the nan area, zeros will result (not sure why)
                        preserve_nans = False

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

                        #focus_ith = tile_ith.squeeze()

                        #except Exception as generic_error:
                        #    print("Impossible to smooth Millan tile with astropy.")
                        #    continue


                        # create xarrays of filtered velocities
                        focus_filter_v50_ar = tile_v.copy(deep=True, data=focus_filter_v50)
                        focus_filter_v100_ar = tile_v.copy(deep=True, data=focus_filter_v100)
                        focus_filter_v150_ar = tile_v.copy(deep=True, data=focus_filter_v150)
                        focus_filter_v300_ar = tile_v.copy(deep=True, data=focus_filter_v300)
                        focus_filter_v450_ar = tile_v.copy(deep=True, data=focus_filter_v450)
                        focus_filter_vfa_ar = tile_v.copy(deep=True, data=focus_filter_af)

                        #fig, ax = plt.subplots()
                        #focus_filter_v50_ar.plot(ax=ax)
                        #plt.show()

                        # Interpolate
                        #ith_data = focus_ith.interp(y=group_northings_ar, x=group_eastings_ar, method="nearest").data
                        v_data = tile_v.interp(y=group_northings_ar, x=group_eastings_ar, method="nearest").data
                        v_filter_50_data = focus_filter_v50_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                        v_filter_100_data = focus_filter_v100_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                        v_filter_150_data = focus_filter_v150_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                        v_filter_300_data = focus_filter_v300_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                        v_filter_450_data = focus_filter_v450_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data
                        v_filter_af_data = focus_filter_vfa_ar.interp(y=group_northings_ar, x=group_eastings_ar, method='nearest').data


                        # Fill dataframe with velocities
                        #points_df.loc[df_group.index,'ith_m'] = ith_data
                        #points_df.loc[df_group.index,'v50'] = v_filter_50_data
                        #points_df.loc[df_group.index,'v100'] = v_filter_100_data
                        #points_df.loc[df_group.index,'v150'] = v_filter_150_data
                        #points_df.loc[df_group.index,'v300'] = v_filter_300_data
                        #points_df.loc[df_group.index,'v450'] = v_filter_450_data
                        #points_df.loc[df_group.index,'vgfa'] = v_filter_af_data

                        # Lets fill only non nans in the interpolation, and we analyse all tiles (no break)
                        points_df.loc[df_group.index[~np.isnan(v_filter_50_data)], 'v50'] = v_filter_50_data[~np.isnan(v_filter_50_data)]
                        points_df.loc[df_group.index[~np.isnan(v_filter_100_data)], 'v100'] = v_filter_100_data[~np.isnan(v_filter_100_data)]
                        points_df.loc[df_group.index[~np.isnan(v_filter_150_data)], 'v150'] = v_filter_150_data[~np.isnan(v_filter_150_data)]
                        points_df.loc[df_group.index[~np.isnan(v_filter_300_data)], 'v300'] = v_filter_300_data[~np.isnan(v_filter_300_data)]
                        points_df.loc[df_group.index[~np.isnan(v_filter_450_data)], 'v450'] = v_filter_450_data[~np.isnan(v_filter_450_data)]
                        points_df.loc[df_group.index[~np.isnan(v_filter_af_data)], 'vgfa'] = v_filter_af_data[~np.isnan(v_filter_af_data)]

                        #fig, ax = plt.subplots()
                        #ax.scatter(x=group_eastings_ar, y=group_northings_ar, s=2, c=v_filter_50_data)
                        #plt.show()

                    # Since we want to loop over ALL unique tiles, we remove the break
                    #break # Since interpolation ith_m has worked not no need to evaluate others


        if verbose:
            print(f"From Millan vx, vy, ith interpolations we have generated no. nans:")
            print(", ".join([f"{col}: {points_df[col].isna().sum()}" for col in cols_millan]))

        if points_df['ith_m'].isna().all():
            print(f"No Millan ith data can be found for rgi {rgi} glacier {glacier_name} at {cenLat} lat {cenLon} lon.") if verbose else None

        return points_df


    if rgi == 5:
        points_df = fetch_millan_data_Gr(points_df)
    elif rgi == 19:
        points_df = fetch_millan_data_An(points_df)
    else:
        points_df = fetch_millan_data(points_df, rgi)

    tmillan2 = time.time()
    tmillan = tmillan2-tmillan1

    """ Add Slopes and Elevation """
    print(f"Calculating slopes and elevations...") if verbose else None
    tslope1 = time.time()

    swlat = points_df['lats'].min()
    swlon = points_df['lons'].min()
    nelat = points_df['lats'].max()
    nelon = points_df['lons'].max()
    deltalat = np.abs(swlat - nelat)
    deltalon = np.abs(swlon - nelon)
    lats_xar = xarray.DataArray(points_df['lats'])
    lons_xar = xarray.DataArray(points_df['lons'])

    eps = 5./3600

    # We now create the mosaic of the dem clipped around the glacier
    t0_load_dem = time.time()
    focus_mosaic_tiles = create_glacier_tile_dem_mosaic(minx=swlon - (deltalon + eps),
                            miny=swlat - (deltalat + eps),
                            maxx=nelon + (deltalon + eps),
                            maxy=nelat + (deltalat + eps),
                             rgi=rgi, path_tandemx=config.tandemx_dir)
    t1_load_dem = time.time()
    print(f"Time to load dem and create mosaic: {t1_load_dem-t0_load_dem}") if verbose else None

    focus = focus_mosaic_tiles.squeeze()

    # ***************** Calculate elevation and slopes in UTM ********************
    # Reproject to utm (projection distortions along boundaries converted to nans)
    # Default resampling is nearest which leads to weird artifacts. Options are bilinear (long) and cubic (very long)
    t0_reproj_dem = time.time()
    focus_utm = focus.rio.reproject(glacier_epsg, resampling=rasterio.enums.Resampling.bilinear, nodata=np.nan)
    t1_reproj_dem = time.time()
    print(f"Time to reproject dem: {t1_reproj_dem - t0_reproj_dem}") if verbose else None

    # Calculate the resolution in meters of the utm focus (resolutions in x and y are the same!)
    res_utm_metres = focus_utm.rio.resolution()[0]

    # Project the points onto the glacier projection
    eastings, northings = Transformer.from_crs("EPSG:4326", glacier_epsg).transform(points_df['lats'], points_df['lons'])
    #eastings, northings, _, _, epsgx = from_lat_lon_to_utm_and_epsg(np.array(points_df['lats']),
    #                                                            np.array(points_df['lons']))

    northings_xar = xarray.DataArray(northings)
    eastings_xar = xarray.DataArray(eastings)

    # clip the utm with a buffer of 2 km in both dimentions. This is necessary since smoothing is otherwise long
    focus_utm = focus_utm.rio.clip_box(
        minx=min(eastings) - 2000,
        miny=min(northings) - 2000,
        maxx=max(eastings) + 2000,
        maxy=max(northings) + 2000)

    num_px_sigma_50 = max(1, round(50 / res_utm_metres))
    num_px_sigma_75 = max(1, round(75 / res_utm_metres))
    num_px_sigma_100 = max(1, round(100 / res_utm_metres))
    num_px_sigma_125 = max(1, round(125 / res_utm_metres))
    num_px_sigma_150 = max(1, round(150 / res_utm_metres))
    num_px_sigma_300 = max(1, round(300 / res_utm_metres))
    num_px_sigma_450 = max(1, round(450 / res_utm_metres))
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
    #dz_dlat_xar, dz_dlon_xar = focus_utm.differentiate(coord='y'), focus_utm.differentiate(coord='x')
    #slope = focus_utm.copy(deep=True, data=(dz_dlat_xar ** 2 + dz_dlon_xar ** 2) ** 0.5)

    #t0_dem_smooth = time.time()
    #preserve_nans = True
    #focus_filter_50_utm = convolve_fft(focus_utm.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #focus_filter_300_utm = convolve_fft(focus_utm.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #focus_filter_af_utm = convolve_fft(focus_utm.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)

    #slope_50 = convolve_fft(slope.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_75 = convolve_fft(slope.values, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_100 = convolve_fft(slope.values, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_125 = convolve_fft(slope.values, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_150 = convolve_fft(slope.values, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_300 = convolve_fft(slope.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_450 = convolve_fft(slope.values, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #slope_af = convolve_fft(slope.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
    #t1_dem_smooth = time.time()
    #print(f"Time to smooth dem: {t1_dem_smooth - t0_dem_smooth}")

    # I first smooth the elevation with 6 kernels and then calculating the slopes.
    # I fear I should first calculate the slope (1 field) and the smooth it using 6 kernels
    run_dem_with_gpu = False
    run_dem_with_cpu = not run_dem_with_gpu

    if run_dem_with_gpu:
        # some weird artifacts are introduced for small glaciers, see e.g. RGI60-13.33257, or RGI60-13.45291,
        # compared to cpu with astropy. i suspect astropy works a little better in dealing with nans due to reprojections
        # GPU
        t0_dem_smooth_cp = time.time()
        focus_utm_cp = cp.asarray(focus_utm.values)
        focus_filter_50_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_50, mode='mirror')#.get()
        focus_filter_75_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_75, mode='mirror')#.get()
        focus_filter_100_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_100, mode='mirror')#.get()
        focus_filter_125_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_125, mode='mirror')#.get()
        focus_filter_150_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_150, mode='mirror')#.get()
        focus_filter_300_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_300, mode='mirror')#.get()
        focus_filter_450_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_450, mode='mirror')#.get()
        focus_filter_af_utm_cp = cupyx.scipy.ndimage.gaussian_filter(focus_utm_cp, sigma=num_px_sigma_af, mode='mirror')#.get()
        t1_dem_smooth_cp = time.time()
        print(f"Time to smooth dem with cupy: {t1_dem_smooth_cp - t0_dem_smooth_cp}") if verbose else None

        # In case of big kernels we need to remove nan artifacts (on cpu we have astropy that does that)
        t_nan0 = time.time()
        mean_dem_elev = cp.nanmean(focus_utm_cp)
        focus_filter_50_utm_cp = cp.nan_to_num(focus_filter_50_utm_cp, nan=mean_dem_elev)
        focus_filter_75_utm_cp = cp.nan_to_num(focus_filter_75_utm_cp, nan=mean_dem_elev)
        focus_filter_100_utm_cp = cp.nan_to_num(focus_filter_100_utm_cp, nan=mean_dem_elev)
        focus_filter_125_utm_cp = cp.nan_to_num(focus_filter_125_utm_cp, nan=mean_dem_elev)
        focus_filter_150_utm_cp = cp.nan_to_num(focus_filter_150_utm_cp, nan=mean_dem_elev)
        focus_filter_300_utm_cp = cp.nan_to_num(focus_filter_300_utm_cp, nan=mean_dem_elev)
        focus_filter_450_utm_cp = cp.nan_to_num(focus_filter_450_utm_cp, nan=mean_dem_elev)
        focus_filter_af_utm_cp = cp.nan_to_num(focus_filter_af_utm_cp, nan=mean_dem_elev)
        t_nan1 = time.time()
        print(f'Time required to fill nans in cupy: {t_nan1 - t_nan0}') if verbose else None

        # slopes with GPU
        s50_lat, s50_lon = cp.gradient(focus_filter_50_utm_cp, -res_utm_metres, res_utm_metres)
        s75_lat, s75_lon = cp.gradient(focus_filter_75_utm_cp, -res_utm_metres, res_utm_metres)
        s100_lat, s100_lon = cp.gradient(focus_filter_100_utm_cp, -res_utm_metres, res_utm_metres)
        s125_lat, s125_lon = cp.gradient(focus_filter_125_utm_cp, -res_utm_metres, res_utm_metres)
        s150_lat, s150_lon = cp.gradient(focus_filter_150_utm_cp, -res_utm_metres, res_utm_metres)
        s300_lat, s300_lon = cp.gradient(focus_filter_300_utm_cp, -res_utm_metres, res_utm_metres)
        s450_lat, s450_lon = cp.gradient(focus_filter_450_utm_cp, -res_utm_metres, res_utm_metres)
        saf_lat, saf_lon = cp.gradient(focus_filter_af_utm_cp, -res_utm_metres, res_utm_metres)

        s50 = cp.sqrt(s50_lat ** 2 + s50_lon ** 2).get()
        s75 = cp.sqrt(s75_lat ** 2 + s75_lon ** 2).get()
        s100 = cp.sqrt(s100_lat ** 2 + s100_lon ** 2).get()
        s125 = cp.sqrt(s125_lat ** 2 + s125_lon ** 2).get()
        s150 = cp.sqrt(s150_lat ** 2 + s150_lon ** 2).get()
        s300 = cp.sqrt(s300_lat ** 2 + s300_lon ** 2).get()
        s450 = cp.sqrt(s450_lat ** 2 + s450_lon ** 2).get()
        saf = cp.sqrt(saf_lat ** 2 + saf_lon ** 2).get()

        # create slope xarrays
        slope_50_xar = focus_utm.copy(data=s50)
        slope_75_xar = focus_utm.copy(data=s75)
        slope_100_xar = focus_utm.copy(data=s100)
        slope_125_xar = focus_utm.copy(data=s125)
        slope_150_xar = focus_utm.copy(data=s150)
        slope_300_xar = focus_utm.copy(data=s300)
        slope_450_xar = focus_utm.copy(data=s450)
        slope_af_xar = focus_utm.copy(data=saf)

        # These are needed for the curvature
        focus_filter_xarray_50_utm = focus_utm.copy(data=focus_filter_50_utm_cp.get())
        focus_filter_xarray_300_utm = focus_utm.copy(data=focus_filter_300_utm_cp.get())
        focus_filter_xarray_af_utm = focus_utm.copy(data=focus_filter_af_utm_cp.get())


    if run_dem_with_cpu:
        # CPU
        t0_dem_smooth = time.time()
        preserve_nans = True
        focus_filter_50_utm = convolve_fft(focus_utm.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_75_utm = convolve_fft(focus_utm.values, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_100_utm = convolve_fft(focus_utm.values, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_125_utm = convolve_fft(focus_utm.values, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_150_utm = convolve_fft(focus_utm.values, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_300_utm = convolve_fft(focus_utm.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_450_utm = convolve_fft(focus_utm.values, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_af_utm = convolve_fft(focus_utm.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        t1_dem_smooth = time.time()
        print(f"Time to smooth dem: {t1_dem_smooth - t0_dem_smooth}") if verbose else None

        # create xarray object of filtered dem
        focus_filter_xarray_50_utm = focus_utm.copy(data=focus_filter_50_utm)
        focus_filter_xarray_75_utm = focus_utm.copy(data=focus_filter_75_utm)
        focus_filter_xarray_100_utm = focus_utm.copy(data=focus_filter_100_utm)
        focus_filter_xarray_125_utm = focus_utm.copy(data=focus_filter_125_utm)
        focus_filter_xarray_150_utm = focus_utm.copy(data=focus_filter_150_utm)
        focus_filter_xarray_300_utm = focus_utm.copy(data=focus_filter_300_utm)
        focus_filter_xarray_450_utm = focus_utm.copy(data=focus_filter_450_utm)
        focus_filter_xarray_af_utm = focus_utm.copy(data=focus_filter_af_utm)

        # create xarray slopes (differentiating an xarray is much slower than using numpy)
        #dz_dlat_xar, dz_dlon_xar = focus_utm.differentiate(coord='y'), focus_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_75, dz_dlon_filter_xar_75 = focus_filter_xarray_75_utm.differentiate(coord='y'), focus_filter_xarray_75_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_125, dz_dlon_filter_xar_125 = focus_filter_xarray_125_utm.differentiate(coord='y'), focus_filter_xarray_125_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_450, dz_dlon_filter_xar_450  = focus_filter_xarray_450_utm.differentiate(coord='y'), focus_filter_xarray_450_utm.differentiate(coord='x')
        #dz_dlat_filter_xar_af, dz_dlon_filter_xar_af = focus_filter_xarray_af_utm.differentiate(coord='y'), focus_filter_xarray_af_utm.differentiate(coord='x')

        # create slope xarrays
        #slope_50_xar = focus_utm.copy(data=(dz_dlat_filter_xar_50 ** 2 + dz_dlon_filter_xar_50 ** 2) ** 0.5)
        #slope_75_xar = focus_utm.copy(data=(dz_dlat_filter_xar_75 ** 2 + dz_dlon_filter_xar_75 ** 2) ** 0.5)
        #slope_100_xar = focus_utm.copy(data=(dz_dlat_filter_xar_100 ** 2 + dz_dlon_filter_xar_100 ** 2) ** 0.5)
        #slope_125_xar = focus_utm.copy(data=(dz_dlat_filter_xar_125 ** 2 + dz_dlon_filter_xar_125 ** 2) ** 0.5)
        #slope_150_xar = focus_utm.copy(data=(dz_dlat_filter_xar_150 ** 2 + dz_dlon_filter_xar_150 ** 2) ** 0.5)
        #slope_300_xar = focus_utm.copy(data=(dz_dlat_filter_xar_300 ** 2 + dz_dlon_filter_xar_300 ** 2) ** 0.5)
        #slope_450_xar = focus_utm.copy(data=(dz_dlat_filter_xar_450 ** 2 + dz_dlon_filter_xar_450 ** 2) ** 0.5)
        #slope_af_xar = focus_utm.copy(data=(dz_dlat_filter_xar_af ** 2 + dz_dlon_filter_xar_af ** 2) ** 0.5)

        # using numpy is much faster than xarray to differentiate
        dz_dlat_np_50, dz_dlon_np_50 = np.gradient(focus_filter_50_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_75, dz_dlon_np_75 = np.gradient(focus_filter_75_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_100, dz_dlon_np_100 = np.gradient(focus_filter_100_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_125, dz_dlon_np_125 = np.gradient(focus_filter_125_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_150, dz_dlon_np_150 = np.gradient(focus_filter_150_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_300, dz_dlon_np_300 = np.gradient(focus_filter_300_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_450, dz_dlon_np_450 = np.gradient(focus_filter_450_utm, -res_utm_metres, res_utm_metres)
        dz_dlat_np_af, dz_dlon_np_af = np.gradient(focus_filter_af_utm, -res_utm_metres, res_utm_metres)

        slope_50_xar = focus_utm.copy(data=(dz_dlat_np_50 ** 2 + dz_dlon_np_50 ** 2) ** 0.5)
        slope_75_xar = focus_utm.copy(data=(dz_dlat_np_75 ** 2 + dz_dlon_np_75 ** 2) ** 0.5)
        slope_100_xar = focus_utm.copy(data=(dz_dlat_np_100 ** 2 + dz_dlon_np_100 ** 2) ** 0.5)
        slope_125_xar = focus_utm.copy(data=(dz_dlat_np_125 ** 2 + dz_dlon_np_125 ** 2) ** 0.5)
        slope_150_xar = focus_utm.copy(data=(dz_dlat_np_150 ** 2 + dz_dlon_np_150 ** 2) ** 0.5)
        slope_300_xar = focus_utm.copy(data=(dz_dlat_np_300 ** 2 + dz_dlon_np_300 ** 2) ** 0.5)
        slope_450_xar = focus_utm.copy(data=(dz_dlat_np_450 ** 2 + dz_dlon_np_450 ** 2) ** 0.5)
        slope_af_xar = focus_utm.copy(data=(dz_dlat_np_af ** 2 + dz_dlon_np_af ** 2) ** 0.5)

        #slat300, slon300 = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_utm.differentiate(coord='x')
        #slope_300_xar_after_dem_conv = slope_300_xar.copy(deep=True, data=(slat300 ** 2 + slon300 ** 2) ** 0.5)

        #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        #slope_300_xar.plot(ax=ax1)
        #focus_filter_xarray_300_utm.plot(ax=ax2, cmap='terrain')
        #slope_300_xar_after_dem_conv.plot(ax=ax3)
        #plt.show()

    # Calculate curvature and aspect using xrspatial
    curv_50 = xrspatial.curvature(focus_filter_xarray_50_utm)
    curv_300 = xrspatial.curvature(focus_filter_xarray_300_utm)
    curv_af = xrspatial.curvature(focus_filter_xarray_af_utm)
    # todo: aspect calculate using DEM instead of importing from OGGM ?
    #ttest0 = time.time()
    #aspect_rad = np.arctan2(dz_dlat_np_50, dz_dlon_np_50)
    #aspect_deg = np.degrees(aspect_rad)
    #aspect_deg = (aspect_deg + 360) % 360
    #flat_areas = (dz_dlat_np_50 == 0) & (dz_dlon_np_50 == 0)
    #aspect_deg[flat_areas] = np.nan
    #mean_aspect = np.nanmean(aspect_deg)
    #ttest1 = time.time()
    #print(gl_df['Aspect'].item(), mean_aspect, ttest1-ttest0)
    #input('wait')

    #aspect_50 = xrspatial.aspect(focus_filter_xarray_50_utm)
    #aspect_300 = xrspatial.aspect(focus_filter_xarray_300_utm)
    #aspect_af = xrspatial.aspect(focus_filter_xarray_af_utm)

    #slope_450_xar.plot()
    #plt.show()

    # interpolate slope and dem (this has to be done on cpu as xarray-cupy does not support interpolation yet)
    t_interp0 = time.time()
    elevation_data = focus_utm.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_50_data = slope_50_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_75_data = slope_75_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_100_data = slope_100_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_125_data = slope_125_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_150_data = slope_150_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_300_data = slope_300_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_450_data = slope_450_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    slope_af_data = slope_af_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
    t_interp1 = time.time()
    print(f"Time to interpolate DEM: {t_interp1 - t_interp0} s") if verbose else None
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
    #aspect_data_50 = aspect_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
    #aspect_data_300 = aspect_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
    #aspect_data_af = aspect_af.interp(y=northings_xar, x=eastings_xar, method='linear').data

    # Hugonnet mass balance
    try:
        glacier_dmdtda = mbdf_rgi.at[glacier_name, 'dmdtda']
    except: # impute the mean
        glacier_dmdtda = mbdf_rgi['dmdtda'].median()
    print(f'Hugonnet mb: {glacier_dmdtda} m w.e/yr') if verbose else None


    # check if any nan in the interpolate data
    contains_nan = any(np.isnan(arr).any() for arr in [elevation_data, slope_50_data, slope_75_data, slope_100_data,
                                                       slope_125_data, slope_150_data, slope_300_data,
                                                       slope_450_data, slope_af_data,
                                                       curv_data_50, curv_data_300, curv_data_af,])
                                                       #aspect_data_50, aspect_data_300, aspect_data_af])

    if contains_nan:
        raise ValueError(f"Nan detected in elevation/slope calc. Check")

    # Fill zmin, zmax, zmed using tandemx interpolated elevation data
    points_df['Zmin'] = np.min(elevation_data)
    points_df['Zmax'] = np.max(elevation_data)
    points_df['Zmed'] = np.median(elevation_data)

    # Fill dataframe with elevation and slopes
    points_df['elevation'] = elevation_data
    points_df['slope50'] = slope_50_data
    points_df['slope75'] = slope_75_data
    points_df['slope100'] = slope_100_data
    points_df['slope125'] = slope_125_data
    points_df['slope150'] = slope_150_data
    points_df['slope300'] = slope_300_data
    points_df['slope450'] = slope_450_data
    points_df['slopegfa'] = slope_af_data
    #points_df['slope_lat'] = slope_lat_data
    #points_df['slope_lon'] = slope_lon_data
    #points_df['slope_lat_gf50'] = slope_lat_data_filter_50
    #points_df['slope_lon_gf50'] = slope_lon_data_filter_50
    #points_df['slope_lat_gf75'] = slope_lat_data_filter_75
    #points_df['slope_lon_gf75'] = slope_lon_data_filter_75
    #points_df['slope_lat_gf100'] = slope_lat_data_filter_100
    #points_df['slope_lon_gf100'] = slope_lon_data_filter_100
    #points_df['slope_lat_gf125'] = slope_lat_data_filter_125
    #points_df['slope_lon_gf125'] = slope_lon_data_filter_125
    #points_df['slope_lat_gf150'] = slope_lat_data_filter_150
    #points_df['slope_lon_gf150'] = slope_lon_data_filter_150
    #points_df['slope_lat_gf300'] = slope_lat_data_filter_300
    #points_df['slope_lon_gf300'] = slope_lon_data_filter_300
    #points_df['slope_lat_gf450'] = slope_lat_data_filter_450
    #points_df['slope_lon_gf450'] = slope_lon_data_filter_450
    #points_df['slope_lat_gfa'] = slope_lat_data_filter_af
    #points_df['slope_lon_gfa'] = slope_lon_data_filter_af
    points_df['curv_50'] = curv_data_50
    points_df['curv_300'] = curv_data_300
    points_df['curv_gfa'] = curv_data_af
    #points_df['aspect_50'] = aspect_data_50
    #points_df['aspect_300'] = aspect_data_300
    #points_df['aspect_gfa'] = aspect_data_af
    points_df['dmdtda_hugo'] = glacier_dmdtda

    calculate_elevation_and_slopes_in_epsg_4326_and_show_differences_wrt_utm = False
    if calculate_elevation_and_slopes_in_epsg_4326_and_show_differences_wrt_utm:
        lon_c = (0.5 * (focus.coords['x'][-1] + focus.coords['x'][0])).to_numpy()
        lat_c = (0.5 * (focus.coords['y'][-1] + focus.coords['y'][0])).to_numpy()
        ris_ang_lon, ris_ang_lat = focus.rio.resolution()
        #print(ris_ang_lon, ris_ang_lat)

        #ris_metre_lon = haversine(lon_c, lat_c, lon_c + ris_ang, lat_c) * 1000
        #ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c + ris_ang) * 1000
        ris_metre_lon = haversine(lon_c, lat_c, lon_c + ris_ang_lon, lat_c) * 1000
        ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c + ris_ang_lat) * 1000

        # calculate slope for restricted dem
        dz_dlat, dz_dlon = np.gradient(focus.values, -ris_metre_lat, ris_metre_lon)  # [m/m]
        dz_dlat_xarray = focus.copy(data=dz_dlat)
        dz_dlon_xarray = focus.copy(data=dz_dlon)

        # interpolate dem and slope
        elevation_data1 = focus.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)
        slope_lat_data1 = dz_dlat_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)
        slope_lon_data1 = dz_dlon_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)

        assert slope_lat_data1.shape == slope_lon_data1.shape == elevation_data1.shape, "Different shapes, something wrong!"

        fig, axes = plt.subplots(2,3, figsize=(10,8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        # elevation
        im1 = focus.plot(ax=ax1, cmap='viridis', vmin=np.nanmin(elevation_data1),
                                  vmax=np.nanmax(elevation_data1), zorder=0)
        s1 = ax1.scatter(x=lons_xar, y=lats_xar, s=50, c=elevation_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(elevation_data1), vmax=np.nanmax(elevation_data1), zorder=1)
        # slope_lat
        im2 = dz_dlat_xarray.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(slope_lat_data1),
                                  vmax=np.nanmax(slope_lat_data1), zorder=0)
        s2 = ax2.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lat_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lat_data1), vmax=np.nanmax(slope_lat_data1), zorder=1)
        # slope_lon
        im3 = dz_dlon_xarray.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(slope_lon_data1),
                                  vmax=np.nanmax(slope_lon_data1), zorder=0)
        s3 = ax3.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lon_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lon_data1), vmax=np.nanmax(slope_lon_data1), zorder=1)
        # utm elevation
        im4 = focus_utm.plot(ax=ax4, cmap='viridis', vmin=np.nanmin(elevation_data),
                                  vmax=np.nanmax(elevation_data), zorder=0)
        s4 = ax4.scatter(x=eastings_xar, y=northings_xar, s=50, c=elevation_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_data), zorder=1)
        # utm slope_lat
        im5 = dz_dlat_xar.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(slope_lat_data),
                                  vmax=np.nanmax(slope_lat_data), zorder=0)
        s5 = ax5.scatter(x=eastings_xar, y=northings_xar, s=50, c=slope_lat_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)
        # utm slope_lon
        im6 = dz_dlon_xar.plot(ax=ax6, cmap='viridis', vmin=np.nanmin(slope_lon_data),
                                  vmax=np.nanmax(slope_lon_data), zorder=0)
        s6 = ax6.scatter(x=eastings_xar, y=northings_xar, s=50, c=slope_lon_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lon_data), vmax=np.nanmax(slope_lon_data), zorder=1)

        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.scatter(x=elevation_data1, y=elevation_data)
        ax2.scatter(x=slope_lon_data1, y=slope_lon_data)
        ax3.scatter(x=slope_lat_data1, y=slope_lat_data)
        l1 = ax1.plot([0, 2000], [0, 2000], color='red', linestyle='--')
        l2 = ax2.plot([-3, 3], [-3, 3], color='red', linestyle='--')
        l3 = ax3.plot([-2, 2], [-2, 2], color='red', linestyle='--')
        plt.show()

    tslope2 = time.time()
    tslope = tslope2-tslope1

    """ Calculate SMB """
    print(f"Calculating SMB...") if verbose else None
    tsmb0 = time.time()

    if rgi in [5,19]:
        print("Mass balance with racmo") if verbose else None
        # Surface mass balance with racmo
        path_RACMO_folder = config.racmo_dir
        if rgi==5:
            racmo_file = "greenland_racmo2.3p2/smb_greenland_mean_1961_1990_RACMO23p2_gf.nc"
        elif rgi==19:
            racmo_file = "antarctica_racmo2.3p2/2km/smb_antarctica_mean_1979_2021_RACMO23p2_gf.nc"
        else: raise ValueError('rgi value for RACMO smb calculation not recognized')

        racmo = rioxarray.open_rasterio(f'{path_RACMO_folder}{racmo_file}')

        eastings_racmo, northings_racmo = (Transformer.from_crs("EPSG:4326", racmo.rio.crs)
                               .transform(points_df['lats'], points_df['lons']))

        # Convert coordinates to racmo projection EPSG:3413 (racmo Greenland) or EPSG:3031 (racmo Antarctica)
        eastings_racmo_ar = xarray.DataArray(eastings_racmo)
        northings_racmo_ar = xarray.DataArray(northings_racmo)

        # Interpolate racmo onto the points
        smb_data = racmo.interp(y=northings_racmo_ar, x=eastings_racmo_ar, method='linear').data.squeeze()

        # If Racmo does not cover this glacier I use Hugonnet-elevation relation
        if np.all(np.isnan(smb_data)):
            print("Using Hugonnet-elevation relation") if verbose else None
            m_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'm']
            q_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'q']
            smb_data_hugo = m_hugo * elevation_data + q_hugo  # m w.e./yr
            smb_data_hugo *= 1.e3  # mm w.e./yr
            smb_data = np.array(smb_data_hugo)


        plot_smb_racmo = False
        if plot_smb_racmo:
            vmin, vmax = racmo.min(), racmo.max()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            racmo.plot(ax=ax1, cmap='hsv', vmin=vmin, vmax=vmax)
            ax1.scatter(x=eastings_racmo_ar, y=northings_racmo_ar, c='k', s=20)
            racmo.plot(ax=ax2, cmap='hsv', vmin=vmin, vmax=vmax)
            ax2.scatter(x=eastings_racmo_ar, y=northings_racmo_ar, c=smb_data, cmap='hsv', vmin=vmin, vmax=vmax, s=20)
            plt.show()

    else:
        print("Using Hugonnet-elevation relation") if verbose else None
        # Surface mass balance with my method in all other regions (BAD METHOD)
        #smb_data = []
        #for (lat, lon, elev) in zip(points_df['lats'], points_df['lons'], elevation_data):
        #    smb = smb_elev_functs(rgi, elev, lat, lon)  # kg/m2s
        #   smb *= 31536000  # kg/m2yr
        #    smb_data.append(smb)
        #smb_data = np.array(smb_data)

        # With Hugonnet regional downscaling
        m_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'm']
        q_hugo = smb_elev_functs_hugo(rgi=rgi).loc[rgi, 'q']
        smb_data_hugo = m_hugo * elevation_data + q_hugo # m w.e./yr = (1000 kg/m2yr)
        smb_data_hugo *= 1.e3 # mm w.e./yr = (kg/m2yr)
        smb_data = np.array(smb_data_hugo)

        plot_smb_my_method = False
        if plot_smb_my_method:
            fig, ax = plt.subplots()
            s = ax.scatter(x=points_df['lons'], y=points_df['lats'], c=smb_data)
            cbar = plt.colorbar(s)
            plt.show()

    print(f'Mean smb: {np.mean(smb_data)} kg/m2yr') if verbose else None
    points_df['smb'] = smb_data

    tsmb1 = time.time()
    tsmb = tsmb1 - tsmb0
    print(f"Finished SMB calculations.") if verbose else None

    """ Calculate ERA5 t2m """
    print(f"Calculating ERA5 t2m...") if verbose else None
    tera5_1 = time.time()

    points_df['t2m'] = np.nan

    tile_era5_t2m = rioxarray.open_rasterio(f"{config.ERA5_t2m_dir}era5land_era5.nc", masked=False)
    tile_era5_t2m = tile_era5_t2m.squeeze()

    #fig, ax = plt.subplots()
    #tile_era5_t2m.plot(ax=ax)
    #ax.scatter(x=points_df['lons'], y=points_df['lats'])
    #plt.show()

    try:
        t2m_data = tile_era5_t2m.interp(y=xarray.DataArray(points_df['lats']),
                                        x=xarray.DataArray(points_df['lons']), method="linear").data

        # Check if there are any NaNs in the interpolated data
        if np.isnan(t2m_data).any():
            raise ValueError("NaN values detected after linear interpolation in temperature")

    # For glaciers close to the lon=-180 border the interpolation fails. Lets redefine the coordinates
    # Triggered in RGI60-10.05038
    except ValueError:
        # If NaNs are detected, perform interpolation with adjusted longitudes
        tile_era5_t2m_adjusted = tile_era5_t2m.assign_coords(x=((tile_era5_t2m['x'] + 360) % 360))
        points_df['lons_adjusted'] = (points_df['lons'] + 360) % 360

        t2m_data = tile_era5_t2m_adjusted.interp(y=xarray.DataArray(points_df['lats']),
                                                 x=xarray.DataArray(points_df['lons_adjusted']), method="linear").data

    points_df['t2m'] = t2m_data

    plot_era5 = False
    if plot_era5:
        fig, ax = plt.subplots()
        ax.scatter(x=points_df['lons'], y=points_df['lats'], s=1, c=t2m_data)
        plt.show()

    tera5_2 = time.time()
    tera5 = tera5_2 - tera5_1

    """ Calculate Farinotti ith_f """
    print(f"Calculating ith_f...") if verbose else None
    tfar1 = time.time()

    points_df['ith_f'] = np.nan

    folder_rgi_farinotti = f"{config.farinotti_icethickness_dir}RGI60-{rgi:02d}/"
    try: # Import farinotti ice thickness file. Note that it contains zero where ice not present.
        file_glacier_farinotti =rioxarray.open_rasterio(f'{folder_rgi_farinotti}{glacier_name}_thickness.tif', masked=False)
        file_glacier_farinotti = file_glacier_farinotti.where(file_glacier_farinotti != 0.0) # replace zeros with nans.
        file_glacier_farinotti.rio.write_nodata(np.nan, inplace=True)

        transformerF = Transformer.from_crs("EPSG:4326", file_glacier_farinotti.rio.crs)
        lons_crs_f, lats_crs_f = transformerF.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())

        ith_f_data = file_glacier_farinotti.interp(y=xarray.DataArray(lats_crs_f), x=xarray.DataArray(lons_crs_f),
                                                   method="nearest").data.squeeze()
        points_df['ith_f'] = ith_f_data
        print(f"From Farinotti ith interpolation we have generated {np.isnan(ith_f_data).sum()} nans.") if verbose else None

        show_farinotti = False
        if show_farinotti:
            fig, (ax1, ax2) = plt.subplots(1,2)
            s1 = ax1.scatter(x=lons_crs_f, y=lats_crs_f, s=1, c=ith_f_data)
            s2 = ax1.scatter(x=lons_crs_f[np.isnan(ith_f_data)], y=lats_crs_f[np.isnan(ith_f_data)], s=1, c='magenta')
            file_glacier_farinotti.plot(ax=ax2, cmap='Blues')
            cmbar = plt.colorbar(s1)
            plt.show()

        no_farinotti_data = False

    except:
        print(f"No Farinotti data can be found for rgi {rgi} glacier {glacier_name} or Farinotti interpolation is problematic.") if verbose else None
        no_farinotti_data = True

    tfar2 = time.time()
    tfar = tfar2-tfar1

    """ Calculate distance_from_border """
    print(f"Calculating the distances using glacier geometries... ") if verbose else None
    tdist0 = time.time()

    # to remove
    def add_new_neighbors(neighbors, df):
        """ I give a list of neighbors and I should return a new list with added neighbors"""
        for id in neighbors:
            #neighbors_wrt_id = df[df['RGIId_1'] == id]['RGIId_2'].unique() # old, less precise
            # new version. We don't care whether the glacier is in 1 or 2. This is accurate as method with graph
            neighbors_wrt_1 = df[df['RGIId_1'] == id]['RGIId_2'].unique()
            neighbors_wrt_2 = df[df['RGIId_2'] == id]['RGIId_1'].unique()
            neighbors_wrt_id = np.concatenate((neighbors_wrt_1, neighbors_wrt_2))
            neighbors = np.append(neighbors, neighbors_wrt_id)
        neighbors = np.unique(neighbors)
        return neighbors

    # to remove
    def find_cluster_RGIIds(id, df):
        neighbors0 = np.array([id])
        len0 = len(neighbors0)
        neighbors1 = add_new_neighbors(neighbors0, df)
        len1 = len(neighbors1)
        while len1 > len0:
            len0 = len1
            neighbors1 = add_new_neighbors(neighbors1, df)
            len1 = len(neighbors1)
        return neighbors1

    # Calculate intersects of all glaciers in the cluster
    # list_cluster_RGIIds = find_cluster_RGIIds(glacier_name, oggm_rgi_intersects)# (SLOW NESTED LOOP METHOD)
    list_cluster_RGIIds = find_cluster_with_graph(rgi_graph, glacier_name, max_depth=graph_max_layer_depth)
    no_glaciers_in_cluster = len(list_cluster_RGIIds)
    print(f"Cluster: {no_glaciers_in_cluster} glaciers created in: {time.time()-tdist0:.3f}") if verbose else None

    # Create Geopandas geoseries objects of glacier geometries (boundary and nunataks) and convert to UTM
    tgeoms0 = time.time()
    cluster_geometry_list = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'].isin(list_cluster_RGIIds), 'geometry'].tolist()

    # Combine into a series of all glaciers in the cluster
    cluster_geometry_4326 = gpd.GeoSeries(cluster_geometry_list, crs="EPSG:4326")

    # Now remove all ice divides
    # Note: .union_all(method='coverage') is faster than 'unary' but may lead to incorrect results if polygons overlap
    #try:
    #    cluster_geometry_no_divides_4326 = gpd.GeoSeries(cluster_geometry_4326.union_all(method='coverage'), crs="EPSG:4326")
    #except GEOSException as e:
    #    print(f"Coverage method failed: {e} Falling back to unary method.") if verbose else None
    #    cluster_geometry_no_divides_4326 = gpd.GeoSeries(cluster_geometry_4326.union_all(method='unary'), crs="EPSG:4326")
    cluster_geometry_no_divides_4326 = gpd.GeoSeries(cluster_geometry_4326.union_all(method='unary'), crs="EPSG:4326")

    cluster_geometry_no_divides_epsg = cluster_geometry_no_divides_4326.to_crs(epsg=glacier_epsg)
    if cluster_geometry_no_divides_epsg.item().geom_type == 'Polygon':
        cluster_exterior_ring = [cluster_geometry_no_divides_epsg.item().exterior]  # shapely.geometry.polygon.LinearRing
        cluster_interior_rings = list(cluster_geometry_no_divides_epsg.item().interiors)  # shapely.geometry.polygon.LinearRing
        multipolygon = False
    elif cluster_geometry_no_divides_epsg.item().geom_type == 'MultiPolygon':
        polygons = list(cluster_geometry_no_divides_epsg.item().geoms)
        cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
        num_multipoly = len(cluster_exterior_ring)
        cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
        cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
        multipolygon = True
    else: raise ValueError("Unexpected geometry type. Please check.")

    # Create a geoseries of all external and internal geometries
    geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings, crs=glacier_epsg)
    no_geometries_in_cluster = len(geoseries_geometries_epsg)
    print(f"Cluster: {no_geometries_in_cluster} geometries created in: {time.time()-tgeoms0:.3f}") if verbose else None

    # Calculate the area of the cluster in km2
    cluster_exterior_ring_gpd = gpd.GeoSeries([cluster_exterior_ring[0]], crs=glacier_epsg).to_crs("EPSG:4326")
    area_cluster, perimeter_cluster = Geod(ellps="WGS84").geometry_area_perimeter(cluster_exterior_ring_gpd.iloc[0])
    area_cluster = abs(area_cluster) * 1e-6 # km^2
    #print(glacier_area, area_cluster)

    # Method that uses KDTree index (best method: found to be same as exact method and ultra fast)
    run_method_KDTree_index = True
    if run_method_KDTree_index:

        td1 = time.time()

        # Extract all utm coordinates of points
        points_coords_array = np.column_stack((eastings, northings)) #(10000,2)

        # Extract all coordinates from the GeoSeries geometries
        geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in geoseries_geometries_epsg.geometry])

        # it appears that when no. geometries is low pykdtree_kdtree is faster, else sklearn KDTree is faster.
        if no_geometries_in_cluster > 2000:
            kdtree = sklearn.neighbors.KDTree(geoms_coords_array)
            print('using sklearn.neighbors.KDTree') if verbose else None
        else:
            kdtree = pykdtree.kdtree.KDTree(geoms_coords_array)
            print('using pykdtree.kdtree.KDTree') if verbose else None

        # Perform nearest neighbor search for each point and calculate minimum distances
        # k can be decreased for speedup to, e.g. k=200. I suspect that k can be somehow as low as 200, and in such
        # case probably pykdtree_kdtree is faster than KDTree
        distances, indices = kdtree.query(points_coords_array, k=min(k_max_geoms,len(geoseries_geometries_epsg)))
        if distances.ndim == 1: distances = distances.reshape(-1, 1) # needed for use_pykdtree_kdtree
        min_distances = np.min(distances, axis=1)

        min_distances /= 1000.

        td2 = time.time()
        print(f"Distances calculated with KDTree in {td2 - td1}") if verbose else None

    plot_minimum_distances = False
    if plot_minimum_distances:
        fig, ax = plt.subplots(figsize=(8,7))
        #ax.plot(*gl_geom.exterior.xy, color='blue')
        ax.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='r')  # first entry is outside border
        for geom in geoseries_geometries_epsg.loc[1:]:
            ax.plot(*geom.xy, lw=1, c='grey')
        #for geom in geoseries_geometries_epsg.loc[20:]:
        #    ax.plot(*geom.xy, lw=1, c='r')
        s1 = ax.scatter(x=points_coords_array[:,0], y=points_coords_array[:,1], s=1, c=min_distances, zorder=0)
        #s1 = ax.scatter(x=points_df['lons'], y=points_df['lats'], s=10, c=min_distances3, alpha=0.5, zorder=0)
        cbar = plt.colorbar(s1, ax=ax)
        cbar.set_label('Distance to closest ice free region (km)', labelpad=15, rotation=90, fontsize=16)
        ax.set_xlabel('Eastings (m)', fontsize=16)
        ax.set_ylabel('Northings (m)', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()

    # Method 2: geopandas spatial indexes (bad method and slow)
    run_method_geopandas_index = False
    if run_method_geopandas_index:
        min_distances = []
        sindex_id = geoseries_geometries_epsg.sindex
        for i, point_epsg in enumerate(geoseries_points_epsg):
            nearest_idx = sindex_id.nearest(point_epsg.bounds)
            nearest_geometries = geoseries_geometries_epsg.iloc[nearest_idx]
            min_distances_ = nearest_geometries.distance(point_epsg)
            min_idx = min_distances_.idxmin()
            min_dist = min_distances_.loc[min_idx]
            min_distances.append(min_dist / 1000.)

    # Method 3: vectorized version with CPU (exact method but slow)
    run_distances_with_geopandas_multicpu = False
    if run_distances_with_geopandas_multicpu:
        def calc_min_distance_to_multi_line(point, multi_line):
            min_dist = point.distance(multi_line)
            return min_dist

        td1 = time.time()
        multiline_geometries_epsg = MultiLineString(list(geoseries_geometries_epsg))
        args_list = [(point, multiline_geometries_epsg) for point in geoseries_points_epsg]
        min_distances = Parallel(n_jobs=-1)(delayed(calc_min_distance_to_multi_line)(*args) for args in args_list)
        min_distances = np.array(min_distances)
        min_distances /= 1000.  # km
        td2 = time.time()
        print(f"Distances using pandas distance and multicpu {td2 - td1}") if verbose else None

    # Method 4: not verctorized version (exact method but very slow)
    run_method_not_vectorized = False
    if run_method_not_vectorized:
        for (i, lon, lat, nunatak) in zip(points_df.index, points_df['lons'], points_df['lats'], points_df['nunataks']):

            # Make a check.
            easting, nothing, zonenum, zonelett, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
            if epsg != glacier_epsg:
                print(f"Note differet UTM zones. Point espg {epsg} and glacier center epsg {glacier_epsg}.") if verbose else None

            # Get shapely Point
            point_epsg = geoseries_points_epsg.iloc[i]

            # Calculate the distances between such point and all glacier geometries
            min_distances_point_geometries = geoseries_geometries_epsg.distance(point_epsg)
            min_dist = np.min(min_distances_point_geometries) # unit UTM: m

            # To debug we want to check what point corresponds to the minimum distance.
            debug_distance = True
            if debug_distance:
                min_distance_index = min_distances_point_geometries.idxmin()
                nearest_line = geoseries_geometries_epsg.loc[min_distance_index]
                nearest_point_on_line = nearest_line.interpolate(nearest_line.project(point_epsg))
                # print(f"{i} Minimum distance: {min_dist:.2f} meters.")

            # Fill dataset
            # note that the generated points cannot be in nunataks so distances are well defined
            points_df.loc[i, 'dist_from_border_km_geom'] = min_dist/1000.

            # Plot
            plot_calculate_distance = False
            if plot_calculate_distance:

                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
                for interior in gl_geom.interiors:
                    ax1.plot(*interior.xy, lw=1, c='blue')

                # Plot boundaries (only external periphery) of all glaciers in the cluster
                if list_cluster_RGIIds is not None:
                    for gl_neighbor_id in list_cluster_RGIIds:
                        gl_neighbor_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_neighbor_id]
                        gl_neighbor_geom = gl_neighbor_df['geometry'].item()  # glacier geometry Polygon
                        ax1.plot(*gl_neighbor_geom.exterior.xy, lw=1, c='orange', zorder=0)

                # intersects of glacier (need only for plotting purposes)
                gl_intersects = oggm.utils.get_rgi_intersects_entities([glacier_name], version='62')
                # Plot intersections of central glacier with its neighbors
                for k, intersect in enumerate(gl_intersects['geometry']):  # Linestring gl_intersects
                    ax1.plot(*intersect.xy, lw=1, color='k')

                # Plot intersections of all glaciers in the cluster
                if list_cluster_RGIIds is not None:
                    cluster_intersects = oggm.utils.get_rgi_intersects_entities(list_cluster_RGIIds,
                                                                                version='62')  # (need only for plotting purposes)
                else:
                    cluster_intersects = None

                if cluster_intersects is not None:
                    for k, intersect in enumerate(cluster_intersects['geometry']):
                        ax1.plot(*intersect.xy, lw=1, color='k') #np.random.rand(3)

                    # Plot cluster ice divides removed
                    if multipolygon:
                        polygons = list(cluster_geometry_no_divides_4326.item().geoms)
                        cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
                        cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
                        cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
                        for exterior in cluster_exterior_ring:
                            ax1.plot(*exterior.xy, lw=1, c='red', zorder=3)
                        for interior in cluster_interior_rings:
                            ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                    else:
                        ax1.plot(*cluster_geometry_no_divides_4326.item().exterior.xy, lw=1, c='red', zorder=3)
                        for interior in cluster_geometry_no_divides_4326.item().interiors:
                            ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                if nunatak: ax1.scatter(lon, lat, s=50, lw=2, c='b')
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


                if nunatak: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='b')
                else: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='r', ec='r')
                if debug_distance: ax2.scatter(*nearest_point_on_line.xy, s=50, lw=2, c='g')

                ax1.set_title('EPSG 4326')
                ax2.set_title(f'EPSG {glacier_epsg}')
                plt.show()

    points_df['dist_from_border_km_geom'] = min_distances
    points_df['Cluster_area'] = area_cluster
    points_df['Cluster_glaciers'] = no_glaciers_in_cluster
    points_df['Cluster_geometries'] = no_geometries_in_cluster
    tdist1 = time.time()
    tdist = tdist1 - tdist0
    print(f"Finished distance calculations.") if verbose else None

    """ Calculate distance_from_ocean """
    print(f"Calculating the distances from ocean... ") if verbose else None
    tdistocean0 = time.time()

    buffer = 1
    box_geoms = coastlines_dataframe.cx[llx-buffer:urx+buffer,lly-buffer:ury+buffer]

    #fig, ax = plt.subplots()
    #box_geoms.plot(ax=ax, linestyle='-', linewidth=1, facecolor='none', edgecolor='red')
    #geoseries_points_4326.plot(ax=ax, c='r', markersize=2)
    #plt.show()

    if len(box_geoms) == 0:
        # We are in the island case in Antarctica, e.g. -73.10288797227037 -105.166778923743
        # It may happen that no geometries gshhg are intercepted
        # In this case we fill dataframe with dist_from_border_km_geom
        points_df['dist_from_ocean'] = points_df['dist_from_border_km_geom']

    else:
        # Reproject to glacier_epsg. This is approximately 0.13 s and the main computational cost for this method
        box_geoms_epsg = box_geoms.to_crs(glacier_epsg)

        # Extract all coordinates of GeoSeries geometries
        geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in box_geoms_epsg.geometry.exterior])

        # Reprojecting very big geometries cause distortion. Let's remove these points. Is this necessary ?
        valid_coords_mask = (
                (geoms_coords_array[:, 0] >= -1e7) & (geoms_coords_array[:, 0] <= 1e7) &
                (geoms_coords_array[:, 1] >= -1e7) & (geoms_coords_array[:, 1] <= 1e7)
        )
        valid_coords = geoms_coords_array[valid_coords_mask]

        #fig, ax = plt.subplots()
        #box_geoms_epsg.plot(ax=ax, linestyle='-', linewidth=1, facecolor='none', edgecolor='k')
        #geoseries_points_epsg.plot(ax=ax, c='k', markersize=2)
        #plt.show()

        # using pykdtree.kdtree.KDTree
        kdtree_ocean = pykdtree.kdtree.KDTree(valid_coords)

        distances_ocean, _ = kdtree_ocean.query(points_coords_array, k=len(box_geoms))
        if distances_ocean.ndim == 1: distances_ocean = distances_ocean.reshape(-1, 1)
        min_distances_ocean = np.min(distances_ocean, axis=1)
        min_distances_ocean /= 1000.

        points_df['dist_from_ocean'] = min_distances_ocean

    tdistocean1 = time.time()
    tdistocean = tdistocean1 - tdistocean0
    print(f"Finished distance from ocean calculations.") if verbose else None

    # Show the result
    show_glacier_with_produced_points = False
    if show_glacier_with_produced_points:
        fig, axes = plt.subplots(2,3, figsize=(10,8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        for ax in (ax1, ax2):
            ax.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
            for interior in gl_geom.interiors:
                ax.plot(*interior.xy, lw=1, c='blue')
            for (lon, lat, nunatak) in zip(points_df['lons'], points_df['lats'], points_df['nunataks']):
                if nunatak: ax.scatter(lon, lat, s=50, lw=2, c='magenta', zorder=2)
                else: ax.scatter(lon, lat, s=50, lw=2, c='r', ec='r', zorder=1)

        # slope_lat
        im1 = dz_dlat_xar.plot(ax=ax1, cmap='gist_gray', vmin=np.nanmin(slope_lat_data),
                                  vmax=np.nanmax(slope_lat_data), zorder=0)
        s1 = ax1.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lat_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)

        # elevation
        im2 = focus.plot(ax=ax2, cmap='gist_gray', vmin=np.nanmin(elevation_data),
                                  vmax=np.nanmax(elevation_data), zorder=0)
        s2 = ax2.scatter(x=lons_xar, y=lats_xar, s=50, c=elevation_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_data), zorder=1)

        # vx
        if no_millan_data is False:
            im3 = focus_vx.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
            s3 = ax3.scatter(x=lons_crs, y=lats_crs, s=50, c=vx_data, ec=(1, 0, 0, 1), cmap='viridis',
                             vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data), zorder=1)
            s3_1 = ax3.scatter(x=lons_crs[np.argwhere(np.isnan(vx_data))], y=lats_crs[np.argwhere(np.isnan(vx_data))], s=50,
                               c='magenta', zorder=1)

        # farinotti
        if no_farinotti_data is False:
            im4 = file_glacier_farinotti.plot(ax=ax4, cmap='inferno', vmin=np.nanmin(file_glacier_farinotti),
                                              vmax=np.nanmax(file_glacier_farinotti))
            s4 = ax4.scatter(x=lons_crs_f, y=lats_crs_f, s=50, c=ith_f_data, ec=(1, 0, 0, 1), cmap='inferno',
                             vmin=np.nanmin(file_glacier_farinotti), vmax=np.nanmax(file_glacier_farinotti), zorder=1)
            s4_1 = ax4.scatter(x=lons_crs_f[np.argwhere(np.isnan(ith_f_data))],
                               y=lats_crs_f[np.argwhere(np.isnan(ith_f_data))], s=50, c='magenta', zorder=2)

        # distance
        if multipolygon:
            for i_poly in range(num_multipoly):
                ax5.plot(*geoseries_geometries_epsg.loc[i_poly].xy, lw=1, c='red')  # first num_multipoly are outside borders
            for inter in geoseries_geometries_epsg.loc[num_multipoly:]:  # all interiors if present
                ax5.plot(*inter.xy, lw=1, c='blue')
        else:
            ax5.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='red')  # first entry is outside border
            for inter in geoseries_geometries_epsg.loc[1:]:  # all interiors if present
                ax5.plot(*inter.xy, lw=1, c='blue')

        ax5.scatter(x=geoseries_points_epsg.x, y=geoseries_points_epsg.y, s=5, lw=2, cmap='cividis',
                    c=points_df['dist_from_border_km_geom'],  vmin=points_df['dist_from_border_km_geom'].min(),
                    vmax=points_df['dist_from_border_km_geom'].max())

        ax6.axis('off')

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------------------------------
    """ Add features """
    points_df['elevation_from_zmin'] = points_df['elevation'] - points_df['Zmin']
    points_df['deltaZ'] = points_df['Zmax'] - points_df['Zmin']

    # ---------------------------------------------------------------------------------------------
    """ Data imputation """
    t0_imputation = time.time()

    # Data imputation for any nan survived in Millan velocities.
    list_vel_cols_for_imputation = ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa']

    median_imputer = SimpleImputer(strategy='median')

    complete_velocity_missing = points_df[list_vel_cols_for_imputation].isna().all().all()
    partial_velocity_missing = points_df[list_vel_cols_for_imputation].isna().any().any()

    # 1. First level velocity imputation: glacier median
    if partial_velocity_missing and not complete_velocity_missing:
        print(f"Some or no velocity data missing. Nans found in v50: {points_df['v50'].isna().sum()}. Progressive imputation.") if verbose else None

        v50_before_knn = points_df['v50']

        points_df[list_vel_cols_for_imputation] = median_imputer.fit_transform(points_df[list_vel_cols_for_imputation])

        plot_velocity_field = False
        if plot_velocity_field:
            fig, (ax1, ax2) = plt.subplots(1,2)

            s1 = ax1.scatter(x=points_df['lons'], y=points_df['lats'], s=2,
                           c=v50_before_knn, norm=LogNorm(), cmap='viridis')
            ax1.scatter(x=points_df[v50_before_knn.isna()]['lons'], y=points_df[v50_before_knn.isna()]['lats'],
                       c='r', s=2)
            cbar1 = plt.colorbar(s1)

            s2 = ax2.scatter(x=points_df['lons'], y=points_df['lats'], s=2, c=points_df['v50'], norm=LogNorm(), cmap='viridis')
            ax2.scatter(x=points_df[points_df['v50'].isna()]['lons'], y=points_df[points_df['v50'].isna()]['lats'],
                       c='r', s=2)
            cbar2 = plt.colorbar(s2)
            plt.show()


    # 2. Second level velocity imputation: regional median
    elif complete_velocity_missing and partial_velocity_missing:
        print(f"No velocity data can be found for rgi {rgi} glacier {glacier_name} "
              f"at {cenLat} lat {cenLon} lon. Regional data imputation.") if verbose else None

        rgi_median_velocities = velocity_median_rgi(rgi=rgi) # 6-vector
        points_df[list_vel_cols_for_imputation] = rgi_median_velocities
        plot_velocity_field = False
        if plot_velocity_field:
            fig, ax = plt.subplots()
            s = ax.scatter(x=points_df['lons'], y=points_df['lats'], c=points_df['v50'], s=2, ) #norm=LogNorm()
            cbar = plt.colorbar(s)
            plt.show()

    # 3. No velocity imputation needed
    else:
        print('No velocity imputation needed.') if verbose else None
        plot_velocity_field = False
        if plot_velocity_field:
            fig, ax = plt.subplots()
            s = ax.scatter(x=points_df['lons'], y=points_df['lats'], c=points_df['v50'], s=2) #norm=LogNorm(),
            cbar = plt.colorbar(s)
            plt.show()

    # Make sure no column is object
    points_df[cols_millan] = points_df[cols_millan].astype('float64')


    # Imputation for smb (should be only needed when interpolating racmo)
    points_df['smb'] = median_imputer.fit_transform(points_df[['smb']])

    t1_imputation = time.time()
    timp = t1_imputation - t0_imputation
    # ---------------------------------------------------------------------------------------------
    """ Drop features and sanity check """

    # Drop features
    columns_vels_to_drop = ['vx', 'vy', 'vx_gf50', 'vx_gf100', 'vx_gf150', 'vx_gf300', 'vx_gf450', 'vx_gfa',
                       'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300', 'vy_gf450', 'vy_gfa',
                       'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy', ]

    columns_slope_to_drop = ['slope_lon', 'slope_lat', 'slope_lon_gf50', 'slope_lat_gf50',
                             'slope_lon_gf75', 'slope_lat_gf75', 'slope_lon_gf100', 'slope_lat_gf100',
                             'slope_lon_gf125', 'slope_lat_gf125', 'slope_lon_gf150', 'slope_lat_gf150',
                             'slope_lon_gf300', 'slope_lat_gf300', 'slope_lon_gf450', 'slope_lat_gf450',
                             'slope_lat_gfa', 'slope_lon_gfa']
    #points_df.drop(columns=columns_slope_to_drop, inplace=True) #columns_vels_to_drop

    print(f"Important: we have generated {points_df['ith_m'].isna().sum()} points where Millan ith is nan.") if verbose else None
    print(f"Important: we have generated {points_df['ith_f'].isna().sum()} points where Farinotti ith is nan.") if verbose else None

    # Sanity check
    # The only survived nans should be only in ith_m, ith_f
    # Check for the presence of nans in the generated dataset.
    assert points_df.drop(columns=['ith_m', 'ith_f']).isnull().any().any() == False, \
        "Nans in generated dataset other than in Millan/Farinotti ice thickness! Something to check."

    tend = time.time()

    if verbose:
        print(f"************** TIMES **************")
        print(f"Geometries generation: {tgeometries:.2f}")
        print(f"Points generation: {tgenpoints:.3f}")
        print(f"Millan: {tmillan:.2f}")
        print(f"Slope: {tslope:.2f}")
        print(f"Smb: {tsmb:.2f}")
        print(f"Temperature: {tera5:.3f}")
        print(f"Farinotti: {tfar:.3f}")
        print(f"Distances: {tdist:.2f}")
        print(f"Distances ocean: {tdistocean:.2f}")
        print(f"Imputation: {timp:.2f}")
        print(f"*******TOTAL FETCHING FEATURES in {tend - tin:.1f} sec *******")
    return points_df


if __name__ == "__main__":

    glacier_name =  'RGI60-17.06074'# 'RGI60-11.01450'# 'RGI60-19.01882' RGI60-02.05515
    # ultra weird: RGI60-02.03411 millan has ith but no ice velocity
    # 'RGI60-05.10315' #RGI60-09.00909

    generated_points_dataframe = populate_glacier_with_metadata(
                                            glacier_name=glacier_name,
                                            n=30000,
                                            k=10000,
                                            seed=42,
                                            verbose=True)