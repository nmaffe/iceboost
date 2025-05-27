import os, sys, time, warnings
from glob import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy
import sklearn.neighbors
import cupy as cp
import cupyx.scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
from rioxarray import merge
import networkx
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import pykdtree.kdtree
from sklearn.impute import SimpleImputer
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon, LineString, MultiLineString, box
from shapely.errors import GEOSException
from pyproj import Proj, Transformer, Geod
import utm

from create_rgi_mosaic_tanxedem import fetch_dem, create_glacier_tile_dem_mosaic
from utils_metadata import *
from imputation_policies import smb_elev_functs, smb_elev_functs_hugo, velocity_median_rgi
import misc as misc

"""
This program generates glacier metadata inside the glacier geometry. 
Input: glacier name (RGIId)
Output: pandas dataframe with features calculated for each generated point. 

# todo: it may be wise to return also other stuff, e.g. some geometries
# todo: also, i may decide not to interpolate farinotti and millan ith for speedup    
"""

def populate_glacier_with_metadata(glacier_name,
                                   config = None,
                                   rgi_products=None,
                                   rgi=None,
                                   mass_balance_df=None,
                                   version=None,
                                   coastlines_dataframe = None,
                                   link_rgi6_rg7_dataframe = None,
                                   seed=None,
                                   verbose=True,
                                   ):


    print(f"******* FETCHING FEATURES FOR GLACIER {glacier_name} *******") if verbose else None

    tin=time.time()

    # unpack config
    n_points_regression_single = config.n_points_regression_single
    n_points_regression_cluster = config.n_points_regression_cluster
    graph_max_layer_depth = config.graph_max_layer_depth
    resolution = config.resolutionXY

    rgi = int(rgi)

    # unpack rgi products
    rgi_glaciers, rgi_graph = rgi_products

    if version == '62':
        name_column_id = 'RGIId'
        name_column_name = 'Name'
    elif version == '70G':
        name_column_id = 'rgi_id'
        name_column_name = 'glac_name'
    else:
        raise ValueError(f"Error: id and-or version not supported.")

    if glacier_name not in rgi_glaciers[name_column_id].values:
        raise ValueError(f"Error: {glacier_name} not present in the glacier dataframe.")

    # get glacier geometry
    gl_df = rgi_glaciers.loc[rgi_glaciers[name_column_id]==glacier_name]
    gl_df.index = [glacier_name] # set the name as index (needed to recording glacier id for generated points)
    gl_geom = gl_df['geometry'].item()  # glacier geometry Polygon
    gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
    gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors]  # list of nunataks Polygons
    assert len(gl_df) == 1, "Glacier id is not unique."

    # Geodataframes of external boundary and all internal nunataks
    gl_geom_nunataks_gdf = gpd.GeoDataFrame(geometry=gl_geom_nunataks_list, crs="EPSG:4326")
    gl_geom_ext_gdf = gpd.GeoDataFrame(geometry=[gl_geom_ext], crs="EPSG:4326")

    # get some features
    glacier_area = gl_df['area'].item()             # km2
    glacier_perimeter = gl_df['perimeter'].item()   # m
    area_noice = gl_df['area_icefree'].item()       # unitless
    cenLon = gl_df['cen_lon'].item()                # degrees east
    cenLat = gl_df['cen_lat'].item()                # degrees north
    glacier_epsg = gl_df['cen_epsg'].item()         # epsg (int)
    glacier_lmax = gl_df['lmax'].item()             # m

    print(f"Glacier {glacier_name} found. Lat: {cenLat}, Lon: {cenLon}") if verbose else None

    tgeometries = time.time() - tin

    # Calculate cluster
    t_cluster0 = time.time()

    # Let's decide to run on cluster only on polar regions
    if config.deploy_mode == 'auto': deploy_mode = 'auto'
    elif config.deploy_mode == 'single': deploy_mode = 'single'
    else: raise ValueError(f"Deploy mode not recognized.")

    cluster_data = False
    if (deploy_mode == 'auto'):
            cluster_data = get_possible_cluster(rgi_graph, glacier_name, glacier_epsg, rgi, rgi_glaciers, name_column_id, config)

    if cluster_data is not False:
        # Case run on cluster
        cluster_area = (cluster_data['area'] * cluster_data['area']).sum() / cluster_data['area'].sum()
        cluster_perimeter = (cluster_data['perimeter'] * cluster_data['area']).sum() / cluster_data['area'].sum()
        cluster_lmax = (cluster_data['lmax'] * cluster_data['area']).sum() / cluster_data['area'].sum()
        list_cluster_RGIIds = cluster_data.index.tolist()
    else:
        # Case: run on normal glacier. Set up a cluster with connectivity 3 as usual
        list_cluster_RGIIds = find_cluster_with_graph(rgi_graph, glacier_name, max_depth=graph_max_layer_depth)

    # deployed_glaciers is the list of glacier IDs for model inference. In 'auto' mode,
    # list_cluster_RGIIds and deployed_glaciers are the same
    deployed_glaciers = [glacier_name] if cluster_data is False else list_cluster_RGIIds

    # return the ids we have consumed
    yield deployed_glaciers

    # list_cluster_RGIIds contains the IDs of the glaciers in the cluster. If we're running as a single glacier,
    # the cluster has a depth of 3. If we're running on auto, all glaciers in the cluster are contained.
    no_glaciers_in_cluster = len(list_cluster_RGIIds)

    # area of the cluster
    area_cluster = rgi_glaciers.loc[rgi_glaciers[name_column_id].isin(list_cluster_RGIIds), 'area'].sum()

    # Create Geopandas geodataframe of glacier geometries (boundary and nunataks)
    cluster_geometry_4326_separate = gpd.GeoDataFrame(geometry=rgi_glaciers.loc[
        rgi_glaciers[name_column_id].isin(list_cluster_RGIIds), 'geometry'], crs="EPSG:4326")

    # Set the glacier ids as the index (needed to keep track of which glacier each generated points belong to)
    cluster_geometry_4326_separate.index = list_cluster_RGIIds

    # Suppress the specific warning about buffering in a geographic CRS
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="Geometry is in a geographic CRS. Results from 'buffer' are likely incorrect")
        buffered_geometries = cluster_geometry_4326_separate.buffer(0.0004)

    # remove ice divides
    cluster_geometry_4326 = gpd.GeoSeries(buffered_geometries.union_all(method='unary'), crs="EPSG:4326")

    # cluster exterior and interior dataframes (likely remove with new grid-generation scheme)
    cluster_ext_gdf = gpd.GeoDataFrame(geometry=[Polygon(cluster_geometry_4326.iloc[0].exterior)], crs="EPSG:4326")
    cluster_nunataks_gdf = gpd.GeoDataFrame(
        geometry=[Polygon(interior) for interior in cluster_geometry_4326.iloc[0].interiors], crs="EPSG:4326")

    #fig, ax = plt.subplots()
    #cluster_geometry_4326_separate.plot(ax=ax, ec='r', fc='none')
    #cluster_geometry_4326.plot(ax=ax, ec='b', fc='none')
    #plt.show()

    print(f"Cluster (4326): {area_cluster:.5f} km2 and {no_glaciers_in_cluster} glaciers created in: {time.time() - t_cluster0:.3f}") if verbose else None

    # Generate points
    tp0 = time.time()
    if cluster_data is not False:
        #print(f"Running on cluster: {list_cluster_RGIIds}")
        print(f"Running on cluster") if verbose else None
        if config.mode_point_generation == 'random':
            points = generate_points(gdf_ext=cluster_ext_gdf, gdf_nuns=cluster_nunataks_gdf, seed=seed, n_points_regression=n_points_regression_cluster)
        elif config.mode_point_generation == 'grid':
            #points_df = generate_points(in_points_df=points_df, gdf=cluster_geometry_4326_separate, epsg_glacier=glacier_epsg, region=rgi)
            points_df = generate_points(gdf=cluster_geometry_4326_separate, epsg_glacier=glacier_epsg, region=rgi, resolution=resolution)
            #points_df = generate_points_on_grid_min_100meter(in_points_df=points_df,
            #                                                 gdf=cluster_geometry_4326_separate,
            #                                                 epsg=glacier_epsg,
            #                                                 region=rgi)
            #points = generate_points_on_grid(gdf_ext=cluster_ext_gdf, gdf_nuns=cluster_nunataks_gdf, max_points=n_points_regression_cluster)
        else: raise ValueError("Unsupported mode for data generation.")
        gl_geom = Polygon(cluster_geometry_4326.iloc[0]) # override (we need this if we have clustered)
        gl_geom_ext = Polygon(gl_geom.exterior)         # override
        gl_geom_nunataks_gdf = cluster_nunataks_gdf     # override
        gl_geom_ext_gdf = cluster_ext_gdf               # override (I guess we need this)
    else:
        print(f"Running single glacier") if verbose else None
        if config.mode_point_generation == 'random':
            points = generate_points(gdf_ext=gl_geom_ext_gdf, gdf_nuns=gl_geom_nunataks_gdf, seed=seed, n_points_regression=n_points_regression_single)
        elif config.mode_point_generation == 'grid':
            #points = generate_points_on_grid(gdf_ext=gl_geom_ext_gdf, gdf_nuns=gl_geom_nunataks_gdf, max_points=n_points_regression_single)
            #points_df = generate_points_on_grid_min_100meter(in_points_df=points_df,
            #                                                 gdf=gl_df,
            #                                                 epsg=glacier_epsg,
            #                                                 region=rgi)
            #points_df = generate_points(in_points_df=points_df, gdf=gl_df, epsg_glacier=glacier_epsg, region=rgi)
            points_df = generate_points(gdf=gl_df, epsg_glacier=glacier_epsg, region=rgi, resolution=resolution)
        else: raise ValueError("Unsupported mode for data generation.")

    plot_gen_points = False
    if plot_gen_points:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        cluster_geometry_4326.plot(ax=ax1, ec='k', fc='none')
        cluster_ext_gdf.plot(ax=ax2, ec='b', fc='none')
        if len(cluster_nunataks_gdf)>0: cluster_nunataks_gdf.plot(ax=ax2, ec='r', fc='none')
        ax2.scatter(x=points_df['lons'], y=points_df['lats'], s=1)
        plt.show()

    #fig, ax = plt.subplots()
    #gl_geom_ext_gdf.plot(ax=ax, ec='blue', fc='none')
    #if len(gl_geom_nunataks_gdf)>0: gl_geom_nunataks_gdf.plot(ax=ax, ec='red', fc='none')
    #ax.scatter(x=points['lons'], y=points['lats'], s=1, c='k')
    #plt.show()

    tp1 = time.time()
    tgenpoints = tp1-tp0
    print(f"We have generated {len(points_df)} points in {tgenpoints:.3f}") if verbose else None

    # Fill these features
    points_df['RGI'] = rgi
    if cluster_data is not False:
        points_df['Area']       = cluster_area      # km^2
        points_df['Perimeter']  = cluster_perimeter # m
        points_df['lmax']       = cluster_lmax      # m
    else:
        points_df['Area']       = glacier_area  # km^2
        points_df['Perimeter']  = glacier_perimeter # m
        points_df['lmax']       = glacier_lmax          # m
    points_df['Area_icefree']   = area_noice  # unitless
    points_df['Cluster_area']   = area_cluster # Note that if I run the cluster, bigger that depth=3, outside training space
    points_df['Cluster_glaciers'] = no_glaciers_in_cluster

    # Calculate the adaptive filter size based on the Area value
    sigma_af_min, sigma_af_max = 100.0, 2000.0
    # OLD
    #try:
    #    area_gl = points_df['Area'][0]
    #    lmax_gl = points_df['Lmax'][0]
    #    a = 1e6 * area_gl / (np.pi * 0.5 * lmax_gl)
    #    sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
    #except Exception as e:
    #    sigma_af = sigma_af_min

    # NEW (we use lmax and not Lmax from RGI)
    area_gl = points_df['Area'][0]
    lmax_gl = points_df['lmax'][0]
    a = 1e6 * area_gl / (np.pi * 0.5 * lmax_gl)
    sigma_af = int(np.clip(a, sigma_af_min, sigma_af_max))

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
                print(f"Found glacier in Millan tile: {file_ith}") if verbose else None
                break

        bedmachine_used = False

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
                    return points_df, bedmachine_used


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

            return points_df, bedmachine_used

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
                bedmachine_used = True
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
                return points_df, bedmachine_used

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

            return points_df, bedmachine_used

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

        bedmachine_used = False
        # Check if Millan ith interpolation is satisfactory. If not, try BedMachine v5
        millan_ith_nan_count_perc = np.isnan(points_df['ith_m']).sum() / len(points_df['ith_m'])
        #print(millan_ith_nan_count_perc)
        if millan_ith_nan_count_perc > .5:
            bedmachine_used = True
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

            #tile_ith.plot(cmap='turbo')
            #plt.show()

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

        return points_df, bedmachine_used

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

        return points_df, False

    if rgi == 5:
        points_df, bedmachine_used = fetch_millan_data_Gr(points_df)
    elif rgi == 19:
        points_df, bedmachine_used = fetch_millan_data_An(points_df)
    else:
        points_df, bedmachine_used = fetch_millan_data(points_df, rgi)

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

    deltalat = np.maximum(deltalat, 0.1) # Ensure deltalat and deltalon are at least 0.1
    deltalon = np.maximum(deltalon, 0.1) # Ensure deltalat and deltalon are at least 0.1

    lats_xar = xarray.DataArray(points_df['lats'])
    lons_xar = xarray.DataArray(points_df['lons'])

    eps = 5./3600 # useless

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

    # Get the glacier geometry in utm
    gl_geom_gdf_utm = gpd.GeoDataFrame(geometry=[gl_geom], crs="EPSG:4326").to_crs(epsg=glacier_epsg)
    gl_geom_utm_polygon = gl_geom_gdf_utm.geometry.iloc[0]

    # Clip the DEM in utm with the glacier geometry also in utm
    dem_glacier_utm = focus_utm.rio.clip([gl_geom_utm_polygon], glacier_epsg, drop=True, invert=False, all_touched=True)

    # Calculate glacier mean slope, aspect, curvature
    glacier_mean_slope_with_dem = xrspatial.slope(dem_glacier_utm).mean().item()
    glacier_mean_aspect_with_dem = xrspatial.aspect(dem_glacier_utm).mean().item()
    glacier_mean_curvature_with_dem = xrspatial.curvature(dem_glacier_utm).mean().item()

    points_df['slope'] = glacier_mean_slope_with_dem
    points_df['aspect'] = glacier_mean_aspect_with_dem
    points_df['curvature'] = glacier_mean_curvature_with_dem

    glacier_zmin_with_dem = np.nanmin(dem_glacier_utm.values)
    glacier_zmax_with_dem = np.nanmax(dem_glacier_utm.values)
    glacier_zmed_with_dem = np.nanmedian(dem_glacier_utm.values)

    plot_zmin_sanity = False
    if plot_zmin_sanity:
        valid_values = dem_glacier_utm.values[~np.isnan(dem_glacier_utm.values)]
        percentiles = np.percentile(valid_values, [1, 5, 50, 99])
        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.0 * iqr
        p1, p5, p50, p99 = percentiles[0], percentiles[1], percentiles[2], percentiles[3]
        std = np.std(valid_values)
        z_3s = p50 - 3 * std # forse z_3s e' l'opzione migliore e poi zmin = max(zmin, z_3s)
        print(f"1st percentile: {p1}")
        print(f"5th percentile: {p5}")
        print(f"lower_bound: {lower_bound}")
        print(f"z_3s: {z_3s}")
        fig, (ax1, ax2) = plt.subplots(1,2)
        dem_glacier_utm.plot(ax=ax1, cmap='bwr')
        ax2.hist(dem_glacier_utm.values.flatten(), bins=100)
        ax2.set_yscale('log')
        ax2.axvline(p1, color='red', linestyle='dashed')
        ax2.axvline(p5, color='blue', linestyle='dashed')
        ax2.axvline(p50, color='blue', linestyle='dashed')
        ax2.axvline(p99, color='blue', linestyle='dashed')
        ax2.axvline(lower_bound, color='yellow', linestyle='dashed')
        ax2.axvline(z_3s, color='orange', linestyle='dashed')
        plt.show()

    # Remove negative zmin outliers due to possible DEM artifacts.
    glacier_zmin_with_dem = max(0.0, glacier_zmin_with_dem)

    points_df['zmin'] = glacier_zmin_with_dem
    points_df['zmax'] = glacier_zmax_with_dem
    points_df['zmed'] = glacier_zmed_with_dem

    # Calculate the resolution in meters of the utm focus (resolutions in x and y are the same!)
    res_utm_metres = focus_utm.rio.resolution()[0]

    # Project the points onto the glacier projection
    eastings, northings = Transformer.from_crs("EPSG:4326", glacier_epsg).transform(points_df['lats'], points_df['lons'])

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

    #print(num_px_sigma_50)
    #print(num_px_sigma_75)
    #print(num_px_sigma_100)
    #print(num_px_sigma_125)
    #print(num_px_sigma_150)
    #print(num_px_sigma_300)
    #print(num_px_sigma_450)
    #print(num_px_sigma_af)
    #input('wait')

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
        #focus_filter_50_utm = convolve_fft(focus_utm.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_75_utm = convolve_fft(focus_utm.values, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_100_utm = convolve_fft(focus_utm.values, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_125_utm = convolve_fft(focus_utm.values, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_150_utm = convolve_fft(focus_utm.values, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_300_utm = convolve_fft(focus_utm.values, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_450_utm = convolve_fft(focus_utm.values, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #focus_filter_af_utm = convolve_fft(focus_utm.values, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        #print(np.sum(np.isnan(focus_utm.values)))

        # Step 1: Calculate padding width from (likely) the biggest kernel's dimensions
        pad_width = kernelaf.array.shape[0] // 2
        # Step 2: Pad the data
        padded_data = np.pad(focus_utm.values, pad_width, mode='edge')

        #focus_filter_50_utm = convolve_fft(focus_utm.values, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan)
        focus_filter_50_utm = convolve_fft(padded_data, kernel50, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_75_utm = convolve_fft(padded_data, kernel75, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_100_utm = convolve_fft(padded_data, kernel100, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_125_utm = convolve_fft(padded_data, kernel125, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_150_utm = convolve_fft(padded_data, kernel150, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_300_utm = convolve_fft(padded_data, kernel300, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_450_utm = convolve_fft(padded_data, kernel450, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]
        focus_filter_af_utm = convolve_fft(padded_data, kernelaf, nan_treatment='interpolate', preserve_nan=preserve_nans, boundary='fill', fill_value=np.nan, allow_huge=True)[pad_width:-pad_width, pad_width:-pad_width]

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

        #fig, ax = plt.subplots()
        #focus_filter_xarray_af_utm.plot(cmap='terrain')
        #plt.show()

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
        dz_dlat_np_af, dz_dlon_np_af = np.gradient(focus_filter_af_utm, -res_utm_metres, res_utm_metres, edge_order=2)

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

        #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
        #focus_utm.plot(ax=ax1, cmap='terrain')
        #focus_filter_xarray_450_utm.plot(ax=ax2, cmap='terrain')
        #focus_filter_xarray_af_utm.plot(ax=ax3, cmap='terrain')
        #slope_50_xar.plot(ax=ax4)
        #slope_af_xar.plot(ax=ax5)
        #slope_300_xar_after_dem_conv.plot(ax=ax3)
        #plt.show()

    # Calculate curvature and aspect using xrspatial
    # Note that artifacts appear if nans are present at the boundary
    curv_50 = xrspatial.curvature(focus_filter_xarray_50_utm)
    curv_100 = xrspatial.curvature(focus_filter_xarray_100_utm)
    curv_150 = xrspatial.curvature(focus_filter_xarray_150_utm)
    curv_300 = xrspatial.curvature(focus_filter_xarray_300_utm)
    curv_450 = xrspatial.curvature(focus_filter_xarray_450_utm)
    curv_af = xrspatial.curvature(focus_filter_xarray_af_utm)

    #fig, ax = plt.subplots()
    #curv_450.plot()
    #plt.show()

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

    #slope_50_xar.plot()
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
    curv_data_100 = curv_100.interp(y=northings_xar, x=eastings_xar, method='linear').data
    curv_data_150 = curv_150.interp(y=northings_xar, x=eastings_xar, method='linear').data
    curv_data_300 = curv_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
    curv_data_450 = curv_450.interp(y=northings_xar, x=eastings_xar, method='linear').data
    curv_data_af = curv_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
    #aspect_data_50 = aspect_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
    #aspect_data_300 = aspect_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
    #aspect_data_af = aspect_af.interp(y=northings_xar, x=eastings_xar, method='linear').data

    # Hugonnet mass balance
    try:
        if version == '62':
            glacier_dmdtda = mass_balance_df.at[glacier_name, 'dmdtda']
        elif version == '70G':
            rgi_id_6 = link_rgi6_rg7_dataframe.at[glacier_name, 'rgi_id_62']
            glacier_dmdtda = mass_balance_df.at[rgi_id_6, 'dmdtda']
    except: # impute the mean
        glacier_dmdtda = mass_balance_df['dmdtda'].median()
    print(f'Hugonnet mb: {glacier_dmdtda} m w.e/yr') if verbose else None


    # check if any nan in the interpolate data
    contains_nan = any(np.isnan(arr).any() for arr in [elevation_data, slope_50_data, slope_75_data, slope_100_data,
                                                       slope_125_data, slope_150_data, slope_300_data,
                                                       slope_450_data, slope_af_data,
                                                       curv_data_50, curv_data_100, curv_data_150, curv_data_300,
                                                       curv_data_450, curv_data_af,])
                                                       #aspect_data_50, aspect_data_300, aspect_data_af])

    if contains_nan:
        raise ValueError(f"Nan detected in elevation/slope calc. Check")

    # Add elevation min-max scaled to 0-1
    elevation_0_1 = normalized_elevation(h=elevation_data, Hmin=glacier_zmin_with_dem, Hmax=glacier_zmax_with_dem)

    # Fill zmin, zmax, zmed using tandemx interpolated elevation data
    #points_df['Zmin'] = np.min(elevation_data)
    #points_df['Zmax'] = np.max(elevation_data)
    #points_df['Zmed'] = np.median(elevation_data)

    # Fill dataframe with elevation and slopes
    points_df['elevation'] = elevation_data
    points_df['elevation_0_1'] = elevation_0_1
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
    points_df['curv_100'] = curv_data_100
    points_df['curv_150'] = curv_data_150
    points_df['curv_300'] = curv_data_300
    points_df['curv_450'] = curv_data_450
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
        if rgi==5:
            racmo_file = config.racmo_file_nc_greenland
        elif rgi==19:
            racmo_file = config.racmo_file_nc_antarctica
        else: raise ValueError('rgi value for RACMO smb calculation not recognized')

        racmo = rioxarray.open_rasterio(f'{racmo_file}')

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
    volumes_farinotti_df = pd.DataFrame(index=deployed_glaciers, columns=['vol_far']).rename_axis('ID')
    folder_rgi_farinotti = f"{config.farinotti_icethickness_dir}RGI60-{rgi:02d}/"

    for n, id in enumerate(deployed_glaciers):

        try: # Import farinotti ice thickness file. Note that it contains zero where ice not present.
            if version == '62':
                file_glacier_farinotti = rioxarray.open_rasterio(f'{folder_rgi_farinotti}{id}_thickness.tif',
                                                                 masked=False)
            elif version == '70G':
                rgi_id_6 = link_rgi6_rg7_dataframe.at[id, 'rgi_id_62']
                file_glacier_farinotti = rioxarray.open_rasterio(f'{folder_rgi_farinotti}{rgi_id_6}_thickness.tif',
                                                             masked=False)

            res_farinotti = file_glacier_farinotti.rio.resolution()[0]
            vol_farinotti_id = 1.e-9 * (res_farinotti ** 2) * np.nansum(file_glacier_farinotti.values)
            volumes_farinotti_df.at[id, 'vol_far'] = vol_farinotti_id

            file_glacier_farinotti = file_glacier_farinotti.where(file_glacier_farinotti != 0.0) # replace zeros with nans.
            file_glacier_farinotti.rio.write_nodata(np.nan, inplace=True)

            transformerF = Transformer.from_crs("EPSG:4326", file_glacier_farinotti.rio.crs)
            lons_crs_f, lats_crs_f = transformerF.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())


            ith_f_data = file_glacier_farinotti.interp(y=xarray.DataArray(lats_crs_f), x=xarray.DataArray(lons_crs_f),
                                                       method="nearest").data.squeeze()


            mask_valid = ~np.isnan(ith_f_data)  # Mask for non-NaN values in ith_f_data
            points_df.loc[mask_valid, 'ith_f'] = ith_f_data[mask_valid]
            #print(np.isnan(ith_f_data).sum(), len(ith_f_data))

            #points_df['ith_f'] = ith_f_data
            #print(f"From Farinotti ith interpolation in {id} we have generated {np.isnan(ith_f_data).sum()} nans.") if verbose else None

            show_farinotti = False
            if show_farinotti:
                fig, (ax1, ax2) = plt.subplots(1,2)
                s1 = ax1.scatter(x=lons_crs_f, y=lats_crs_f, s=1, c=ith_f_data)
                s2 = ax1.scatter(x=lons_crs_f[np.isnan(ith_f_data)], y=lats_crs_f[np.isnan(ith_f_data)], s=1, c='magenta')
                file_glacier_farinotti.plot(ax=ax2, cmap='Blues')
                cmbar = plt.colorbar(s1)
                plt.show()

        except:
            print(f"No Farinotti data can be found for rgi {rgi} glacier {id} or Farinotti interpolation is problematic.") if verbose else None

    tfar2 = time.time()
    tfar = tfar2-tfar1

    """ Calculate distance_from_border """
    print(f"Calculating the distances using glacier geometries... ") if verbose else None
    tdist0 = time.time()

    is_inside_ice_sheet = False
    if rgi in [5, 19]:
        if rgi == 5:
            # Get Greenland ice sheet boundary. EPSG:3413
            ice_sheet = gpd.read_file(config.ice_sheet_boundary_greenland_shp)
        if rgi == 19:
            # Get Antarctic ice sheet boundary. EPSG:3031
            ice_sheet = gpd.read_file(config.ice_sheet_boundary_antarctica_shp)

        #print(ice_sheet)

        ice_sheet_epsg = ice_sheet.to_crs(epsg=glacier_epsg)

        # calculate if the glacier is mostly inside the ice sheet
        glacier_ext_epsg = gl_geom_ext_gdf.to_crs(epsg=glacier_epsg)
        intersection = glacier_ext_epsg.intersection(ice_sheet_epsg)

        area_intersection = intersection.area.item() * 1e-6
        area_glacier_ext = glacier_ext_epsg.area.item() * 1e-6

        # decide if glacier is inside the ice sheet if area contained for at least 90%
        is_inside_ice_sheet = (area_intersection / area_glacier_ext) > 0.9
        print(f"Glacier inside ice sheet: {is_inside_ice_sheet}") if verbose else None

    tgeoms0 = time.time()
    # Reproject cluster geometries in utm
    cluster_geometry_epsg = cluster_geometry_4326.to_crs(epsg=glacier_epsg)

    # NEW
    # Ensure the geometry is a valid MultiPolygon (it will automatically handle both Polygon and MultiPolygon)
    geometries = list(
        cluster_geometry_epsg.item().geoms) if cluster_geometry_epsg.item().geom_type == 'MultiPolygon' else [
        cluster_geometry_epsg.item()]

    # Create exterior and interior rings, regardless of whether it's a single Polygon or a MultiPolygon
    cluster_exterior_ring = [polygon.exterior for polygon in geometries]  # List of LinearRing objects
    cluster_interior_rings = [ring for polygon in geometries for ring in
                              polygon.interiors]  # List of all interior rings

    #OLD
    '''
    if cluster_geometry_epsg.item().geom_type == 'Polygon':
        cluster_exterior_ring = [cluster_geometry_epsg.item().exterior]  # shapely.geometry.polygon.LinearRing
        cluster_interior_rings = list(cluster_geometry_epsg.item().interiors)  # shapely.geometry.polygon.LinearRing
        multipolygon = False
    elif cluster_geometry_epsg.item().geom_type == 'MultiPolygon':
        polygons = list(cluster_geometry_epsg.item().geoms)
        cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
        num_multipoly = len(cluster_exterior_ring)
        cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
        cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
        multipolygon = True
    else: raise ValueError("Unexpected geometry type. Please check.")
    '''

    # Create a geoseries of all external and internal geometries
    if is_inside_ice_sheet is False:
        geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings, crs=glacier_epsg)
    else:
        geoseries_geometries_epsg = gpd.GeoSeries([ice_sheet_epsg.geometry.iloc[0].exterior] + cluster_interior_rings, crs=glacier_epsg)
    no_geometries_in_cluster = len(geoseries_geometries_epsg)

    #fig, (ax1, ax2) = plt.subplots(1,2)
    #cluster_geometry_epsg.geometry.exterior.plot(ax=ax1, ec='k', fc='none')
    #geoseries_geometries_epsg.plot(ax=ax2, ec='k', fc='none')
    #plt.show()
    print(f"Cluster (UTM): {no_geometries_in_cluster} geometries created in: {time.time()-tgeoms0:.3f}") if verbose else None

    #fig, ax = plt.subplots(figsize=(8, 7))
    #ax.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='r')  # first entry is outside border
    #for geom in geoseries_geometries_epsg.loc[1:]:
    #    ax.plot(*geom.xy, lw=1, c='grey')
    #plt.show()

    # Method that uses KDTree index (best method: found to be same as exact method and ultra fast)
    run_method_KDTree_index = True
    if run_method_KDTree_index:

        td1 = time.time()

        # Extract all utm coordinates of points
        points_coords_array = np.column_stack((eastings, northings)) #(N,2)

        if geoseries_geometries_epsg.has_z.any():
            # Remove the third dimension by stripping out the z-values.
            geoms_coords_array = np.concatenate(geoseries_geometries_epsg.geometry.apply(lambda geom: np.array(geom.xy).T))

        else:
            geoms_coords_array = np.concatenate(geoseries_geometries_epsg.geometry.apply(lambda geom: np.array(geom.coords)))

        #kdtree = sklearn.neighbors.KDTree(geoms_coords_array)
        kdtree = pykdtree.kdtree.KDTree(geoms_coords_array)
        distances, _ = kdtree.query(points_coords_array, k=1)
        assert distances.ndim == 1, "Bad distances matrix."
        min_distances = distances / 1000.

        td2 = time.time()
        print(f"Distances calculated with KDTree in {td2 - td1}") if verbose else None

    plot_minimum_distances = False
    if plot_minimum_distances:
        fig, ax = plt.subplots(figsize=(8,7))
        #ax.plot(*gl_geom.exterior.xy, color='blue')
        #ax.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='r')  # first entry is outside border
        #for geom in geoseries_geometries_epsg.loc[1:]:
        #    ax.plot(*geom.xy, lw=1, c='grey')
        #ice_sheet_epsg.plot(ax=ax, edgecolor='red', facecolor='none')
        geoseries_geometries_epsg.loc[[0]].plot(ax=ax, edgecolor='blue', facecolor='none')
        if len(geoseries_geometries_epsg)>1:
            geoseries_geometries_epsg.loc[1:].plot(ax=ax, edgecolor='orange', facecolor='none')

        #glacier_ext_epsg.plot(ax=ax, edgecolor='green', facecolor='none', linewidth=1, zorder=2)

        #geoseries_geometries_epsg.plot(ax=ax)
        #sgeom = ax.scatter(x=geoms_coords_array[:, 0], y=geoms_coords_array[:, 1], c='r', zorder=0)

        s1 = ax.scatter(x=points_coords_array[:,0], y=points_coords_array[:,1], s=1, c=min_distances, zorder=0)
        #s1 = ax.scatter(x=points_df['lons'], y=points_df['lats'], s=10, c=min_distances, alpha=0.5, zorder=0)
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
        from joblib import Parallel, delayed
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


    points_df['dist_from_border_km_geom'] = min_distances
    points_df['Cluster_geometries'] = no_geometries_in_cluster
    tdist1 = time.time()
    tdist = tdist1 - tdist0
    print(f"Finished distance calculations.") if verbose else None

    """ Calculate distance_from_ocean """
    print(f"Calculating the distances from ocean... ") if verbose else None
    tdistocean0 = time.time()

    # Greenland
    if rgi == 5:
        coastal_geoms = gpd.read_file(config.ice_sheet_coastlines_greenland_gpkg)  # EPSG:3413
        assert len(coastal_geoms) == 1, "The coastal geometry should be a 1-line multipolygon"

        coastal_geoms_epsg = coastal_geoms.to_crs(epsg=glacier_epsg)

        glacier_center_epsg = gpd.GeoDataFrame(geometry=gpd.points_from_xy([cenLon], [cenLat]), crs="EPSG:4326").to_crs(glacier_epsg)
        is_inside_coastal_geoms = glacier_center_epsg.geometry.iloc[0].within(coastal_geoms_epsg.geometry.iloc[0])
        is_outside_coastal_geoms = not is_inside_coastal_geoms

        print(f"The glacier is inside from coastal geometries: {is_inside_coastal_geoms}") if verbose else None

    # Antarctica
    elif rgi == 19:
        coastal_geoms = gpd.read_file(config.ice_sheet_coastlines_antarctica_gpkg) # EPSG:3031
        assert len(coastal_geoms) == 1, "The coastal geometry should be a 1-line multipolygon"

        coastal_geoms_epsg = coastal_geoms.to_crs(epsg=glacier_epsg)

        glacier_center_epsg = gpd.GeoDataFrame(geometry=gpd.points_from_xy([cenLon], [cenLat]), crs="EPSG:4326").to_crs(glacier_epsg)
        is_inside_coastal_geoms = glacier_center_epsg.geometry.iloc[0].within(coastal_geoms_epsg.geometry.iloc[0])
        is_outside_coastal_geoms = not is_inside_coastal_geoms

        print(f"The glacier is inside from coastal geometries: {is_inside_coastal_geoms}") if verbose else None

    # All other regions
    else:
        buffer = 1
        llx, lly, urx, ury = gl_geom.bounds  # geometry bounds
        coastal_geoms = coastlines_dataframe.cx[llx-buffer:urx+buffer,lly-buffer:ury+buffer] # EPSG:4326
        glacier_center_4326 = gpd.GeoDataFrame(geometry=gpd.points_from_xy([cenLon], [cenLat]), crs="EPSG:4326")
        is_inside_coastal_geoms = coastal_geoms.geometry.contains(glacier_center_4326.geometry.iloc[0]).any() # FASTER
        #is_inside_coastal_geoms = glacier_center_4326.geometry.iloc[0].within(coastal_geoms.union_all(method='unary')) # SLOWER
        is_outside_coastal_geoms = not is_inside_coastal_geoms

        print(f"The glacier is inside from coastal geometries: {is_inside_coastal_geoms}") if verbose else None
        # OLD - too restrictive
        # is_outside_coastal_geoms = not any(gl_geom.within(box_geom) for box_geom in coastal_geoms.geometry)

        #fig, ax = plt.subplots()
        #coastal_geoms.plot(ax=ax, linestyle='-', linewidth=1, facecolor='none', edgecolor='red')
        #ax.plot(*gl_geom.exterior.xy, "k-")
        #gl_df.plot(ax=ax)
        #plt.show()

    # The fact that if the glacier is outside the coastal geometries we use the distance from border
    # is an approximate solution. It may be improved.
    if len(coastal_geoms) == 0 or is_outside_coastal_geoms:
        points_df['dist_from_ocean'] = points_df['dist_from_border_km_geom']

    else:
        # Reproject to glacier_epsg. This is approximately 0.13 s and the main computational cost for this method
        coastal_geoms_epsg = coastal_geoms.to_crs(epsg=glacier_epsg)

        coastal_geoms_epsg = coastal_geoms_epsg.geometry.explode(index_parts=True)

        # Extract all coordinates of GeoSeries geometries (0.02 s)
        geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in coastal_geoms_epsg.geometry.exterior])

        # Reprojecting very big geometries cause distortion. Let's remove these points. (0.008s)
        # Is this necessary ?
        valid_coords_mask = (
                (geoms_coords_array[:, 0] >= -1e7) & (geoms_coords_array[:, 0] <= 1e7) &
                (geoms_coords_array[:, 1] >= -1e7) & (geoms_coords_array[:, 1] <= 1e7)
        )
        valid_coords = geoms_coords_array[valid_coords_mask]

        #fig, ax = plt.subplots()
        #coastal_geoms_epsg.plot(ax=ax, linestyle='-', linewidth=1, facecolor='none', edgecolor='k')
        #geoseries_points_epsg.plot(ax=ax, c='k', markersize=2)
        #plt.show()

        kdtree_ocean = pykdtree.kdtree.KDTree(valid_coords)
        distances_ocean, _ = kdtree_ocean.query(points_coords_array, k=1)
        assert distances_ocean.ndim == 1, "Bad ocean distances vector."
        min_distances_ocean = distances_ocean / 1000.

        points_df['dist_from_ocean'] = min_distances_ocean

    plot_dist_from_ocean = False
    if plot_dist_from_ocean:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(geoms_coords_array[:, 0], geoms_coords_array[:, 1], s=1, c='k')
        ax.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='r')  # first entry is outside border
        for geom in geoseries_geometries_epsg.loc[1:]:
            ax.plot(*geom.xy, lw=1, c='grey')
        s1 = ax.scatter(x=points_coords_array[:, 0], y=points_coords_array[:, 1], s=1, c=points_df['dist_from_ocean'], zorder=0)
        cbar = plt.colorbar(s1, ax=ax)
        cbar.set_label('Distance to ocean (km)', labelpad=15, rotation=90, fontsize=16)
        ax.set_xlabel('Eastings (m)', fontsize=16)
        ax.set_ylabel('Northings (m)', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()

    tdistocean1 = time.time()
    tdistocean = tdistocean1 - tdistocean0
    print(f"Finished distance from ocean calculations.") if verbose else None


    # ---------------------------------------------------------------------------------------------
    """ Add features """
    points_df['elevation_from_zmin'] = points_df['elevation'] - points_df['zmin']
    points_df['elevation_to_zmax'] =  points_df['zmax'] - points_df['elevation']
    points_df['deltaz'] = points_df['zmax'] - points_df['zmin']
    # ---------------------------------------------------------------------------------------------
    """ Data imputation """
    t0_imputation = time.time()

    # Data imputation for any nan survived in the velocity features.
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

    # Imputation for slope, aspect and curvature (when xarray struggles bad geometry and produces nans, i.e. with RGI60-19.01285
    points_df['slope'] = points_df['slope'].fillna(points_df['slope100'].median())
    points_df['aspect'] = points_df['aspect'].fillna(0)
    points_df['curvature'] = points_df['curvature'].fillna(0)

    t1_imputation = time.time()
    timp = t1_imputation - t0_imputation
    # ---------------------------------------------------------------------------------------------
    """ Drop features and sanity check """

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

    # Let's create a dataframe in which we store some information for each glacier we have processed.
    # The ID is the glacier name.
    # Note that the order of the index (deployed_glaciers) is random.

    # Extract the names in the same order as deployed_glaciers
    filtered_df = rgi_glaciers.set_index(name_column_id).reindex(deployed_glaciers)
    popular_names = filtered_df[name_column_name].to_list()


    if cluster_data is False:
        info_df = pd.DataFrame({
            'Area': points_df['Area'].mean(),
            'Name': popular_names,
            'bedmachine': bedmachine_used,
            'vol_far': volumes_farinotti_df['vol_far']}, index=deployed_glaciers).rename_axis('ID')
    else:
        info_df = pd.DataFrame({
            'Area': cluster_data['area'],
            'Name': popular_names,
            'bedmachine': bedmachine_used,
            'vol_far': volumes_farinotti_df['vol_far']}, index=deployed_glaciers).rename_axis('ID')

    #print(info_df)

    yield info_df, points_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to yaml config file")
    args = parser.parse_args()

    config = misc.get_config(args.config)  # import from config.yaml

    glacier_name = 'RGI60-05.10315'  # 'RGI60-11.01450'# 'RGI60-19.01882' RGI60-02.05515
    # ultra weird: RGI60-02.03411 millan has ith but no ice velocity
    # 'RGI60-05.10315' #RGI60-09.00909 RGI60-05.10315 RGI60-05.10315
    # 'RGI60-19.01285' bad slope
    # RGI60-07.01394 RGI60-07.00027 RGI60-05.13501

    test_glacier_rgi, version = get_version_and_rgi_from_id(glacier_name)
    rgi_products = get_rgi_products(test_glacier_rgi, version=version)
    coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
    link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')

    data_generator = populate_glacier_with_metadata(glacier_name=glacier_name,
                                                  config=config,
                                                  rgi_products=rgi_products,
                                                  rgi=test_glacier_rgi,
                                                  version=version,
                                                  coastlines_dataframe=coastline_dataframe,
                                                  link_rgi6_rg7_dataframe=link_ids_rgi6_rgi7,
                                                  seed=42,
                                                  verbose=True,
                                                  )

    deployed_ids = data_generator.send(None)

    info, data = data_generator.send(None)
    print(deployed_ids)


