import argparse, time
import os, yaml
import random
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
import pandas as pd
import earthpy.spatial
import geopandas as gpd
from glob import glob
import xarray, rioxarray
from oggm import utils
import imageio

from scipy import stats
from scipy.interpolate import griddata, NearestNDInterpolator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
import catboost as cb
import optuna
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata, get_rgi_products, get_coastline_dataframe
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import *
import misc as misc
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to yaml config file")
args = parser.parse_args()

config = misc.get_config(args.config)  # import from config.yaml

file_deploy = pd.read_csv(f'{config.model_input_dir}{config.filename_csv_deploy}', index_col='rgi')
all_glacier_ids = file_deploy.values.flatten().tolist()

glathida_rgis = pd.read_csv(config.metadata_csv_file, low_memory=False)
glathida_rgis.loc[glathida_rgis['RGIId'] == 'RGI60-19.01406', 'THICKNESS'] /= 10.

# Load the model(s)
model_xgb_filename = config.model_input_dir + config.model_filename_xgb
iceboost_xgb = xgb.Booster()
iceboost_xgb.load_model(model_xgb_filename)

model_cat_filename = config.model_input_dir + config.model_filename_cat
iceboost_cat = cb.CatBoostRegressor()
iceboost_cat.load_model(model_cat_filename, format='cbm')

# *********************************************
# Model deploy
# *********************************************

#glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-11.01450', rgi=5, area=100, seed=None)
run_deploy_from_csv_list = True
if run_deploy_from_csv_list:
    for n, glacier_name_for_generation in enumerate(tqdm(all_glacier_ids)):

        glacier_name_for_generation = 'RGI60-19.01410' #'RGI60-01.13696'# 'RGI60-11.01450' #'RGI60-13.33257'
        # RGI60-07.00832
        print(n, glacier_name_for_generation)
        #if f"{glacier_name_for_generation}.png" in os.listdir(f"{config.model_output_results_dir}"):
        #    print(f"{glacier_name_for_generation} already in there.")
        #    continue

        test_glacier_rgi = glacier_name_for_generation[6:8]
        rgi_products = get_rgi_products(test_glacier_rgi)
        coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)

        test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                                      config=config,
                                                      rgi_products=rgi_products,
                                                      coastlines_dataframe=coastline_dataframe,
                                                      seed=42,
                                                      verbose=True)

        h_wgs84 = test_glacier['elevation'].to_numpy()
        lats = test_glacier['lats'].to_numpy()
        lons = test_glacier['lons'].to_numpy()
        h_egm2008 = calc_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)

        # Begin to extract all necessary things to plot the result
        oggm_rgi_shp = glob(f"{config.oggm_dir}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
        glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()
        glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['Area'].item()
        exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
        x0, y0, x1, y1 = exterior_ring.bounds
        dx, dy = x1 - x0, y1 - y0
        glacier_nunataks_list = [nunatak for nunatak in glacier_geometry.interiors]

        swlat = test_glacier['lats'].min()
        swlon = test_glacier['lons'].min()
        nelat = test_glacier['lats'].max()
        nelon = test_glacier['lons'].max()
        deltalat = np.abs(swlat - nelat)
        deltalon = np.abs(swlon - nelon)
        eps = 5./3600
        focus_mosaic_tiles = create_glacier_tile_dem_mosaic(minx=swlon - (deltalon + eps),
                                    miny=swlat - (deltalat + eps),
                                    maxx=nelon + (deltalon + eps),
                                    maxy=nelat + (deltalat + eps),
                                     rgi=test_glacier_rgi, path_tandemx=config.tandemx_dir)
        focus = focus_mosaic_tiles.squeeze()

        X_test_glacier = test_glacier[config.features]
        y_test_glacier_m = test_glacier[config.millan]
        y_test_glacier_f = test_glacier[config.farinotti]

        no_millan_data = np.isnan(y_test_glacier_m).all()
        no_farinotti_data = np.isnan(y_test_glacier_f).all()

        dtest = xgb.DMatrix(data=X_test_glacier)

        y_preds_glacier_xgb = iceboost_xgb.predict(dtest)
        y_preds_glacier_cat = iceboost_cat.predict(X_test_glacier)

        # ensemble
        y_preds_glacier = 0.5 * (y_preds_glacier_xgb + y_preds_glacier_cat)

        # Set negative predictions to zero
        y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

        # Calculate the glacier volume using the 3 models
        vol_ML, err_vol_ML, _ = calc_volume_glacier(y1=y_preds_glacier_xgb, y2=y_preds_glacier_cat,
                                                 area=glacier_area, h_egm2008=h_egm2008)
        vol_millan = calc_volume_glacier(y1=y_test_glacier_m, area=glacier_area)
        vol_farinotti = calc_volume_glacier(y1=y_test_glacier_f, area=glacier_area)
        print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, "
              f"volML: {vol_ML:.4g} km3 "
              f"volMil: {vol_millan:.4g} km3 "
              f"volFar: {vol_farinotti:.4g} km3")

        print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

        vmin = min(y_preds_glacier)
        vmax = max(y_preds_glacier)

        # 2d interpolation
        NNinterpolator = NearestNDInterpolator(np.column_stack((lons, lats)), y_preds_glacier)

        resolution = 3./3600
        n_bins_lon = max(10, int(dx/resolution))
        n_bins_lat = max(10, int(dy/resolution))
        lon_grid, lat_grid = np.meshgrid(np.linspace(x0, x1, n_bins_lon),np.linspace(y0, y1, n_bins_lat))

        thickness_grid = NNinterpolator(lon_grid, lat_grid)

        # create an xarray DataArray
        data_array = xarray.DataArray(
            thickness_grid,
            coords=[('y', lat_grid[:, 0]), ('x', lon_grid[0, :])],
            name='thickness'
        ).rio.write_crs("EPSG:4326", inplace=True).rio.set_nodata(np.nan, inplace=True)
        data_array = data_array.rio.clip(geometries=[glacier_geometry], crs="EPSG:4326", drop=False, invert=False, all_touched=False)

        plot_for_gif = False
        if plot_for_gif:

            fig = plt.figure(figsize=(9, 8), facecolor='none')
            gs = GridSpec(1, 2, width_ratios=[1, 0.05])  # Adjust the width ratios

            # Create the axes
            ax = fig.add_subplot(gs[0])
            cax = fig.add_subplot(gs[1])  # Colorbar axis

            dx, dy = x1 - x0, y1 - y0
            hillshade = copy.deepcopy(focus)
            hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
            hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

            im1 = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

            ax.plot(*exterior_ring.xy, c='k')
            for nunatak in glacier_nunataks_list:
                ax.plot(*nunatak.xy, c='k', lw=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)
            ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)
            ax.set_title('')

            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.tick_params(axis='both', labelsize=16)

            # Frame 0
            # Create a fake scatter plot (empty) just for the colorbar in frame 0
            #s1 = ax.scatter([], [], s=1, c=[], cmap='jet', zorder=1, vmin=vmin, vmax=vmax)
            # Frame 1
            s1 = ax.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
                                            cmap='jet', label='ML', zorder=1, vmin=vmin, vmax=vmax)

            # Create the colorbar
            cbar1 = plt.colorbar(s1, cax=cax)
            cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)
            cbar1.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
            cbar1.ax.tick_params(labelsize=16)

            plt.tight_layout()
            plt.show()

        plot_fancy_ML_prediction = False
        if plot_fancy_ML_prediction:
            fig, axes = plt.subplots(1,2, figsize=(8,6))
            ax, ax3 = axes
            hillshade = copy.deepcopy(focus)
            hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
            hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)

            im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

            s1 = ax.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
                             cmap='jet', label='ML', zorder=1, vmin=vmin,vmax=vmax)
            s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                    cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin,vmax=vmax)


            #s2 = ax2.contourf(lon_grid, lat_grid, thickness_grid, levels=100, cmap='jet')
            #cbar2 = plt.colorbar(s2, ax=ax2)
            #cbar2.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)

            im = hillshade.plot(ax=ax3, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
            im3 = data_array.plot(ax=ax3, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)

            cbar = plt.colorbar(s1, ax=ax)
            cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
            cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            for ax in (ax,): #axes
                ax.plot(*exterior_ring.xy, c='k')
                for nunatak in glacier_nunataks_list:
                    ax.plot(*nunatak.xy, c='k', lw=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)
                ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)
                ax.set_title('')

                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax.tick_params(axis='both', labelsize=16)

            plt.tight_layout()
            #plt.savefig('/home/maffe/Downloads/RGI60-1313574_CCAI.png', dpi=200)
            plt.show()


        plot_fancy_ML_Mil_Far_prediction = True
        if plot_fancy_ML_Mil_Far_prediction:
            #fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 6))
            fig = plt.figure(figsize=(15, 6))
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust the width ratios

            # Create the axes
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            cax = fig.add_subplot(gs[3])  # Colorbar axis

            dx, dy = x1 - x0, y1 - y0
            hillshade = copy.deepcopy(focus)
            hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
            hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

            im1 = hillshade.plot(ax=ax1, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
            im2 = hillshade.plot(ax=ax2, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
            im3 = hillshade.plot(ax=ax3, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

            s1 = ax1.scatter(x=lons, y=lats, s=2, c=y_preds_glacier, cmap='jet', label='ML', vmin=vmin, vmax=vmax)
            if not no_millan_data:
                s2 = ax2.scatter(x=lons, y=lats, s=2, c=y_test_glacier_m, cmap='jet',
                                 label='Millan', vmin=vmin, vmax=vmax)
            if not no_farinotti_data:
                s3 = ax3.scatter(x=lons, y=lats, s=2, c=y_test_glacier_f, cmap='jet',
                                 label='Farinotti', vmin=vmin, vmax=vmax)

            ax1.set_title(f"IceBoost: {vol_ML:.4g} km$^3$", fontsize=16)
            ax2.set_title(f"Model1: {vol_millan:.4g} km$^3$", fontsize=16)
            ax3.set_title(f"Model2: {vol_farinotti:.4g} km$^3$", fontsize=16)

            for ax in (ax1, ax2, ax3):
                ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                        cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)

            cbar1 = plt.colorbar(s1, cax=cax)#ax=ax1)
            cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)#vmax=vmax
            cbar1.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
            cbar1.ax.tick_params(labelsize=16)#11

            '''
            if not no_millan_data:
                cbar2 = plt.colorbar(s2, ax=ax2)
                cbar2.mappable.set_clim(vmin=vmin, vmax=vmax)
                cbar2.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
                cbar2.ax.tick_params(labelsize=11)
            if not no_farinotti_data:
                cbar3 = plt.colorbar(s3, ax=ax3)
                cbar3.mappable.set_clim(vmin=vmin, vmax=vmax)
                cbar3.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
                cbar3.ax.tick_params(labelsize=11)
            '''

            for ax in (ax1, ax2, ax3):
                ax.plot(*exterior_ring.xy, c='k')
                for nunatak in glacier_nunataks_list:
                    ax.plot(*nunatak.xy, c='k', lw=0.8)

                # ax.legend(fontsize=14, loc='upper left')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)#14
                ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)#14
                ax.tick_params(axis='both', labelsize=16)#12
                #ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax2.axis('off')
            ax3.axis('off')

            plt.tight_layout()

            if config.deploy_save_figs:
                plt.savefig(f"{config.model_output_results_dir}{glacier_name_for_generation}.png", dpi=100)

            plt.show()
            #plt.close()
            #exit()


####################################
# Regional simulation
def run_rgi_simulation(rgi=None):

    print(f"Begin regional simulation for region {rgi}")

    # Get rgi products
    rgi_products = get_rgi_products(rgi)
    oggm_rgi_glaciers, oggm_rgi_intersects, rgi_graph, mbdf_rgi = rgi_products
    # Get coastlines
    coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)

    # load xgb, cat models (by default they run on cpu)
    iceboost_xgb, iceboost_cat = load_models(config)

    # Get glaciers and order them in decreasing order by Area. First glaciers will be bigger and slower to process.
    oggm_rgi_glaciers = oggm_rgi_glaciers.sort_values(by='Area', ascending=False)

    def process_glacier(gl_id):

        # 0. get some useful ingredients
        glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_id]['geometry'].item()
        glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_id]['Area'].item()
        glacier_name = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_id]['Name'].item()

        # 1. generate features
        glacier_data = populate_glacier_with_metadata(glacier_name=gl_id,
                                                      config=config,
                                                      rgi_products=rgi_products,
                                                      coastlines_dataframe=coastline_dataframe,
                                                      seed=42,
                                                      verbose=False)

        h_wgs84 = glacier_data['elevation'].to_numpy()
        lats = glacier_data['lats'].to_numpy()
        lons = glacier_data['lons'].to_numpy()
        h_egm2008 = calc_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)

        # 2. run model
        X_test_glacier = glacier_data[config.features]
        y_test_glacier_m = glacier_data[config.millan]
        y_test_glacier_f = glacier_data[config.farinotti]

        dtest = xgb.DMatrix(data=X_test_glacier)

        y_preds_glacier_xgb = iceboost_xgb.predict(dtest)
        y_preds_glacier_cat = iceboost_cat.predict(X_test_glacier)

        # ensemble
        y_preds_glacier = 0.5 * (y_preds_glacier_xgb + y_preds_glacier_cat)

        # set negative predictions to zero
        y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

        # 3. calculate volumes
        vol_iceboost, err_vol_iceboost, vol_iceboost_bsl = calc_volume_glacier(y1=y_preds_glacier_xgb, y2=y_preds_glacier_cat,
                                                 area=glacier_area, h_egm2008=h_egm2008)
        vol_millan = calc_volume_glacier(y1=y_test_glacier_m, area=glacier_area)
        vol_farinotti = calc_volume_glacier(y1=y_test_glacier_f, area=glacier_area)

        # 4. produce xarray
        exterior_ring = glacier_geometry.exterior
        x0, y0, x1, y1 = exterior_ring.bounds
        dx, dy = x1 - x0, y1 - y0

        interpolator = NearestNDInterpolator(np.column_stack((lons, lats)), y_preds_glacier)
        tif_resolution = 2./3600

        xmin, xmax = tif_resolution * np.floor(x0/tif_resolution), tif_resolution * np.ceil(x1/tif_resolution)
        ymin, ymax = tif_resolution * np.floor(y0/tif_resolution), tif_resolution * np.ceil(y1/tif_resolution)

        lon_range = np.arange(xmin, xmax + tif_resolution, tif_resolution)
        lat_range = np.arange(ymin, ymax + tif_resolution, tif_resolution)

        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        thickness_grid = interpolator(lon_grid, lat_grid)

        assert not np.isnan(thickness_grid).any(), f'Thickness with some nans: glacier {gl_id}'

        data_array = xarray.DataArray(
            thickness_grid,
            coords=[('y', lat_range), ('x', lon_range)],
            name='thickness'
        ).rio.write_crs("EPSG:4326", inplace=True).rio.set_nodata(np.nan, inplace=True)

        data_array = data_array.rio.clip(geometries=[glacier_geometry], crs="EPSG:4326", drop=False, invert=False,
                                         all_touched=True)

        # enforce precise resolution
        data_array = data_array.rio.reproject(dst_crs=data_array.rio.crs, resolution=tif_resolution)

        assert data_array.rio.resolution()[0] == tif_resolution, 'Created array with unexpected resolution.'

        # add attributes
        data_array.attrs['id'] = gl_id
        data_array.attrs['name'] = glacier_name
        data_array.attrs['lat'] = lats.mean()
        data_array.attrs['lon'] = lons.mean()
        data_array.attrs['volume'] = vol_iceboost
        data_array.attrs['err'] = err_vol_iceboost
        data_array.attrs['volume_bsl'] = vol_iceboost_bsl
        data_array.attrs['volume_millan'] = vol_millan
        data_array.attrs['volume_farinotti'] = vol_farinotti
        data_array.attrs['thickness_units'] = 'm'
        data_array.attrs['volume_units'] = 'km3'
        data_array.attrs['method'] = 'gradient-boosted tree ensamble iceboost'
        data_array.attrs['author'] = 'Niccolo Maffezzoli, University of California Irvine'
        #data_array.plot(cmap='jet')
        #plt.show()

        # 5. save .tif
        PATH_OUT = config.model_output_global_deploy_dir
        file_out_tif = f'{PATH_OUT}rgi{rgi}/{gl_id}.tif'
        data_array.rio.to_raster(file_out_tif, compress="deflate")


    multicpu = True
    if multicpu:
        Parallel(n_jobs=8, timeout=300)(delayed(process_glacier)(gl_id) for gl_id in tqdm(oggm_rgi_glaciers['RGIId'],
                                                                                    desc=f"rgi {rgi} glaciers",
                                                                                    leave=True))
    else:
        for i, gl_id in tqdm(enumerate(oggm_rgi_glaciers['RGIId']), total=len(oggm_rgi_glaciers),
                             desc=f"rgi {rgi} glaciers", leave=True):
            process_glacier(gl_id)
    print(f"Finished regional simulation for rgi {rgi}.")

run_rgi_simulation_YN = False
if run_rgi_simulation_YN:
    run_rgi_simulation(rgi=5)