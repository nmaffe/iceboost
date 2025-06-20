import argparse, time
import os, yaml
import random
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import earthpy.spatial
import geopandas as gpd
from glob import glob
import xarray, rioxarray
from oggm import utils
from scipy import stats
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock, Queue, Pool, Semaphore, current_process

import xgboost as xgb
import catboost as cb
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import *
import misc as misc

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to yaml config file")
args = parser.parse_args()

# import from config.yaml
config = misc.get_config(args.config)

feature_human_names = {
        'Area': 'Area',  'Perimeter': 'Perimeter', 'zmin': r'z$_{min}$', 'zmax': r'z$_{max}$', 'zmed': r'z$_{med}$',
        'slope': 'Slope', 'aspect': 'Aspect', 'curvature': 'Curvature', 'lmax': 'Lmax', 'elevation': 'z',
        'elevation_from_zmin': r'z-z$_{min}$', 'dist_from_border_km_geom': r'd$_{noice}$', 'slope50': r's$_{50}$',
        'slope75': r's$_{75}$', 'slope100': r's$_{100}$', 'slope125': r's$_{125}$', 'slope150': r's$_{150}$',
        'slope300': r's$_{300}$', 'slope450': r's$_{450}$', 'slopegfa': r's$_{gfa}$', 'curv_50': r'c$_{50}$',
        'curv_100': r'c$_{100}$', 'curv_150': r'c$_{150}$', 'curv_300': r'c$_{300}$', 'curv_450': r'c$_{450}$',
        'curv_gfa': r'c$_{gfa}$', 'dmdtda_hugo': 'MB', 'deltaz': r'$\Delta$z', 'smb': 'mb', 't2m': 't2m',
        'dist_from_ocean': r'd$_{ocean}$', 'Cluster_area': r'A$_{cluster}$', 'elevation_0_1': r'z$_{01}$',
        'v50': r'v$_{50}$', 'v100': r'v$_{100}$', 'v150': r'v$_{150}$',
        'v300': r'v$_{300}$', 'v450': r'v$_{450}$', 'vgfa': r'v$_{gfa}$',
    }

file_deploy = pd.read_csv(f'/home/maffe/PycharmProjects/iceboost/saved_iceboost/iceboost_deploy_list_id.csv', index_col='rgi')
all_glacier_ids = file_deploy.values.flatten().tolist()

glathida_rgis = pd.read_csv(config.metadata_csv_file, low_memory=False)

# Load the model(s)
iceboost_xgb, iceboost_cat = load_models(config)


# *********************************************
# Model deploy
# *********************************************
#all_glacier_ids = ['AntPen_0', 'AntPen_1', 'AntPen_2', 'AntPen_3', 'AntPen_4', 'AntPen_5', 'AntPen_6',
#                   'AntPen_7', 'AntPen_8', 'AntPen_9', 'AntPen_10', 'AntPen_11', 'AntPen_12', 'AntPen_13',
#                   'AntPen_14', 'AntPen_15', 'AntPen_16', 'AntPen_17', 'AntPen_18', 'AntPen_19',
#                   'AntPen_20', 'AntPen_21', 'AntPen_22']
#all_glacier_ids = glathida_rgis.loc[glathida_rgis['RGI'] == 1, 'RGIId'].unique()
all_glacier_ids = ['AntPen_' + str(n) for n in range(0, 50)]
run_deploy_from_csv_list = False
if run_deploy_from_csv_list:
    for n, glacier_name_for_generation in enumerate(tqdm(all_glacier_ids)):

        glacier_name_for_generation = get_random_glacier_rgiid(name='RGI2000-v7.0-G-17-30629', rgi=13, version='62', area=0, seed=None)
        #print(n, glacier_name_for_generation)

        #if f"{glacier_name_for_generation}.png" in os.listdir(f"{config.model_output_results_dir}"):
        #    print(f"{glacier_name_for_generation} already in there.")
        #    continue

        test_glacier_rgi, version = get_version_and_rgi_from_id(glacier_name_for_generation)
        #test_glacier_rgi, version = 19, '62'

        rgi_products = get_rgi_products(region=test_glacier_rgi, version=version,
                                        add_glacier_geom_file=None,#config.antarctic_peninsula_gpkg,
                                        add_glacier_intersects_geom_file=None)#config.antarctic_peninsula_intersects_gpkg)
        rgi_glaciers, rgi_graph = rgi_products
        rgi_glaciers = add_regional_features(rgi_glaciers)
        coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
        link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')
        mbdf_rgi = get_mass_balance_df(region=test_glacier_rgi)

        data_generator = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                                      config=config,
                                                      rgi_products=rgi_products,
                                                      rgi=test_glacier_rgi,
                                                      mass_balance_df=mbdf_rgi,
                                                      version=version,
                                                      coastlines_dataframe=coastline_dataframe,
                                                      link_rgi6_rg7_dataframe=link_ids_rgi6_rgi7,
                                                      seed=42,
                                                      verbose=True,
                                                      )

        deployed_ids = data_generator.send(None)
        info, data = data_generator.send(None)

        deploy_ids = info.index.tolist()
        deploy_area = info['Area'].sum()

        # We will use the deployed geometries
        glacier_geometries = get_glacier_geometries_4326(ids=deploy_ids, glacier_geo_df=rgi_glaciers, version=f'{version}')

        h_wgs84 = data['elevation'].to_numpy()
        lats = data['lats'].to_numpy()
        lons = data['lons'].to_numpy()
        #h_egm2008 = calc_orthometric_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)
        h_ortho, n_geoid = calc_ortho_and_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84, geoid_tif=config.eigen6c4_tif)

        #fig, ax = plt.subplots()
        #s = ax.scatter(x=lons, y=lats, s=2, c=h_egm2008-h_egm2008_2, cmap='bwr')
        #cb = plt.colorbar(s)
        #plt.show()

        x0, y0, x1, y1 = lons.min(), lats.min(), lons.max(), lats.max()
        dx, dy = x1 - x0, y1 - y0

        swlat = data['lats'].min()
        swlon = data['lons'].min()
        nelat = data['lats'].max()
        nelon = data['lons'].max()
        deltalat = np.abs(swlat - nelat)
        deltalon = np.abs(swlon - nelon)
        eps = 5./3600
        focus = create_glacier_tile_dem_mosaic(minx=swlon - (deltalon + eps),
                                    miny=swlat - (deltalat + eps),
                                    maxx=nelon + (deltalon + eps),
                                    maxy=nelat + (deltalat + eps),
                                     rgi=test_glacier_rgi, path_tandemx=config.tandemx_dir)

        X_test_glacier = data[config.features]
        y_test_glacier_m = data[config.millan]
        y_test_glacier_f = data[config.farinotti]

        no_millan_data = np.isnan(y_test_glacier_m).all()
        no_farinotti_data = np.isnan(y_test_glacier_f).all()

        dtest = xgb.DMatrix(data=X_test_glacier)

        y_preds_glacier_xgb = iceboost_xgb.predict(dtest)
        y_preds_glacier_cat = iceboost_cat.predict(X_test_glacier)

        # ensemble
        y_preds_glacier = 0.5 * (y_preds_glacier_xgb + y_preds_glacier_cat)

        # Do you want to see the features ?
        # plot_feature_scatter(config.features, data)

        # Set negative predictions to zero
        y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

        # Calculate the glacier volume using the 3 models
        vol_montecarlo, err_vol_montecarlo, _ = calc_volume_glacier(y=y_preds_glacier, area=deploy_area, H=h_ortho)
        vol_millan_montecarlo, _, _ = calc_volume_glacier(y=y_test_glacier_m, area=deploy_area, H=h_ortho)
        vol_farinotti_montecarlo, _, _ = calc_volume_glacier(y=y_test_glacier_f, area=deploy_area, H=h_ortho)
        print(f"Glacier {glacier_name_for_generation} Area: {deploy_area:.2f} km2, "
              f"volML: {vol_montecarlo:.4g} km3 "
              f"volMil: {vol_millan_montecarlo:.4g} km3 "
              f"volFar: {vol_farinotti_montecarlo:.4g} km3")

        print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

        vmin = min(y_preds_glacier)
        vmax = max(y_preds_glacier)

        #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        #s1 = ax1.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier, s=1, vmin=vmin, vmax=vmax,cmap='turbo')
        #s2 = ax2.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier_xgb, s=1, vmin=vmin, vmax=vmax, cmap='turbo')
        #s3 = ax3.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier_cat, s=1, vmin=vmin, vmax=vmax, cmap='turbo')
        #cb1 = plt.colorbar(s1)
        #cb2 = plt.colorbar(s2)
        #cb3 = plt.colorbar(s3)
        #plt.show()

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

            for geometry in glacier_geometries:
                x, y = geometry.exterior.xy
                ax.plot(x, y, c='k', lw=1)
                for interior in geometry.interiors:
                    x, y = interior.xy
                    ax.plot(x, y, c='k', lw=0.8)
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
            #s1 = ax.scatter([], [], s=1, c=[], cmap='turbo', zorder=1, vmin=vmin, vmax=vmax)
            # Frame 1
            s1 = ax.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
                                            cmap='turbo', label='ML', zorder=1, vmin=vmin, vmax=vmax)

            # Create the colorbar
            cbar1 = plt.colorbar(s1, cax=cax)
            cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)
            cbar1.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
            cbar1.ax.tick_params(labelsize=16)

            plt.tight_layout()
            plt.show()

        plot_fancy_ML_prediction = False
        if plot_fancy_ML_prediction:

            resolution = 1. / 3600
            n_bins_lon = max(10, int(dx / resolution))
            n_bins_lat = max(10, int(dy / resolution))
            lon_grid, lat_grid = np.meshgrid(np.linspace(x0, x1, n_bins_lon), np.linspace(y0, y1, n_bins_lat))

            thickness_grid = griddata(np.column_stack((lons, lats)), y_preds_glacier, (lon_grid, lat_grid),
                                      method='nearest')

            # create an xarray DataArray
            data_array = xarray.DataArray(
                thickness_grid,
                coords=[('y', lat_grid[:, 0]), ('x', lon_grid[0, :])],
                name='thickness'
            ).rio.write_crs("EPSG:4326", inplace=True).rio.set_nodata(np.nan, inplace=True)
            data_array = data_array.rio.clip(geometries=glacier_geometries, crs="EPSG:4326", drop=False, invert=False,
                                             all_touched=False)

            # data_array.plot(cmap='turbo')
            # plt.show()

            fig, axes = plt.subplots(1,1, figsize=(8,6))
            #ax, ax3 = axes
            ax3 = axes
            hillshade = copy.deepcopy(focus)
            hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
            #hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)
            hillshade = hillshade.rio.clip_box(minx=x0-dx/8, miny=y0-dy/8, maxx=x1+dx/8, maxy=y1+dy/8)

            #im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

            #s1 = ax.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
            #                 cmap='turbo', label='ML', zorder=1, vmin=vmin,vmax=vmax)
            #s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
            #                        cmap='turbo', ec='grey', lw=0.5, s=35, vmin=vmin,vmax=vmax)


            #s2 = ax2.contourf(lon_grid, lat_grid, thickness_grid, levels=100, cmap='turbo')
            #cbar2 = plt.colorbar(s2, ax=ax2)
            #cbar2.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)

            im = hillshade.plot(ax=ax3, cmap='grey', alpha=1, zorder=0, add_colorbar=False)
            #im3 = data_array.plot(ax=ax3, cmap='turbo', alpha=0.3, vmin=vmin, vmax=vmax, add_colorbar=False)
            s1 = ax3.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
                                             cmap='turbo', label='ML', zorder=1, vmin=vmin,vmax=vmax)
            s_glathida3 = ax3.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'],
                                    c=glathida_rgis['THICKNESS'],
                                    cmap='turbo', ec='k', lw=0.5, s=40, vmin=vmin, vmax=vmax)

            #box_text = f"a) {glacier_name_for_generation} (Greenland)"
            #box_text = f"b) {glacier_name_for_generation} (Karakoram range)"
            #box_text = f"c) {glacier_name_for_generation} (Swiss Alps)"
            #box_text = f"d) {glacier_name_for_generation} (Devon Ice Cap, Canada)"
            #ax3.text(0.02, 0.93, box_text, fontsize=16, color='black',
            #         bbox=dict(facecolor='lightgrey', alpha=1, boxstyle='round'), transform=ax3.transAxes)

            #cbar = plt.colorbar(s1, ax=ax)
            cbar = plt.colorbar(s1, ax=ax3)
            cbar.set_label('Thickness [m]', labelpad=15, rotation=90, fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            #for ax in (ax, ax3): #axes
            for ax in (ax3, ):  # axes
                for geometry in glacier_geometries:
                    x, y = geometry.exterior.xy
                    ax.plot(x, y, c='k', lw=1)
                    for interior in geometry.interiors:
                        x, y = interior.xy
                        ax.plot(x, y, c='k', lw=0.8)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xlabel('Lon [$^{\\circ}$E]', fontsize=16)
                ax.set_ylabel('Lat [$^{\\circ}$N]', fontsize=16)
                ax.set_title('')

                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                ax.tick_params(axis='both', labelsize=16)
                props = dict(boxstyle='round', facecolor='white', alpha=0.9)
                ax.text(0.02, 0.93, f"{glacier_name_for_generation}", fontsize=20, color='black',
                        bbox=props, transform=ax.transAxes)
                #ax.text(0.5, 0.82, f"Unteraargletscher \nMB = -1.0 m w.e / yr \nV = 2.934 km3",
                #        fontsize=16, color='black',bbox=props, transform=ax.transAxes)

            plt.tight_layout()
            #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/artifact_{glacier_name_for_generation}.png", dpi=100,
            #            transparent=False)
            #plt.savefig(f"/home/maffe/Downloads/appendix/mb_minus1.png", dpi=100,
            #            transparent=False)
            plt.show()

        plot_fancy_ML_Mil_Far_prediction = True
        if plot_fancy_ML_Mil_Far_prediction:
            fig = plt.figure(figsize=(15, 6))
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

            # Create the axes
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            cax = fig.add_subplot(gs[3])

            dx, dy = x1 - x0, y1 - y0
            hillshade = copy.deepcopy(focus)
            hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
            hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

            hillshade.values = np.clip(hillshade.values, 0, 255)
            hillshade.values = hillshade.values.astype(np.uint8)

            im1 = hillshade.plot(ax=ax1, alpha=1.0, cmap='gray', zorder=0, add_colorbar=False, rasterized=True)
            im2 = hillshade.plot(ax=ax2, alpha=1.0, cmap='gray', zorder=0, add_colorbar=False, rasterized=True)
            im3 = hillshade.plot(ax=ax3, alpha=1.0, cmap='gray', zorder=0, add_colorbar=False, rasterized=True)

            s1 = ax1.scatter(x=lons, y=lats, s=2, c=y_preds_glacier, cmap='turbo', label='ML', vmin=vmin, vmax=vmax)
            if not no_millan_data:
                s2 = ax2.scatter(x=lons, y=lats, s=2, c=y_test_glacier_m, cmap='turbo',
                                 label='Millan', vmin=vmin, vmax=vmax)
            if not no_farinotti_data:
                s3 = ax3.scatter(x=lons, y=lats, s=2, c=y_test_glacier_f, cmap='turbo',
                                 label='Farinotti', vmin=vmin, vmax=vmax)

            ax1.set_title(f"IceBoost: {vol_montecarlo:.4g} km$^3$", fontsize=16)
            is_bedmachine = info['bedmachine'].iloc[0]
            if is_bedmachine: ax2.set_title(f"BedMachine: {vol_millan_montecarlo:.4g} km$^3$", fontsize=16)
            else:  ax2.set_title(f"Millan et al. (2022): {vol_millan_montecarlo:.4g} km$^3$", fontsize=16)
            ax3.set_title(f"Farinotti et al. (2019): {vol_farinotti_montecarlo:.4g} km$^3$", fontsize=16)

            for ax in (ax1, ax2, ax3):
                ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                        cmap='turbo', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)

            cbar = plt.colorbar(s1, cax=cax)
            cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
            cbar.ax.tick_params(labelsize=16)#11

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
                for geometry in glacier_geometries.geometry:
                    x, y = geometry.exterior.xy
                    ax.plot(x, y, c='k', lw=1)
                    for interior in geometry.interiors:
                        x, y = interior.xy
                        ax.plot(x, y, c='k', lw=0.8)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)
                ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)
                ax.tick_params(axis='both', labelsize=16)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax2.axis('off')
            ax3.axis('off')

            plt.tight_layout()
            #plt.savefig(f"/home/maffe/Downloads/peninsula/{glacier_name_for_generation}.png", dpi=100)
            #plt.savefig(f"/home/maffe/Downloads/peninsula/{glacier_name_for_generation}.png", dpi=100)
            #plt.close()

            if config.deploy_save_figs:
                plt.savefig(f"{config.model_output_results_dir}{glacier_name_for_generation}.png", dpi=100, transparent=False)
                #from PIL import Image
                #image = Image.open(f"{config.model_output_results_dir}{glacier_name_for_generation}.png")
                #image = image.convert("RGB")
                #image = image.resize((1300,520))
                #image.save(f"{config.model_output_results_dir}{glacier_name_for_generation}.jpg", optimize=True, quality=75)
                plt.close(fig)

            #plt.close(fig)
            plt.show()

        run_shap_single_glacier = False
        if run_shap_single_glacier:
            print(f"Running SHAP for glacier {glacier_name_for_generation}...")

            data["xgb"] = y_preds_glacier_xgb
            data["cat"] = y_preds_glacier_cat

            NO_SHAP_POINTS = len(data)
            print(f"Shap points: {NO_SHAP_POINTS}")
            data_glacier_sample = data.sample(n=NO_SHAP_POINTS, random_state=42)

            explainer = shap.explainers.Tree(iceboost_xgb, data_glacier_sample[config.features])
            shap_values = explainer(data_glacier_sample[config.features], check_additivity=False) # (500, 39)
            shap_features = ['shap_' + feature for feature in config.features]
            shap_values.feature_names = shap_features

            shap_df = pd.DataFrame(
                data=shap_values.values,
                columns=shap_features
            )

            shap_avrg_df = pd.DataFrame(
                np.abs(shap_df.values).mean(axis=0),
                index=shap_df.columns,
                columns=["Mean |SHAP value|"]
            )
            shap_avrg_df = shap_avrg_df.sort_values(by="Mean |SHAP value|", ascending=False)
            # Get top-5 features
            top_shap_avrg_df = shap_avrg_df.head(5)
            top_shap_avrg_df = top_shap_avrg_df.copy()
            # Add the human names
            top_shap_avrg_df["human_names"] = [feature_human_names[feature.replace("shap_", "")] for feature in
                                                      top_shap_avrg_df.index]
            top_features = top_shap_avrg_df.index
            #print(top_features)
            #print(shap_avrg_df)

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 6))

            s1 = ax1.scatter(x=data_glacier_sample['lons'].to_numpy(), y=data_glacier_sample['lats'].to_numpy(),
                             c=np.abs(shap_df.values).sum(axis=1), s=2, cmap='binary')

            s2 = ax2.scatter(x=data_glacier_sample['lons'].to_numpy(), y=data_glacier_sample['lats'].to_numpy(),
                              c=np.abs(data_glacier_sample['xgb'] - data_glacier_sample['cat']), s=2, cmap='Reds')

            cb1 = plt.colorbar(s1)
            cb2 = plt.colorbar(s2)

            cb1.set_label(r'$\sum_{f}$ |SHAP(f)| [m]', labelpad=15, rotation=90, fontsize=16)
            cb2.set_label('|XGBoost - CatBoost| [m]', labelpad=15, rotation=90, fontsize=16)

            for cb in (cb1, cb2):
                cb.ax.tick_params(which='both', length=0)
                cb.ax.tick_params(labelsize=14)

            for ax in (ax1, ax2):
                ax.set_xlabel('Lon [$^{\\circ}$E]', fontsize=16)
                ax.set_ylabel('Lat [$^{\\circ}$N]', fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for geometry in glacier_geometries:
                    x, y = geometry.exterior.xy
                    ax.plot(x, y, c='k', lw=1)
                    for interior in geometry.interiors:
                        x, y = interior.xy
                        ax.plot(x, y, c='k', lw=0.8)

            for ax in (ax1, ax2):
                ax.set_xlim(-81.4, None)
                ax.set_ylim(None, 77.12)

            ax1.text(0.04, 0.93, "a)", fontsize=18, color='black', transform=ax1.transAxes)
            ax2.text(0.04, 0.93, "b)", fontsize=17, color='black', transform=ax2.transAxes)

            ax_inset = inset_axes(ax1, width="100%", height="100%", loc="lower left",
                                  bbox_to_anchor=(0.12, 0.1, 0.3, 0.2), bbox_transform=ax1.transAxes)

            palettes_4bars = ['#8B4513', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']
            bars = top_shap_avrg_df.plot(kind='barh', ax=ax_inset, color='gray')
            for i, bar in enumerate(bars.patches):
                bar.set_facecolor(palettes_4bars[::-1][i])

            ax_inset.set_facecolor('none')
            ax_inset.set_yticklabels(top_shap_avrg_df['human_names'], fontsize=13)
            ax_inset.tick_params(axis='x', labelsize=12)
            ax_inset.yaxis.set_tick_params(length=0)
            ax_inset.xaxis.set_tick_params(length=0)
            ax_inset.invert_yaxis()
            ax_inset.set_xlabel("Mean |SHAP| [m]", fontsize=12)
            ax_inset.get_legend().set_visible(False)
            ax_inset.spines['top'].set_visible(False)
            ax_inset.spines['left'].set_visible(False)
            ax_inset.spines['right'].set_visible(False)

            plt.tight_layout()
            #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/fig_shap_mittie.png", dpi=100)
            #plt.show()



            """Create a figure and 3D axis"""
            lons = data_glacier_sample['lons'].values
            lats = data_glacier_sample['lats'].values
            lon_min, lon_max = min(lons), max(lons)
            lat_min, lat_max = min(lats), max(lats)
            grid_lons, grid_lats = np.meshgrid(
                np.linspace(lon_min, lon_max, 500),  # Adjust grid resolution as needed
                np.linspace(lat_min, lat_max, 500)
            )
            grid_points = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])
            # Create mask based on geometry
            mask = np.array([any(glacier_geometries.contains(Point(xy))) for xy in grid_points])
            mask = mask.reshape(grid_lons.shape)

            fig, ax = plt.subplots(figsize=(8, 8),
                                   subplot_kw=dict(projection='3d'),
                                   constrained_layout=True,
                                   gridspec_kw=dict(top=1, bottom=0, left=0, right=1))
            ax.view_init(elev=20, azim=65, vertical_axis='y')

            # Loop through the top 5 features to plot them in 3D
            # Plot each feature's SHAP values with a different Z-offset for stacking
            palettes_4shap = [get_cmap('white_to_brown'), 'Blues', 'Greens', 'Purples', get_cmap('white_to_orange')]
            for i, feature in enumerate(top_features[::-1]):

                shap_values_feature = np.abs(shap_df[feature].values)

                grid_shap_values_feature = griddata(
                    points=(lons, lats), values=shap_values_feature,
                    xi=(grid_lons, grid_lats), method='linear'
                )
                masked_grid_shap_values_feature = np.ma.masked_where(~mask, grid_shap_values_feature)

                offset = 2*i

                contour = ax.contourf(grid_lons, grid_lats, masked_grid_shap_values_feature,
                                      cmap=palettes_4shap[i], levels=30, zdir='z', offset=offset, alpha=1, linestyles='none')

                ax.text(-80.8955, 77, offset-0.5, top_shap_avrg_df["human_names"].loc[feature],
                        ha='center', va='center', fontsize=20, color='black', weight='normal')

            #ax.set_xlim(lon_min, lon_max)
            #ax.set_ylim(lat_min, lat_max)
            #ax.set_zlim(0, offset)
            ax.set_xlim3d(left=lon_min, right=lon_max)
            ax.set_ylim3d(bottom=lat_min, top=lat_max)
            ax.set_zlim3d(bottom=0, top=offset)

            ax.xaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
            ax.yaxis.pane.set_facecolor((0.5, 0.5, 0.5, 0.9))
            ax.grid(True)

            ax.set_xticks([])  # Removes x-axis ticks
            ax.set_yticks([])  # Removes y-axis ticks
            ax.set_zticks([])  # Removes z-axis ticks
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # Generate grid lines on the floor
            x_vals = np.linspace(lon_min, lon_max, 5)
            for x in x_vals:
                ax.plot([x, x], [lat_min, lat_min], [0, offset], color='lightgrey', linestyle='--')
            z_vals = np.linspace(0, offset, 5)
            for z in z_vals:
                ax.plot([lon_min, lon_max], [lat_min, lat_min], [z, z], color='lightgrey', linestyle='--')

            title_text = r"$\longleftarrow$ Top-5 Feature |SHAP| ───"
            ax.text(lon_max-0.2, lat_min, offset/2, title_text, zdir='z',
                    ha='center', va='center', fontsize=20)

            #ax.set_axis_off()
            #plt.tight_layout()
            #ax.set_box_aspect([1, 1, 4])

            #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/fig_shap_mittie_all.png", dpi=600, bbox_inches='tight')
            #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/fig_shap_mittie_all.pdf", bbox_inches='tight')
            plt.show()


            exit()


####################################
# Regional simulation

def process_glacier(split_IDS, process_idx=None, processed_ids=None, lock=None):
    """
    :param split_IDS: list of ids to process
    :param process_idx: integer of process number
    :param processed_ids: dictionary in shared memory to keep track of processed ids among multiple processors
    :param lock: shared lock to manage accessing shared memory
    :return: None
    """

    #if processed_glaciers_shared is not None:
        #print(f"N: {len(processed_glaciers_shared)}")

        #if gl_id in processed_glaciers_shared:
            #print(f"{gl_id} already processed")
            #return None  # Skip already processed ID

    # Get the name of the current process
    process_name = current_process().name
    #print(process_name)


    with (tqdm(total=len(split_IDS), desc=f"Process {process_name}", position=1+process_idx, leave=False) as pbar):

        for gl_id in split_IDS:

            if processed_ids is not None:
                if gl_id in processed_ids:
                    #print(f"{gl_id} already processed")
                    continue
                with lock:
                    #print(f"Processing glacier {gl_id} in process: {process_name}")
                    processed_ids[gl_id] = True  # mark gl_id as processed

            # 1. generate features: info, data
            data_generator = populate_glacier_with_metadata(glacier_name=gl_id,
                                                          config=config,
                                                          rgi_products=rgi_products,
                                                          rgi=rgi,
                                                          mass_balance_df=mbdf_rgi,
                                                          version=version,
                                                          coastlines_dataframe=coastline_dataframe,
                                                          link_rgi6_rg7_dataframe=link_ids_rgi6_rgi7,
                                                          seed=42,
                                                          verbose=False,
                                                          )
            deployed_ids = data_generator.send(None)

            pbar.update(len(deployed_ids))

            if processed_ids is not None:
                with lock:
                    for id in deployed_ids:
                        processed_ids[id] = True  # mark additional ids as processed


            info, data = data_generator.send(None)

            deploy_ids = info.index.tolist()
            deploy_area = info['Area'].sum()
            no_glaciers = len(info)
            #print(f'We have {no_glaciers} glaciers produced from {gl_id}')
            glacier_geometries = get_glacier_geometries_4326(ids=deploy_ids, glacier_geo_df=rgi_glaciers, version=version)
            grid_crs = data.crs
            glacier_geometries_grid_crs = glacier_geometries.to_crs(crs=grid_crs)

            h_wgs84 = data['elevation'].to_numpy()
            lats = data['lats'].to_numpy()
            lons = data['lons'].to_numpy()
            h_ortho, n_geoid = calc_ortho_and_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84,
                                                            geoid_tif=config.eigen6c4_tif)
            data['h_ortho'] = h_ortho
            data['n_geoid'] = n_geoid

            # 2. run model
            X_test_glacier = data[config.features]
            y_test_glacier_m = data[config.millan]
            y_test_glacier_f = data[config.farinotti]

            dtest = xgb.DMatrix(data=X_test_glacier)

            y_preds_glacier_xgb = iceboost_xgb.predict(dtest)
            y_preds_glacier_cat = iceboost_cat.predict(X_test_glacier)

            # ensemble
            y_preds_glacier = 0.5 * (y_preds_glacier_xgb + y_preds_glacier_cat)

            # set negative predictions to zero
            y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)
            data['thickness'] = y_preds_glacier

            # calculate error on thickness
            err_y = np.abs(y_preds_glacier_xgb - y_preds_glacier_cat)

            # 3. calculate volumes with Montecarlo
            vol_montecarlo, err_vol_montecarlo, vol_montecarlo_bsl = calc_volume_glacier(y=y_preds_glacier, area=deploy_area, H=h_ortho)
            vol_montecarlo_millan, _, _ = calc_volume_glacier(y=y_test_glacier_m, area=deploy_area, H=h_ortho)
            vol_montecarlo_farinotti, _, _ = calc_volume_glacier(y=y_test_glacier_f, area=deploy_area, H=h_ortho)

            #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
            #s1 = ax1.scatter(x=lons, y=lats, c=y_preds_glacier_xgb, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            #s2 = ax2.scatter(x=lons, y=lats, c=y_preds_glacier_cat, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            #s3 = ax3.scatter(x=lons, y=lats, c=y_preds_glacier, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            #s4 = ax4.scatter(x=lons, y=lats, c=err_y, cmap='binary')
            #cb1 = plt.colorbar(s1)
            #cb2 = plt.colorbar(s2)
            #cb3 = plt.colorbar(s3)
            #cb4 = plt.colorbar(s4)
            #plt.show()


            # 4. Create grid
            grid_eastings = data.geometry.x.values
            grid_northings = data.geometry.y.values
            grid_res = data.attrs["grid_resolution"]
            grid_e0, grid_e1 = grid_eastings.min(), grid_eastings.max()
            grid_n0, grid_n1 = grid_northings.min(), grid_northings.max()
            xmin, xmax = np.floor(grid_e0 / grid_res) * grid_res, np.ceil(grid_e1 / grid_res) * grid_res
            ymin, ymax = np.floor(grid_n0 / grid_res) * grid_res, np.ceil(grid_n1 / grid_res) * grid_res
            x_range = np.arange(xmin, xmax + grid_res, grid_res)
            y_range = np.arange(ymin, ymax + grid_res, grid_res)
            x_grid, y_grid = np.meshgrid(x_range, y_range)


            # 5. Create grid of thickness values
            points = np.column_stack((data.geometry.x.values, data.geometry.y.values))
            tree = cKDTree(points)
            distances, indexes = tree.query(np.column_stack((x_grid.ravel(), y_grid.ravel())))
            thickness_grid = y_preds_glacier[indexes].reshape(x_grid.shape)
            err_thickness_grid = err_y[indexes].reshape(x_grid.shape)
            h_wgs84_grid = h_wgs84[indexes].reshape(x_grid.shape)
            n_geoid_grid = n_geoid[indexes].reshape(x_grid.shape)

            assert not np.isnan(thickness_grid).any(), f'Thickness with some nans: glacier {gl_id}'
            assert not np.isnan(err_thickness_grid).any(), f'Thickness error with some nans: glacier {gl_id}'
            assert not np.isnan(h_wgs84_grid).any(), f'h_wgs84 with some nans: glacier {gl_id}'
            assert not np.isnan(n_geoid_grid).any(), f'n_geoid with some nans: glacier {gl_id}'

            # 6. Create the Dataset
            data_dataset = xarray.Dataset({
                'thickness': (('y', 'x'), np.flip(thickness_grid, axis=0)),
                'thickness_err': (('y', 'x'), np.flip(err_thickness_grid, axis=0)),
                'h_wgs84': (('y', 'x'), np.flip(h_wgs84_grid, axis=0)),
                'n_geoid': (('y', 'x'), np.flip(n_geoid_grid, axis=0))
            },
                coords={
                    'y': np.flip(y_range),
                    'x': x_range
                }).rio.write_crs(grid_crs, inplace=True)

            data_dataset['thickness'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['thickness_err'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['h_wgs84'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['n_geoid'].rio.write_nodata(np.nan, inplace=True)

            # 7. Crop the data using all geometries at once
            data_dataset = data_dataset.map(
                lambda da: da.rio.clip(
                    geometries=glacier_geometries_grid_crs.geometry,
                    crs=grid_crs,
                    drop=False,
                    invert=False,
                    all_touched=True,
                )
            )

            #fig, ax = plt.subplots()
            #data_dataset['thickness'].plot(ax=ax, cmap='turbo')
            #glacier_geometries_grid_crs.plot(ax=ax, ec='k', fc='none')
            #plt.show()

            # 8. Get individual glaciers using individual geometries
            for n, glacierID in enumerate(info.index):
                #tqdm.write(f"{n+1}/{len(info)} ID: {glacierID} process {process_name}")

                areaID = info.at[glacierID, 'Area']
                nameID = info.at[glacierID, 'Name']
                geomID = glacier_geometries_grid_crs.loc[glacierID, 'geometry']
                geomID_4326 = glacier_geometries.loc[glacierID, 'geometry']

                arrayID = data_dataset.map(
                    lambda da: da.rio.clip(
                        geometries=[geomID],
                        crs=grid_crs,
                        drop=True,
                        invert=False,
                        all_touched=True,
                    )
                )

                # Sanity check
                assert arrayID.rio.resolution() == (grid_res, -grid_res), "Mismatch in resolution."

                # Calculate volume from produced data points
                dataID = data.loc[data["polygon"] == glacierID]
                f = 0.001 * areaID / len(dataID)
                volID = dataID["thickness"].sum() * f
                volID_bsl = np.where(dataID['h_ortho'] - dataID['thickness'] > 0, 0.0,
                                     dataID['thickness'] - dataID['h_ortho']).sum() * f

                # Get ground truth measurements
                glathida_rgis_ID = glathida_rgis.loc[glathida_rgis['RGIId'] == glacierID]
                ground_truth_lons = glathida_rgis_ID['POINT_LON'].to_list()
                ground_truth_lats = glathida_rgis_ID['POINT_LAT'].to_list()
                ground_truth_meas = glathida_rgis_ID['THICKNESS'].to_list()

                # Add attributes
                arrayID.attrs['id'] = glacierID
                arrayID.attrs['name'] = nameID
                arrayID.attrs['lat'] = geomID_4326.representative_point().y
                arrayID.attrs['lon'] = geomID_4326.representative_point().x
                arrayID.attrs['area'] = areaID
                arrayID.attrs['volume'] = volID
                arrayID.attrs['volume_bsl'] = volID_bsl
                arrayID.attrs['ground_truth_lons'] = json.dumps(ground_truth_lons)
                arrayID.attrs['ground_truth_lats'] = json.dumps(ground_truth_lats)
                arrayID.attrs['ground_truth_meas'] = json.dumps(ground_truth_meas)
                arrayID.attrs['resX'] = grid_res
                arrayID.attrs['resY'] = grid_res
                arrayID.attrs['crs'] = arrayID.rio.crs.to_string()
                arrayID.attrs['h_wgs84'] = 'Tandem-X Edited DEM, 30m'
                arrayID.attrs['n_geoid'] = 'EIGEN-6C4 geoid height, m'
                arrayID.attrs['units_thickness'] = 'm'
                arrayID.attrs['units_volume'] = 'km3'
                arrayID.attrs['units_area'] = 'km2'
                arrayID.attrs['method'] = 'ICEBOOST v1.1 model'
                arrayID.attrs['data_citation'] = ("Maffezzoli, N., et al. 'A gradient-boosted tree framework to "
                                                  "model the ice thickness of the world's glaciers (IceBoost v1.1).' "
                                                  "Geoscientific Model Development 18.9 (2025): 2545-2568.")
                arrayID.attrs['author'] = 'Niccolò Maffezzoli, University of California Irvine'
                arrayID.attrs['production_date'] = datetime.today().strftime("%d-%B-%Y")
                #print(arrayID)
                #print(arrayID.rio.crs)

                #fig, ax = plt.subplots()
                #arrayID['thickness'].plot(ax=ax, cmap='turbo')
                #gpd.GeoSeries([geomID]).plot(ax=ax, ec='k', fc='none')
                #plt.show()

                # 9. Save individual glacier .tif
                if config.deploy_global_save_figs:
                    PATH_OUT = config.model_output_global_deploy_dir
                    file_out_tif = f'{PATH_OUT}RGI{version}/rgi{rgi}/{glacierID}.tif'
                    arrayID.rio.to_raster(file_out_tif, compress="deflate", dtype="float32")


run_rgi_simulation_YN = True
if run_rgi_simulation_YN:
    t0 = time.time()
    rgi = 3
    version = '70G'

    print(f"Begin regional simulation for region {rgi}, version {version}")


    rgi_products = get_rgi_products(region=rgi,
                                    version=version,
                                    add_glacier_geom_file=None,#config.antarctic_peninsula_gpkg,
                                    add_glacier_intersects_geom_file=None)#config.antarctic_peninsula_intersects_gpkg)
    rgi_glaciers, rgi_graph = rgi_products
    rgi_glaciers = add_regional_features(rgi_glaciers)
    mbdf_rgi = get_mass_balance_df(region=rgi)
    coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
    link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')

    if version == '62':
        name_column_id = 'RGIId'
        name_column_area = 'Area'
    elif version == '70G':
        name_column_id = 'rgi_id'
        name_column_area = 'area_km2'

    # Get glaciers and order them in decreasing order by Area. First glaciers will be bigger and slower to process.
    rgi_glaciers = rgi_glaciers.sort_values(by=name_column_area, ascending=False)
    total_no_glaciers = len(rgi_glaciers)

    # load xgb, cat models (by default they run on cpu)
    iceboost_xgb, iceboost_cat = load_models(config)

    # decide if multiprocessing is used
    multicpu = config.n_jobs > 1
    print(f"MultiCPU: {multicpu}")

    if multicpu:
        # List of lists of ids for multithread
        splits_ids = geographic_split_adaptive(glaciers_df=rgi_glaciers, n_jobs=config.n_jobs, version=version)
        for n, split in enumerate(splits_ids):
            print(f"Split {n}: {len(split)}")

        with Manager() as manager:
            shared_processed_ids = manager.dict()
            shared_lock = manager.Lock()

            print("="*20, "Grab a beer!", "="*20)
            print(f"Processing rgi {rgi} v.{version} - {total_no_glaciers} glaciers")
            #with tqdm(total=len(split), desc=f"Processing rgi {rgi} - {total_no_glaciers} glaciers", position=0, leave=True) as outer_bar:
            Parallel(n_jobs=config.n_jobs)(
                delayed(process_glacier)(i_split, idx, shared_processed_ids, shared_lock)
                for idx, i_split in enumerate(splits_ids)
            )

            # The Parallel function now runs the processing in parallel
            #Parallel(n_jobs=config.n_jobs)(
                #    delayed(process_glacier)(i_split, shared_processed_ids, shared_lock)
                #for i_split in tqdm(splits_ids, desc=f"rgi {rgi} glaciers", leave=True)
            #)

    else:
        target = ['AntPen_18', 'AntPen_21'] # RGI60-07.00027 RGI60-07.01514 RGI60-07.00027
        #target = ['AntPen_7', 'AntPen_8', 'AntPen_9', 'AntPen_10', 'AntPen_11', 'AntPen_13', 'AntPen_14', 'AntPen_15', 'AntPen_16',
        #          'AntPen_20', 'AntPen_21', 'AntPen_22']
        glaciers_for_deploy = rgi_glaciers.loc[rgi_glaciers[name_column_id].isin(target)]

        pbar = tqdm(enumerate(glaciers_for_deploy[name_column_id]), total=len(glaciers_for_deploy), leave=True)
        #pbar = tqdm(enumerate(rgi_glaciers[name_column_id]), total=len(rgi_glaciers), leave=True)
        for i, gl_id in pbar:
            pbar.set_description(f"rgi {rgi} glaciers - ID: {gl_id}")
            process_glacier([gl_id], process_idx=0)

    print(f"Finished regional simulation for rgi {rgi}, version {version} in {(time.time()-t0)/60} min.")