import argparse, time
import os, yaml
import random
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
from scipy.interpolate import griddata, NearestNDInterpolator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock, Queue, Pool, Semaphore, current_process

import xgboost as xgb
import catboost as cb
import optuna
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata, get_rgi_products, get_coastline_dataframe
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import *
import misc as misc

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to yaml config file")
args = parser.parse_args()

config = misc.get_config(args.config)  # import from config.yaml

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

file_deploy = pd.read_csv(f'{config.model_input_dir}{config.filename_csv_deploy}', index_col='rgi')
all_glacier_ids = file_deploy.values.flatten().tolist()

glathida_rgis = pd.read_csv(config.metadata_csv_file, low_memory=False)

fig_stats = False
if fig_stats:
    min_val, max_val = glathida_rgis['THICKNESS'].min(), glathida_rgis['THICKNESS'].max()
    mean_val = glathida_rgis['THICKNESS'].mean()
    median_val = glathida_rgis['THICKNESS'].median()
    std_val = glathida_rgis['THICKNESS'].std()
    stats_text = (
        f"Min: {min_val:.0f} m\n"
        f"Max: {max_val:.0f} m\n"
        f"Mean: {mean_val:.0f} m\n"
        f"Median: {median_val:.0f} m\n"
        f"Std: {std_val:.0f} m"
    )
    print(glathida_rgis['THICKNESS'].describe())
    fig, ax = plt.subplots()
    ax.hist(glathida_rgis['THICKNESS'], bins=np.logspace(np.log10(min_val), np.log10(max_val), num=100),
            alpha=0.7, lw=2, edgecolor='black', facecolor='blue', histtype='stepfilled')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(
        0.06, 0.9, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='top', bbox=props
    )
    ax.set_xscale('log')
    ax.set_xlabel('Thickness [m]', fontsize=16)
    ax.set_ylabel('No. training points', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='k', linewidth=0.5, alpha=1)#True, which='both', linestyle='--', color='k', linewidth=0.5, alpha=1)
    plt.tight_layout()
    plt.show()

# Load the model(s)
iceboost_xgb, iceboost_cat = load_models(config)


# *********************************************
# Model deploy
# *********************************************
run_deploy_from_csv_list = True
if run_deploy_from_csv_list:
    for n, glacier_name_for_generation in enumerate(tqdm(all_glacier_ids)):

        glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-11.01450', rgi=4, version='70G', area=0, seed=None)
        #print(n, glacier_name_for_generation)

        #if f"{glacier_name_for_generation}.png" in os.listdir(f"{config.model_output_results_dir}"):
        #    print(f"{glacier_name_for_generation} already in there.")
        #    continue

        test_glacier_rgi, version = get_version_and_rgi_from_id(glacier_name_for_generation)
        rgi_products = get_rgi_products(test_glacier_rgi, version=version)
        coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
        link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')

        data_generator = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
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

        deploy_ids = info.index.tolist()
        deploy_area = info['Area'].sum()

        h_wgs84 = data['elevation'].to_numpy()
        lats = data['lats'].to_numpy()
        lons = data['lons'].to_numpy()
        h_egm2008 = calc_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)

        # Begin to extract all necessary things to plot the result
        oggm_rgi_glaciers, oggm_rgi_intersects, rgi_graph, mbdf_rgi = rgi_products
        if version == '62': name_column_id = 'RGIId'
        elif version == '70G': name_column_id = 'rgi_id'

        # We will use the deployed geometries
        glacier_geometries = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id].isin(deploy_ids), 'geometry']
        glacier_geometries = glacier_geometries.to_crs("EPSG:4326")

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

        #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        #s1 = ax1.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier, s=1, vmin=y_preds_glacier.min(), vmax=y_preds_glacier.max(),cmap='turbo')
        #s2 = ax2.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier_xgb, s=1, vmin=y_preds_glacier.min(), vmax=y_preds_glacier.max(), cmap='turbo')
        #s3 = ax3.scatter(x=data['lons'], y=data['lats'], c=y_preds_glacier_cat, s=1, vmin=y_preds_glacier.min(), vmax=y_preds_glacier.max(), cmap='turbo')
        #cb1 = plt.colorbar(s1)
        #cb2 = plt.colorbar(s2)
        #cb3 = plt.colorbar(s3)
        #plt.show()

        # Do you want to see the features ?
        #plot_feature_scatter(config, data)

        # Set negative predictions to zero
        y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

        # Calculate the glacier volume using the 3 models
        vol_montecarlo, err_vol_montecarlo, _ = calc_volume_glacier(y=y_preds_glacier, area=deploy_area, h_egm2008=h_egm2008)
        vol_millan_montecarlo, _, _ = calc_volume_glacier(y=y_test_glacier_m, area=deploy_area, h_egm2008=h_egm2008)
        vol_farinotti_montecarlo, _, _ = calc_volume_glacier(y=y_test_glacier_f, area=deploy_area, h_egm2008=h_egm2008)
        print(f"Glacier {glacier_name_for_generation} Area: {deploy_area:.2f} km2, "
              f"volML: {vol_montecarlo:.4g} km3 "
              f"volMil: {vol_millan_montecarlo:.4g} km3 "
              f"volFar: {vol_farinotti_montecarlo:.4g} km3")

        print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

        vmin = min(y_preds_glacier)
        vmax = max(y_preds_glacier)

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

        plot_fancy_ML_prediction = True
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
            hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)

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

            plt.tight_layout()
            #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/artifact_{glacier_name_for_generation}.png", dpi=100,
            #            transparent=False)
            plt.show()

        plot_fancy_ML_Mil_Far_prediction = False
        if plot_fancy_ML_Mil_Far_prediction:
            fig = plt.figure(figsize=(15, 6))
            #fig = plt.figure(figsize=(10, 6))
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust the width ratios
            #gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # Adjust the width ratios

            # Create the axes
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            #cax = fig.add_subplot(gs[2])  # Colorbar axis
            cax = fig.add_subplot(gs[3])  # Colorbar axis

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
            #for ax in (ax1, ax2):
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
            #for ax in (ax1, ax2):
                for geometry in glacier_geometries:
                    x, y = geometry.exterior.xy
                    ax.plot(x, y, c='k', lw=1)
                    for interior in geometry.interiors:
                        x, y = interior.xy
                        ax.plot(x, y, c='k', lw=0.8)

                # ax.legend(fontsize=14, loc='upper left')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)
                ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)
                ax.tick_params(axis='both', labelsize=16)
                #ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax2.axis('off')
            ax3.axis('off')

            plt.tight_layout()

            if config.deploy_save_figs:
                from PIL import Image
                plt.savefig(f"{config.model_output_results_dir}{glacier_name_for_generation}.png", dpi=100, transparent=False)
                image = Image.open(f"{config.model_output_results_dir}{glacier_name_for_generation}.png")
                image = image.convert("RGB")
                image = image.resize((1300,520))
                image.save(f"{config.model_output_results_dir}{glacier_name_for_generation}.jpg", optimize=True, quality=75)
                plt.close()

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


    with tqdm(total=len(split_IDS), desc=f"Process {process_name}", position=1+process_idx, leave=False) as pbar:

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

            h_wgs84 = data['elevation'].to_numpy()
            lats = data['lats'].to_numpy()
            lons = data['lons'].to_numpy()
            h_egm2008 = calc_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)

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

            # calculate error on thickness
            err_y = np.abs(y_preds_glacier_xgb - y_preds_glacier_cat)

            # 3. calculate volumes with Montecarlo
            vol_montecarlo, err_vol_montecarlo, vol_montecarlo_bsl = calc_volume_glacier(y=y_preds_glacier, area=deploy_area, h_egm2008=h_egm2008)
            vol_montecarlo_millan, _, _ = calc_volume_glacier(y=y_test_glacier_m, area=deploy_area, h_egm2008=h_egm2008)
            vol_montecarlo_farinotti, _, _ = calc_volume_glacier(y=y_test_glacier_f, area=deploy_area, h_egm2008=h_egm2008)

            # 4. produce xarray
            glacier_geometries = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id].isin(deploy_ids), 'geometry']
            glacier_names = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id].isin(deploy_ids), name_column_name]

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
            s1 = ax1.scatter(x=lons, y=lats, c=y_preds_glacier_xgb, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            s2 = ax2.scatter(x=lons, y=lats, c=y_preds_glacier_cat, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            s3 = ax3.scatter(x=lons, y=lats, c=y_preds_glacier, vmin=min(y_preds_glacier), vmax=max(y_preds_glacier), cmap='turbo')
            s4 = ax4.scatter(x=lons, y=lats, c=err_y, cmap='binary')
            cb1 = plt.colorbar(s1)
            cb2 = plt.colorbar(s2)
            cb3 = plt.colorbar(s3)
            cb4 = plt.colorbar(s4)
            plt.show()

            x0, y0, x1, y1 = lons.min(), lats.min(), lons.max(), lats.max()
            dx, dy = x1 - x0, y1 - y0

            tif_resolution = 2./3600

            xmin, xmax = tif_resolution * np.floor(x0/tif_resolution), tif_resolution * np.ceil(x1/tif_resolution)
            ymin, ymax = tif_resolution * np.floor(y0/tif_resolution), tif_resolution * np.ceil(y1/tif_resolution)

            lon_range = np.arange(xmin, xmax + tif_resolution, tif_resolution)
            lat_range = np.arange(ymin, ymax + tif_resolution, tif_resolution)

            lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

            #interpolator_thickness = NearestNDInterpolator(np.column_stack((lons, lats)), y_preds_glacier)
            #interpolator_h_wgs84 = NearestNDInterpolator(np.column_stack((lons, lats)), h_wgs84)
            #interpolator_h_egm2008 = NearestNDInterpolator(np.column_stack((lons, lats)), h_egm2008)
            #thickness_grid = interpolator_thickness(lon_grid, lat_grid)

            thickness_grid = griddata(np.column_stack((lons, lats)), y_preds_glacier, (lon_grid, lat_grid), method='nearest')
            err_thickness_grid = griddata(np.column_stack((lons, lats)), err_y, (lon_grid, lat_grid), method='nearest')
            h_wgs84_grid = griddata(np.column_stack((lons, lats)), h_wgs84, (lon_grid, lat_grid), method='nearest')
            h_egm2008_grid = griddata(np.column_stack((lons, lats)), h_egm2008, (lon_grid, lat_grid), method='nearest')

            assert not np.isnan(thickness_grid).any(), f'Thickness with some nans: glacier {gl_id}'
            assert not np.isnan(err_thickness_grid).any(), f'Thickness error with some nans: glacier {gl_id}'
            assert not np.isnan(h_wgs84_grid).any(), f'h_wgs84 with some nans: glacier {gl_id}'
            assert not np.isnan(h_egm2008_grid).any(), f'h_egm2008 with some nans: glacier {gl_id}'


            # Create the Dataset
            data_dataset = xarray.Dataset({
                    'thickness': (('y', 'x'), thickness_grid),
                    'thickness_err': (('y', 'x'), err_thickness_grid),
                    'h_wgs84': (('y', 'x'), h_wgs84_grid),
                    'h_egm2008': (('y', 'x'), h_egm2008_grid),
                },
                coords={
                    'y': lat_range,
                    'x': lon_range
                }).rio.write_crs("EPSG:4326", inplace=True)

            data_dataset['thickness'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['thickness_err'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['h_wgs84'].rio.write_nodata(np.nan, inplace=True)
            data_dataset['h_egm2008'].rio.write_nodata(np.nan, inplace=True)

            # cropping with all geometries at once (it may be unnecessary)
            data_dataset = data_dataset.map(
                lambda da: da.rio.clip(
                    geometries=glacier_geometries,
                    crs="EPSG:4326",
                    drop=False,
                    invert=False,
                    all_touched=True,
                )
            )

            # enforce precise resolution
            data_dataset = data_dataset.rio.reproject(dst_crs=data_dataset.rio.crs, resolution=tif_resolution)
            assert data_dataset.rio.resolution()[0] == tif_resolution, 'Created dataset with unexpected resolution.'

            vol_from_array = 0
            area_from_array = 0
            for n, glacierID in enumerate(info.index):
                #tqdm.write(f"{n+1}/{len(info)} ID: {glacierID} process {process_name}")
                #print(f"{n+1}/{len(info)} ID: {glacierID} process {process_name}")

                areaID = info.at[glacierID, 'Area']
                nameID = info.at[glacierID, 'Name']
                volID_far = info.at[glacierID, 'vol_far']
                geomID = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id] == glacierID, 'geometry'].item()

                arrayID = data_dataset.map(
                    lambda da: da.rio.clip(
                        geometries=[geomID],
                        crs="EPSG:4326",
                        drop=True,
                        invert=False,
                        all_touched=True,
                    )
                )
                arrayID = arrayID.rio.reproject(dst_crs=data_dataset.rio.crs, resolution=tif_resolution)
                assert arrayID.rio.resolution()[0] == tif_resolution, 'Created dataset with unexpected resolution.'

                # Calculate the volume
                volID, volID_bsl = calc_volume_glacier_from_ar(ar=arrayID, area=areaID)
                #print(volID, volID_bsl, vol_montecarlo, vol_montecarlo_bsl)

                # Get ground truth measurements
                glathida_rgis_ID = glathida_rgis.loc[glathida_rgis['RGIId'] == glacierID]
                ground_truth_lons = glathida_rgis_ID['POINT_LON'].to_list()
                ground_truth_lats = glathida_rgis_ID['POINT_LAT'].to_list()
                ground_truth_meas = glathida_rgis_ID['THICKNESS'].to_list()

                # Add attributes - take inspiration from BedMachine
                arrayID.attrs['id'] = glacierID
                arrayID.attrs['name'] = nameID
                arrayID.attrs['lat'] = arrayID.coords['y'].mean().item()
                arrayID.attrs['lon'] = arrayID.coords['x'].mean().item()
                arrayID.attrs['area'] = areaID
                arrayID.attrs['volume'] = volID if no_glaciers > 1 else vol_montecarlo
                arrayID.attrs['volume_bsl'] = volID_bsl if no_glaciers > 1 else vol_montecarlo_bsl
                arrayID.attrs['volume_err'] = 0.1 * volID if no_glaciers > 1 else 0.1 * vol_montecarlo
                #arrayID.attrs['volume_farinotti'] = volID_far
                arrayID.attrs['GT_lons'] = json.dumps(ground_truth_lons)
                arrayID.attrs['GT_lats'] = json.dumps(ground_truth_lats)
                arrayID.attrs['GT_meas'] = json.dumps(ground_truth_meas)
                arrayID.attrs['units_thickness'] = 'm'
                arrayID.attrs['units_volume'] = 'km3'
                arrayID.attrs['units_area'] = 'km2'
                arrayID.attrs['resolution'] = tif_resolution
                arrayID.attrs['method'] = 'iceboost: gradient-boosted tree ensemble'
                arrayID.attrs['author'] = 'Niccolo Maffezzoli, University of California Irvine'
                #print(arrayID)

                fig, (ax1, ax2) = plt.subplots(1,2)
                arrayID['thickness'].plot(ax=ax1, cmap='turbo')
                gdf = gpd.GeoDataFrame({"geometry": [geomID]})
                gdf.plot(ax=ax1, ec='k', fc='none')
                arrayID['thickness_err'].plot(ax=ax2, cmap='binary')
                gdf.plot(ax=ax2, ec='k', fc='none')
                plt.show()

                vol_from_array += volID
                area_from_array += areaID
                #print(n, '\t', gl_id, '\t', glacierID, '\t', areaID, '\t', volID, '\t', volID_bsl)

                # 5. save .tif
                if config.deploy_global_save_figs:
                    PATH_OUT = config.model_output_global_deploy_dir
                    file_out_tif = f'{PATH_OUT}RGI{version}/rgi{rgi}/{glacierID}.tif'
                    arrayID.rio.to_raster(file_out_tif, compress="deflate")

            #print(f"Check areas: {deploy_area}, {area_from_array}")
            #print(f"Check volumes: {vol_montecarlo}, {vol_from_array}")


            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #data_dataset['thickness'].plot(ax=ax1, cmap='turbo')
            #glacier_geometries.plot(ax=ax1, ec='k', fc='none')
            #s = ax2.scatter(x=lons, y=lats, c=y_preds_glacier, s=1, cmap='turbo')
            #cb = plt.colorbar(s)
            #plt.show()


run_rgi_simulation_YN = False
if run_rgi_simulation_YN:
    t0 = time.time()
    rgi = 3
    version = '62'

    print(f"Begin regional simulation for region {rgi}, version {version}")

    rgi_products = get_rgi_products(rgi, version=version)
    coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
    link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')

    oggm_rgi_glaciers, oggm_rgi_intersects, rgi_graph, mbdf_rgi = rgi_products
    if version == '62':
        name_column_id = 'RGIId'
        name_column_area = 'Area'
        name_column_name = 'Name'
        name_column_lon, name_column_lat = 'CenLon', 'CenLat'
    elif version == '70G':
        name_column_id = 'rgi_id'
        name_column_area = 'area_km2'
        name_column_name = 'glac_name'
        name_column_lon, name_column_lat = 'cenlon', 'cenlat'

    # Get glaciers and order them in decreasing order by Area. First glaciers will be bigger and slower to process.
    oggm_rgi_glaciers = oggm_rgi_glaciers.sort_values(by=name_column_area, ascending=False)
    total_no_glaciers = len(oggm_rgi_glaciers)

    # load xgb, cat models (by default they run on cpu)
    iceboost_xgb, iceboost_cat = load_models(config)

    # List of lists of ids for multithread
    splits_ids = geographic_split_adaptive(oggm_rgi_glaciers, config.n_jobs, version)
    for n, split in enumerate(splits_ids):
        print(f"Split {n}: {len(split)}")

    multicpu = False
    if multicpu:

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
        #oggm_rgi_glaciers = oggm_rgi_glaciers.loc[(oggm_rgi_glaciers[name_column_id] == 'RGI60-06.00475') |
        #                                          (oggm_rgi_glaciers[name_column_id] == 'RGI60-06.00416')]
        #oggm_rgi_glaciers = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id] == 'RGI60-11.01450']
        #oggm_rgi_glaciers = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id] == 'RGI2000-v7.0-G-11-02596']
        #target = ['RGI60-07.00607', 'RGI60-07.00031', 'RGI60-07.00608', 'RGI60-07.00682', 'RGI60-07.00551',
        #           'RGI60-07.00552', 'RGI60-07.00027', 'RGI60-07.00062', 'RGI60-07.00028', 'RGI60-07.00423',
        #           'RGI60-07.00425', 'RGI60-07.00025', 'RGI60-07.00030', 'RGI60-07.00424', 'RGI60-07.00727',
        #           'RGI60-07.00681', 'RGI60-07.00026', 'RGI60-07.00061', 'RGI60-07.00683', 'RGI60-07.00728',
        #           'RGI60-07.00029']
        #target = ['RGI60-07.00073'] #['RGI60-06.00475']
        #target = ['RGI60-05.13567', 'RGI60-05.13501', 'RGI60-05.13722', 'RGI60-05.13726', 'RGI60-05.13495',
        #          'RGI60-05.13564', 'RGI60-05.13717', 'RGI60-05.13499', 'RGI60-05.13575', 'RGI60-05.13536',
        #          'RGI60-05.13667', 'RGI60-05.13437', 'RGI60-05.13440', 'RGI60-05.13568', 'RGI60-05.13616',
        #          'RGI60-05.13565', 'RGI60-05.13495', 'RGI60-05.13496', 'RGI60-05.13728', 'RGI60-05.13451',
        #          'RGI60-05.14872', 'RGI60-05.13505', 'RGI60-05.13429', 'RGI60-05.13499', 'RGI60-05.13655',
        #          'RGI60-05.13426', 'RGI60-05.14147', 'RGI60-05.13663', 'RGI60-05.13457']
        #target = ['RGI60-05.13501']
        #target = ['RGI60-11.01450']
        target = ['RGI60-03.01517']
        #target = ['RGI60-16.01389']

        glaciers_for_deploy = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id].isin(target)]

        pbar = tqdm(enumerate(glaciers_for_deploy[name_column_id]), total=len(glaciers_for_deploy), leave=True)
        for i, gl_id in pbar:
            pbar.set_description(f"rgi {rgi} glaciers - ID: {gl_id}")
            process_glacier([gl_id], process_idx=0)

    print(f"Finished regional simulation for rgi {rgi}, version {version} in {(time.time()-t0)/60} min.")