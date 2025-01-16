import argparse, time
import random

from pandas.core.common import random_state
from tqdm import tqdm
import copy, math
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
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE

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
parser.add_argument('--save_model', type=int, default=0, help="Save trained model or not.")
parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to yaml config file")
args = parser.parse_args()

config = misc.get_config(args.config)  # import from config.yaml

utils.get_rgi_dir(version='62')

def custom_loss(elevation):
    def loss(y_true, y_pred):
        residual = y_pred - elevation
        penalty = np.maximum(residual, 0) #** 2
        grad = 2 * (y_pred - y_true) + 2 * penalty
        hess = 2 * np.ones_like(y_true) + 2 * (residual > 0)
        return grad, hess
    return loss

def xgb_custom_obj(elevation):
    def obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        grad, hess = custom_loss(elevation)(y_true, y_pred)
        return grad, hess
    return obj

class CFG:
    features_not_used = ['Form', 'sia', 'RGI', 'lats', 'Area_icefree', 'aspect_50', 'aspect_300', 'aspect_gfa', 'Form'
                         'Slope', 'Lmax', 'Aspect', 'Cluster_glaciers', 'Cluster_geometries',
                         'Zmin', 'Zmax', 'Zmed', 'elevation_from_Zmin', 'deltaZ', 'TermType',
                           ]

    featuresSmall = ['Area',  'Perimeter', 'zmin', 'zmax', 'zmed', 'slope', 'aspect', 'curvature', 'lmax',
                'elevation', 'elevation_from_zmin', 'dist_from_border_km_geom',
                   'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
                 'curv_50',  'curv_100', 'curv_150', 'curv_300', 'curv_450', 'curv_gfa', 'dmdtda_hugo',  'deltaz',
                     'smb', 't2m', 'dist_from_ocean', 'Cluster_area', 'elevation_0_1']
    featuresBig = featuresSmall + ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa', ]

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

    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'

    xgb_params = {'tree_method': "hist",
                   'device': 'cuda',
                    'lambda': 76.814,#0.00878,
                    'alpha': 76.374, #6.3
                    'colsample_bytree': 0.9388, ## 0.8459,
                    'subsample': 0.741501, #0.809,
                    'learning_rate': 0.079244, #0.07,
                    'max_depth': 20, # 15
                    'min_child_weight': 19, #3,
                    'gamma': 0.18611, #0.0803458919901354,
                    'objective': 'reg:squarederror' #placeholder if custom loss is used
                    }
    cat_params = {
        'iterations': 10000,
        'depth': 6, #11,
        'learning_rate': 0.1, #0.12393,
        #'min_data_in_leaf': 44,
        #'l2_leaf_reg': 74.48
    }

    n_rounds = 1
    n_points_regression = 30000
    run_umap_tsne = False
    run_shap = False
    features = featuresBig

# Import the training dataset
glathida_rgis = pd.read_csv(config.metadata_csv_file, low_memory=False)
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.01517'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-01.13696'])] # Malaspina
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.10315'])] # Flade Isblink ice cap
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-19.01406'])] # suspeciously high measurements
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.13726'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.02442'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.02467'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.04255'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.04288'])]

# Regional statistics for Millan and Farinotti
calc_regional_stats_millan_and_farinotti = False
if calc_regional_stats_millan_and_farinotti:
    for rgi in sorted(glathida_rgis['RGI'].unique()):
        df = glathida_rgis.loc[glathida_rgis['RGI']==rgi]

        rmse_millan = np.sqrt(((df['THICKNESS'] - df['ith_m']) ** 2).mean())
        rmse_farinotti = np.sqrt(((df['THICKNESS'] - df['ith_f']) ** 2).mean())

        print(f"{rgi}\t{rmse_millan:.2f}\t{rmse_farinotti:.2f}")


# Add some features
glathida_rgis['lats'] = glathida_rgis['POINT_LAT']
#glathida_rgis['elevation_from_Zmin'] = glathida_rgis['elevation'] - glathida_rgis['Zmin']
#glathida_rgis['deltaZ'] = glathida_rgis['Zmax'] - glathida_rgis['Zmin']
# new
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation'] - glathida_rgis['zmin']
glathida_rgis['deltaz'] = glathida_rgis['zmax'] - glathida_rgis['zmin']


# = glathida_rgis.loc[(glathida_rgis['zmin']>0) & (glathida_rgis['zmin']<200)
#                                          & (glathida_rgis['Zmin']<1100) & (glathida_rgis['Zmin']>800)]
#print(glathida_rgis_inspect['RGIId'].unique())
#data_RGI60_0504288 = glathida_rgis.loc[glathida_rgis['RGIId']=='RGI60-05.00800']
#print(data_RGI60_0504288['zmax'].mean())
#print(data_RGI60_0504288['Zmax'].mean())
#print(data_RGI60_0504288['POINT_LAT'].mean())
#print(data_RGI60_0504288['POINT_LON'].mean())

#fig, ax = plt.subplots()
#ax.scatter(x=glathida_rgis['Zmin'], y=glathida_rgis['zmin'], s=3)
#plt.show()

# Remove nans (if any)
glathida_rgis = glathida_rgis.dropna(subset=CFG.features + ['THICKNESS'])

print(f"Overall dataset: {len(glathida_rgis)} rows, {glathida_rgis['RGI'].value_counts()} regions and {glathida_rgis['RGIId'].nunique()} glaciers.")

# umap and tsne
if CFG.run_umap_tsne:
    print(f"Begin umap and tsne")
    import umap
    from sklearn.manifold import TSNE

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, n_components=2, metric='euclidean')
    embedding_umap = reducer.fit_transform(glathida_rgis[CFG.features])
    embeddeding_tsne = TSNE(n_components=2).fit_transform(glathida_rgis[CFG.features])
    print(embedding_umap.shape)
    print(embeddeding_tsne.shape)

    fig, (ax1, ax2) = plt.subplots(1,2)
    s1 = ax1.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=glathida_rgis[CFG.target], cmap='gnuplot', s=5)
    cbar1 = plt.colorbar(s1, ax=ax1, alpha=1)
    cbar1.set_label('THICKNESS (m)', labelpad=15, rotation=270)
    s2 = ax2.scatter(embeddeding_tsne[:, 0], embeddeding_tsne[:, 1], c=glathida_rgis[CFG.target], cmap='gnuplot', s=5)
    cbar2 = plt.colorbar(s2, ax=ax2, alpha=1)
    cbar2.set_label('THICKNESS (m)', labelpad=15, rotation=270)
    plt.show()
    input('wait')

def compute_scores(y, predictions, verbose=False):
    '''returns mae, rmse, mu, med, std, slope, intercept'''
    if np.isnan(predictions).all():
        res = {'mae': np.nan, 'rmse': np.nan, 'mu': np.nan, 'med': np.nan, 'std': np.nan, 'mfit': np.nan, 'qfit': np.nan}

    else:
        # Remove NaNs from both vectors
        mask = ~np.isnan(y) & ~np.isnan(predictions)

        # Filter the vectors
        y = y[mask]
        predictions = predictions[mask]

        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = mean_squared_error(y, predictions, squared=False)
        mu = np.mean(y - predictions)
        med = np.median(y - predictions)
        std = np.std(y - predictions)
        r_squared = r2_score(y, predictions)
        slope, intercept, r_value, p_value, std_err = stats.linregress(y,predictions)
        res = {'mae': mae, 'rmse': rmse, 'mu': mu, 'med': med, 'std': std, 'mfit': slope, 'qfit': intercept}
    if verbose:
        for key in res: print(f"{key}: {res[key]:.2f}", end=", ")
    return tuple(res.values())

def objective_xgb(trial):

    # Suggest values of the hyperparameters using a trial object.
    params = {
        # To select which parameters to optimize, please look at the XGBoost documentation:
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        "objective": 'reg:squarederror',
        'tree_method': "gpu_hist",
        "n_estimators": 1000, #trial.suggest_int("n_estimators", 1, 2000),
        "early_stopping_rounds": 50, #100
        "verbosity": 0,
        'lambda': trial.suggest_float('lambda', 1e-3, 100.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 100, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
    }

    train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=0.2, seed=None) #42
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_preds, squared=False)
    return rmse

def objective_cat(trial):
    params_cat = {
        'iterations': 10000,
        'depth': trial.suggest_int('depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        #'subsample': trial.suggest_float('subsample', 0.05, 1.0),
        #'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100),
        'task_type': 'GPU',
        'devices': '0',
    }

    train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=0.2, seed=None)  # 42
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]

    cat = cb.CatBoostRegressor(**params_cat, silent=True)
    cat.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    y_preds = cat.predict(X_test)

    rmse = mean_squared_error(y_test, y_preds, squared=False)
    return rmse

optimize_xgb, optimize_cat = False, True
optune_optimize = False
if optune_optimize:
    study = optuna.create_study(direction='minimize')
    if optimize_xgb:
        study.optimize(objective_xgb, n_trials=200, n_jobs=-1)
    elif optimize_cat:
        study.optimize(objective_cat, n_trials=100)

    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)

    input('Continue')

stds_ML, meds_ML, slopes_ML, rmses_ML, rmses_xgb, rmses_cat = [], [], [], [], [], []
stds_Mil, meds_Mil, slopes_Mil, rmses_Mil = [], [], [], []
stds_Far, meds_Far, slopes_Far, rmses_Far = [], [], [], []

best_model_xgb = None
best_model_cat = None
best_rmse = 9999

for i in range(CFG.n_rounds):

    # Train, val, and test
    train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=0.2, seed=None)

    create_val = False
    if create_val:
        val = glathida_rgis.drop(test.index).sample(n=500)
        train = glathida_rgis.drop(test.index).drop(val.index)

    print(f"Iter {i} Train/Test: {len(train)}/{len(test)}, Train no. glaciers: {train['RGIId'].nunique()}, Test no. glaciers: {test['RGIId'].nunique()}")

    plot_train_test = False
    if plot_train_test:
        fig, ax = plt.subplots()
        s1 = ax.scatter(x=train['POINT_LON'], y=train['POINT_LAT'], s=10, c='b')
        s2 = ax.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=5, c='r')
        plt.show()

    plot_some_train_features = False
    if plot_some_train_features:
        print(train['sia'].describe())
        fig, ax = plt.subplots()
        #ax.hist(train['slope50'], bins=np.arange(train['slope50'].min(), train['slope50'].max(), 0.01), color='k', ec='k', alpha=.4)
        ax.hist(train['sia'], bins=200, color='k', ec='k', alpha=.4)
        plt.show()

    # Prepare the Data
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]
    y_test_m = test[CFG.millan]
    y_test_f = test[CFG.farinotti]

    # Step 4: Create DMatrix for training and testing
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # Wrap the custom objective function with the training elevation data
    # If I want to use the custom loss:
    # elevation_train, elevation_test = train['elevation'], test['elevation']
    # custom_obj = xgb_custom_obj(elevation_train)

    # Train the model
    model_xgb = xgb.train(
        CFG.xgb_params,
        dtrain,
        #obj=custom_obj, # If I want to use the custom loss:
        num_boost_round=1000, #537 #1000
        evals=[(dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=0#10
    )
    y_preds_xgb = model_xgb.predict(dtest)

    model_cat = cb.CatBoostRegressor(
        **CFG.cat_params,
        loss_function='RMSE',
        task_type="GPU",
        verbose=0#100
    )

    model_cat.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    y_preds_cat = model_cat.predict(X_test)

    # ensemble
    y_preds = 0.5 * (y_preds_xgb + y_preds_cat)

    # Shap Analysis
    if CFG.run_shap:

        print(f"Running SHAP...")
        '''Note: for reproducibility set seed=42 in create_train_test() and also random_state=42 below'''

        #explainer = shap.explainers.GPUTree(model_xgb, X_test)
        explainer = shap.explainers.Tree(model_xgb, X_test)
        shap_values = explainer(X_test.sample(2000, random_state=42), check_additivity=False)

        list_new_feature_names = [CFG.feature_human_names.get(col) for col in X_test.columns]

        fig, ax = plt.subplots()

        shap_values.feature_names = list_new_feature_names
        #shap.plots.bar(shap_values, max_display=len(CFG.features))
        shap.plots.beeswarm(shap_values, max_display=len(CFG.features), color=get_cmap('black_electric_green'), show=False)#len(CFG.features) plt.get_cmap('winter')
        cbar = fig.axes[-1]
        cbar.set_ylabel('Feature value', fontsize=16, color='grey')
        cbar.tick_params(labelsize=16, colors='grey')

        # Set the y-axis labels font size
        ax.tick_params(axis='y', labelsize=18, labelcolor='grey')
        ax.tick_params(axis='x', labelsize=16, labelcolor='grey')

        ax.set_xlabel('SHAP value', fontsize=16, color='grey')#ax.get_xlabel()

        for line in ax.lines: line.set_color('k')

        plt.tight_layout()
        plt.show()

        plot_nice_shape = True
        if plot_nice_shape:

            # Retrieve the SHAP values in a format suitable for plotting
            shap_summary_values = np.abs(shap_values.values) # (2000, 35)

            # Sort the SHAP values for better presentation in the bar plot
            sorted_indices = np.argsort(shap_summary_values.mean(axis=0))[::-1]

            # Prepare data for plotting (all features)
            all_shap_values = shap_summary_values[:, sorted_indices]
            all_feature_names = np.array(list_new_feature_names)[sorted_indices]

            # Plotting all features as a bar chart
            fig, ax = plt.subplots(figsize=(8, 10))
            bars = ax.barh(all_feature_names, all_shap_values.mean(axis=0), color='grey', alpha=0.3)

            ax.invert_yaxis()  # Invert y-axis to show highest importance at the top
            ax.set_xlabel('Mean |SHAP Value|', fontsize=16)

            ax.set_yticks(range(len(all_feature_names)))  # Set the y-tick positions
            ax.set_yticklabels(all_feature_names, fontsize=18, color='grey')  # Set the y-tick labels
            ax.tick_params(axis='x', labelsize=16)

            ax.set_ylim(ax.get_ylim()[0] - 1, ax.get_ylim()[1] + 1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

    # benchmarks
    _, rmse_xgb, _, _, _, _, _ = compute_scores(y_test, y_preds_xgb, verbose=False)
    _, rmse_cat, _, _, _, _, _ = compute_scores(y_test, y_preds_cat, verbose=False)

    mae_ML, rmse_ML, mu_ML, med_ML, std_ML, mfit_ML, qfit_ML = compute_scores(y_test, y_preds, verbose=False)
    mae_mil, rmse_mil, mu_mil, med_mil, std_mil, mfit_mil, qfit_mil = compute_scores(y_test, y_test_m, verbose=False)
    mae_far, rmse_far, mu_far, med_far, std_far, mfit_far, qfit_far = compute_scores(y_test, y_test_f, verbose=False)

    if rmse_ML < best_rmse:
        best_rmse = rmse_ML
        best_model_xgb = model_xgb
        best_model_cat = model_cat

    print(f'{i} Benchmarks ML, Millan and Farinotti: {rmse_ML:.2f} {rmse_mil:.2f} {rmse_far:.2f}')

    stds_ML.append(std_ML)
    meds_ML.append(med_ML)
    slopes_ML.append(mfit_ML)
    rmses_ML.append(rmse_ML)
    rmses_xgb.append(rmse_xgb)
    rmses_cat.append(rmse_cat)
    stds_Mil.append(std_mil)
    meds_Mil.append(med_mil)
    slopes_Mil.append(mfit_mil)
    rmses_Mil.append(rmse_mil)
    stds_Far.append(std_far)
    meds_Far.append(med_far)
    slopes_Far.append(mfit_far)
    rmses_Far.append(rmse_far)


print(f"Res. medians {np.mean(meds_ML):.2f}({np.std(meds_ML):.2f}) {np.mean(meds_Mil):.2f}({np.std(meds_Mil):.2f}) {np.mean(meds_Far):.2f}({np.std(meds_Far):.2f})")
print(f"Res. stdevs {np.mean(stds_ML):.2f}({np.std(stds_ML):.2f}) {np.mean(stds_Mil):.2f}({np.std(stds_Mil):.2f}) {np.mean(stds_Far):.2f}({np.std(stds_Far):.2f})")
print(f"Res. slopes {np.mean(slopes_ML):.2f}({np.std(slopes_ML):.2f}) {np.mean(slopes_Mil):.2f}({np.std(slopes_Mil):.2f}) {np.mean(slopes_Far):.2f}({np.std(slopes_Far):.2f})")
print(f"Rmse {np.mean(rmses_ML):.2f}({np.std(rmses_ML):.2f}) {np.mean(rmses_Mil):.2f}({np.std(rmses_Mil):.2f}) {np.mean(rmses_Far):.2f}({np.std(rmses_Far):.2f})")
print(f"Rmse xgb {np.mean(rmses_xgb):.2f}({np.std(rmses_xgb):.2f})")
print(f"Rmse cat {np.mean(rmses_cat):.2f}({np.std(rmses_cat):.2f})")
print(f"Rmse {100*(np.nanmean(rmses_Mil)-np.nanmean(rmses_ML))/np.nanmean(rmses_Mil):.1f}% better than Millan")
print(f"Rmse {100*(np.nanmean(rmses_Far)-np.nanmean(rmses_ML))/np.nanmean(rmses_Far):.1f}% better than Farinotti")

print(f"At the end of cv the best rmse is {best_rmse}")

if args.save_model:
    date_n_time = time.strftime("%Y%m%d", time.localtime())
    fileout_xgb = f"{config.model_input_dir}iceboost_xgb_{date_n_time}.json"
    fileout_cat = f"{config.model_input_dir}iceboost_cat_{date_n_time}.cbm"
    best_model_xgb.save_model(fileout_xgb)
    best_model_cat.save_model(fileout_cat, format="cbm")
    print(f'saved models {fileout_xgb} and {fileout_cat}')

# ************************************
# plot
# ************************************
plot_last_cv_iteration = False
if plot_last_cv_iteration:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))

    xmax = max(np.max(y_test), np.max(y_preds), np.max(y_test_m), np.max(y_test_f))
    max_misfit = max_misfit = max((y_test - y_preds).abs().max(),(y_test - y_test_m).abs().max(),(y_test - y_test_f).abs().max())
    #bins=np.arange(-max_misfit, max_misfit, 20)
    bins=np.arange(-500, 500, 10)

    #ax1.hist(y_test - y_preds, bins=bins, label='IceBoost', color='tab:gray', alpha=.3, histtype='stepfilled', zorder=2)
    ax1.hist(y_test - y_preds, bins=bins, edgecolor='tab:blue', facecolor='none', histtype='stepfilled', linewidth=2, zorder=2)
    #ax1.hist(y_test - y_test_m, bins=bins, label='Millan', color='tab:green', alpha=.6, histtype='stepfilled', zorder=1)
    ax1.hist(y_test - y_test_m, bins=bins, edgecolor='tab:green', facecolor='none', histtype='stepfilled', linewidth=2, zorder=1)
    #ax1.hist(y_test - y_test_f, bins=bins, label='Farinotti', color='tab:orange', alpha=.6, histtype='stepfilled', zorder=1)
    ax1.hist(y_test - y_test_f, bins=bins, edgecolor='tab:orange', facecolor='none', histtype='stepfilled', linewidth=2, zorder=1)

    s2 = ax2.scatter(x=y_test, y=y_preds, s=5, c='tab:blue', ec='tab:blue', alpha=.5)
    s3 = ax3.scatter(x=y_test, y=test['ith_m'], s=5, c='tab:green', ec='tab:green', alpha=.5)
    s4 = ax4.scatter(x=y_test, y=test['ith_f'], s=5, c='tab:orange', ec='tab:orange', alpha=.5)

    # Linear Regressions
    #fit_ML_plot = ax2.plot([0.0, xmax], [qfit_ML, qfit_ML+xmax*mfit_ML], c='b')
    #fit_millan_plot = ax3.plot([0.0, xmax], [qfit_mil, qfit_mil+xmax*mfit_mil], c='lime')
    #fit_farinotti_plot = ax4.plot([0.0, xmax], [qfit_far, qfit_far+xmax*mfit_far], c='r')

    # text
    text_ml = f"IceBoost \n(w/o spervision)\n$\\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\nrmse = {rmse_ML:.1f}"
    text_millan = f"Millan, 2022\n$\\mu$ = {mu_mil:.1f}\nmed = {med_mil:.1f}\nrmse = {rmse_mil:.1f}"
    text_farinotti = f"Farinotti, 2019\n$\\mu$ = {mu_far:.1f}\nmed = {med_far:.1f}\nrmse = {rmse_far:.1f}"
    # text boxes
    props_ML = dict(boxstyle='round', facecolor='tab:blue', ec='tab:blue', alpha=0.2)
    props_millan = dict(boxstyle='round', facecolor='tab:green', ec='tab:green', alpha=0.2)
    props_farinotti = dict(boxstyle='round', facecolor='tab:orange', ec='tab:orange', alpha=0.2)
    ax1.text(0.05, 0.98, text_ml, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props_ML)
    ax1.text(0.05, 0.68, text_millan, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props_millan)
    ax1.text(0.05, 0.43, text_farinotti, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props_farinotti)

    props_ex = dict(boxstyle='round', facecolor='none', ec='black')
    text_ex = f"Arctic Canada N. (rgi=3)\nValidation set\nno.glaciers = {test['RGIId'].nunique()}\nno.points = {len(test)}"
    ax1.text(0.6, 0.98, text_ex, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props_ex)

    ax1.set_xlabel('GT - model [m]', fontsize=13)
    ax1.set_ylabel('No. training points', fontsize=13)
    #ax1.legend(loc='best')

    for ax in (ax2, ax3, ax4):
        ax.plot([0.0, xmax], [0.0, xmax], c='tab:gray')
        ax.axis([None, xmax, None, xmax])
        ax.set_xlabel('GT ice thickness [m]', fontsize=13)
    ax2.set_ylabel('IceBoost thickness [m]', fontsize=13)
    ax3.set_ylabel('Millan et al. (2022) thickness [m]', fontsize=13)
    ax4.set_ylabel('Farinotti et al. (2019) thickness [m]', fontsize=13)

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

plot_spatial_test_predictions = False
if plot_spatial_test_predictions:

    # Note that it works for training in rgi=3, seed=42
    # Visualize test predictions
    test_glaciers_names = test['RGIId'].unique().tolist()
    print(f"Test dataset: {len(test)} points and {len(test_glaciers_names)} glaciers")

    glacier_geometries = []
    for glacier_name in test_glaciers_names:
        rgi = glacier_name[6:8]
        oggm_rgi_shp = glob(f"{config.oggm_dir}rgi/RGIV62/{rgi}*/{rgi}*.shp")[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
        glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == glacier_name]['geometry'].item()
        # print(glacier_geometry)
        glacier_geometries.append(glacier_geometry)

    fig, axes = plt.subplots(3,2, figsize=(9,10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    x0, y0, x1, y1 = -82, 77.9, -75.4, 78.7
    test_glacier_rgi_plot = 3
    focus = create_glacier_tile_dem_mosaic(minx=x0, miny=y0, maxx=x1, maxy=y1,
                                           rgi=test_glacier_rgi_plot, path_tandemx=config.tandemx_dir)


    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=90)
    hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

    y_min = min(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
    y_max = max(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
    y_min_diff = min(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
    y_max_diff = max(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
    absmax = max(abs(y_min_diff), abs(y_max_diff))

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        hillshade.plot(ax=ax, cmap='grey', alpha=0.65, zorder=0, add_colorbar=False, add_labels=False)
        for geom in glacier_geometries:
            ax.plot(*geom.exterior.xy, c='k')

    s1 = ax1.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=y_test, cmap='turbo', label='Ground Truth', vmin=y_min,vmax=y_max)
    s2 = ax2.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=y_preds, cmap='turbo', label='IceBoost', vmin=y_min,vmax=y_max)
    s3 = ax3.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=y_test_m, cmap='turbo', label='Millan et al. (2022)', vmin=y_min,vmax=y_max)
    s4 = ax4.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=y_test_f, cmap='turbo', label='Farinotti et al. (2019)', vmin=y_min,vmax=y_max)
    s5 = ax5.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=(y_test-y_preds), cmap='bwr', label='GT-IceBoost',vmin=-absmax, vmax=absmax)
    s6 = ax6.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=15, c=(y_test-y_test_f), cmap='bwr', label='GT-Farinotti',vmin=-absmax, vmax=absmax)

    cbbox1 = inset_axes(ax1, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax1.transAxes)
    cbbox2 = inset_axes(ax2, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax2.transAxes)
    cbbox3 = inset_axes(ax3, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax3.transAxes)
    cbbox4 = inset_axes(ax4, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax4.transAxes)
    cbbox5 = inset_axes(ax5, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax5.transAxes)
    cbbox6 = inset_axes(ax6, width="50%", height="20%", loc='lower left', borderpad=0, bbox_to_anchor=(0.03, 0.05, 1, 1), bbox_transform=ax6.transAxes)

    for cbbox in (cbbox1, cbbox2, cbbox3, cbbox4, cbbox5, cbbox6):
        for k in cbbox.spines: cbbox.spines[k].set_visible(False)
        cbbox.tick_params(axis='both',left=False,top=False,right=False,bottom=False,labelleft=False,labeltop=False,labelright=False,labelbottom=False)
        cbbox.set_facecolor([1, 1, 1, 0.7])

    axins1 = inset_axes(cbbox1, '90%', '20%', loc='center')
    axins2 = inset_axes(cbbox2, '90%', '20%', loc='center')
    axins3 = inset_axes(cbbox3, '90%', '20%', loc='center')
    axins4 = inset_axes(cbbox4, '90%', '20%', loc='center')
    axins5 = inset_axes(cbbox5, '90%', '20%', loc='center')
    axins6 = inset_axes(cbbox6, '90%', '20%', loc='center')

    cbar1 = plt.colorbar(s1, cax=axins1, orientation="horizontal")
    cbar1.set_label('Thickness [m]', labelpad=5, loc='left', fontsize=14)

    cbar2 = plt.colorbar(s2, cax=axins2, orientation="horizontal")
    cbar2.set_label('Thickness [m]', labelpad=5, loc='left', fontsize=14)

    cbar3 = plt.colorbar(s3, cax=axins3, orientation="horizontal")
    cbar3.set_label('Thickness [m]', labelpad=5, loc='left', fontsize=14)

    cbar4 = plt.colorbar(s4, cax=axins4, orientation="horizontal")
    cbar4.set_label('Thickness [m]', labelpad=5, loc='left', fontsize=14)

    cbar5 = plt.colorbar(s5, cax=axins5, orientation="horizontal")
    cbar5.set_label('GT-IceBoost [m]', labelpad=5, loc='left', fontsize=14)

    cbar6 = plt.colorbar(s6, cax=axins6, orientation="horizontal")
    cbar6.set_label('GT-Farinotti [m]', labelpad=5, loc='left', fontsize=14)

    #cbar1 = plt.colorbar(s1, ax=ax1)
    #cbar2 = plt.colorbar(s2, ax=ax2)
    #cbar3 = plt.colorbar(s3, ax=ax3)
    #cbar4 = plt.colorbar(s4, ax=ax4)
    #cbar5 = plt.colorbar(s5, ax=ax5)
    #cbar6 = plt.colorbar(s6, ax=ax6)

    for cbar in (cbar1, cbar2, cbar3, cbar4, cbar5, cbar6):
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=11)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.03, 0.96, "a) GT", transform=ax1.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    ax2.text(0.03, 0.96, "b) IceBoost (this work)", transform=ax2.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    ax3.text(0.03, 0.96, "c) Millan et al. (2022)", transform=ax3.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    ax4.text(0.03, 0.96, "d) Farinotti et al. (2019)", transform=ax4.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    ax5.text(0.03, 0.96, "e)", transform=ax5.transAxes, fontsize=13, verticalalignment='top', bbox=props)
    ax6.text(0.03, 0.96, "f) ", transform=ax6.transAxes, fontsize=13, verticalalignment='top', bbox=props)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    plt.tight_layout()
    #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/fig_2_supp_info.png", dpi=100)
    plt.show()


# *********************************************
# Model deploy
# *********************************************
glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-11.01450', rgi=11, version='70G', area=0, seed=None)
#glacier_name_for_generation = 'RGI2000-v7.0-G-11-02596'
# RGI60-01.13696, RGI60-03.01517
# RGI60-19.00748 molto bello
# 'RGI60-13.37753', RGI60-13.43528 RGI60-13.54431
# glacier_name_for_generation = 'RGI60-07.00228' #RGI60-07.00027 'RGI60-11.01450' RGI60-07.00552,
# glacier_name_for_generation = 'RGI60-07.00832' very nice
# RGI60-03.00832 la differenza tra i modelli è molto interessante
# RGI60-03.01708
#'RGI60-03.01632', 'RGI60-07.01482' ML simile agli altri 2 in termini di alte profondita
# 'RGI60-03.00251' Dobbin Bay, 'RGI60-07.00124' Renardbreen, 'RGI60-11.01328' Unteraargletscher, 'RGI60-11.01478'
# 'RGI60-03.02469', 'RGI60-03.01483, RGI60-03.01517 ML << Millan/Farinotti. This is super interesting. These are marine-term glaciers,
# I remember Millan mentioning that marine term glaciers have a bias towards bigger thickness (high speed, low slope)
# 'RGI60-03.01466' RGI60-04.05745 << M-F
# 'RGI60-03.00228'
# Barnes Ice Cap e' molto interessante confrontare le predizioni perche' c'e' area senza ghiaccio attorno per confrontare
# i bedrock! E i volumi di ghiaccio sono enormi: RGI60-04.06187
# 'RGI60-11.01492' we can see millan's effect of velocity products on ice thickness calculation
# RGI60-07.00027 biggest in Svalbard
# RGI60-03.01710 biggest in Arctic Canada (Wykeham Glacier South)
# 'RGI60-07.01464' Holtedahlfonna
# in 'RGI60-08.01657' and RGI60-08.01641 I see Millan having gaps (in v hence in ith_m).
# no Millan data: RGI60-08.03159, RGI60-08.03084 controlla questo
# RGI60-03.00862 and RGI60-03.04229 have produced points in another utm zone wrt glacier center (issue?)
# RGI60-03.02811 in interesting since on top Millan has no data, so what is the effect or modeling with/without v ?
#RGI60-07.00174 Farinotti has a gap ?
# RGI60-07.01575 has no millan data
# RGI60-13.54431
# 'RGI60-14.06794' Baltoro glacier
# 'RGI60-04.04988' potrebbe essere emblematico di quanto Far sovrastimi ?
# RGI60-04.05758, RGI60-04.05748 look at the small spatial scales which ML can appreciated (look in relation to hillshade)
# 'RGI60-05.11268' has tandem-x with bugs. The bugs are reflected in my solution (since the model strongly uses the slope)
# RGI60-05.10988 I think Millan and Farinotti solutions are wrong.
# RGI60-05.10743 Farinotti looks badly wrong.
# RGI60-05.15702 big difference between Millan and IceBoost-Farinotti
# RGI60-05.10148 big difference between IceBoost and Millan-Farinotti
# RGI60-09.00909 RGI60-09.00520 I think iceboost is very wrong
# high frequency features in 'RGI60-14.16214' or RGI60-15.04541 RGI60-16.00244 RGI60-16.00776 RGI60-18.02210
# Probably caused by some features, check curv_50 or elevation


# Generate points for one glacier
test_glacier_rgi, version = get_version_and_rgi_from_id(glacier_name_for_generation)
rgi_products = get_rgi_products(test_glacier_rgi, version=version)
coastline_dataframe = get_coastline_dataframe(config.coastlines_gshhg_dir)
link_ids_rgi6_rgi7 = pd.read_csv(config.link_ids_rgi6_rgi7_csv, index_col='rgi_id_7')


data_generator  = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
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

glacier_area = info['Area'].sum()

h_wgs84 = data['elevation'].to_numpy()
lats = data['lats'].to_numpy()
lons = data['lons'].to_numpy()
h_egm2008 = calc_geoid_heights(lons=lons, lats=lats, h_wgs84=h_wgs84)


X_test_glacier = data[CFG.features]
perturbate_features = False
if perturbate_features:
    X_test_glacier = X_test_glacier + np.random.normal(
        loc=0, scale=0.2 * np.abs(X_test_glacier),  # 10% of each feature's value
    )

y_test_glacier_m = data[CFG.millan]
y_test_glacier_f = data[CFG.farinotti]
dtest = xgb.DMatrix(data=X_test_glacier)

no_millan_data = np.isnan(y_test_glacier_m).all()
no_farinotti_data = np.isnan(y_test_glacier_f).all()

y_preds_glacier_xgb = best_model_xgb.predict(dtest)
y_preds_glacier_cat = best_model_cat.predict(X_test_glacier)

y_preds_glacier = 0.5 * (y_preds_glacier_xgb + y_preds_glacier_cat)

y_preds_diff_xgb_cat = np.abs(y_preds_glacier_xgb-y_preds_glacier_cat)

#fig, (ax1, ax2, ax3) = plt.subplots(1,3)
#s1=ax1.scatter(x=lons, y=lats, c=y_preds_glacier_xgb, s=1)
#s2=ax2.scatter(x=lons, y=lats, c=y_preds_glacier_cat, s=1)
#s3=ax3.scatter(x=lons, y=lats, c=y_preds_diff_xgb_cat, s=1)
#cb1 = plt.colorbar(s1, cmap='turbo')
#cb2 = plt.colorbar(s2, cmap='turbo')
#cb3 = plt.colorbar(s3, cmap='viridis')
#plt.show()

#plot_feature_scatter(config, test_glacier)

# Set negative predictions to zero
y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

# Calculate the bedrock elevations
bedrock_elevations_ML = data['elevation'] - y_preds_glacier
bedrock_elevations_Millan = data['elevation'] - y_test_glacier_m
bedrock_elevations_Far = data['elevation'] - y_test_glacier_f

# Begin to extract all necessary things to plot the result
#oggm_rgi_shp = glob(f"{config.oggm_dir}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
#oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
oggm_rgi_glaciers, oggm_rgi_intersects, rgi_graph, mbdf_rgi = rgi_products
if version == '62': name_column_id = 'RGIId'
elif version == '70G': name_column_id = 'rgi_id'
#glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()
glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers[name_column_id] == glacier_name_for_generation, 'geometry'].item()
#glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['Area'].item()
#glacier_area = test_glacier.iloc[0]['Area']

exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
x0, y0, x1, y1 = exterior_ring.bounds
dx, dy = x1 - x0, y1 - y0
glacier_nunataks_list = [nunatak for nunatak in glacier_geometry.interiors]

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

# Get Farinotti file and calculate the volume
#todo: currently import farinotti only if i'm running rgi62. Therefore this does not work for rgi7
if (version =='62' and not no_farinotti_data):
    test_glacier_folder_farinotti = glob(f"{config.farinotti_icethickness_dir}/*{test_glacier_rgi}/")[0]
    ice_farinotti = rioxarray.open_rasterio(test_glacier_folder_farinotti+glacier_name_for_generation+'_thickness.tif')
    res_farinotti = ice_farinotti.rio.resolution()[0]
    vol_farinotti_published = 1.e-9 * (res_farinotti ** 2) * np.nansum(ice_farinotti.values) # Volume Farinotti km3
else:
    print(f"Farinotti glacier {glacier_name_for_generation} not found in OGGM V62 database.")
    vol_farinotti_published = np.nan

# Calculate the glacier volume using the 3 models
vol_ML, _, _ = calc_volume_glacier(y=y_preds_glacier, area=glacier_area, h_egm2008=h_egm2008)
err_vol_ML = np.sqrt(np.sum(y_preds_diff_xgb_cat**2)) * 0.001 * glacier_area / len(y_preds_diff_xgb_cat)
vol_millan, _, _ = calc_volume_glacier(y=y_test_glacier_m, area=glacier_area, h_egm2008=h_egm2008)
vol_farinotti, _, _ = calc_volume_glacier(y=y_test_glacier_f, area=glacier_area, h_egm2008=h_egm2008)

print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, "
      f"volML: {vol_ML:.4g} km3 pm {err_vol_ML:.4g} km3 "
      f"volMil: {vol_millan:.4g} km3 "
      f"volFar: {vol_farinotti:.4g} km3 volFar published: {vol_farinotti_published:.4g} km3"
      f"Far mismatch {100*abs(vol_farinotti-vol_farinotti_published)/vol_farinotti_published:.2f}%")

print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

# Visualize test predictions of specific glacier
y_min = min(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))
y_max = max(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))

vmin = min(y_preds_glacier)
vmax = max(y_preds_glacier)

create_tif_file = False
if create_tif_file:
    lons = test_glacier['lons'].to_numpy()
    lats = test_glacier['lats'].to_numpy()

    # Create the grid
    grid_res = 0.001  # Adjust as needed
    lat_grid = np.arange(lats.min(), lats.max(), grid_res)
    lon_grid = np.arange(lons.min(), lons.max(), grid_res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate the z values onto the grid
    z_grid = griddata((lons, lats), y_preds_glacier, (lon_grid, lat_grid), method='linear')

    data_array = xarray.DataArray(z_grid, coords=[lat_grid[:, 0], lon_grid[0, :]], dims=['lat', 'lon'])
    data_array = data_array.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    data_array = data_array.rio.write_crs("EPSG:4326")
    data_array.rio.to_raster("ex.tif")

    plt.imshow(z_grid, cmap='turbo')
    plt.show()

plot_fancy_ML_prediction = True
if plot_fancy_ML_prediction:
    fig, ax = plt.subplots(figsize=(8,6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)

    im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

    s1 = ax.scatter(x=lons, y=lats, s=1, c=y_preds_glacier,
                     cmap='turbo', label='ML', zorder=1, vmin=vmin,vmax=vmax)
    s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                            cmap='turbo', ec='grey', lw=0.5, s=35, vmin=vmin,vmax=vmax)

    cbar = plt.colorbar(s1, ax=ax)
    cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
    cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
    #cbar.set_label(r'mass balance (mm w.e. yr$^{-1}$)', labelpad=15, rotation=90, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
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

    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    #plt.savefig('/home/maffe/Downloads/RGI60-1313574_CCAI.png', dpi=200)
    plt.show()

plot_fancy_ML_Mil_Far_prediction = False
if plot_fancy_ML_Mil_Far_prediction:
    #fig = plt.figure(figsize=(15, 6))
    fig = plt.figure(figsize=(4.5, 7))
    #gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust the width ratios
    #gs = GridSpec(4, 2, width_ratios=[1, 0.05], height_ratios=[1, 0.02, 1, 0.02])
    #gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])
    gs = GridSpec(3, 1, height_ratios=[0.03, 1, 1])

    # Create the axes
    #ax1 = fig.add_subplot(gs[0])
    #ax2 = fig.add_subplot(gs[1])
    #ax3 = fig.add_subplot(gs[2])
    #cax = fig.add_subplot(gs[3])  # Colorbar axis

    #ax1 = fig.add_subplot(gs[0, 0])
    #cax1 = fig.add_subplot(gs[0, 1])
    #ax2 = fig.add_subplot(gs[1, 0])
    #cax2 = fig.add_subplot(gs[1, 1])

    cax = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)
    hillshade.values = earthpy.spatial.hillshade(hillshade, azimuth=315, altitude=0)

    im1 = hillshade.plot(ax=ax1, cmap='grey', alpha=0.8, zorder=0, add_colorbar=False)
    im2 = hillshade.plot(ax=ax2, cmap='grey', alpha=0.8, zorder=0, add_colorbar=False)
    #im3 = hillshade.plot(ax=ax3, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

    vmin, vmax = 0, 750
    s1 = ax1.scatter(x=lons, y=lats, s=1, c=y_preds_glacier, cmap='turbo', label='ML', vmin=vmin, vmax=vmax)
    if not no_millan_data:
        s2 = ax2.scatter(x=lons, y=lats, s=1, c=y_test_glacier_m, cmap='turbo', label='Millan', vmin=vmin, vmax=vmax)
    #if not no_farinotti_data:
    #    s2 = ax2.scatter(x=lons, y=lats, s=1, c=y_test_glacier_f, cmap='turbo', label='Farinotti', vmin=vmin, vmax=vmax)

    #ax1.set_title(f"IceBoost: {vol_ML:.4g} km$^3$", fontsize=16)
    #ax2.set_title(f"Millan et al. (2022): {vol_millan:.4g} km$^3$", fontsize=16)
    #ax3.set_title(f"Farinotti et al. (2019): {vol_farinotti:.4g} km$^3$", fontsize=16)

    #for ax in (ax1, ax2, ax3):
    ax1.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                cmap='turbo', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)
    glathida_rgis = pd.read_csv(config.metadata_csv_file, low_memory=False)
    ax2.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
               cmap='turbo', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)


    #cbar1 = plt.colorbar(s1, cax=cax)#ax=ax1)
    #cbar1 = plt.colorbar(s1, cax=cax1)#ax=ax1)
    #cbar1.set_label('Thickness [m]', labelpad=15, rotation=90, fontsize=13)
    #cbar1.ax.tick_params(labelsize=13)

    #cbar2 = plt.colorbar(s2, cax=cax2)
    #cbar2.set_label('Thickness [m]', labelpad=15, rotation=90, fontsize=13)#16
    #cbar2.ax.tick_params(labelsize=13)#16

    cbar = plt.colorbar(s1, cax=cax, orientation='horizontal')
    cbar.set_label('Thickness [m]', labelpad=5, fontsize=12, loc='center')

    cbar.ax.xaxis.set_label_position('top')  # Position the labels on top
    cbar.ax.tick_params(which='both', length=0)  # Set tick lengths to 0
    cbar.ax.tick_params(labelsize=12)

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

    #for ax in (ax1, ax2, ax3):
    for ax in (ax1, ax2):
        ax.plot(*exterior_ring.xy, c='k')
        for nunatak in glacier_nunataks_list:
            ax.plot(*nunatak.xy, c='k', lw=0.8)

        # ax.legend(fontsize=14, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Lon [$^{\\circ}$E]', fontsize=12)#14 #16
        ax.set_ylabel('Lat [$^{\\circ}$N]', fontsize=12)#14 #16
        ax.tick_params(axis='both', labelsize=12)#12#16
        #ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xlabel("")

    #ax1.yaxis.set_label_position('right')  # Move the y-axis label
    #ax1.yaxis.tick_right()  # Move the y-ticks and their labels
    #ax2.yaxis.set_label_position('right')  # Move the y-axis label
    #ax2.yaxis.tick_right()  # Move the y-ticks and their labels

    ax1.set_title("")
    ax2.set_title("")

    # Text boxes
    iceboost_text = (f"a) IceBoost w/o supervision")
    other_text = (f"c) Millan et al. (2022)")
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.03, 0.97, iceboost_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax2.text(0.03, 0.97, other_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    ax1.text(0.03, 0.1, f"Vol = {vol_ML:.4g} km$^3$", transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax2.text(0.03, 0.1, f"Vol = {vol_millan:.4g} km$^3$", transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    #ax2.axis('off')
    #ax3.axis('off')

    #cax1.set_visible(False)
    #cax2.set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    #ax2.set_position([ax2.get_position().x0, ax2.get_position().y0 + 0.05, ax2.get_position().width, ax2.get_position().height])
    #plt.subplots_adjust(hspace=0.1)
    #plt.savefig(f"{config.model_output_results_dir}{glacier_name_for_generation}.png", dpi=100)
    #plt.savefig(f"/home/maffe/Downloads/new_figures_iceboost_paper/fig4_without_sup.png", dpi=100)
    plt.show()

run_shap_single_glacier = True
if run_shap_single_glacier:
    print(f"Running SHAP on single glacier...")
    '''Note: for reproducibility set seed=42 in create_train_test() and also random_state=42 below'''

    data["xgb"] = y_preds_glacier_xgb
    data["cat"] = y_preds_glacier_cat
    #data_glacier_sample = data.sample(frac=1, random_state=42)
    data_glacier_sample = data.sample(n=500, random_state=42)
    #explainer = shap.explainers.GPUTree(model_xgb, X_test)
    explainer = shap.explainers.Tree(best_model_xgb, data_glacier_sample[CFG.features])
    shap_values = explainer(data_glacier_sample[CFG.features], check_additivity=False)

    print(type(shap_values))
    print(shap_values.shape)
    print(type(shap_values.values))


    list_new_feature_names = [CFG.feature_human_names.get(col) for col in data_glacier_sample[CFG.features].columns]
    shap_values.feature_names = list_new_feature_names

    # Add SHAP values as new columns to X_test_glacier_sample
    for feature_name, shap_value in zip(data_glacier_sample[CFG.features].columns, shap_values.values.T):
        data_glacier_sample[f'shap_{feature_name}'] = shap_value

    print(list(data_glacier_sample))

    fig, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.scatter(x=data_glacier_sample['lons'].to_numpy(), y=data_glacier_sample['lats'].to_numpy(),
               c=np.abs(shap_values.values).sum(axis=1), s=2, cmap='Blues')
    cb1 = plt.colorbar(im1)
    cb1.set_label('Sum(abs(shap)) (a.u.)')
    im2 = ax2.scatter(x=data_glacier_sample['lons'].to_numpy(), y=data_glacier_sample['lats'].to_numpy(),
               c=np.abs(data_glacier_sample['xgb']-data_glacier_sample['cat']), s=2, cmap='Reds')
    cb2 = plt.colorbar(im2)
    cb2.set_label('|xgb-cat| (meters)')
    plt.show()

    fig, ax = plt.subplots()
    #shap.plots.bar(shap_values, max_display=len(CFG.features))
    shap.plots.beeswarm(shap_values, max_display=len(CFG.features), color=get_cmap('black_electric_green'), show=False)#len(CFG.features) plt.get_cmap('winter')
    cbar = fig.axes[-1]
    cbar.set_ylabel('Feature value', fontsize=16, color='grey')
    cbar.tick_params(labelsize=16, colors='grey')

    # Set the y-axis labels font size
    ax.tick_params(axis='y', labelsize=18, labelcolor='grey')
    ax.tick_params(axis='x', labelsize=16, labelcolor='grey')

    ax.set_xlabel('SHAP value', fontsize=16, color='grey')#ax.get_xlabel()

    for line in ax.lines: line.set_color('k')

    plt.tight_layout()
    plt.show()

    plot_nice_shape = True
    if plot_nice_shape:
        # Retrieve the SHAP values in a format suitable for plotting
        shap_summary_values = np.abs(shap_values.values)  # (2000, 35)

        # Sort the SHAP values for better presentation in the bar plot
        sorted_indices = np.argsort(shap_summary_values.mean(axis=0))[::-1]

        # Prepare data for plotting (all features)
        all_shap_values = shap_summary_values[:, sorted_indices]
        all_feature_names = np.array(list_new_feature_names)[sorted_indices]

        # Plotting all features as a bar chart
        fig, ax = plt.subplots(figsize=(8, 10))
        bars = ax.barh(all_feature_names, all_shap_values.mean(axis=0), color='grey', alpha=0.3)

        ax.invert_yaxis()  # Invert y-axis to show highest importance at the top
        ax.set_xlabel('Mean |SHAP Value|', fontsize=16)

        ax.set_yticks(range(len(all_feature_names)))  # Set the y-tick positions
        ax.set_yticklabels(all_feature_names, fontsize=18, color='grey')  # Set the y-tick labels
        ax.tick_params(axis='x', labelsize=16)

        ax.set_ylim(ax.get_ylim()[0] - 1, ax.get_ylim()[1] + 1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()