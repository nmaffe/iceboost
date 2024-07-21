import argparse, time
import random
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
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
import optuna
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import calc_volume_glacier, get_random_glacier_rgiid, create_train_test, get_cmap

#import warnings
#warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/media/maffe/nvme/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata35_hmineq0.0_tmin20050000_mean_grid_100.csv", help="Training dataset.")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/media/maffe/nvme/Farinotti/composite_thickness_RGI60-all_regions/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--mosaic', type=str,default="/media/maffe/nvme/Tandem-X-EDEM/", help="Path to Tandem-X-EDEM")
parser.add_argument('--oggm', type=str,default="/home/maffe/OGGM/", help="Path to OGGM folder")
parser.add_argument('--save_model', type=int, default=0, help="Save the model")
parser.add_argument('--save_outdir', type=str, default="/home/maffe/PycharmProjects/skynet/metadata/saved_iceboost/", help="Saved model dir.")
parser.add_argument('--save_outname', type=str, default="iceboost", help="Saved model name.")

args = parser.parse_args()
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
                         ]

    featuresSmall = ['Area',  'Perimeter', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'Aspect',  'TermType',
                'elevation', 'elevation_from_zmin', 'dist_from_border_km_geom',
                   'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
                 'curv_50', 'curv_300', 'curv_gfa', 'dmdtda_hugo',  'deltaZ',
                     'smb', 't2m', 'dist_from_ocean', ]
    featuresBig = featuresSmall + ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa', ]

    feature_human_names = {
        'Area': 'Area', 'Zmin': r'H$_{min}$', 'Zmax': r'H$_{max}$', 'Zmed': r'H$_{med}$', 'Slope': 'Slope', 'Lmax': 'Lmax',
        'Form': 'Form', 'TermType': 'TermType', 'Aspect': 'Aspect', 'elevation': 'h', 'elevation_from_zmin': r'h-H$_{min}$',
        'dist_from_border_km_geom': r'd$_{noice}$', 'slope50': r's$_{50}$', 'slope75': r's$_{75}$', 'slope100': r's$_{100}$',
        'slope125': r's$_{125}$', 'slope150': r's$_{150}$', 'slope300': r's$_{300}$', 'slope450': r's$_{450}$',
        'slopegfa': r's$_{gfa}$', 'curv_50': r'c$_{50}$',
        'curv_300': r'c$_{300}$', 'curv_gfa': r'c$_{gfa}$', 'aspect_50': r'a$_{50}$', 'aspect_300': r'a$_{300}$', 'aspect_gfa': r'a$_{gfa}$',
        'dmdtda_hugo': 'MB', 'smb': 'mb', 't2m': 't2m', 'v50': r'v$_{50}$', 'v100': r'v$_{100}$', 'v150': r'v$_{150}$',
        'v300': r'v$_{300}$', 'v450': r'v$_{450}$', 'vgfa': r'v$_{gfa}$', 'Area_icefree': 'Area icefree', 'Perimeter': 'Perimeter',
        'deltaZ': r'$\Delta$H', 'RGI': 'RGI', 'dist_from_ocean': r'd$_{ocean}$'
    }

    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'

    xgb_params = {'tree_method': "hist",
                   'device': 'cuda',
                    'lambda': 0.00878,
                    'alpha': 6.3, #6.3
                    'colsample_bytree': 0.8459,
                    'subsample': 0.809,
                    'learning_rate': 0.07,
                    #'n_estimators': 537,#537
                    #'num_boost_round': 537,
                    'max_depth': 15, # 15
                    'min_child_weight': 3,
                    'gamma': 0.0803458919901354,
                    'objective': 'reg:squarederror' #placeholder if custom loss is used
                    } #
    #model = xgb.XGBRegressor(n_estimators=537, max_depth=15, learning_rate=0.07, min_child_weight=8,
    #                        subsample=0.808, gamma=2.303, alpha=0.698, reg_lambda=5.009,
    #                         objective=xgbloss, tree_method="gpu_hist")
    n_rounds = 1
    n_points_regression = 30000
    run_umap_tsne = False
    run_shap = False
    features = featuresBig

# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
glathida_rgis.loc[glathida_rgis['RGIId'] == 'RGI60-19.01406', 'THICKNESS'] /= 10.
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-19.01406'])] # suspeciously high measurements
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-01.13696'])] # Malaspina
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.10315'])] # Flade Isblink ice cap
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-05.13726'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.02442'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.01517'])]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-03.02467'])]

# Replace zeros
#glathida_rgis_specific = glathida_rgis.loc[glathida_rgis['RGIId'] == 'RGI60-03.01517']
#fig, ax = plt.subplots()
#s = ax.scatter(x=glathida_rgis_specific['POINT_LON'], y=glathida_rgis_specific['POINT_LAT'],
#               c=glathida_rgis_specific['THICKNESS'], s=50, cmap='jet', vmin=0, vmax=750)
#cbar = plt.colorbar(s)
#plt.show()

# Remove zeros by replacing them with Millan and Farinotti average
#glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, 'THICKNESS'] = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)
glathida_rgis_zeros = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0.0]
#glathida_rgis = glathida_rgis[glathida_rgis['THICKNESS'] != 0.0]
glathida_nan = glathida_rgis[glathida_rgis.isna().any(axis=1)]

glathida_19 = glathida_rgis[glathida_rgis['RGI'].isin([19])]
#fig, ax = plt.subplots()
#s1 = ax.scatter(x=glathida_19['POINT_LON'], y=glathida_19['POINT_LAT'], s=2, c=glathida_19['THICKNESS'])
#s2 = ax.scatter(x=glathida_nan['POINT_LON'], y=glathida_nan['POINT_LAT'], s=2, c='r')#c=np.sqrt(glathida_nan['vx']**2+glathida_nan['vy']**2))
#cbar = plt.colorbar(s1)
#plt.show()

# Regional statistics for Millan and Farinotti
calc_regional_stats_millan_and_farinotti = True
if calc_regional_stats_millan_and_farinotti:
    for rgi in sorted(glathida_rgis['RGI'].unique()):
        df = glathida_rgis.loc[glathida_rgis['RGI']==rgi]

        rmse_millan = np.sqrt(((df['THICKNESS'] - df['ith_m']) ** 2).mean())
        rmse_farinotti = np.sqrt(((df['THICKNESS'] - df['ith_f']) ** 2).mean())

        print(f"{rgi}\t{rmse_millan:.2f}\t{rmse_farinotti:.2f}")


# Filter out some portion of it
#glathida_rgis = glathida_rgis.loc[glathida_rgis['THICKNESS']>=CFG.min_thick_value_train]
#glathida_rgis = glathida_rgis.loc[(glathida_rgis['RGI'] == 3) | (glathida_rgis['RGI'] == 7)]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGI'].isin([19])]
#glathida_rgis = glathida_rgis[glathida_rgis['RGI'].isin([19])]

# Add some features
glathida_rgis['lats'] = glathida_rgis['POINT_LAT']
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation'] - glathida_rgis['Zmin']
glathida_rgis['deltaZ'] = glathida_rgis['Zmax'] - glathida_rgis['Zmin']
#glathida_rgis['hbahrm'] = 0.03*(glathida_rgis['Area']**0.375)*1000 # Bahr's approximation: h in meters
glathida_rgis['hbahrm2'] = np.sqrt(glathida_rgis['Area'])*glathida_rgis['dist_from_border_km_geom']/glathida_rgis['Lmax']
A = 24 * np.power(10, -25.0) #s−1 Pa−3
rho, g, n = 917., 9.81, 3
#glathida_rgis['sia1'] = ((glathida_rgis['v100']*(1-0.1)*(n+1))/(2*A*(rho*g*glathida_rgis['slope100'])**3))**(1./(n+1))
#glathida_rgis['sia2'] = ((glathida_rgis['v100']*(1-0.2)*(n+1))/(2*A*(rho*g*glathida_rgis['slope100'])**3))**(1./(n+1))
#glathida_rgis['sia3'] = ((glathida_rgis['v100']*(1-0.5)*(n+1))/(2*A*(rho*g*glathida_rgis['slope100'])**3))**(1./(n+1))
glathida_rgis['sia'] = glathida_rgis['v100']/(glathida_rgis['slope100']**3)

# Remove nans (this is an overkill - i want ideally to remove nans only in the training features)
#glathida_rgis = glathida_rgis.dropna()
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

def objective(trial):

    # Suggest values of the hyperparameters using a trial object.
    params = {
        # To select which parameters to optimize, please look at the XGBoost documentation:
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        "objective": 'reg:squarederror',
        'tree_method': "gpu_hist",
        "n_estimators": 1000, #trial.suggest_int("n_estimators", 1, 2000),
        "verbosity": 0,
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        #'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0), # some say either lambda or alpha is enough
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0), # [0.7, 1] are usually the best
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0), # [0.5,1] usually seem to work best
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200), # some say [1, 200]
    }

    train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=0.2, seed=None) #42
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    y_preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_preds, squared=False)
    return rmse

optune_optimize = False
if optune_optimize:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)

    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)

    input('Continue')

stds_ML, meds_ML, slopes_ML, rmses_ML = [], [], [], []
stds_Mil, meds_Mil, slopes_Mil, rmses_Mil = [], [], [], []
stds_Far, meds_Far, slopes_Far, rmses_Far = [], [], [], []

best_model = None
best_rmse = 9999

for i in range(CFG.n_rounds):

    # Train, val, and test
    train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=.2, seed=None)

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

    elevation_train, elevation_test = train['elevation'], test['elevation']

    # Step 4: Create DMatrix for training and testing
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # Wrap the custom objective function with the training elevation data
    # If I want to use the custom loss:
    # custom_obj = xgb_custom_obj(elevation_train)

    # Train the model
    model = xgb.train(
        CFG.xgb_params,
        dtrain,
        #obj=custom_obj, # If I want to use the custom loss:
        num_boost_round=537,
        evals=[(dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    y_preds = model.predict(dtest)

    '''
    ### sklearn-api: initialize the model
    model = xgb.XGBRegressor(**CFG.xgb_params)
    #model = CFG.model

    #model.fit(X_train, y_train)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    y_preds = model.predict(X_test)
    '''

    # Shap Analysis
    if CFG.run_shap:

        explainer = shap.explainers.GPUTree(model, X_test)
        shap_values = explainer(X_test.sample(2000), check_additivity=False)

        list_new_feature_names = [CFG.feature_human_names.get(col) for col in X_test.columns]

        fig, ax = plt.subplots()

        shap_values.feature_names = list_new_feature_names
        #shap.plots.bar(shap_values, max_display=len(CFG.features))
        shap.plots.beeswarm(shap_values, max_display=len(CFG.features), color=get_cmap('black_electric_green'), show=False)#len(CFG.features) plt.get_cmap('winter')
        cbar = fig.axes[-1]
        cbar.set_ylabel('Feature value', fontsize=16, color='grey')
        cbar.tick_params(labelsize=14)
        cbar.tick_params(labelsize=14, colors='grey')

        # Set the y-axis labels font size
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, color='grey')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, color='grey')
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
            fig, ax = plt.subplots(figsize=(8, 8))
            bars = ax.barh(all_feature_names, all_shap_values.mean(axis=0), color='grey', alpha=0.3)

            ax.invert_yaxis()  # Invert y-axis to show highest importance at the top

            ax.set_yticklabels(all_feature_names, fontsize=14, color='grey')
            ax.set_xlabel('Mean |SHAP Value|', fontsize=16)
            ax.tick_params(axis='x', labelsize=14)

            ax.set_ylim(ax.get_ylim()[0] - 1, ax.get_ylim()[1] + 1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

    # benchmarks
    mae_ML, rmse_ML, mu_ML, med_ML, std_ML, mfit_ML, qfit_ML = compute_scores(y_test, y_preds, verbose=False)
    mae_mil, rmse_mil, mu_mil, med_mil, std_mil, mfit_mil, qfit_mil = compute_scores(y_test, y_test_m, verbose=False)
    mae_far, rmse_far, mu_far, med_far, std_far, mfit_far, qfit_far = compute_scores(y_test, y_test_f, verbose=False)

    #*** Note: here it is very important since this is the policy to decide which model will be selected for deploy
    #rmse = model_metrics_ML['rmse']
    if rmse_ML < best_rmse:
        best_rmse = rmse_ML
        best_model = model

    print(f'{i} Benchmarks ML, Millan and Farinotti: {rmse_ML:.2f} {rmse_mil:.2f} {rmse_far:.2f}')

    stds_ML.append(std_ML)
    meds_ML.append(med_ML)
    slopes_ML.append(mfit_ML)
    rmses_ML.append(rmse_ML)
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
print(f"Rmse {100*(np.nanmean(rmses_Mil)-np.nanmean(rmses_ML))/np.nanmean(rmses_Mil):.1f}% better than Millan")
print(f"Rmse {100*(np.nanmean(rmses_Far)-np.nanmean(rmses_ML))/np.nanmean(rmses_Far):.1f}% better than Farinotti")

print(f"At the end of cv the best rmse is {best_rmse}")

if args.save_model==1:
    date_n_time = time.strftime("%Y%m%d", time.localtime())
    fileout = f"{args.save_outdir}{args.save_outname}_{date_n_time}.json"
    best_model.save_model(fileout)
    print(f'saved: {fileout}')

# ************************************
# plot
# ************************************
plot_last_cv_iteration = False
if plot_last_cv_iteration:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    s1 = ax1.scatter(x=y_test, y=test['ith_m'], s=5, c='g', alpha=.3)
    s1 = ax1.scatter(x=y_test, y=test['ith_f'], s=5, c='r', alpha=.3)
    s1 = ax1.scatter(x=y_test, y=y_preds, s=5, c='aliceblue', ec='b', alpha=.5)

    xmax = max(np.max(y_test), np.max(y_preds), np.max(y_test_m), np.max(y_test_f))

    fit_ML_plot = ax1.plot([0.0, xmax], [qfit_ML, qfit_ML+xmax*mfit_ML], c='b')
    fit_millan_plot = ax1.plot([0.0, xmax], [qfit_mil, qfit_mil+xmax*mfit_mil], c='lime')
    fit_farinotti_plot = ax1.plot([0.0, xmax], [qfit_far, qfit_far+xmax*mfit_far], c='r')
    s2 = ax1.plot([0.0, xmax], [0.0, xmax], c='k')
    ax1.axis([None, xmax, None, xmax])

    ax2.hist(y_test-y_preds, bins=np.arange(-xmax, xmax, 10), label='ML', color='lightblue', ec='blue', alpha=.4, zorder=2)
    ax2.hist(y_test-y_test_m, bins=np.arange(-xmax, xmax, 10), label='Millan', color='green', ec='green', alpha=.3, zorder=1)
    ax2.hist(y_test-y_test_f, bins=np.arange(-xmax, xmax, 10), label='Farinotti', color='red', ec='red', alpha=.3, zorder=1)

    # text
    text_ml = f"ML\n$\\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\n$\\sigma$ = {std_ML:.1f}"
    text_millan = f"Millan\n$\\mu$ = {mu_mil:.1f}\nmed = {med_mil:.1f}\n$\\sigma$ = {std_mil:.1f}"
    text_farinotti = f"Farinotti\n$\\mu$ = {mu_far:.1f}\nmed = {med_far:.1f}\n$\\sigma$ = {std_far:.1f}"
    # text boxes
    props_ML = dict(boxstyle='round', facecolor='lightblue', ec='blue', alpha=0.4)
    props_millan = dict(boxstyle='round', facecolor='lime', ec='green', alpha=0.4)
    props_farinotti = dict(boxstyle='round', facecolor='salmon', ec='red', alpha=0.4)
    ax2.text(0.05, 0.95, text_ml, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_ML)
    ax2.text(0.05, 0.7, text_millan, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_millan)
    ax2.text(0.05, 0.45, text_farinotti, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_farinotti)

    ax1.set_xlabel('GT ice thickness (m)')
    ax1.set_ylabel('Modelled ice thickness (m)')
    ax2.set_xlabel('GT - Model (m)')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

plot_spatial_test_predictions = False
if plot_spatial_test_predictions:

    # Visualize test predictions
    test_glaciers_names = test['RGIId'].unique().tolist()
    print(f"Test dataset: {len(test)} points and {len(test_glaciers_names)} glaciers")

    glacier_geometries = []
    for glacier_name in test_glaciers_names:
        rgi = glacier_name[6:8]
        oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{rgi}*/{rgi}*.shp")[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
        glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == glacier_name]['geometry'].item()
        # print(glacier_geometry)
        glacier_geometries.append(glacier_geometry)

    fig, axes = plt.subplots(2,3, figsize=(10,7))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    y_min = min(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
    y_max = max(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
    y_min_diff = min(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
    y_max_diff = max(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
    absmax = max(abs(y_min_diff), abs(y_max_diff))

    s1 = ax1.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test, cmap='Blues', label='GT')
    s2 = ax2.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_preds, cmap='Blues', label='ML')
    s3 = ax3.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_m, cmap='Blues', label='Millan')
    s4 = ax4.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_f, cmap='Blues', label='Farinotti')
    s5 = ax5.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=(y_test-y_preds)/y_test, cmap='bwr', label='GT-ML')
    s6 = ax6.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=(y_test-y_test_f)/y_test, cmap='bwr', label='GT-Farinotti')

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        for geom in glacier_geometries:
            ax.plot(*geom.exterior.xy, c='k')

    cbar1 = plt.colorbar(s1, ax=ax1)
    cbar2 = plt.colorbar(s2, ax=ax2)
    cbar3 = plt.colorbar(s3, ax=ax3)
    cbar4 = plt.colorbar(s4, ax=ax4)
    cbar5 = plt.colorbar(s5, ax=ax5)
    cbar6 = plt.colorbar(s6, ax=ax6)

    for cbar in (cbar1, cbar2, cbar3, cbar4):
        cbar.mappable.set_clim(vmin=y_min,vmax=y_max)
        cbar.set_label('THICKNESS (m)', labelpad=15, rotation=270)
    for cbar in (cbar5, cbar6):
        cbar.mappable.set_clim(vmin=-3, vmax=3)

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6): ax.legend(loc='upper left')

    plt.show()

# *********************************************
# Model deploy
# *********************************************
glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-03.02467', rgi=7, area=30, seed=None)
#'RGI60-13.37753', RGI60-13.43528 RGI60-13.54431
#glacier_name_for_generation = 'RGI60-07.00228' #RGI60-07.00027 'RGI60-11.01450' RGI60-07.00552,
#glacier_name_for_generation = 'RGI60-07.00832' very nice
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

# check if RGI60-06.00475 and RGI60-06.00481 blend together

# Generate points for one glacier
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                              n=CFG.n_points_regression, seed=42, verbose=True)
test_glacier_rgi = glacier_name_for_generation[6:8]

# Add features
#test_glacier['sia'] = ((0.3*test_glacier['v50']*(3+1))/(2*A*(rho*g*test_glacier['slope50'])**3))**(1./4)
test_glacier['sia'] = test_glacier['v100']/(test_glacier['slope100']**3)
test_glacier['deltaZ'] = test_glacier['Zmax'] - test_glacier['Zmin']
#test_glacier['hbahrm2'] = np.sqrt(test_glacier['Area'])*test_glacier['dist_from_border_km_geom']/test_glacier['Lmax']

#fig, (ax1, ax2, ax3) = plt.subplots(1,3)
#s = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=test_glacier['dist_from_border_km_geom'])
#s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=test_glacier['dist_from_ocean'])
#s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=test_glacier['curv_50'])
#cbar1 = plt.colorbar(s)
#cbar2 = plt.colorbar(s2)
#cbar3 = plt.colorbar(s3)
#plt.show()


X_test_glacier = test_glacier[CFG.features]
y_test_glacier_m = test_glacier[CFG.millan]
y_test_glacier_f = test_glacier[CFG.farinotti]
dtest = xgb.DMatrix(data=X_test_glacier)

no_millan_data = np.isnan(y_test_glacier_m).all()
no_farinotti_data = np.isnan(y_test_glacier_f).all()

#y_preds_glacier = best_model.predict(X_test_glacier) # sklearn-api
y_preds_glacier = best_model.predict(dtest) # Native API

# Set negative predictions to zero
y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

# Calculate the bedrock elevations
bedrock_elevations_ML = test_glacier['elevation'] - y_preds_glacier
bedrock_elevations_Millan = test_glacier['elevation'] - y_test_glacier_m
bedrock_elevations_Far = test_glacier['elevation'] - y_test_glacier_f

# Begin to extract all necessary things to plot the result
oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()
glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['Area'].item()
exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
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
                             rgi=test_glacier_rgi, path_tandemx=args.mosaic)
focus = focus_mosaic_tiles.squeeze()


# Get Farinotti file and calculate the volume
if not no_farinotti_data:
    test_glacier_folder_farinotti = glob(f"{args.farinotti_icethickness_folder}/*{test_glacier_rgi}/")[0]
    ice_farinotti = rioxarray.open_rasterio(test_glacier_folder_farinotti+glacier_name_for_generation+'_thickness.tif')
    res_farinotti = ice_farinotti.rio.resolution()[0]
    vol_farinotti = 1.e-9 * (res_farinotti ** 2) * np.nansum(ice_farinotti.values) # Volume Farinotti km3
else:
    print(f"Farinotti glacier {glacier_name_for_generation} not found in OGGM V62 database.")
    vol_farinotti = np.nan

# Calculate the glacier volume using the 3 models
vol_ML = calc_volume_glacier(y_preds_glacier, glacier_area)
vol_millan = calc_volume_glacier(y_test_glacier_m, glacier_area)
vol_farinotti2 = calc_volume_glacier(y_test_glacier_f, glacier_area)
print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, "
      f"volML: {vol_ML:.4g} km3 "
      f"volMil: {vol_millan:.4g} km3 "
      f"volFar: {vol_farinotti:.4g} km3 volFar2: {vol_farinotti2:.4g} km3"
      f"Far mismatch {100*abs(vol_farinotti2-vol_farinotti)/vol_farinotti:.2f}%")

print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

# Visualize test predictions of specific glacier
y_min = min(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))
y_max = max(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))

vmin = min(y_preds_glacier) # 0 # min(y_preds_glacier)#test_glacier['smb'].min()
vmax = max(y_preds_glacier) #1600 #max(y_preds_glacier)#test_glacier['smb'].max()

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

    plt.imshow(z_grid, cmap='jet')
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
    #im = ax.imshow(hillshade, cmap='grey', alpha=0.9, zorder=0)
    #im = ax.imshow(hillshade, cmap='grey', vmin=np.nanmin(focus), vmax=np.nanmax(focus), alpha=0.15, zorder=0)
    # y_preds_glacier
    s1 = ax.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=y_preds_glacier,
                     cmap='jet', label='ML', zorder=1, vmin=vmin,vmax=vmax)
    s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                            cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin,vmax=vmax)

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

plot_fancy_ML_Mil_Far_prediction = True
if plot_fancy_ML_Mil_Far_prediction:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

    im1 = hillshade.plot(ax=ax1, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    im2 = hillshade.plot(ax=ax2, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    im3 = hillshade.plot(ax=ax3, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

    s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_preds_glacier, cmap='jet', label='ML', vmin=vmin, vmax=vmax)
    if not no_millan_data:
        s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_m, cmap='jet',
                         label='Millan', vmin=vmin, vmax=vmax)
    if not no_farinotti_data:
        s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_f, cmap='jet',
                         label='Farinotti', vmin=vmin, vmax=vmax)

    ax1.set_title(f"IceBoost: {vol_ML:.4g} km$^3$", fontsize=16)
    ax2.set_title(f"Model1: {vol_millan:.4g} km$^3$", fontsize=16)
    ax3.set_title(f"Model2: {vol_farinotti:.4g} km$^3$", fontsize=16)

    for ax in (ax1, ax2, ax3):
        ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)

    cbar1 = plt.colorbar(s1, ax=ax1)
    cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)
    cbar1.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
    cbar1.ax.tick_params(labelsize=11)
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

    for ax in (ax1, ax2, ax3):
        ax.plot(*exterior_ring.xy, c='k')
        for nunatak in glacier_nunataks_list:
            ax.plot(*nunatak.xy, c='k', lw=0.8)

        # ax.legend(fontsize=14, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=14)
        ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        #ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax2.axis('off')
    ax3.axis('off')

    #fig.suptitle(f'{glacier_name_for_generation}', fontsize=13)
    plt.tight_layout()
    plt.show()



# Plot comparison between ML, Millan and Farinotti
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(8,4))
s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_preds_glacier, cmap='jet', label='ML')
if not no_millan_data:
    s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_m, cmap='jet',
                     label='Millan')
if not no_farinotti_data:
    s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_f, cmap='jet', label='Farinotti')

ax1.set_title(f"ML {vol_ML:.4g} km3")
ax2.set_title(f"Millan {vol_millan:.4g} km3")
ax3.set_title(f"Farinotti {vol_farinotti:.4g} km3")

cbar1 = plt.colorbar(s1, ax=ax1)
cbar1.mappable.set_clim(vmin=y_min,vmax=y_max)
cbar1.set_label('Thickness (m)', labelpad=15, rotation=270)
if not no_millan_data:
    cbar2 = plt.colorbar(s2, ax=ax2)
    cbar2.mappable.set_clim(vmin=y_min, vmax=y_max)
    cbar2.set_label('Thickness (m)', labelpad=15, rotation=270)
if not no_farinotti_data:
    cbar3 = plt.colorbar(s3, ax=ax3)
    cbar3.mappable.set_clim(vmin=y_min,vmax=y_max)
    cbar3.set_label('Thickness (m)', labelpad=15, rotation=270)

for ax in (ax1, ax2, ax3):
    ax.plot(*exterior_ring.xy, c='k')
    for nunatak in glacier_nunataks_list:
        ax.plot(*nunatak.xy, c='k', lw=0.8)

    #ax.legend(fontsize=14, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    #ax.axis("off")
#ax4.set_title("")
#ax4.axis("off")

fig.suptitle(f'{glacier_name_for_generation}', fontsize=13)
plt.tight_layout()
plt.show()

####################################
# Regional simulation
def run_rgi_simulation(rgi):
    print("Begin regional simulation ")
    oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
    # Get glaciers and order them in decreasing order by Area. First glaciers will be bigger and slower to process.
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio').sort_values(by='Area', ascending=False)
    # Main loop over glaciers
    for i, gl_id in tqdm(enumerate(oggm_rgi_glaciers['RGIId']), total=len(oggm_rgi_glaciers), desc=f"rgi {rgi} Glaciers", leave=True):
        # Fetch glacier data
        glacier_data = populate_glacier_with_metadata(glacier_name=gl_id, n=CFG.n_points_regression, seed=None, verbose=True)
    print(f"Finished regional simulation for rgi {rgi}.")

run_rgi_simulation_YN = False
if run_rgi_simulation_YN:
    run_rgi_simulation(11)