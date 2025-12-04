import time

import rioxarray
import utm
import warnings
import scipy
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj, Transformer, Geod, CRS
from sklearn.neighbors import KDTree
from scipy.spatial import distance_matrix
from shapely.geometry import Point, Polygon, LineString, MultiLineString, box
from oggm import utils
import xgboost as xgb
import catboost as cb
from PIL import Image
import matplotlib.pyplot as plt
import networkx

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Determines return value units.
    return c * r

def haversine_vectorized(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two sets of points
    on the earth (specified in decimal degrees). Works with arrays for vectorized calculations.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1[:, np.newaxis]  # Broadcasting lon1 across lon2
    dlat = lat2 - lat1[:, np.newaxis]  # Broadcasting lat1 across lat2

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, np.newaxis]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r  # Returns distance in kilometers

def lmax_with_covex_hull(geometry, glacier_epsg):
    '''
    This method calculates lmax using the geometry convex hull.
    It should be exactly equivalent to lmax_imputer with KDTree but much faster.
    '''
    geometry_epsg = geometry.to_crs(epsg=glacier_epsg) # Geodataframe
    gl_geom = geometry_epsg.iloc[0].geometry  # Polygon

    # Compute the convex hull
    convex_hull = gl_geom.convex_hull
    # Extract coordinates from the convex hull's exterior
    coords_hull = np.array(convex_hull.exterior.coords)
    # Compute pairwise distances between all points on the convex hull
    dist_matrix = distance_matrix(coords_hull, coords_hull)
    lmax = np.max(dist_matrix)

    ifplot = False
    if ifplot:
        # Only for plotting purposes
        gdf = gpd.GeoDataFrame({
            'geometry': [gl_geom, convex_hull],
            'type': ['Glacier geometry', 'Convex Hull']
        })
        # Find the indices of the maximum distance
        max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        point1, point2 = coords_hull[max_idx[0]], coords_hull[max_idx[1]]
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, color=['lightgrey', 'none'], edgecolor=['k', 'blue'], linestyle=['-', '--'])
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue', linestyle='-', linewidth=2,
                label=f'Lmax = {lmax:.2f}')
        plt.legend(handles=[
            plt.Line2D([0], [0], color='k', label='Glacier geometry'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Convex Hull'),
            plt.Line2D([0], [0], color='blue', label=f'Lmax = {lmax:.0f} m')], fontsize=14, loc='upper left')
        plt.tick_params(labelsize=14)
        ax.set_xlabel("Eastings [m]", fontsize=16)
        ax.set_ylabel("Northings [m]", fontsize=16)
        plt.show()

    return lmax

def lmax_imputer(geometry, glacier_epsg):
    '''
    geometry: glacier external geometry as pandas geodataframe in 4326 prjection
    glacier_epsg: glacier epsg
    return: lmax in meters
    '''
    geometry_epsg = geometry.to_crs(epsg=glacier_epsg)
    glacier_vertices = np.array(geometry_epsg.iloc[0].geometry.exterior.coords)
    tree_lmax = KDTree(glacier_vertices)
    dists, _ = tree_lmax.query(glacier_vertices, k=len(glacier_vertices))
    lmax = np.max(dists)

    return lmax

def from_lat_lon_to_utm_and_epsg(lat, lon):
    """https://github.com/Turbo87/utm"""
    # Note lat lon can be also NumPy arrays.
    # In this case zone letter and number will be calculate from first entry.
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    southern_hemisphere_TrueFalse = True if zone_letter < 'N' else False
    epsg_code = 32600 + zone_number + southern_hemisphere_TrueFalse * 100
    return (easting, northing, zone_number, zone_letter, epsg_code)

def gaussian_filter_with_nans(U, sigma, trunc=4.0):
    # Since the reprojection into utm leads to distortions (=nans) we need to take care of this during filtering
    # From David in https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    WW[WW == 0] = np.nan
    filtered_U = VV / WW
    return filtered_U

def get_cmap(name):
    from matplotlib.colors import LinearSegmentedColormap

    if name == 'white_electric_blue':
        # (0.11764706, 0.56470588, 1.0)] dodgerblue
        colors = [(1, 1, 1), (0.0, 0.0, .8)]  # White to electric blue
        cm = LinearSegmentedColormap.from_list(name, colors)

    if name == 'black_electric_green':
        cm = LinearSegmentedColormap.from_list(name, ['#000000', '#00FF00'])

    if name == 'black_electric_blue':
        cm = LinearSegmentedColormap.from_list(name, ['#000000', '#0000CC'])

    if name == 'dark_green_to_blue':
        # Dark green to grey, then white, light blue, and blue
        colors = ['#006400', '#808080', '#FFFFFF', '#ADD8E6', '#0000FF']  # Dark green, grey, white, light blue, blue
        cm = LinearSegmentedColormap.from_list(name, colors)

    if name == 'dark_green_to_purple':
        colors = ['#006400', '#808080', '#ADD8E6', '#0000FF', '#800080']  # Dark green, grey, light blue, blue, purple
        cm = LinearSegmentedColormap.from_list(name, colors)

    elif name == 'grey_to_blue_orange':
        colors = ['#808080', '#A0D8E6', '#3F00FF', '#800080', '#FFA500']  # Grey, light blue, blue, purple, orange
        cm = LinearSegmentedColormap.from_list(name, colors)

    elif name == 'white_to_brown':
        colors = [(1, 1, 1), '#8B4513']
        cm = LinearSegmentedColormap.from_list(name, colors)

    elif name == 'white_to_orange':
        colors = [(1, 1, 1), '#f47200']
        cm = LinearSegmentedColormap.from_list(name, colors)


    return cm


def calc_orthometric_heights(lons=None, lats=None, h_wgs84=None):
    """
    :param h_wgs84: ellipsoid height
    :return: orthometric height (Hmsl)
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3855", always_xy=True)
    _, _, h_egm2008 = transformer.transform(lons, lats, h_wgs84)
    return h_egm2008

def calc_ortho_and_geoid_heights(lons=None, lats=None, h_wgs84=None, geoid_tif=None):
    """
    :param h_wgs84: ellipsoid height (i.e. DEM elevation)
    :return: orthometric height (H) and geoid height (N)
    """
    # Ellipsoidal input CRS
    wgs84 = CRS.from_epsg(4979)
    geoid_crs = CRS.from_proj4(f"+proj=latlong +datum=WGS84 +geoidgrids={geoid_tif}")

    transformer = Transformer.from_crs(geoid_crs, wgs84, always_xy=True)

    z = np.zeros_like(lons)
    _, _, N = transformer.transform(lons, lats, z)

    H = h_wgs84 - N

    return H, N

def calc_ellipsoid_heights(lons=None, lats=None, H=None):
    """
    :param H: orthometric height (Hmsl)
    :return: ellipsoid height (hwgs84)
    """
    lons = np.array(lons)
    lats = np.array(lats)
    H = np.array(H)

    zeros = np.zeros(len(lons))
    transformer = Transformer.from_crs("epsg:4326", "epsg:3855", always_xy=True)
    _, _, N = transformer.transform(lons, lats, zeros) # this yields N = -H
    N = N * -1
    h_wgs84 = H + N
    return h_wgs84

def calc_volume_glacier(y=None, area=0, H=None):
    '''
    :param y: numpy.ndarray. Ice thickness [m]
    :param area: area: float [km2]
    :param H: numpy.ndarray. Orthometric height [m]
    :return: volume [km3]
    '''
    #y_xgb = y1
    #y_cat = y2
    #N = len(y1)
    N = len(y)
    f = 0.001 * area / N

    # Millan or Farinotti
    #if y2 is None:
    #    volume = np.sum(y1) * f
    #    return volume

    # iceboost
    #else:
    #y_mean = 0.5 * (y_xgb + y_cat)
    #y_mean = np.where(y_mean < 0, 0, y_mean)

    # volume ice
    #volume = np.sum(y_mean) * f
    volume = np.sum(y) * f
    # volume ice above sea level
    #volume_af = np.sum(np.where(h_egm2008 - y_mean > 0, y_mean, h_egm2008)) * f
    # volume ice below sea level
    #volume_bsl = np.sum(np.where(h_egm2008 - y_mean > 0, 0.0, y_mean - h_egm2008)) * f
    volume_bsl = np.sum(np.where(H - y > 0, 0.0, y - H)) * f

    #err_points = np.std((y_xgb, y_cat), axis=0)
    err_points = 0
    # This error considers the point-wise spread between the models
    err_volume_points = np.sqrt(np.sum(err_points**2)) * f
    # This error is the semi-difference of the 2 modeled volumes.
    #err_volume_range = 0.5 * np.abs(np.sum(y_xgb) - np.sum(y_cat)) * f
    err_volume_range = 0
    # Add in quadrature the two errors
    err_volume = np.sqrt(err_volume_points**2 + err_volume_range**2)

    return volume, err_volume, volume_bsl

def calc_volume_glacier_from_ar(ar=None, area=0):
    """
    :param ar: xarray with 'thickness' and 'h_egm2008' fields
    :param area: glacier area
    :return: volume (km3) and volume below sea level (km3)
    """
    vals_thickness_np = ar['thickness'].values
    vals_h_egm2008_np = ar['h_egm2008'].values
    N = np.count_nonzero(~np.isnan(vals_thickness_np))
    f = 0.001 * area / N

    # Calculate the volume
    vol = np.nansum(vals_thickness_np) * f
    vol_bsl = np.nansum(
        np.where(vals_h_egm2008_np - vals_thickness_np > 0, 0.0, vals_thickness_np - vals_h_egm2008_np)) * f

    return vol, vol_bsl

def get_random_glacier_rgiid(name=None, rgi=11, version=None, area=None, seed=None):
    """Provide a rgi number and seed. This method returns a
    random glacier rgiid name.
    If not rgi is passed, any rgi region is good.
    """
    if version == '62':
        name_column_id = 'RGIId'
        area_column_id = 'Area'
    elif version == '70G':
        name_column_id = 'rgi_id'
        area_column_id = 'area_km2'

    if name is not None: return name
    if seed is not None:
        np.random.seed(seed)
    if rgi is not None:
        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version=version)
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
    if area is not None:
        oggm_rgi_glaciers = oggm_rgi_glaciers[oggm_rgi_glaciers[area_column_id] > area]
    rgi_ids = oggm_rgi_glaciers[name_column_id].dropna().unique().tolist()
    rgiid = np.random.choice(rgi_ids)
    return rgiid


def create_train_test(df, rgi=None, frac=0.1, full_shuffle=None, seed=None):
    """
    - rgi se voglio creare il test in una particolare regione
    - frac: quanto lo voglio grande in percentuale alla grandezza del rgi
    """
    if seed is not None:
        random.seed(seed)

    if rgi is not None and full_shuffle is True:
        df_rgi = df[df['RGI'] == rgi]
        test = df_rgi.sample(frac=frac, random_state=seed)
        train = df.drop(test.index)
        return train, test

    if full_shuffle is True:
        test = df.sample(frac=frac, random_state=seed)
        train = df.drop(test.index)
        return train, test

    # create test based on rgi
    if rgi is not None:
        df_rgi = df[df['RGI']==rgi]
    else:
        df_rgi = df

    minimum_test_size = round(frac * len(df_rgi))

    unique_glaciers = df_rgi['RGIId'].unique()
    random.shuffle(unique_glaciers)
    selected_glaciers = []
    n_total_points = 0
    #print(unique_glaciers)

    for glacier_name in unique_glaciers:
        if n_total_points < minimum_test_size:
            selected_glaciers.append(glacier_name)
            n_points = df_rgi[df_rgi['RGIId'] == glacier_name].shape[0]
            n_total_points += n_points
            #print(glacier_name, n_points, n_total_points)
        else:
            #print('Finished with', n_total_points, 'points, and', len(selected_glaciers), 'glaciers.')
            break

    test = df_rgi[df_rgi['RGIId'].isin(selected_glaciers)]
    train = df.drop(test.index)
    #print(test['RGI'].value_counts())
    #print(test['RGIId'].value_counts())
    #print('Total test size: ', len(test))
    #print(train.describe().T)
    #input('wait')
    return train, test

def load_models(config_file):

    #model_xgb_filename = config_file.model_input_dir + config_file.model_filename_xgb
    #iceboost_xgb = xgb.Booster()
    #iceboost_xgb.load_model(model_xgb_filename)

    #model_cat_filename = config_file.model_input_dir + config_file.model_filename_cat
    #iceboost_cat = cb.CatBoostRegressor()
    #iceboost_cat.load_model(model_cat_filename, format='cbm')

    # return iceboost_xgb, iceboost_cat

    filename_xgb_with_v = config_file.model_input_dir + config_file.model_filename_xgb_with_v
    filename_xgb_without_v = config_file.model_input_dir + config_file.model_filename_xgb_without_v
    filename_xgb_without_v_with_lmax = config_file.model_input_dir + config_file.model_filename_xgb_without_v_with_lmax

    filename_cat_with_v = config_file.model_input_dir + config_file.model_filename_cat_with_v
    filename_cat_without_v = config_file.model_input_dir + config_file.model_filename_cat_without_v
    filename_cat_without_v_with_lmax = config_file.model_input_dir + config_file.model_filename_cat_without_v_with_lmax

    iceboost_xgb_with_v = xgb.Booster()
    iceboost_xgb_with_v.load_model(filename_xgb_with_v)

    iceboost_xgb_without_v = xgb.Booster()
    iceboost_xgb_without_v.load_model(filename_xgb_without_v)

    iceboost_xgb_without_v_with_lmax = xgb.Booster()
    iceboost_xgb_without_v_with_lmax.load_model(filename_xgb_without_v_with_lmax)

    iceboost_cat_with_v = cb.CatBoostRegressor()
    iceboost_cat_with_v.load_model(filename_cat_with_v, format='cbm')

    iceboost_cat_without_v = cb.CatBoostRegressor()
    iceboost_cat_without_v.load_model(filename_cat_without_v, format='cbm')

    iceboost_cat_without_v_with_lmax = cb.CatBoostRegressor()
    iceboost_cat_without_v_with_lmax.load_model(filename_cat_without_v_with_lmax, format='cbm')

    dict_models = {'xgb_v': iceboost_xgb_with_v, 'xgb_without_v': iceboost_xgb_without_v, 'xgb_without_v_with_lmax': iceboost_xgb_without_v_with_lmax,
                   'cat_v': iceboost_cat_with_v, 'cat_without_v': iceboost_cat_without_v, 'cat_without_v_with_lmax': iceboost_cat_without_v_with_lmax}

    return dict_models


def compute_monte_carlo_error(dataset=None, features=None, rgi=None, model_xgb=None, model_cat=None):

    assert rgi in range(1, 20), f"rgi must be an integer between 1 and 19, got {rgi}"

    n_simul = 50
    y_preds_xgb_all = []
    y_preds_cat_all = []

    noise_rules = {
        'curv_50': 100*17.88/(50**2),           # [1/m] 0.01
        'curv_100': 100*17.88/(100**2),         # [1/m] 0.01
        'curv_150': 100*17.88/(150**2),         # [1/m] 0.01
        'curv_300': 100*17.88/(300**2),         # [1/m] 0.01
        'curv_450': 100*17.88/(450**2),         # [1/m] 0.01
        'curv_gfa': 100*17.88/(500**2),         # [1/m] 0.01
        't2m': 1,                               # [Kelvin] 1
    }

    slope_features = ['slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa']
    slope_steps_meters = {
        'slope50': 50,
        'slope75': 75,
        'slope100': 100,
        'slope125': 125,
        'slope150': 150,
        'slope300': 300,
        'slope450': 450,
        'slopegfa': 500  # *** simplify ***
    }

    velocity_features = ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa']
    if rgi in (5, 19):  # Greenland or Antarctica
        sigma_velocity = 18. # [m/yr]
    else:
        sigma_velocity = 10. # [m/yr]

    for n in range(n_simul):
        #print(n)
        X_noisy = dataset[features].copy()

        # sigma z is 2.0 m for gentle terrain (<0.2), else 4.0 m
        scale_elevation = np.where(np.abs(dataset['slope50']) < 0.2, 2.0, 4.0)

        for f in features:

            # Errors on velocity: 10-18 m/yr depending on region
            if f in velocity_features:
                X_noisy[f] += np.random.normal(loc=0, scale=sigma_velocity, size=len(dataset))
                X_noisy[f] = np.clip(X_noisy[f], 0, None)

            elif f == 'elevation':
                X_noisy[f] += np.random.normal(loc=0, scale=scale_elevation, size=len(dataset))

            # Errors on slope
            elif f in slope_features:

                sigma_delta_z = np.sqrt(4**2 + 4**2)  # [m] # # Vertical error in height *difference* (sqrt(4^2 + 4^2))
                step_x = slope_steps_meters[f]
                scale = sigma_delta_z / (2*step_x) # factor 2 from finite differences
                #print(f, scale)

                X_noisy[f] += np.random.normal(loc=0, scale=scale, size=len(dataset))
                X_noisy[f] = np.clip(X_noisy[f], 0, None)

            # Error on smb is taken as the 10% [mm w.e. yr-1]. Note absolute value.
            elif f == 'smb':
                X_noisy[f] += np.random.normal(loc=0, scale=0.1 * np.abs(dataset[f]), size=len(dataset))

            # Error on dist_from_border_km_geom is taken as 100 meters.
            elif f == 'dist_from_border_km_geom':
                X_noisy[f] += np.random.normal(loc=0, scale=0.1, size=len(dataset))
                X_noisy[f] = np.clip(X_noisy[f], 0, None)

            # Error on dist_from_ocean is taken as 100 meters.
            elif f == 'dist_from_ocean':
                X_noisy[f] += np.random.normal(loc=0, scale=0.1, size=len(dataset))
                X_noisy['dist_from_ocean'] = np.clip(X_noisy['dist_from_ocean'], 0, None)

            # Error on lmax is 5 percent of glacier length [m]
            elif f == 'lmax':
                sigma_lmax = 0.05 * dataset['lmax'].iloc[0]
                X_noisy[f] += np.random.normal(loc=0, scale=sigma_lmax, size=len(dataset))
                X_noisy['lmax'] = np.clip(X_noisy['lmax'], 0, None)

            elif f in noise_rules:
                X_noisy[f] += np.random.normal(loc=0, scale=noise_rules[f], size=len(dataset))

        # Construct dataset and predict
        dtest_xgb = xgb.DMatrix(data=X_noisy)
        y_preds_glacier_xgb = model_xgb.predict(dtest_xgb)
        y_preds_glacier_cat = model_cat.predict(X_noisy)

        y_preds_xgb_all.append(y_preds_glacier_xgb)
        y_preds_cat_all.append(y_preds_glacier_cat)

    # finally compute the ensemble
    y_preds_xgb_all = np.stack(y_preds_xgb_all)  # shape: (n_simul, n_points)
    y_preds_cat_all = np.stack(y_preds_cat_all)  # shape: (n_simul, n_points)

    # Stack both model predictions together along a new axis
    y_all = np.stack([y_preds_xgb_all, y_preds_cat_all], axis=0)  # shape: (2, n_simul, n_points)

    # Compute mean across models and simulations
    mean_all = y_all.mean(axis=(0, 1))  # shape: (n_points,)

    # Compute std across models and simulations
    std_all = y_all.std(axis=(0, 1))  # shape: (n_points,)

    #fig, (ax1, ax2) = plt.subplots(1,2)
    #s1 = ax1.scatter(dataset['lons'], dataset['lats'], c=mean_all, s=1, cmap='turbo')
    #s2 = ax2.scatter(dataset['lons'], dataset['lats'], c=std_all, s=1, cmap='viridis')
    #cb1 = plt.colorbar(s1)
    #cb2 = plt.colorbar(s2)
    #plt.show()

    return mean_all, std_all


def create_PIL_image(array, png_resolution=None):
    """
    Given 2d numpy ndarray returns PIL image for .png
    """
    array = np.flipud(array)
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    array_normalized = (array - array_min) / (array_max - array_min) * 255
    array_normalized = np.nan_to_num(array_normalized, nan=0).astype(np.uint8)
    colormap = plt.cm.jet
    colored_array = colormap(array_normalized)
    colored_array = (colored_array[:, :, :3] * 255).astype(np.uint8)
    alpha_channel = np.where(np.isnan(array), 0, 255).astype(np.uint8)
    rgba_array = np.dstack((colored_array, alpha_channel))
    image = Image.fromarray(rgba_array)
    image_resized = image.resize((png_resolution, png_resolution), Image.Resampling.LANCZOS)
    return image_resized

def get_rgi_products(region=None, version=None, add_glacier_geom_file=None, add_glacier_intersects_geom_file=None):
    """
    :param region: rgi region 1 to 19
    :param version: rgi version '62', '70G'
    :param add_glacier_geom_file: if this gpkg file is provided, it will be added to the rgi dataframe
    :param add_glacier_intersects_geom_file: if this gpkg file is provided, it will be added to the rgi dataframe
    :return: regional glacier dataframe and regional graph of glacier connectivity
    """

    if region is None: raise ValueError("You need to specify the region number as string. Exit.")

    if version not in ('62', '70G'): raise ValueError("Accepted RGI versions are 62 or 70G. Exit.")

    if not isinstance(region, str): region = f"{region:02d}"

    FILE_SHP_RGI = utils.get_rgi_region_file(region=region, version=version)
    FILE_INTERSECTS_SHP_RGI = utils.get_rgi_intersects_region_file(region=region, version=version)

    # get dataset of glaciers and intersects from rgi
    rgi_glaciers = gpd.read_file(FILE_SHP_RGI, engine='pyogrio')
    rgi_intersects = gpd.read_file(FILE_INTERSECTS_SHP_RGI, engine='pyogrio')

    # if user has provided a glacier and intersect shp files, concatenate with rgi
    if add_glacier_geom_file is not None:
        rgi_glaciers_user_input = gpd.read_file(add_glacier_geom_file, engine='pyogrio')
        rgi_intersects_user_input = gpd.read_file(add_glacier_intersects_geom_file, engine='pyogrio')

        if not rgi_glaciers_user_input.crs == "EPSG:4326":
            rgi_glaciers_user_input = rgi_glaciers_user_input.to_crs("EPSG:4326")
        if not rgi_intersects_user_input.crs == "EPSG:4326":
            rgi_intersects_user_input = rgi_intersects_user_input.to_crs("EPSG:4326")

        assert set(rgi_glaciers_user_input.columns).issubset(rgi_glaciers.columns), \
            "Incompatible concatenation with custom glacier dataframes"
        assert set(rgi_intersects_user_input.columns).issubset(rgi_intersects.columns), \
            "Incompatible concatenation with custom intersect glacier dataframes"

        rgi_glaciers = pd.concat([rgi_glaciers, rgi_glaciers_user_input], ignore_index=True)
        rgi_intersects = pd.concat([rgi_intersects, rgi_intersects_user_input], ignore_index=True)

    # create graph of connectivity needed for distance calculations
    rgi_graph = networkx.Graph()
    if version == '62':
        edges = rgi_intersects[['RGIId_1', 'RGIId_2']].values
    if version == '70G':
        edges = rgi_intersects[['rgi_g_id_1', 'rgi_g_id_2']].values

    rgi_graph.add_edges_from(edges)

    rgi_products = (rgi_glaciers, rgi_graph)

    return rgi_products


def add_regional_features(df=None):
    """
    :param df: regional dataframe with glacier geometries
    :return: df with added 'area', 'area_icefree', 'perimeter', 'cen_lat', 'cen_lon', 'cen_epsg'
    """

    geod = Geod(ellps="WGS84")

    def calc_feats(geometry):
        area, perimeter = geod.geometry_area_perimeter(geometry)
        area = abs(area) * 1e-6
        geometry_ext = Polygon(geometry.exterior)
        gl_geom_ext_gdf = gpd.GeoDataFrame(geometry=[geometry_ext], crs="EPSG:4326")
        area_ice_and_noince, _ = geod.geometry_area_perimeter(geometry_ext)
        area_ice_and_noince = abs(area_ice_and_noince) * 1e-6

        # Calculate area of nunataks in percentage to the total area
        area_noice = 1 - area / area_ice_and_noince

        glacier_centroid = geometry_ext.centroid
        cenLon, cenLat = glacier_centroid.x, glacier_centroid.y
        _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)

        lmax = lmax_with_covex_hull(gl_geom_ext_gdf, glacier_epsg)

        return (area,  # Area in km²,
                area_noice,  # unitless,
                perimeter,  # perimeter in meters
                lmax,    # m
                cenLat,  # degrees north
                cenLon,  # degrees east
                glacier_epsg)  # epsg

    # apply the function and unpack results
    results = np.array(df['geometry'].apply(calc_feats).to_list())
    df[['area', 'area_icefree', 'perimeter', 'lmax', 'cen_lat', 'cen_lon', 'cen_epsg']] = results
    df['cen_epsg'] = df['cen_epsg'].astype(int)

    return df

def get_mass_balance_df(region=None):
    # mass balance rgi dataframe
    mbdf = utils.get_geodetic_mb_dataframe()
    mbdf = mbdf.loc[mbdf['period'] == '2000-01-01_2020-01-01']
    mbdf_rgi = mbdf.loc[mbdf['reg'] == int(region)]
    assert len(mbdf_rgi)>0, "Mass balance dataframe error in import."

    return mbdf_rgi

def get_glacier_geometries_4326(ids=None, glacier_geo_df=None, version=None):

    if version == '62':
        name_column_id = 'RGIId'
        name_column_name = 'Name'
    elif version == '70G':
        name_column_id = 'rgi_id'
        name_column_name = 'glac_name'
    else:
        raise ValueError("Version not supported.")

    # Extract id and geometry
    glacier_geometries = glacier_geo_df.loc[glacier_geo_df[name_column_id].isin(ids), [name_column_id, 'geometry']]
    # Set the glacier id as the dataframe index
    glacier_geometries = glacier_geometries.set_index(name_column_id).rename_axis('polygon')

    assert glacier_geometries.crs == "EPSG:4326", "Unexpected projection in get_glacier_geometries_4326 method."
    assert len(glacier_geometries) > 0, "Geometries not found."

    return glacier_geometries

def get_coastline_dataframe(GSHHG_folder):
    gdf16 = gpd.read_file(f'{GSHHG_folder}GSHHS_f_L1_L6.shp', engine='pyogrio')
    return gdf16


def find_cluster_with_graph(graph, start_node, max_depth=None):
    if not graph.has_node(start_node):
        return [start_node]
    # Find all nodes in the connected component
    #neighbors = networkx.node_connected_component(graph, start_node)
    #return list(neighbors)
    # Use a BFS traversal to get nodes up to the max_depth
    nodes_at_depth = set()
    nodes_to_visit = [(start_node, 0)]  # (node, current_depth)

    while nodes_to_visit:
        current_node, current_depth = nodes_to_visit.pop(0)

        # If max_depth is not None and current_depth exceeds max_depth, stop processing this branch
        if max_depth is not None and current_depth > max_depth:
            continue

        # Add the current node to the set of nodes to return
        nodes_at_depth.add(current_node)

        # Add neighbors to the list with incremented depth
        neighbors = list(graph.neighbors(current_node))
        for neighbor in neighbors:
            if neighbor not in nodes_at_depth:
                nodes_to_visit.append((neighbor, current_depth + 1))

    return list(nodes_at_depth)

def get_possible_cluster(graph, start_node, glacier_epsg, rgi, rgi_glaciers, name_column_id, config):

    # Get rgi clustering limits from config file
    rgi_clustering_limits = config.rgi_clustering_limits.get(f"rgi_{rgi}", False)

    # if rgi does not contemplate clustering, return False
    if rgi_clustering_limits is False:
        return False

    # we are in the case of a rgi that contemplate clustering

    # Return False if the start node is not in the graph (isolated glacier)
    if not graph.has_node(start_node):
        return False

    # Step 1: Create a cluster, i.e. all glaciers connected to the start glacier
    cluster_nodes = networkx.node_connected_component(graph, start_node)

    # Step 2: Extract cluster info
    cluster = graph.subgraph(cluster_nodes)
    cluster_min_depth = networkx.radius(cluster)
    cluster_no_nodes = cluster.number_of_nodes()
    cluster_no_edges = cluster.number_of_edges()
    #print(cluster_min_depth, cluster_no_nodes, cluster_no_edges)

    # Step 3: create cluster dataframe with ids, area, perimeter and lmax
    df_cluster = rgi_glaciers.loc[rgi_glaciers[name_column_id].isin(cluster_nodes), [name_column_id, 'area', 'perimeter', 'lmax']]
    df_cluster = df_cluster.set_index(name_column_id).rename_axis('cluster_IDs')
    cluster_area = float(df_cluster['area'].sum())
    max_area_for_clustering = 10000  # km2

    # Step 4: calculate cluster complexity based on graph parameters and cluster total area
    is_low_complexity = (
        cluster_min_depth <= rgi_clustering_limits["min_depth"]
        and cluster_no_nodes <= rgi_clustering_limits["no_nodes"]
        and cluster_no_edges <= rgi_clustering_limits["no_edges"]
        and cluster_area < max_area_for_clustering
    )
    #print(rgi_clustering_limits)
    #print(cluster_min_depth, cluster_no_nodes, cluster_no_edges, cluster_area)
    #input('wait')

    # Step 5: if cluster too complex, return False, otherwise return cluster dataframe
    if is_low_complexity is False:
        return False
    else:
        return df_cluster


def normalized_elevation(h, Hmin, Hmax):
    '''
    :param h: elevation (accepted types are scalar, numpy array or pandas series)
    :param Hmin: glacier minimum elevation
    :param Hmax: glacier maximum elevation
    :return: normalized height 0 to 1
    '''
    h_np = np.asarray(h)  # Convert h to a NumPy array
    if np.all(Hmin == Hmax): # handles scalars and arrays
        result = np.zeros_like(h_np)
    else:
        result = np.clip((h_np - Hmin) / (Hmax - Hmin), 0, 1)

    # Handle return type for different types of h
    if isinstance(h, (np.ndarray, pd.Series)):
        return result.astype(h.dtype)
    elif isinstance(h, list):
        return result.tolist()
    else:
        return float(result)


def get_version_and_rgi_from_id(id):
    '''
    :param id: name of the glacier (string)
    :return: rgi and version ('62 or '70G')
    '''

    if id.startswith('RGI60'):
        version_rgi = '62'
        rgi = id[6:8]

    elif id.startswith('RGI2000'):
        version_rgi = '70G'
        rgi = id[15:17]

    else: raise ValueError("Glacier id starts with unexpected string. Exit.")

    return rgi, version_rgi


def plot_feature_scatter(feats, test_glacier):
    """
    Plot scatter plots for features in the given configuration against the glacier data.

    Parameters:
    - config: An object that contains the 'features' attribute.
    - test_glacier: A DataFrame containing 'lons', 'lats', and feature data.

    """
    num_feats = len(feats)  # Number of features
    cols = 6
    rows = (num_feats // cols) + (num_feats % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.ravel()

    for idx, feat in enumerate(feats):
        sc = axes[idx].scatter(x=test_glacier['lons'], y=test_glacier['lats'], c=test_glacier[feat], s=1, cmap='jet')
        cb = plt.colorbar(sc)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].tick_params(labelbottom=False, labelleft=False)
        axes[idx].text(0.05, 0.95, feat, transform=axes[idx].transAxes,
                       fontsize=10, verticalalignment='top', color='black', weight='bold')

    # Hide any unused subplots
    for i in range(len(feats), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def choose_grid_epsg(epsg_glacier=None, region=None, lat_max=None):
    epsg_grid = None
    if region == 5:
        epsg_grid = 3413
    elif region == 19 and lat_max < -60:
        epsg_grid = 3031
    else:
        epsg_grid = epsg_glacier
    assert isinstance(epsg_grid, int), f"Problems with choice of projection."
    return epsg_grid

def generate_points(gdf=None, epsg_glacier=None, region=None, resolution=None):

    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

    # Decide crs of grid
    epsg_grid = choose_grid_epsg(epsg_glacier=epsg_glacier, region=region, lat_max=max_lat)

    # Reproject geometries to grid crs
    gdf_grid_crs = gdf.to_crs(epsg=epsg_grid)
    minx, miny, maxx, maxy = gdf_grid_crs.total_bounds

    spatial_posting = resolution + 1
    minimum_no_points_in_box = 1e3
    no_glaciers_covered = -999
    points_inside = -999

    while no_glaciers_covered != len(gdf_grid_crs) and spatial_posting > 1.:
        spatial_posting -= 1.

        # Create grid coordinates
        xs = np.arange(minx+1, maxx-1, spatial_posting)
        ys = np.arange(miny+1, maxy-1, spatial_posting)
        xx, yy = np.meshgrid(xs, ys)
        Nx, Ny = len(xs), len(ys)
        if (Nx * Ny) < minimum_no_points_in_box:
            continue

        # Create dataframe of points in bounding box
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=epsg_grid)

        # Get only points inside the glacier(s)
        points_inside = gpd.sjoin(points_gdf, gdf_grid_crs, predicate="within", how="inner")
        # print(points_inside)

        no_glaciers_covered = points_inside['index_right'].nunique()


    assert not isinstance(points_inside, int), f"Problems with grid generation {gdf.index}."
    assert no_glaciers_covered == len(gdf), "The generated grid does not cover all glaciers."

    # Rename the index
    points_inside = points_inside.rename(columns={'index_right': 'polygon'})

    # Keep only the geometries and the glacier ids
    points_inside = points_inside[['geometry', 'polygon']]

    # We need to get the grid points in lat and lon
    points_inside_4326 = points_inside.to_crs(epsg=4326)

    # debug
    #print("Final: ", spatial_posting, len(points_inside))
    #fig, (ax1, ax2) = plt.subplots(1,2)
    #gdf_grid_crs.plot(ax=ax1, edgecolor='black', facecolor='none')
    #points_inside.plot(ax=ax1, marker='o', color='k', markersize=1)
    #gdf.plot(ax=ax2, edgecolor='black', facecolor='none')
    #points_inside_4326.plot(ax=ax2, marker='o', color='k', markersize=1)
    #plt.show()

    # Add lats and lons
    points_inside["lons"] = points_inside_4326.geometry.x.values
    points_inside["lats"] = points_inside_4326.geometry.y.values

    # Store grid resolution in meters
    points_inside.attrs["grid_resolution"] = spatial_posting

    # reset index
    points_inside = points_inside.reset_index(drop=True)

    return points_inside

def generate_points_on_grid_min_100meter(in_points_df=None, gdf=None, epsg=None, region=None):

    #print(f"We have to generate grid points inside {len(gdf)} glacier(s)")

    # Get the bounding box
    minx, miny, maxx, maxy = gdf.total_bounds
    delta_lon = maxx - minx
    delta_lat = maxy - miny
    mean_lat = 0.5 * (miny + maxy)
    mean_lon = 0.5 * (minx + maxx)
    #print(minx, miny, maxx, maxy)

    spatial_posting = 100.
    minimum_no_points_in_box = 1e4
    lat_res = spatial_posting / 111320  # Latitude resolution in degrees
    lon_res = spatial_posting / (111320 * np.cos(np.deg2rad(mean_lat)))  # Longitude resolution in degrees
    Nx = int(delta_lon / lon_res)
    Ny = int(delta_lat / lat_res)
    no_glaciers_covered = -999
    points_inside = -999

    # If not all glaciers covered by grid, decrease grid spacing
    #while (Nx * Ny) < minimum_no_points_in_box and spatial_posting > 2.:
    while no_glaciers_covered != len(gdf) and spatial_posting > 1.:
        spatial_posting -= 1.  # Decrease spatial posting
        lat_res = spatial_posting / 111320  # Recalculate lat resolution
        lon_res = spatial_posting / (111320 * np.cos(np.deg2rad(mean_lat)))  # Recalculate lon resolution
        Nx = int(delta_lon / lon_res)  # Recalculate the number of points in the x direction
        Ny = int(delta_lat / lat_res)  # Recalculate the number of points in the y direction
        #print(f"lat_res: {lat_res} lon_res: {lon_res} posting {spatial_posting} {Nx*Ny}")
        if (Nx * Ny) < minimum_no_points_in_box:
            continue

        # Create grid coordinates
        xs = np.arange(minx, maxx, lon_res)
        ys = np.arange(miny, maxy, lat_res)
        xx, yy = np.meshgrid(xs, ys)

        # Create dataframe of points in bounding box
        points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=gdf.crs)

        # Get only points inside the glacier(s)
        points_inside = gpd.sjoin(points_gdf, gdf, predicate="within", how="inner")
        #print(points_inside)

        no_glaciers_covered = points_inside['index_right'].nunique()

    #print(f"lat_res: {lat_res} lon_res: {lon_res} posting {spatial_posting}")
    #print(points_inside, no_glaciers_covered)
    assert not isinstance(points_inside, int), f"Problems with grid generation {gdf.index}."

    # Rename the index
    points_inside = points_inside.rename(columns={'index_right': 'polygon_index'})

    assert len(points_inside) > 300, f"Generated too few points. Check point generation on grid: {mean_lat}-{mean_lon}"
    assert len(points_inside) < 3e6, f"Generated too many points. Not a problem but carefully check if i need so many."
    assert no_glaciers_covered == len(gdf), "The generated grid does not cover all glaciers."

    #fig, ax = plt.subplots()
    #points_inside.plot(ax=ax, color='red', markersize=1, alpha=0.2)
    #gdf.plot(ax=ax, ec='b', fc='none')
    #plt.show()

    # Fill dataframe for output
    in_points_df["lons"] = points_inside.geometry.x.values
    in_points_df["lats"] = points_inside.geometry.y.values
    in_points_df["polygon"] = points_inside.polygon_index.values
    in_points_df["nunataks"] = 0.0

    return in_points_df

def generate_points_on_grid(gdf_ext=None, gdf_nuns=None, max_points=None):

    # Get the bounding box of the external boundary
    llx, lly, urx, ury = gdf_ext.total_bounds

    # Calculate bounding box width and height
    bbox_width = urx - llx
    bbox_height = ury - lly
    bbox_height_meters = 1000 * haversine(llx, lly, llx, ury)
    bbox_width_meters = 1000 * haversine(llx, lly, urx, lly)
    aspect_ratio = bbox_height_meters / bbox_width_meters
    # print("Box height and width:", bbox_height_meters, bbox_width_meters, "meters")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bbox_area = gdf_ext.geometry.area.iloc[0] # area in deg squared

    # Calculate Base resolution derived from total area and max points
    base_resolution = np.sqrt(bbox_area / max_points)
    #print("Base resolution:", base_resolution, "Aspect ratio:", aspect_ratio)

    # Adjust grid resolution for longitude and latitude for aspect ratio
    lon_resolution = base_resolution / np.sqrt(aspect_ratio)
    lat_resolution = base_resolution * np.sqrt(aspect_ratio)
    #print("Grid resolution:", lon_resolution, lat_resolution)

    # Generate grid points
    x_coords = np.arange(llx-0.001, urx+0.001, lon_resolution)
    y_coords = np.arange(lly-0.001, ury+0.001, lat_resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:4326")

    # Filter points inside the glacier boundary
    points_in_glacier_gdf = gpd.sjoin(grid_points, gdf_ext, how="inner", predicate="within").drop(columns=['index_right'])

    # Remove points inside nunataks
    if gdf_nuns is not None and not gdf_nuns.empty:
        points_not_in_nunataks_gdf = gpd.sjoin(points_in_glacier_gdf, gdf_nuns, how="left", predicate="within")
        valid_points_gdf = points_not_in_nunataks_gdf[points_not_in_nunataks_gdf.index_right.isna()].drop(
            columns=['index_right'])
    else:
        valid_points_gdf = points_in_glacier_gdf

    # Prepare dictionary output
    points = {
        "lons": valid_points_gdf.geometry.x.tolist(),
        "lats": valid_points_gdf.geometry.y.tolist(),
        "nunataks": [0.0] * len(valid_points_gdf)
    }

    plot_gen_points = False
    if plot_gen_points:
        fig, ax = plt.subplots()
        gdf_ext.plot(ax=ax, ec='k', fc='none', linewidth=2)
        if len(gdf_nuns)>0: gdf_nuns.plot(ax=ax, ec='orange', fc='none')
        valid_points_gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=1)
        plt.show()

    #print(f"We have generated: {len(points["lons"])} points")

    return points


def generate_points_old(gdf_ext=None, gdf_nuns=None, n_points_regression=None, seed=None):

    points = {'lons': [], 'lats': [], 'nunataks': []}
    if seed is not None: np.random.seed(seed)

    llx, lly, urx, ury = gdf_ext.total_bounds # geometry bounds

    while (len(points['lons']) < n_points_regression):
        batch_size = min(n_points_regression, n_points_regression - len(points['lons']))  # Adjust batch size as needed
        r_lons = np.random.uniform(llx, urx, batch_size)
        r_lats = np.random.uniform(lly, ury, batch_size)
        points_batch_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(r_lons, r_lats), crs="EPSG:4326")

        # A bit faster
        # 1) Select only those points generated inside the external polygon
        # points_in_glacier_gdf = gpd.sjoin(points_batch_gdf, gl_geom_ext_gdf, how="inner", predicate="within").drop(columns=['index_right'])
        # 2) Exclude points that are inside any internal polygons
        # points_in_internal_polygons_gdf = gpd.sjoin(points_in_glacier_gdf, gl_geom_nunataks_gdf, how="left", predicate="within")
        # points_not_in_nunataks_gdf = points_in_internal_polygons_gdf[points_in_internal_polygons_gdf.index_right.isna()]

        # A bit slower
        # 1) First we select only those points generated inside the glacier
        points_yes_no_ext_gdf = gpd.sjoin(points_batch_gdf, gdf_ext, how="left", predicate="within")
        points_in_glacier_gdf = points_yes_no_ext_gdf[~points_yes_no_ext_gdf.index_right.isna()].drop(
            columns=['index_right'])
        indexes_of_points_inside = points_in_glacier_gdf.index
        # 2) Then we get rid of all those generated inside nunataks
        points_yes_no_nunataks_gdf = gpd.sjoin(points_batch_gdf.loc[indexes_of_points_inside], gdf_nuns,
                                               how="left", predicate="within")
        points_not_in_nunataks_gdf = points_yes_no_nunataks_gdf[points_yes_no_nunataks_gdf.index_right.isna()].drop(
            columns=['index_right'])

        points['lons'].extend(points_not_in_nunataks_gdf['geometry'].x.tolist())
        points['lats'].extend(points_not_in_nunataks_gdf['geometry'].y.tolist())
        points['nunataks'].extend([0.0] * len(points_not_in_nunataks_gdf))


    plot_gen_points = False
    if plot_gen_points:
        points_in_nunataks_gdf = points_yes_no_nunataks_gdf[~points_yes_no_nunataks_gdf.index_right.isna()].drop(
            columns=['index_right'])
        fig, ax = plt.subplots()
        #ax.plot(*gl_geom.exterior.xy, color='blue')
        gdf_ext.plot(ax=ax, ec='k', fc='none', linewidth=2)
        gdf_nuns.plot(ax=ax, ec='orange', fc='none')
        ax.scatter(x=points['lons'], y=points['lats'], c='b', alpha=0.5, s=1)
        #gl_geom_nunataks_gdf.plot(ax=ax, color='orange', alpha=0.5)
        #points_in_glacier_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
        #points_not_in_nunataks_gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=20, zorder=2)
        #points_in_nunataks_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
        plt.show()

    return points


def geographic_split_adaptive(glaciers_df=None, n_jobs=None, version=None):
    """
    Split glaciers into n_jobs geographic chunks, adapting to aspect ratio.
    Args:
        glaciers_df (pd.DataFrame): DataFrame from RGI (imported via OGGM).
        n_jobs (int): Number of geographic chunks to create.
    Returns:
        list of pd.DataFrame: list of lists. Each sublist contains glacier ids in a geographic chunk.
    """

    if version == '62':
        name_column_id = 'RGIId'
        #name_column_area = 'Area'
        #name_column_name = 'Name'
        name_column_lon, name_column_lat = 'CenLon', 'CenLat'
    elif version == '70G':
        name_column_id = 'rgi_id'
        #name_column_area = 'area_km2'
        #name_column_name = 'glac_name'
        name_column_lon, name_column_lat = 'cenlon', 'cenlat'

    # Since some custom glacier ids may be present in glaciers_df (with no centers from rgi), let's calculate glacier centers
    name_column_lon, name_column_lat = "centroid_lon", "centroid_lat"
    glaciers_df[name_column_lon] = glaciers_df.geometry.representative_point().x
    glaciers_df[name_column_lat] = glaciers_df.geometry.representative_point().y

    if n_jobs == 1:
        split_ids = glaciers_df[name_column_id].to_list()
        return [split_ids]

    # Extract geographic bounds
    lon_min, lon_max = glaciers_df[name_column_lon].min(), glaciers_df[name_column_lon].max()
    lat_min, lat_max = glaciers_df[name_column_lat].min(), glaciers_df[name_column_lat].max()
    #print(lon_min, lon_max, lat_min, lat_max)

    # Calculate aspect ratio
    delta_lon = lon_max - lon_min
    delta_lat = lat_max - lat_min

    # Calculate the desired aspect ratio
    aspect_ratio = delta_lon / delta_lat

    # Function to generate all factor pairs of n_jobs
    def get_factor_pairs(n_jobs):
        factors = []
        for i in range(1, int(np.sqrt(n_jobs)) + 1):
            if n_jobs % i == 0:
                factors.append((i, n_jobs // i))
        return factors

    # Function to select the best split pair based on aspect ratio
    def find_best_split_pair(factors, aspect_ratio):
        best_pair = None
        best_ratio_diff = float('inf')

        for (a, b) in factors:
            # Calculate aspect ratio of the pair
            pair_aspect_ratio = a / b
            # Find the difference between the pair's aspect ratio and the desired one
            ratio_diff = abs(pair_aspect_ratio - aspect_ratio)

            # If this pair is closer to the desired aspect ratio, choose it
            if ratio_diff < best_ratio_diff:
                best_pair = (a, b)
                best_ratio_diff = ratio_diff

        return best_pair

    # Generate factor pairs of n_jobs
    factors = get_factor_pairs(n_jobs)

    # Find the best split pair that respects the aspect ratio
    splits_lon, splits_lat = find_best_split_pair(factors, aspect_ratio)

    # Add assertion to verify the split consistency
    assert splits_lon * splits_lat == n_jobs, f"Splits ({splits_lon}x{splits_lat}) do not equal the number of jobs ({n_jobs})."
    #print(splits_lon, ' x ', splits_lat)

    # Define grid edges (buffer of 0.5deg)
    lon_bins = np.linspace(lon_min-0.5, lon_max+0.5, splits_lon + 1)
    lat_bins = np.linspace(lat_min-0.5, lat_max+0.5, splits_lat + 1)

    # Initialize chunks
    # Assign glaciers to chunks
    chunks = []
    for i in range(splits_lon):
        for j in range(splits_lat):
            in_chunk = glaciers_df[
                (glaciers_df[name_column_lon] >= lon_bins[i]) & (glaciers_df[name_column_lon] < lon_bins[i + 1]) &
                (glaciers_df[name_column_lat] >= lat_bins[j]) & (glaciers_df[name_column_lat] < lat_bins[j + 1])
                ]

            # Skip empty chunks
            if not in_chunk.empty:
                chunks.append(in_chunk)

            #print(i,j,len(in_chunk))

    # Calculate how many more chunks are needed
    no_missing = n_jobs - len(chunks)

    while no_missing > 0:
        #print(no_missing)
        # Find the largest chunk
        largest_chunk_idx = np.argmax([len(chunk) for chunk in chunks])
        largest_chunk = chunks.pop(largest_chunk_idx)  # Remove the largest chunk

        # Split the largest chunk into two parts (based on latitude or longitude)
        if (largest_chunk[name_column_lon].max() - largest_chunk[name_column_lon].min()) > (
                largest_chunk[name_column_lat].max() - largest_chunk[name_column_lat].min()):
            # Split along longitude
            mid_lon = largest_chunk[name_column_lon].median()
            left_chunk = largest_chunk[largest_chunk[name_column_lon] <= mid_lon]
            right_chunk = largest_chunk[largest_chunk[name_column_lon] > mid_lon]
        else:
            # Split along latitude
            mid_lat = largest_chunk[name_column_lat].median()
            left_chunk = largest_chunk[largest_chunk[name_column_lat] <= mid_lat]
            right_chunk = largest_chunk[largest_chunk[name_column_lat] > mid_lat]

        # Add the split chunks back to the list
        chunks.append(left_chunk)
        chunks.append(right_chunk)

        # Decrease the missing chunks count by 1, since we replaced one chunk with two
        no_missing -= 1

    assert len(chunks) == n_jobs, "Final number of chunks does not match n_jobs"

    # List of lists. Every sublist contains the ids for multiproc.
    split_ids = [chunk[name_column_id].tolist() for chunk in chunks]

    # check
    flattened_ids = [item for sublist in split_ids for item in sublist]
    assert set(flattened_ids) == set(glaciers_df[name_column_id].values), "Some ids went missing."

    return split_ids
