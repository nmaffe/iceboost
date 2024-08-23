import utm
import scipy
import random
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from sklearn.neighbors import KDTree
from scipy.spatial import distance_matrix
from oggm import utils
import xgboost as xgb
import catboost as cb
from PIL import Image
import matplotlib.pyplot as plt

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
    return lmax

def lmax_imputer(geometry, glacier_epsg):
    '''
    geometry: glacier external geometry as pandas geodataframe in 4326 prjection
    glacier_epsg: glacier espg
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

    return cm


def calc_geoid_heights(lons=None, lats=None, h_wgs84=None):
    '''Calculates orthometric heights'''
    transformer = Transformer.from_crs("epsg:4326", "epsg:3855", always_xy=True)
    _, _, h_egm2008 = transformer.transform(lons, lats, h_wgs84)
    return h_egm2008

def calc_volume_glacier(y1=None, y2=None, area=0, h_egm2008=None):
    '''
    :param y1: numpy.ndarray. Ice thickness [m]
    :param y2: numpy.ndarray. Ice thickness [m]
    :param area: float [km2]
    :return: volume [km3].
    '''
    y_xgb = y1
    y_cat = y2
    N = len(y1)
    f = 0.001 * area / N

    # Millan or Farinotti
    if y2 is None:
        volume = np.sum(y1) * f
        return volume

    # iceboost
    else:
        y_mean = 0.5 * (y_xgb + y_cat)
        y_mean = np.where(y_mean < 0, 0, y_mean)

        # volume ice
        volume = np.sum(y_mean) * f
        # volume ice above sea level
        #volume_af = np.sum(np.where(h_egm2008 - y_mean > 0, y_mean, h_egm2008)) * f
        # volume ice below sea level
        volume_bsl = np.sum(np.where(h_egm2008 - y_mean > 0, 0.0, y_mean - h_egm2008)) * f

        err_points = np.std((y_xgb, y_cat), axis=0)
        # This error considers the point-wise spread between the models
        err_volume_points = np.sqrt(np.sum(err_points**2)) * f
        # This error is the semi-difference of the 2 modeled volumes.
        err_volume_range = 0.5 * np.abs(np.sum(y_xgb) - np.sum(y_cat)) * f
        # Add in quadrature the two errors
        err_volume = np.sqrt(err_volume_points**2 + err_volume_range**2)

        return volume, err_volume, volume_bsl


def get_random_glacier_rgiid(name=None, rgi=11, area=None, seed=None):
    """Provide a rgi number and seed. This method returns a
    random glacier rgiid name.
    If not rgi is passed, any rgi region is good.
    """
    # setup oggm version
    utils.get_rgi_dir(version='62')
    utils.get_rgi_intersects_dir(version='62')

    if name is not None: return name
    if seed is not None:
        np.random.seed(seed)
    if rgi is not None:
        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
    if area is not None:
        oggm_rgi_glaciers = oggm_rgi_glaciers[oggm_rgi_glaciers['Area'] > area]
    rgi_ids = oggm_rgi_glaciers['RGIId'].dropna().unique().tolist()
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

    model_xgb_filename = config_file.model_input_dir + config_file.model_filename_xgb
    iceboost_xgb = xgb.Booster()
    iceboost_xgb.load_model(model_xgb_filename)

    model_cat_filename = config_file.model_input_dir + config_file.model_filename_cat
    iceboost_cat = cb.CatBoostRegressor()
    iceboost_cat.load_model(model_cat_filename, format='cbm')

    return iceboost_xgb, iceboost_cat

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