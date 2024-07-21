import utm
import scipy
import random
import numpy as np
import geopandas as gpd
from sklearn.neighbors import KDTree
from oggm import utils

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


# shap.plots.bar(shap_values, max_display=len(CFG.features))


def calc_volume_glacier(points_thickness, area=0):
    '''points_thickness is a <class 'pandas.core.series.Series'>
    If the series contains only nan, the returned volume will be zero'''
    # A potential drawback of this method is that I am randomly sampling in epsg:4326. In a utm projection
    # such sampling does not turn out to be uniform. Returned volume in km3.
    N = len(points_thickness)
    volume = np.sum(points_thickness) * 0.001 * area / N
    return volume


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
