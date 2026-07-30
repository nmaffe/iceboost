import time, glob
import rasterio, rioxarray
from rioxarray import merge
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
import matplotlib.pyplot as plt
import json

def get_rgi_products(region=None, which_rgi_version=None):
    """
    :param region: rgi region 1 to 19
    :param which_rgi_version: 6 or 7
    :return: regional dataframes
    """

    if region is None: raise ValueError("You need to specify the region number as string. Exit.")

    if not isinstance(region, str): region = f"{region:02d}"

    rgi_glaciers, rgi_intersects, rgi_complexes, rgi_complexes_to_glaciers = None, None, None, None

    if which_rgi_version == 6:
        FILE_SHP_RGI_G = utils.get_rgi_region_file(region=region, version='62')
        FILE_INTERSECTS_SHP_RGI = utils.get_rgi_intersects_region_file(region=region, version='62')
        rgi_glaciers = gpd.read_file(FILE_SHP_RGI_G, engine='pyogrio')
        rgi_intersects = gpd.read_file(FILE_INTERSECTS_SHP_RGI, engine='pyogrio')

    elif which_rgi_version == 7:
        FILE_SHP_RGI_G = utils.get_rgi_region_file(region=region, version='70G')
        FILE_INTERSECTS_SHP_RGI = utils.get_rgi_intersects_region_file(region=region, version='70G')
        FILE_SHP_RGI_C = utils.get_rgi_region_file(region=region, version='70C')
        FILE_CtoG_links_json = FILE_SHP_RGI_C.replace(".shp", "-CtoG_links.json")

        print(FILE_SHP_RGI_G)
        print(FILE_INTERSECTS_SHP_RGI)
        print(FILE_SHP_RGI_C)
        print(FILE_CtoG_links_json)

        rgi_glaciers = gpd.read_file(FILE_SHP_RGI_G, engine='pyogrio')
        rgi_intersects = gpd.read_file(FILE_INTERSECTS_SHP_RGI, engine='pyogrio')
        rgi_complexes = gpd.read_file(FILE_SHP_RGI_C, engine='pyogrio')
        # parse json
        with open(FILE_CtoG_links_json, 'r', encoding='utf-8') as f:
            rgi_complexes_to_glaciers = json.load(f)
        #for complex_id, product_ids in list(rgi_complexes_to_glaciers.items())[:2]:
        #    print(f"Complex ID: {complex_id}")
        #    print(f"Glacier Products: {product_ids}\n")

        # Sanity check that complex ids are the same as in the json file
        assert rgi_complexes['rgi_id'].to_list() == list(rgi_complexes_to_glaciers.keys()), "CHECK !"

    else:
        raise ValueError(f"{which_rgi_version} not supported. Exit.")

    return rgi_glaciers, rgi_intersects, rgi_complexes, rgi_complexes_to_glaciers


rgi_6_or_7 = 7
rgi = 3

(rgi_glaciers, rgi_intersects,
 rgi_complexes, rgi_complexes_to_glaciers) = get_rgi_products(region=rgi,
                                                                which_rgi_version=rgi_6_or_7)

PATH_ICEBOOST_GLACIERS = glob.glob(f"/media/maffe/nvme/iceboost_global_deploy/iceboost_20251009/RGI{rgi_6_or_7}*")[0]
print(PATH_ICEBOOST_GLACIERS)

print(f'Analyzing Region {rgi}')
for n, (id_c, ids_g) in enumerate(list(rgi_complexes_to_glaciers.items())):
    print(f'{n}/{len(rgi_complexes_to_glaciers)}', '\t', id_c, '\t', ids_g)

    #print(f'\t Creating complex id {id_c}')

    # Get the tif files belonging to this id_c
    list_tifs_for_id_c = []

    #print(f'\t Extracting individual tif files ...')
    for id_g in ids_g:
        file_id_g = f'{PATH_ICEBOOST_GLACIERS}/rgi{rgi}/{id_g}.tif'
        tif = rioxarray.open_rasterio(f'{file_id_g}').astype("float32")
        list_tifs_for_id_c.append(tif)

        #print('\t\t', id_g)

    # Let's create a dictionary of all glaciers in the complex id
    dict_complex_id = {
        'complex_id': id_c,
        'complex_glacier_ids': [],
        'complex_areas': [],
        'complex_epsgs': [],
        'complex_GT_lats': [],
        'complex_GT_lons': [],
        'complex_GT_meas': [],
        'complex_volume': [],
        'complex_volume_error': [],
        'complex_volume_bsl': [],
        'complex_volume_bsl_error': [],
    }

    for tif in list_tifs_for_id_c:

        # pull values from glacier raster
        g_id = tif.attrs['id']
        g_area = tif.attrs['area']
        g_epsg = tif.rio.crs.to_epsg()
        g_volume = tif.attrs['volume']
        g_volume_error = tif.attrs['volume_error']
        g_volume_bsl = tif.attrs['volume_bsl']
        g_volume_bsl_error = tif.attrs['volume_bsl_error']

        # fill the dictionary
        dict_complex_id['complex_glacier_ids'].append(g_id)
        dict_complex_id['complex_areas'].append(g_area)
        dict_complex_id['complex_volume'].append(g_volume)
        dict_complex_id['complex_volume_error'].append(g_volume_error)
        dict_complex_id['complex_volume_bsl'].append(g_volume_bsl)
        dict_complex_id['complex_volume_bsl_error'].append(g_volume_bsl_error)
        dict_complex_id['complex_epsgs'].append(g_epsg)
        dict_complex_id['complex_GT_lats'] += json.loads(tif.attrs['ground_truth_lats'])
        dict_complex_id['complex_GT_lons'] += json.loads(tif.attrs['ground_truth_lons'])
        dict_complex_id['complex_GT_meas'] += json.loads(tif.attrs['ground_truth_meas'])


    #print(dict_complex_id)
    #print(dict_complex_id['complex_volume'])

    # if epsgs of all glaciers are the same, simply merge
    if len(set(dict_complex_id['complex_epsgs'])) == 1:

        # merge
        complex_epsg = dict_complex_id['complex_epsgs'][0]
        complex = merge.merge_arrays(list_tifs_for_id_c, method='max', nodata=np.nan, res=(100, 100))

    else:
        # if epsgs of the glaciers are not the same, chose that corresponding to max area
        # 0. find the index of the largest area and extract the corresponding epsg
        max_area_index = dict_complex_id['complex_areas'].index(max(dict_complex_id['complex_areas']))
        max_area_in_complex = dict_complex_id['complex_areas'][max_area_index]
        complex_epsg = dict_complex_id['complex_epsgs'][max_area_index] # int

        #print(set(dict_complex_id['complex_epsgs']))
        #print(complex_epsg)

        # 1. reproject if necessary
        ready_tifs = []
        for tif in list_tifs_for_id_c:
            if tif.rio.crs.to_epsg() != complex_epsg:
                # Reproject to the chosen EPSG if it doesn't match
                tif = tif.rio.reproject(complex_epsg, resampling=rasterio.enums.Resampling.bilinear,
                                                nodata=np.nan)
            ready_tifs.append(tif)

        # 2. merge
        complex = merge.merge_arrays(ready_tifs, method='max', nodata=np.nan, res=(100, 100))


    # Wipe all inherited attributes clean
    complex.attrs.clear()

    complex = complex.rio.write_nodata(np.nan)

    # Write fresh attributes
    complex.attrs['complex_id'] = id_c
    complex.attrs['crs'] = complex.rio.crs.to_string()
    complex.attrs['area'] = sum(dict_complex_id['complex_areas'])
    complex.attrs['volume'] = sum(dict_complex_id['complex_volume'])
    complex.attrs['volume_error'] = sum(dict_complex_id['complex_volume_error'])
    complex.attrs['volume_bsl'] = sum(dict_complex_id['complex_volume_bsl'])
    complex.attrs['volume_bsl_error'] = sum(dict_complex_id['complex_volume_bsl_error'])
    complex.attrs['glacier_ids'] = json.dumps(dict_complex_id['complex_glacier_ids'])
    complex.attrs['ground_truth_lats'] = json.dumps(dict_complex_id['complex_GT_lats'])
    complex.attrs['ground_truth_lons'] = json.dumps(dict_complex_id['complex_GT_lons'])
    complex.attrs['ground_truth_meas'] = json.dumps(dict_complex_id['complex_GT_meas'])

    # Verify that the CRS exists and matches the chosen EPSG code
    assert complex.rio.crs is not None, "Spatial CRS was not registered on the complex array."
    assert complex.rio.crs.to_epsg() == complex_epsg, (f"CRS mismatch! Expected {complex_epsg}, "
                                                       f"got {complex.rio.crs.to_epsg()}")
    # Verify that Nodata is np.nan
    assert np.isnan(complex.rio.nodata), f"Expected nodata to be None or np.nan, but got {complex.rio.nodata}"
    #print(complex.attrs['ground_truth_meas'])

    # Verify that resolution is 100 meters
    assert complex.rio.resolution() == (100., -100.), f"Unexpected resolution value {complex.rio.resolution()}"

    # save
    save = False
    if save:
        #print(PATH_ICEBOOST_GLACIERS)
        out = f"{PATH_ICEBOOST_GLACIERS}/complex_RGI_C/rgi{rgi}/{id_c}.tif"
        complex.rio.to_raster(out, compress="deflate", dtype="float32", windowed=True)

    plot_test = False
    if plot_test:
        test_tif = rioxarray.open_rasterio(out)
        print(test_tif.rio.crs)
        print(test_tif.rio.nodata)
        print(test_tif.rio.resolution())
        print(test_tif.shape)

        first_entry = test_tif.isel(band=0)  # ice thickness
        first_entry_np = first_entry.values  # ice thickness

        # 2. Plot the 2D spatial grid
        #first_entry.plot(cmap="viridis")
        im = plt.imshow(first_entry_np)
        cb = plt.colorbar(im)

        # 3. Display the plot
        plt.title("First Entry Visualization")
        plt.show()
