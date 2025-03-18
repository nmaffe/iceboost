import argparse
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

"""
This program imports the generated metadata dataset from create_metadata.py and:
1. Processing:
    - Remove old measurements
    - Remove possible bad data
    - Retain only some final features and remove any nan.
2. Gridding:
    - gridding is done computing the mean of each feature in all pixels that form the grid, which is specified by
    the parameter --nbins_grid_latlon (default=100)
    
The processed and gridded dataframe is finally saved.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--tmin', type=int, default=20050000, help="Keep only measurements after this year.")
parser.add_argument('--hmin', type=float, default=1.0, help="Keep only measurements with thickness greater than this.")
parser.add_argument('--method_grid', type=str, default='mean', help="Supported options: mean, median")
parser.add_argument('--nbins_grid_latlon', type=int, default=100, help="How many bins in the lat/lon directions")
parser.add_argument('--save', type=int, default=0, help="Save final dataset or not.")
args = parser.parse_args()

# Input datasets
GLATHIDA_FOLDER = "/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/"
GLATHIDA_FILE = "glathida42.csv"
POLAR_FOLDER = "/media/maffe/nvme/polar_ice_thickness_data/"
POLAR_FILE = "polar_ice_thick_train_iceboost3.parquet"

# save options
OUT_SAVE_FOLDER = "/media/maffe/nvme/iceboost_train_dataset"
today = datetime.today().strftime('%Y%m%d')
filename_out = f"iceboost_train_{today}_hmineq{args.hmin}_tmin{args.tmin}_{args.method_grid}_grid_{args.nbins_grid_latlon}.csv"
print(f"Training dataset will be saved as {filename_out}")
print("*"*100)

""" Import ungridded dataset """
glathida = pd.read_csv(f"{GLATHIDA_FOLDER}{GLATHIDA_FILE}", low_memory=False)
glathida['THICKNESS'] = glathida['THICKNESS'].astype(float)

# This glacier has a factor 10 too much.
glathida.loc[glathida['RGIId'] == 'RGI60-19.01406', 'THICKNESS'] /= 10.

# Remove glaciers containing bad data (id, min, max)
bad_data_to_remove = [
    ('RGI60-03.02756', 1, np.nan),
    ('RGI60-04.05541', 7, np.nan),
    ('RGI60-04.05595', 15, np.nan),
    ('RGI60-05.00808', np.nan, 300),
    ('RGI60-05.00814', np.nan, 300),
    ('RGI60-05.01906', np.nan, 200),
    ('RGI60-05.02244', np.nan, 200),
    ('RGI60-05.03731', np.nan, np.nan),
    ('RGI60-05.04255', 6, np.nan),
    ('RGI60-05.04276', np.nan, 50),
    ('RGI60-05.04288', np.nan, np.nan),
    ('RGI60-05.04304', np.nan, np.nan),
    ('RGI60-05.04309', np.nan, np.nan),
    ('RGI60-05.04339', np.nan, np.nan),
    ('RGI60-05.04786', np.nan, np.nan),
    ('RGI60-05.04959', np.nan, np.nan),
    ('RGI60-05.05000', np.nan, np.nan),
    ('RGI60-05.05191', 1, np.nan),
    ('RGI60-05.05412', np.nan, np.nan),
    ('RGI60-05.07490', np.nan, np.nan),
    ('RGI60-05.07542', 3, np.nan),
    ('RGI60-05.07545', 6, np.nan),
    ('RGI60-05.12322', np.nan, np.nan),
    ('RGI60-05.12325', np.nan, np.nan),
    ('RGI60-05.12532', np.nan, np.nan),
    ('RGI60-05.12761', np.nan, np.nan),
    ('RGI60-05.12783', np.nan, np.nan),
    ('RGI60-05.13058', np.nan, np.nan),
    ('RGI60-05.13564', 5, np.nan),
    ('RGI60-05.13612', np.nan, np.nan),
    ('RGI60-05.13651', np.nan, np.nan),
    ('RGI60-05.13693', np.nan, np.nan),
    ('RGI60-05.13713', np.nan, np.nan),
    ('RGI60-05.13722', 4, np.nan),
    ('RGI60-05.13756', np.nan, np.nan),
    ('RGI60-05.13785', 1, np.nan),
    ('RGI60-05.13961', 33, np.nan),
    ('RGI60-05.13983', 16, np.nan),
    ('RGI60-05.14147', 150, np.nan),
    ('RGI60-05.14783', np.nan, 50),
    ('RGI60-05.14816', np.nan, np.nan),
    ('RGI60-11.02739', np.nan, np.nan),
    ('RGI60-19.00137', 50, np.nan),
    ('RGI60-19.00139', 7, np.nan),
    ('RGI60-19.00396', 5, np.nan),
    ('RGI60-19.00416', 5, np.nan),
    ('RGI60-19.00422', np.nan, 400),
    ('RGI60-19.00459', np.nan, np.nan),
    ('RGI60-19.00461', np.nan, np.nan),
    ('RGI60-19.00474', np.nan, np.nan),
    ('RGI60-19.00492', np.nan, np.nan),
    ('RGI60-19.00501', 1, np.nan),
    ('RGI60-19.01172', np.nan, np.nan),
    ('RGI60-19.01294', np.nan, 600),
]
bad_data_to_remove_df = pd.DataFrame(bad_data_to_remove, columns=['ID', 'THICK_min', 'THICK_max'])

# Remove old data and apply minimum ice thickness
cond = ((glathida['SURVEY_DATE'] > args.tmin) & (glathida['DATA_FLAG'].isna()) & (glathida['THICKNESS']>=args.hmin))
glathida = glathida[cond]
print(f"Glathida after some constraints: {len(glathida)}")

"""Import polar and calculate the extra glaciers to add"""
""" Note that this is a very important policy. I am deciding to only consider those ids that are not 
present in GlaThiDa. Another option would be to contemplate all measurements in both datasets. """
unique_rgiid_glathida = glathida['RGIId'].unique()
polar = pd.read_parquet(f"{POLAR_FOLDER}{POLAR_FILE}")
polar_extra = polar[~polar['RGIId'].isin(unique_rgiid_glathida)]
cond = ((polar_extra['THICKNESS']>=args.hmin))
polar_extra = polar_extra[cond]
unique_rgiid_polar_extra = polar_extra['RGIId'].unique()
print(f"Polar data: {len(polar_extra)}")

# Concatenate glathida with icebridge
glathida = pd.concat([glathida, polar_extra], axis=0, ignore_index=True)
print(f"Glathida + Polar data: {len(glathida)}")

# Now based on manual visual investigations, we have identified the following data to remove
# Merge datasets
glathida = glathida.merge(bad_data_to_remove_df,
                          left_on='RGIId',
                          right_on='ID',
                          how='left')
# Remove from glathida the bad data
glathida = glathida[
    ~(
        glathida['ID'].notna() & (
            (glathida['THICK_min'].notna() & (glathida['THICKNESS'] < glathida['THICK_min'])) |
            (glathida['THICK_max'].notna() & (glathida['THICKNESS'] > glathida['THICK_max'])) |
            (glathida['THICK_min'].isna() & glathida['THICK_max'].isna())  # Remove all rows with this ID
        )
    )
]
glathida.drop(['ID', 'THICK_min', 'THICK_max'], axis=1, inplace=True)
print(f"After glathida merged with Polar data and after removing bad data: {len(glathida)}")

# Now we again apply minimum (it should not be necessary)
cond = ((glathida['THICKNESS']>=args.hmin))
glathida = glathida[cond]
print(f"Unique ids: {len(glathida['RGIId'].unique())}")


# A.2 Keep only these columns
cols_not_used = ['Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'Form', 'Aspect', 'TermType',]
cols = ['RGI', 'RGIId', 'POINT_LAT', 'POINT_LON', 'THICKNESS', 'Area', 'Area_icefree', 'Perimeter',
        'elevation', 'dmdtda_hugo', 'smb', 'dist_from_border_km_geom', 'ith_m', 'ith_f',
        'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
        'v50', 'v100', 'v150', 'v300', 'v450', 'vgfa',
        'curv_50', 'curv_100', 'curv_150', 'curv_300', 'curv_450', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa',
        't2m', 'dist_from_ocean', 'zmin', 'zmax', 'zmed', 'slope', 'aspect', 'curvature', 'lmax', 'Cluster_area',
        'Cluster_glaciers', 'Cluster_geometries', 'elevation_0_1']

glathida = glathida[cols]

# A.3 Remove nans
# RGI60-19.00707 will be removed from the dataset because velocities are zero
# In general, missing velocity is the first cause for deleting otherwise good ground truth data (40k points)
cols_dropna = [col for col in cols if col not in ('ith_m', 'ith_f')]
glathida = glathida.dropna(subset=cols_dropna)
print(f'After having removed nans in all features except for ith_m and ith_f we have {len(glathida)} rows')

""" B. Grid the dataset """
# We loop over all unique glacier ids; for each unique glacier we grid every feature.
print(f"Begin gridding.")
rgi_ids = glathida['RGIId'].unique().tolist()
print(f'We have {len(rgi_ids)} unique glaciers and {len(glathida)} rows')
#for rgi in glathida['RGI'].unique():
#    glathida_rgi = glathida.loc[glathida['RGI']==rgi]
#    print(rgi, len(glathida_rgi['RGIId'].unique().tolist()), len(glathida_rgi))
#print(glathida['RGI'].value_counts())

gridded_data_list = []

# These features are the local ones that I have to average
features_to_grid = ['THICKNESS', 'elevation', 'smb', 'dist_from_border_km_geom',
        'ith_m', 'ith_f', 'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
        'v50', 'v100', 'v150', 'v300', 'v450', 'vgfa',
        'curv_50', 'curv_100', 'curv_150', 'curv_300', 'curv_450', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa',
        't2m', 'dist_from_ocean', 'elevation_0_1']

list_num_measurements_before_grid = []
list_num_measurements_after_grid = []

# loop over unique glaciers
for n, rgiid in tqdm(enumerate(rgi_ids), total=len(rgi_ids), desc=f"Glacier", leave=True):

    #rgiid = 'RGI60-19.00707'

    glathida_id = glathida.loc[glathida['RGIId'] == rgiid]
    glathida_id_grid = pd.DataFrame(columns=glathida_id.columns)

    lons = glathida_id['POINT_LON'].to_numpy()
    lats = glathida_id['POINT_LAT'].to_numpy()

    # Those are the glacier-wide constant features
    area = glathida_id['Area'].iloc[0]
    area_noice = glathida_id['Area_icefree'].iloc[0]
    perimeter = glathida_id['Perimeter'].iloc[0]
    rgi = glathida_id['RGI'].iloc[0]
    #zmin = glathida_id['Zmin'].iloc[0]
    #zmax = glathida_id['Zmax'].iloc[0]
    #zmed = glathida_id['Zmed'].iloc[0]
    #Slope = glathida_id['Slope'].iloc[0]
    #lmax = glathida_id['Lmax'].iloc[0]
    #form = glathida_id['Form'].iloc[0]
    #aspect = glathida_id['Aspect'].iloc[0]
    #termtype = glathida_id['TermType'].iloc[0]
    dmdtda = glathida_id['dmdtda_hugo'].iloc[0]

    # new glacier-wide constant features calculated using dem
    zmin_with_dem       = glathida_id['zmin'].iloc[0]
    zmax_with_dem       = glathida_id['zmax'].iloc[0]
    zmed_with_dem       = glathida_id['zmed'].iloc[0]
    slope_with_dem      = glathida_id['slope'].iloc[0]
    aspect_with_dem     = glathida_id['aspect'].iloc[0]
    curvature_with_dem  = glathida_id['curvature'].iloc[0]
    lmax_with_dem       = glathida_id['lmax'].iloc[0]
    cluster_area        = glathida_id['Cluster_area'].iloc[0]
    cluster_no_glaciers = glathida_id['Cluster_glaciers'].iloc[0]
    cluster_no_geometries = glathida_id['Cluster_geometries'].iloc[0]

    # make same checks
    if not glathida_id['Area'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique Area.")
    if not glathida_id['RGI'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGI.")
    if not glathida_id['RGIId'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGIId.")

    print(f'{n}/{len(rgi_ids)}, {rgiid}, Tot. meas to be gridded: {len(glathida_id)}')

    list_num_measurements_before_grid.append(len(glathida_id))

    # if only one measurement, append that line as is
    if (len(glathida_id) == 1):
        list_num_measurements_after_grid.append(len(glathida_id))
        gridded_data_list.append(glathida_id) # Append data to list
        continue

    # if more than one measurement, calculate the rectangular domain for gridding
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    eps = 1.e-4
    binsx = np.linspace(min_lon-eps, max_lon+eps, num=args.nbins_grid_latlon)
    binsy = np.linspace(min_lat-eps, max_lat+eps, num=args.nbins_grid_latlon)
    assert len(binsx) == len(binsy) == args.nbins_grid_latlon, "Number of bins unexpected."

    # loop over each feature and grid
    for feature in features_to_grid:

        feature_array = glathida_id[feature].to_numpy()
        #print(feature, type(feature_array), feature_array.shape,
        #      lons.shape, lats.shape, len(binsx), len(binsy), np.isnan(feature_array).sum())

        #if np.isnan(feature_array).any(): raise ValueError('Watch out, nan in feature vector')

        if args.method_grid == 'mean':
            statistic = np.nanmean
        elif args.method_grid == 'median':
            statistic = np.nanmedian
        else: raise ValueError("method not supported.")

        # grid the feature
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(x=lons, y=lats, values=feature_array,
                                                                 statistic=statistic, bins=[binsx, binsy])

        # calculate the latitude and longitude of the grid
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2

        # new version: keep all values
        indices = np.indices(H.shape)
        x_indices = indices[0].flatten() # These are instead the indexes of H
        y_indices = indices[1].flatten() # These are instead the indexes of H

        xs = xcenters[x_indices]
        ys = ycenters[y_indices]
        # In the old version we only store non-nans. In the new one we keep them and
        # remove in the end from all features except for ith_m, ith_f
        #zs = H[non_nan_mask]
        zs = H[x_indices,y_indices]

        # check how many values we have produced
        new_gl_nmeas = np.count_nonzero(~np.isnan(H))
        #print(feature, H.shape, new_gl_nmeas, xedges.shape, xs.shape, ys.shape, zs.shape)

        # Fill gridded feature
        glathida_id_grid['POINT_LON'] = xs  # unnecessarily overwriting each loop
        glathida_id_grid['POINT_LAT'] = ys  # unnecessarily overwriting each loop
        glathida_id_grid[feature] = zs

        #if feature == 'THICKNESS':
        #    print(np.sum(feature_array == 0), len(feature_array), np.sum(feature_array == 0)/len(feature_array))
        #    print(np.sum(zs == 0), np.count_nonzero(zs>=0), np.sum(zs == 0)/np.count_nonzero(zs>=0))
        #    input('wait')

        # plot
        ifplot = False
        if (ifplot and feature == 'THICKNESS' and rgiid == 'RGI60-19.00707'):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            s1 = ax1.scatter(x=lons, y=lats, c=feature_array, s=50, cmap='jet', vmin=0, vmax=750)
            cbar1 = plt.colorbar(s1, ax=ax1, alpha=1)
            cbar1.set_label(feature, labelpad=15, rotation=270)

            s2 = ax2.scatter(x=xs, y=ys, c=zs, s=50, cmap='jet', vmin=0, vmax=750)
            cbar2 = plt.colorbar(s2, ax=ax2, alpha=1)
            cbar2.set_label(feature, labelpad=15, rotation=270)

            for x_edge in xedges:
                ax1.axvline(x_edge, color='gray', linestyle='--', linewidth=0.1)
                ax2.axvline(x_edge, color='gray', linestyle='--', linewidth=0.1)
            for y_edge in yedges:
                ax1.axhline(y_edge, color='gray', linestyle='--', linewidth=0.1)
                ax2.axhline(y_edge, color='gray', linestyle='--', linewidth=0.1)
            plt.show()

    # add these features that are constant for each glacier
    glathida_id_grid['RGI'] = rgi
    glathida_id_grid['RGIId'] = rgiid
    glathida_id_grid['Area'] = area
    glathida_id_grid['Area_icefree'] = area_noice
    glathida_id_grid['Perimeter'] = perimeter
    #glathida_id_grid['Zmin'] = zmin
    #glathida_id_grid['Zmax'] = zmax
    #glathida_id_grid['Zmed'] = zmed
    #glathida_id_grid['Slope'] = Slope
    #glathida_id_grid['Lmax'] = lmax
    #glathida_id_grid['Form'] = form
    #glathida_id_grid['TermType'] = termtype
    #glathida_id_grid['Aspect'] = aspect
    glathida_id_grid['dmdtda_hugo'] = dmdtda
    glathida_id_grid['zmin'] = zmin_with_dem
    glathida_id_grid['zmax'] = zmax_with_dem
    glathida_id_grid['zmed'] = zmed_with_dem
    glathida_id_grid['slope'] = slope_with_dem
    glathida_id_grid['aspect'] = aspect_with_dem
    glathida_id_grid['curvature'] = curvature_with_dem
    glathida_id_grid['lmax'] = lmax_with_dem

    glathida_id_grid['Cluster_area'] = cluster_area
    glathida_id_grid['Cluster_glaciers'] = cluster_no_glaciers
    glathida_id_grid['Cluster_geometries'] = cluster_no_geometries

    # Append data to list
    gridded_data_list.append(glathida_id_grid) # faster method

    list_num_measurements_after_grid.append(len(glathida_id_grid))


# Remove all nans from all features except for ith_m and ith_f
glathida_gridded = pd.concat(gridded_data_list, ignore_index=True).dropna(subset=cols_dropna)
#print(glathida_gridded.isna().sum())

print(f"Finished. No. original measurements {len(glathida)} down to {len(glathida_gridded)}, divided into:")
print(f"{glathida_gridded['RGI'].value_counts()}")

if args.save:
    glathida_gridded.to_csv(f"{OUT_SAVE_FOLDER}/{filename_out}", index=False)
    print(f"Training dataset saved: {OUT_SAVE_FOLDER}/{filename_out}")


