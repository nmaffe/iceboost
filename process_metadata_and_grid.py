import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Suppress RuntimeWarning
#warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
This program imports the generated metadata dataset from create_metadata.py and:
1. Processing:
    - Remove old measurements
    - Remove possible bad data
    - Retain only some final features and remove any nan.
2. Gridding:
    - gridding is done computing the mean of each feature in all pixels that form the grid, which is specified by
    the parameter --nbins_grid_latlon (default=20)
    
The processed and gridded dataframe is finally saved.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_metadata_folder', type=str,
                    default="/media/maffe/nvme/glathida/glathida-3.1.0/glathida-3.1.0/data/",
                    help="Input metadata folder to be gridded")
parser.add_argument('--input_metadata_file', type=str,
                    default="metadata35.csv", help="Input metadata file to be gridded")
parser.add_argument('--tmin', type=int, default=20050000, help="Keep only measurements after this year.")
parser.add_argument('--hmin', type=float, default=0.0, help="Keep only measurements with thickness greater than this.")
parser.add_argument('--method_grid', type=str, default='mean', help="Supported options: mean, median")
parser.add_argument('--nbins_grid_latlon', type=int, default=100, help="How many bins in the lat/lon directions")
parser.add_argument('--save', type=int, default=0, help="Save final dataset or not.")

args = parser.parse_args()

""" Import ungridded dataset """
glathida = pd.read_csv(f"{args.input_metadata_folder}{args.input_metadata_file}", low_memory=False)

""" A. Work on the dataset """
# A.1 Remove old (-er than 2005) measurements and erroneous data (if DATA_FLAG is not nan)
cond = ((glathida['SURVEY_DATE'] > args.tmin) & (glathida['DATA_FLAG'].isna()) & (glathida['THICKNESS']>=args.hmin))
glathida = glathida[cond]
print(f'Original columns: {list(glathida)} \n')

# A.2 Keep only these columns
cols = ['RGI', 'RGIId', 'POINT_LAT', 'POINT_LON', 'THICKNESS', 'Area', 'Area_icefree', 'Perimeter',
        'elevation', 'dmdtda_hugo', 'smb', 'dist_from_border_km_geom',
       'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'ith_m', 'ith_f',
        'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
        'Form', 'Aspect', 'TermType', 'v50', 'v100', 'v150', 'v300', 'v450', 'vgfa',
        'curv_50', 'curv_300', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa', 't2m', 'dist_from_ocean']


glathida = glathida[cols]
print(f'We keep only the following columns: \n {list(glathida)} \n{len(glathida)} rows')

# A.3 Remove nans
cols_dropna = [col for col in cols if col not in ('ith_m', 'ith_f')]
glathida = glathida.dropna(subset=cols_dropna)

#print(glathida.isna().sum())
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

glathida_gridded = pd.DataFrame(columns=glathida.columns)

# These features are the local ones that I have to average
features_to_grid = ['THICKNESS', 'elevation', 'smb', 'dist_from_border_km_geom',
        'ith_m', 'ith_f', 'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
        'v50', 'v100', 'v150', 'v300', 'v450', 'vgfa',
        'curv_50', 'curv_300', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa', 't2m', 'dist_from_ocean']

list_num_measurements_before_grid = []
list_num_measurements_after_grid = []

# loop over unique glaciers
for n, rgiid in enumerate(rgi_ids):

    #rgiid = 'RGI60-03.01517'

    glathida_id = glathida.loc[glathida['RGIId'] == rgiid]
    glathida_id_grid = pd.DataFrame(columns=glathida_id.columns)

    lons = glathida_id['POINT_LON'].to_numpy()
    lats = glathida_id['POINT_LAT'].to_numpy()

    # Those are the glacier-wide constant features
    area = glathida_id['Area'].iloc[0]
    area_noice = glathida_id['Area_icefree'].iloc[0]
    perimeter = glathida_id['Perimeter'].iloc[0]
    rgi = glathida_id['RGI'].iloc[0]
    zmin = glathida_id['Zmin'].iloc[0]
    zmax = glathida_id['Zmax'].iloc[0]
    zmed = glathida_id['Zmed'].iloc[0]
    Slope = glathida_id['Slope'].iloc[0]
    lmax = glathida_id['Lmax'].iloc[0]
    form = glathida_id['Form'].iloc[0]
    aspect = glathida_id['Aspect'].iloc[0]
    termtype = glathida_id['TermType'].iloc[0]
    dmdtda = glathida_id['dmdtda_hugo'].iloc[0]

    # make same checks
    if not glathida_id['Area'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique Area.")
    if not glathida_id['RGI'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGI.")
    if not glathida_id['RGIId'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGIId.")

    print(f'{n}/{len(rgi_ids)}, {rgiid}, Tot. meas to be gridded: {len(glathida_id)}')

    list_num_measurements_before_grid.append(len(glathida_id))

    # if only one measurement, append that line as is
    if (len(glathida_id) == 1):
        glathida_gridded = pd.concat([glathida_gridded, glathida_id], ignore_index=True)
        list_num_measurements_after_grid.append(len(glathida_id))
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

        # old version: remove all nans.
        non_nan_mask = ~np.isnan(H) # boolean matrix of H of only non-nans
        #x_indices, y_indices = np.where(non_nan_mask) # these are the indexes of non-non H
        #print(x_indices.shape, y_indices.shape)

        # new version: keep all values
        indices = np.indices(H.shape)
        x_indices = indices[0].flatten() # These are instead the indexes of H
        y_indices = indices[1].flatten() # These are instead the indexes of H

        xs = xcenters[x_indices]
        ys = ycenters[y_indices]
        # In the old version we only store non-nans. In the new one we keep them and remove in the end from all features except for ith_m, ith_f
        #zs = H[non_nan_mask]
        zs = H[x_indices,y_indices]

        # check how many values we have produced
        new_gl_nmeas = np.count_nonzero(~np.isnan(H))
        #print(feature, H.shape, new_gl_nmeas, xedges.shape, xs.shape, ys.shape, zs.shape)

        # Fill gridded feature
        glathida_id_grid['POINT_LON'] = xs  # unnecessarily overwriting each loop
        glathida_id_grid['POINT_LAT'] = ys  # unnecessarily overwriting each loop
        glathida_id_grid[feature] = zs

        # plot
        ifplot = False
        if (ifplot and feature == 'THICKNESS' and  rgiid=='RGI60-03.01517'):

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
    glathida_id_grid['Zmin'] = zmin
    glathida_id_grid['Zmax'] = zmax
    glathida_id_grid['Zmed'] = zmed
    glathida_id_grid['Slope'] = Slope
    glathida_id_grid['Lmax'] = lmax
    glathida_id_grid['Form'] = form
    glathida_id_grid['TermType'] = termtype
    glathida_id_grid['Aspect'] = aspect
    glathida_id_grid['dmdtda_hugo'] = dmdtda

    # append glacier gridded dataset to main gridded dataset
    # In the first passage glathida_gridded will be empty so we copy glathida_id_grid
    if glathida_gridded.empty:
        glathida_gridded = glathida_id_grid.copy()
    else:
        glathida_gridded = pd.concat([glathida_gridded, glathida_id_grid], ignore_index=True)

    list_num_measurements_after_grid.append(len(glathida_id_grid))

print(f"Finished. No. original measurements {len(glathida)} down to {len(glathida_gridded)}.")
#print(glathida_gridded.isna().sum())

# Remove all nans from all features except for ith_m and ith_f
# todo: i see a huge amount of nans are removed, which are neither ith_m nor ith_f. what are they ?
glathida_gridded = glathida_gridded.dropna(subset=cols_dropna)

print(f"Finished. No. original measurements {len(glathida)} down to {len(glathida_gridded)}, divided into:")
print(f"{glathida_gridded['RGI'].value_counts()}")

if args.save:
    ext = args.input_metadata_file[args.input_metadata_file.rfind('.'):]
    filename_out = (args.input_metadata_folder +
                    args.input_metadata_file.replace(ext, f'_hmineq{args.hmin}_tmin{args.tmin}_{args.method_grid}_grid_{args.nbins_grid_latlon}{ext}'))
    glathida_gridded.to_csv(filename_out, index=False)

    print(f"Gridded dataframe saved: {filename_out}")


