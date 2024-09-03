<div align="center">
    <img alt="Logo" src="img/logo_iceboost.svg" style="width: 100%; height: auto;" />
</div>


a Gradient-Boosted Tree framework 
to model the ice thickness of the World's glaciers

---
## Prepare model inputs

#### 1. Setup OGGM
Install [OGGM](https://oggm.org/). ICEBOOST uses OGGM's glacier geometries, v62. They are a slight revision-improvement 
of the official, RGI v6 repository, with some additional glaciers added as well.
Once installed, specify its location in the ``` config/config.yaml``` file,
under the ```oggm_dir``` argument.


#### 2. Tandem-X EDEM tiles
The model needs a Digital Elevation Model. We use Tandem-X EDEM. Ensure that you have enough storage space (~600.0 GB).
To automatically download all but only the necessary tiles that contain glaciers,
- Run the following:
```
python produce_txtfile_tandemx_rgi_tile_urls --rgi 
```
The script generates .txt file containing the url links pointing to the Tandem-X tiles (as .zip files), for region ```--rgi```.
Adjust the ```--save``` and ```--outfolder``` options to specify where to save the txt file.

- Run the code using the ```--rgi``` argument from 1 to 19 to generate the 19 txt files. The tiles are chosen such 
that they contain all RGI v.6 glaciers with a buffer of 1/8 degree. You can increase this option as well.

Once you have all ```.txt``` files, setup a [DLR account at the EOWEB Geoportal](https://eoweb.dlr.de/egp/). 

Now, create a structure of empty folders like the following. The path of the root folder Tandem-X-EDEM should 
be specified  in the ```config.yaml``` file, under the ```tandemx_dir``` argument.
The directory structure should be organized with 19 subdirectories as follows:
```
Tandem-X-EDEM/
├── RGI_01/
├── RGI_02/
├── RGI_03/
├── ...
└── RGI_19/
```

Place the txt files inside the respective folders. 
- Run the following to download 
all the tiles specified in the txt files:

```xargs -a TDM30_EDEM-url-list.txt -L1 curl -O -u 'usr:pass'```

Repeat for all 19 txt files. 
Great. You should have all zip tiles in all folders. 
Now the last step is unpacking them.

#### 3. Prepare ERA-5 temperature
The model needs a temperature field over all glaciers. We use t2m from ERA5-Land and ERA5 merged together 
and averaged over 2000-2010. 
Download these 2 products from the [Copernicus Climate Change Service C3S Climate Date Store](https://cds.climate.copernicus.eu/cdsapp#!/home) 
from 2000 to 2010 at monthly resolution.

- Run the following while specifying the input and output paths: 
```
python create_ERA5_T2m.py --ERA5Land_folder --ERA5_folder --ERA5_outfolder 
```
The ```--ERA5Land_folder``` and ```--ERA5_folder``` arguments should point to the folders containing 
the downloaded ```ERA5-land``` and ```ERA5``` t2m products, while ```--ERA5_outfolder``` points the 
destination folder for the generated file.

In the code, set ```save=True``` to save the generated ```era5land_era5.nc``` temperature field.

#### 4. Prepare the ice velocity products
ICEBOOST uses surface ice velocity from [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c), 
[Joughin et al. 2016 (Greenland, prod. NSIDC-0670)](https://nsidc.org/data/nsidc-0670/versions/1),
and [Mouginot et al. 2019 (Antarctica, prod. NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1). 

Setup a directory structure like the following, download tiles and place them in the respective folders.
- [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c)
- Greenland_NSIDC: ice velocity of Greenland from [NSIDC](https://nsidc.org/data/nsidc-0670/versions/1)
- Antarctica_NSIDC: Ice velocity of Antarctica from [NSIDC](https://nsidc.org/data/nsidc-0754/versions/1)
```
Millan/
├── thickness/             
└── velocity/           
    ├── RGI-1-2/        
    ├── RGI-3/
    ├── RGI-4/
    ├── RGI-5/
    ├── ...
    ├── RGI-12/
    ├── RGI-13-15/
    ├── RGI-16/
    ├── ...
    └── RGI-19/
Greenland_NSIDC
├── thickness/
└── velocity/
Antarctica_NSIDC
├── thickness/NSIDC-0756/
└── velocity/NSIDC-0754/ 
```

Note: place Millan et al. (2022) tiles in RGI 1-2 and 13-14-15 together. 

From NSIDC download the ```greenland_vel_mosaic250_vx_v1.tif``` and ```greenland_vel_mosaic250_vy_v1.tif``` files
and place them in ```Greenland_NSIDC/velocity/```. 

From NSIDC download the ```antarctic_ice_vel_phase_map_v01.nc``` file 
and place it in ```Antarctica_NSIDC/velocity/NSIDC-0754/```. 


Specify the location of these folders in the ``` config/config.yaml``` file, under the arguments:
```millan_velocity_dir```, ```NSIDC_velocity_Greenland_dir```, and ```NSIDC_velocity_Antarctica_dir```.

#### 5. Prepare the world's coastlines product
ICEBOOST uses a global coastline product, obtained from the Global Self-consistent, Hierarchical, High-resolution 
Geography Database, Version 2.3.7. Download the shoreline polygons product at 'f' (full) resolution from [here](https://www.soest.hawaii.edu/pwessel/gshhg/).
We only care about land-and-ocean boundaries, therefore we only need the following two files:

- ```GSHHS_f_L1.shp```: boundary between land and ocean, except Antarctica.
- ```GSHHS_f_L6.shp```: boundary between Antarctica grounding-line and ocean.

Merge these two datasets to generate a final dataset of global coastline product. 
You can use the following code snippet:
```
import pandas as pd
import geopandas as gpd

gdf1 = gpd.read_file('/YOUR_IN_PATH/GSHHS_f_L1.shp', engine='pyogrio')
gdf6 = gpd.read_file('/YOUR_IN_PATH/GSHHS_f_L6.shp', engine='pyogrio')
gdf16 = pd.concat([gdf1, gdf6], ignore_index=True)
gdf16.to_file('/YOUR_OUT_PATH/GSHHS_f_L1_L6.shp', driver='ESRI Shapefile')
```
Place the generated ```GSHHS_f_L1_L6.shp``` file in a folder specified in ``` config/config.yaml``` file, 
under the argument ```coastlines_gshhg_dir/```.

#### 6. Prepare the RACMO surface mass balance product
Over Greenland and Antarctica, ICEBOOST uses RACMO mass balance.
- Greenland: RACMO2.3, Downscaled at 1 km, from [Noël, B., & van Kampenhout, L. (2019)](https://zenodo.org/records/3367211).
- Antarctica: : RACMO2.3, Downscaled at 2 km, from [Noël, B., et al. (2023)](https://zenodo.org/records/10007855)

Greenland:
1. Download the ```SMB_rec_RACMO2.3p2_1km_1961-1990.nc``` file. 
2. Run ```python create_racmo_greenland.py```, with ```save=True``` to generate the final 1961-1990 averaged mass 
balance product: ```smb_greenland_mean_1961_1990_RACMO23p2_gf.nc```

Antarctica: 
1. Download the ```smb_rec.1979-2021.RACMO2.3p2_ANT27_ERA5-3h.AIS.2km.YY.nc``` file. 
2. Run ```python create_racmo_antarctica.py```, with ```save=True``` to generate the final 1979-2021 averaged mass 
balance product: ```smb_antarctica_mean_1979_2021_RACMO23p2_gf.nc```

Setup the following folder structure and place the generated files in the relevant folders:
```
racmo/
├── antarctica_racmo2.3p2/smb_antarctica_mean_1979_2021_RACMO23p2_gf.nc
└── greenland_racmo2.3p2/smb_greenland_mean_1961_1990_RACMO23p2_gf.nc
```
In ``` config/config.yaml``` file, specify the location of the racmo root folder under the argument ```racmo_dir/```.

#### 7. Prepare all other models' ice thickness solutions for comparisons
ICEBOOST code uses the following products of ice thickness distributions for comparisons:
- [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c): all regions
- [BedMachine Greenland, v5](https://nsidc.org/data/idbmg4/versions/5): Greenland
- [BedMachine Antarctica, v3](https://nsidc.org/data/nsidc-0756/versions/3): Antarctica
- [Farinotti et al. (2019)](https://www.research-collection.ethz.ch/handle/20.500.11850/315707): all regions

Download all ice thickness tiles from [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c)
and place them inside the ```Millan/thickness/``` folder, following the same structure described for the velocity tiles (point 4).

From NSIDC download ```BedMachineGreenland-v5.nc``` and place it in ```Greenland_NSIDC/thickness/```. 
From NSIDC download ```BedMachineAntarctica-v3.nc``` and place it in ```Antarctica_NSIDC/thickness/NSIDC-0756/```. 

From [Farinotti et al. (2019)](https://www.research-collection.ethz.ch/handle/20.500.11850/315707), download 
the ```composite_thickness_RGI60-all_regions.zip``` archive and extract its content in a folder ```Farinotti/```.

Finally, in ``` config/config.yaml```, specify the locations of the following folders: ```millan_icethickness_dir```,
```NSIDC_icethickness_Greenland_dir```, ```NSIDC_icethickness_Antarctica_dir```, ```farinotti_icethickness_dir```.

## Create the training dataset 🏋️

---

## Process training dataset and downscale 🏋️

---

## Train model ensemble 🤖

---

## Model inference 🔮

---

## Acknowledgments

<p align="left">
  <a href="https://marie-sklodowska-curie-actions.ec.europa.eu/">
    <img alt="EU" src="img/logo_MSCA.png" height="70" />
  </a>
  <a href="https://www.climatechange.ai/">
    <img alt="CCAI" src="img/logo_CCAI.png" height="70" />
  </a>
</p>

This work has received funding from the European Union’s Horizon 2020
research and innovation programme, under the Marie Skłodowska-Curie 
grant agreement No 101066651, project SKYNET. This work was also funded by 
the Climate Change  AI Innovation Grants program, under the project ICENET.