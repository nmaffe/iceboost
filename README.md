<div align="center">
    <img alt="Logo" src="img/logo_iceboost.svg" style="width: 100%; height: auto;" />
</div>


a Gradient-Boosted Tree framework 
to model the ice thickness of the World's glaciers

---
### Prepare model inputs
Specify the location of the following inputs in the ``` config/config.yaml``` file.

#### 1. Tandem-X EDEM tiles
The model needs a Digital Elevation Model. We use Tandem-X EDEM. To automatically download 
all but only the necessary tiles that contain glaciers. Ensure that you have enough storage space (~600.0 GB).

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

#### 2. Prepare ERA-5 temperature
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

#### 3. Prepare the ice velocity products
ICEBOOST uses surface ice velocity from [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c), 
[Joughin et al. 2016 (Greenland, prod. NSIDC-0670)](https://nsidc.org/data/nsidc-0670/versions/1),
and [Mouginot et al. 2019 (Antarctica, prod. NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1). 
Download these products and specify the folders in the ``` config/config.yaml``` file. 
Put Millan's rgi-1-2 tiles in the same folder. Similarly, rgi-13-14-15 tiles together. 
- - 
#### 4. Prepare the world's coastlines product

### Create the training dataset 🏋️

---

### Process training dataset and downscale 🏋️

---

### Train model ensemble 🤖

---

### Model inference 🔮

---

### Acknowledgments

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