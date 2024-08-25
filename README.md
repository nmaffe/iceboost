<div align="center">
    <img alt="Logo" src="img/logo_iceboost.svg" style="width: 100%; height: auto;" />
</div>

🚧🚧 UNDER CONSTRUCTION 🚧🚧

# ICEBOOST

a Gradient-Boosted Tree framework 
to model the ice thickness of the World's glaciers

#### Prepare the Tandem-X EDEM tiles
The model needs a Digital Elevation Model. We use Tandem-X EDEM. To automatically download 
all but only the necessary tiles that contain glaciers, run the following script.
```
python produce_txtfile_tandemx_rgi_tile_urls --rgi 
```
The script produces .txt files containing the url links pointing to the Tandem-X zip tiles, for each region.
Adjust the ```--save``` and ```--outfolder``` options to specify where to save the txt file.
Specify the region from 1 to 19 using the ```--rgi``` argument. The tiles are chosen such that they 
contain RGI v.6 glaciers with a buffer of 1/8 degree. You can increase this option as well.

Afterwards, run the following to download the tiles from https://download.geoservice.dlr.de:

```xargs -a TDM30_EDEM-url-list.txt -L1 curl -O -u 'usr:pass'```

#### Prepare ERA-5 temperature
The model needs a temperature field over glaciers. We use t2m from ERA5-Land and ERA5 merged together 
and averaged over 2000-2010. Download these 2 products from Copernicus and run the following while specifying the input 
and output paths: 
```
python create_ERA5_T2m.py --ERA5Land_folder --ERA5_folder --ERA5_outfolder 
```
to produce a 50 MB ```era5land_era5.nc``` file that will be needed.

#### Prepare the ice velocity products
ICEBOOST uses surface ice velocity from [Millan et al. (2022)](https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c), 
[Joughin et al. 2016 (Greenland, prod. NSIDC-0670)](https://nsidc.org/data/nsidc-0670/versions/1),
and [Mouginot et al. 2019 (Antarctica, prod. NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1). 
Download all these products. Put Millan's rgi-1-2 tiles in the same folder. Similarly, rgi-13-14-15 tiles together. 

#### Prepare the world's coastlines product

#### Create the training dataset 🏋️

---

#### Process training dataset and downscale 🏋️

---

#### Train model ensemble 🤖

---

#### Model inference 🔮

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