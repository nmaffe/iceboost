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