# Evaluation of Preprocessing Methods of Sentinel-2 Data and their Impact on Traditional and Modern Empirical Satellite Derived Bathymetry Estimations

The goal of this project is to look at the influence of different preprocessing methods for Sentinel-2 data products and
their influence on traditional algorithms like the Stumpf Log-Ratio Method (Stumpf et al., 2003) in contrast to modern
approaches like LightGBM (Ke et al., 2017).

## ⚠ WIP Warning ⚠

The content of this repository is still work in progress. The modules and notebooks still need to be documented and
the conference paper still needs to be finished and linked correctly.

## Analysis Areas of Interest

The analysis looks at three different areas:

- Section of shallow ocean water near the north-west corner
  of the Bahamas `BBox: (25.23467352,-78.43272685,25.31877266,-78.23940804)`
- Section of shallow ocean water the west coast of Puerto Rico
  `BBox: (18.14442526,-67.24112119,18.17335221,-67.18944271)`
- [Mille Lacs Lake](https://en.wikipedia.org/wiki/Mille_Lacs_Lake) in Minnesota, USA
  `BBox: (46.099296265601545,-93.83878319721899,46.377612102131366,-93.44756526336063)`

## Analysis Data

The data used for this environment consists of:

- Shapefiles for certain AOIs created in QGIS
- Bathymetry maps from various sources
- Sentinel-2 L1C scenes and derived L2A and Acolite products

I will share the shapefiles as well as the analysis ready EOPatches as part of the Zenodo artifacts. As I am not the owner
of the original source data I will not distribute but rather document their origin for interested readers to reproduce.

If you experience problems reproducing the source data set reach out to me at `gregor_beyerle (at) outlook (dot) com` and I will
do my best to provide you with the access you need.

The Bathymetry Sources are:

- Mille Lacs Lake: [Lakes Data for Minnesota](https://www.mngeo.state.mn.us/chouse/water_lakes.html)
  [Bathybase Entry](http://www.bathybase.org/Data/800-899/895/)
- Puerto Rico: Grid Export [NOAA NCEI Data Viewer](https://www.ncei.noaa.gov/maps/bathymetry/)
- Bahamas: handed down from previous project. The source reference is unfortunately lost.

## Computing Environment

This project was mainly executed on a Laptop PC (Lenovo ThinkPad E14 Gen 2, Intel Core(TM) i7-1165G7, 32 GB RAM, Windows 10 21 H2).
While especially the modelling notebooks can make good use of additional CPU resources a machine with lower specs should be still
sufficient to repeat all processing steps. Windows users should be able to directly recreate the
[conda](https://docs.conda.io/en/latest/) from the `environment.yml` file in this repository. Linux and macOS users will need to
adapt the environment as some transitive dependencies are currently locked at Windows specific versions.

## Interpretation of Notebook Order

In the `notebooks` directory of this repository you will find numerated Jupyter Notebooks which can be subdivided into the following
process steps:

- Bathymetry Map Preprocessing (
  [00 - Puerto Rico](notebooks/00__preparing_puerto_rico_bathy_aoi.ipynb),
  [01 - Bahamas](notebooks/01__preparing_bahamas_bathy_aoi.ipynb),
  [03 - Mille Lacs Lake](notebooks/03__preparing_mille_lacs_bathy_aoi.ipynb))
- Sentinel-2 Data Preprocessing and Dataset Merge (
  [04 - Puerto Rico](notebooks/04__dataset_preparation_puerto_rico.ipynb),
  [05 - Bahamas](notebooks/05__dataset_preparation_bahamas.ipynb),
  [06 - Mille Lacs Lake](notebooks/06__dataset_preparation_mille_lacs.ipynb))
- Stumpf Log-Regression Fitting and Evaluation (
  [07 - Puerto Rico](notebooks/07__stumpf_log_regression_puerto_rico.ipynb),
  [08 - Bahamas](notebooks/08__stumpf_log_regression_bahamas.ipynb),
  [09 - Mille Lacs Lake](notebooks/09__stumpf_log_regression_mille_lacs.ipynb))
- LightGBM Fitting and Evaluation (
  [10 - Puerto Rico](notebooks/10__lgbm_calibration_puerto_rico.ipynb),
  [11 - Bahamas](notebooks/11__lgbm_calibration_bahamas.ipynb),
  [12 - Mille Lacs Lake](notebooks/12__lgbm_calibration_mille_lacs.ipynb))

Each notebook includes a detailed description of the current context and each taken step. If you wish to read a more condensed
writeup of the project please feel free to follow the link to my conference paper.

## Python Sources

While working on this project I produced a rather generic `eolearn_extras` module which contains some eo-learn tasks which could
be useful to others and a less generic collection of helper code in the `notebooks/sdb_utils` directory. All the code is available
freely under the MIT license. If you find any bugs or need further assistance please don't hesitate to open an issue.

## Approach

The general analysis approach can be seen in <a href="#fig-1">Figure 1</a>. As both the traditional as well as the modern model
are supervised learning algorithms we need to provide ground truth values for training. Those values can be extracted from
bathymetry maps which represent the depth profile (or underwater topography) of areas of inland or ocean water.
Two possible repositories are [Bathybase](http://www.bathybase.org/) and the National Oceanic and Atmospheric
Administration's (NOAA) National Centers for Environmental Information (NCEI)
[bathymetry portal](https://www.ncei.noaa.gov/maps/bathymetry/).

For a given area of interest (AOI) which either includes the extent of the whole bathymetry map or a particular subsection
we search for Sentinel-2 scenes which contain the AOI at a time with no cloud obstruction and - in the case of regions which
experience low temperatures - no ice formation. Once a fitting scene is found we download the complete Standard Archive
Format for Europe (SAFE) archive and store it for further preprocessing. It is essential not to use partial downloads
(e.g. with the [sentinelsat](https://sentinelsat.readthedocs.io/en/stable/index.html) Python package) because further
preprocessing methods assume that the SAFE archives are complete.

In this project two preprocessing methods for atmospheric correction are evaluated against the top of atmosphere (TOA)
L1C product. One is the L2A product generated by using the Sen2Cor processor (Main-Knorn et al., 2017) while the other
is the resulting data product produced by applying the [Acolite](https://github.com/acolite/acolite)
(Vanhellmont and Ruddick, 2016) processor.
<a href="#table-1">Table 1</a> shows the exact version of the used operating system (OS) as well as the versions of the
processors.

| Software   | Version                                                     |
| ---------- | ----------------------------------------------------------- |
| Windows OS | 21H2 Build 19044.1706                                       |
| Sen2Cor    | 2.10.01-win64                                               |
| Acolite    | Generic Git - Hash dafc2d4bced4864f0bc111b9e0d3348ff16a5336 |
<p align="center">
    <b id="table-1">Table 1: Used software for executing preprocessor</b>
</p>

All further processing of the acquired raster images to create analysis ready data (ARD) is done using
the [eo-learn framework](https://eo-learn.readthedocs.io/en/latest/). We either use the complete bounding
box of the bathymetry map or a prepared bounding box (supplied as a shape file) to crop or subset the
ground truth image and all other consecutive images. Depending on the Sentinel-2 scenes coordinate reference
system (CRS) - Sentinel-2 L1C scenes use the most fitting Universal Transverse Mercator (UTM) projection
in relation to the scene location - we reproject the bathymetry data to avoid further reprojections down
then line with the Sentinel-2 products. Sentinel-2 bands have at most a resolution of ten-by-ten meters so we
also resample the bathymetry data accordingly.

![General Approach Schematic](diagrams/diagram_exports/approach_schematic.drawio.png)
<p align="center">
    <b id="fig-1">Fig 1: General analysis approach</b>
</p>

## Original Dataset Description

In case you wand to reproduce the exact data directory you will most likely find this tree representation
helpful. The information
[encoded into Sentinel-2 product names](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention)
should allow you to find and retrieve them. Should you need additional information do not hesitate to reach out.

```bash
sdb_datasets
├── eopatches
│   ├── bahamas_bathy_eop
│   ├── bahamas_sentinel_merged
│   ├── mille_lacs_bathy_eop
│   ├── mille_lacs_sentinel_merged
│   ├── puerto_rico_bathy_eop
│   └── puerto_rico_sentinel_merged
├── mille_lacs.tiff
├── ncei_nos_bag_puerto_rico.tiff
├── sentinel_bahamas
│   ├── S2A_MSIL1C_20211019T155531_N0301_R011_T17RQJ_20211019T192647.SAFE
│   ├── S2A_MSIL2A_20211019T155531_N9999_R011_T17RQJ_20220423T204757.SAFE
│   └── S2A_MSI_20211019T155531_ACOLITE_SUBSET
├── sentinel_mille_lacs
│   ├── S2A_20210429T170851_ACOLITE_SUBSET
│   ├── S2A_MSIL1C_20210429T170851_N0300_R112_T15TVM_20210429T215623.SAFE
│   ├── S2A_MSIL1C_20210916T170941_N0301_R112_T15TVM_20210916T210302.SAFE
│   ├── S2A_MSIL1C_20211016T171311_N0301_R112_T15TVM_20211016T191815.SAFE
│   └── S2A_MSIL2A_20210429T170851_N9999_R112_T15TVM_20220501T111454.SAFE
├── sentinel_puerto_rico
│   ├── S2B_MSIL1C_20210502T150719_N0300_R082_T19QFA_20210502T164912.SAFE
│   ├── S2B_MSIL2A_20210502T150719_N9999_R082_T19QFA_20220425T153242.SAFE
│   └── S2B_MSI_20210502T150719_ACOLITE_SUBSET
├── shapes
│   ├── sbd_bahamas_aoi.cpg
│   ├── sbd_bahamas_aoi.dbf
│   ├── sbd_bahamas_aoi.prj
│   ├── sbd_bahamas_aoi.qmd
│   ├── sbd_bahamas_aoi.shp
│   ├── sbd_bahamas_aoi.shx
│   ├── sdb_puerto_rico_aoi.cpg
│   ├── sdb_puerto_rico_aoi.dbf
│   ├── sdb_puerto_rico_aoi.prj
│   ├── sdb_puerto_rico_aoi.shp
│   ├── sdb_puerto_rico_aoi.shx
│   ├── vectorized_outline_mille_lacs.cpg
│   ├── vectorized_outline_mille_lacs.dbf
│   ├── vectorized_outline_mille_lacs.prj
│   ├── vectorized_outline_mille_lacs.shp
│   ├── vectorized_outline_mille_lacs.shx
│   ├── vectorized_outline_mille_lacs_minus_30m_buffer.cpg
│   ├── vectorized_outline_mille_lacs_minus_30m_buffer.dbf
│   ├── vectorized_outline_mille_lacs_minus_30m_buffer.prj
│   ├── vectorized_outline_mille_lacs_minus_30m_buffer.shp
│   └── vectorized_outline_mille_lacs_minus_30m_buffer.shx
└── target_bahamas.tif
```
