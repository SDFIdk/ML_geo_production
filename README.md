# ML_geo_production

**ML_geo_production** is a geospatial ML pipeline that runs one or more
PyTorch semantic-segmentation models over GeoTIFF imagery, ensembles
their outputs, and produces combined, normalized probability arrays for
areas of interest. It processes only the area of the raster that intersect a
defined bounding box, and can optionally compare
predictions with polygon data (GeoPackage) for change detection.

For questions about the repo, email rajoh@kds.dk 

------------------------------------------------------------------------
![change_detection_pipeline](https://github.com/user-attachments/assets/f3fc07ab-0b9c-422d-b236-9dcb5c798fc2)
------------------------------------------------------------------------
## Features

-   Model ensembling for multiple PyTorch semantic-segmentation models\
-   Processes only GeoTIFF regions that intersect the area of interest\
-   Outputs a single combined & normalized probability array for the
    AOI\
-   Optional comparison against a GeoPackage of polygons for change
    detection\
-   Example configs work with [https://github.com/SDFIdk/multi_channel_dataset_creation]dataset and pretrained models

------------------------------------------------------------------------

## Quick Start (step-by-step)

### **1. Install this repository**

``` bash
mamba env create -f environment.yml
mamba activate ML_geo_production
pip install -e .
```

------------------------------------------------------------------------

### **2. Clone the example dataset repo side-by-side**

Clone the dataset repository so it sits next to this repository in the
same parent directory:

``` bash
git clone https://github.com/SDFIdk/multi_channel_dataset_creation
```

The example config files in `config_files/` work **out of the box** with
the dataset included in that repository.

------------------------------------------------------------------------

### **3. Download example models**

``` bash
python src/ML_geo_production/download_example_models.py
```

**Note:**\
The example models were trained using the training code from:\
 https://github.com/SDFIdk/ML_model_training

------------------------------------------------------------------------

## Basic Usage

Main processing script:

``` bash
python src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json
```
Example above use an ensamble of three models, save both probs, preds and change detection.


Example workflows:

``` bash
python src/ML_geo_production/do_change_detection.py --json config_files/change_detection.json
python src/ML_geo_production/create_prediction_raster.py --json config_files/raster_production.json
python src/ML_geo_production/process_many_areas.py --json config_files/process_many_areas.json  --shapefile ../multi_channel_dataset_creation/example_dataset/shape_files/many_areas.shp
```
The process_many_areas.py example above shows an example of how to process many areas after each other.

------------------------------------------------------------------------

## Config Files

See examples in `config_files/` --- these are ready to run using the
dataset from\
https://github.com/SDFIdk/multi_channel_dataset_creation

------------------------------------------------------------------------

## Inputs & Requirements

-   GeoTIFFs must share a compatible CRS\
-   Models must output probability tensors\
-   Optional polygon layers must match the CRS (or will be reprojected
    automatically)

------------------------------------------------------------------------

## Outputs

-   Combined, normalized probability arrays\
-   Prediction raster outputs\
-   Change-detection results when polygon comparison is enabled

------------------------------------------------------------------------

## License (MIT)

    MIT License

    Copyright (c) 2025

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
