# ML_geo_production

**ML_geo_production** is a geospatial ML pipeline that runs one or more
PyTorch semantic-segmentation models over GeoTIFF imagery, ensembles
their outputs, and produces combined, normalized probability arrays for
areas of interest. It processes only the area of the raster that intersect a
defined bounding box, and can optionally compare
predictions with polygon data (e.g GeoPackage) for change detection.

For questions about the repo, email rajoh@kds.dk 

------------------------------------------------------------------------

![change_detection2](https://github.com/user-attachments/assets/d9607467-81fb-4e05-b0ac-246103c8c07a)

------------------------------------------------------------------------
## Features

-   Model ensembling for multiple PyTorch semantic-segmentation models\
-   Processes only GeoTIFF regions that intersect the area of interest\
-   Outputs a single combined & normalized probability array for the
    AOI\
-   Optional comparison against reference polygons for change
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

## Verify that everything works
Run the Quick Start instructions 
run 

python src/ML_geo_production/process_images.py --json config_files/save_probs_preds_and_change_detection.json
there shouold be no errors in output 

------------------------------------------------------------------------
## Model evaluation and summarization

### evaluate_models.py

Runs one or more configs over shapefile-defined areas: builds label rasters from the config’s geopackage, runs inference, and writes per-area classification stats (IoU, pixel accuracy, F1, etc.) plus prediction and difference rasters.

**Arguments:**

-   `--config`: One or more JSON config paths; glob patterns supported (e.g. `path/to/change_detection_5_models_2026_SOTA_*`). Default: `config_files/change_detection.json`.
-   `--shape`: One or more `.shp` or `.gpkg` paths (required).
-   `--image_folder`: Path to folder containing input images (required).
-   `--output_folder`: Path where label, stats, prediction and diff files are written (required).

**Example:**

``` bash
python src/ML_geo_production/evaluate_models.py \
  --config config_files/change_detection.json \
  --shape /path/to/areas.shp \
  --image_folder /path/to/rooftop_rgb \
  --output_folder /path/to/evaluations
```

**Outputs:** For each (config, shape, feature): label `.tif`, stats `.md`, prediction `.tif` (`_pred_im.tif`), and difference `.tif` (`_label_pred_diff_im.tif`; 0=agree, 1=FP, 2=FN, 3=wrong class). The config must include a `geopackage` key for label creation.

### summarize_evaluations.py

Reads evaluation `.md` files from a folder, extracts a chosen metric and inference time, and writes a summary markdown table (score, inference minutes, filename) plus model-index mapping.

**Arguments:**

-   `--folder`: Folder containing evaluation `.md` files (default: `/mnt/T/mnt/ML_output/building_change_detection_2026/evaluations`).
-   `--area`: Substring to filter files, e.g. `parcellhuse` (default: `parcellhuse`).
-   `--output_directory`: Where to write the summary `.md` (default: same as folder).
-   `--statistic`: Metric to extract and sort by (default: `Pixel accuracy`).
-   `--original_config`: JSON with `model_names` for the index mapping (default: `config_files/change_detection_5_models_2026_SOTA.json`).

**Example:**

``` bash
python src/ML_geo_production/summarize_evaluations.py \
  --folder /path/to/evaluations --area parcellhuse \
  --statistic "Pixel accuracy"
```

**Output:** One markdown file per area/statistic (e.g. `parcellhuse-Pixel_accuracy.md`) with a table of score, inference (min), and filename, plus a model index mapping section.

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
