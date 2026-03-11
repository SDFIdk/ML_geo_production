"""
Evaluate models over areas defined by shapefiles: create label rasters from geopackage,
run inference, and write classification statistics (IoU, accuracy, F1, etc.) per area.

Outputs include *_label_pred_diff_im.tif: uint8 error-type raster per pixel:
  0 = agree (label == pred), 1 = FP (pred positive, label not),
  2 = FN (label positive, pred not), 3 = wrong class (both positive, label != pred).
"""

import argparse
import glob
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import geopandas as gpd
import numpy as np

from ML_geo_production import model_utils
from ML_geo_production.geo_utils import (
    filter_images_by_bounds,
    get_image_bounds,
    normalize_tile_id_name,
)
from ML_geo_production.get_classification_stats import (
    _format_stats_markdown,
    get_classification_stats,
)
from ML_geo_production.processing_utils import save_predictions_data
from multi_channel_dataset_creation import geopackage_to_label_v2

import process_images

# Ignore value used in label rasters (pixels to exclude from metric computation)
LABEL_IGNORE_VALUE = 0

# Error-type encoding for label-vs-prediction diff raster (_label_pred_diff_im.tif):
# 0 = agree (label == pred)
# 1 = FP   (pred positive, label not: pred != 0 and label == 0)
# 2 = FN   (label positive, pred not: label != 0 and pred == 0)
# 3 = wrong class (both positive but different: label != pred, both != 0)
DIFF_AGREE = 0
DIFF_FP = 1
DIFF_FN = 2
DIFF_WRONG_CLASS = 3


def run_evaluation(
    config_paths,
    shape_paths,
    image_folder: str,
    output_folder: str,
):
    """
    For each (config, shapefile) pair and each feature in the shapefile, run the
    inference pipeline, create label rasters from the config's geopackage, and
    write classification stats and label GeoTIFFs to output_folder.

    Parameters
    ----------
    config_paths : list of str or Path
        Paths to JSON config files (must contain saved_models, geopackage, etc.).
    shape_paths : list of str or Path
        Paths to shapefiles (.shp or .gpkg) defining areas (one or more features each).
    image_folder : str
        Directory containing input images; overrides config path_to_images for filtering.
    output_folder : str
        Directory where label .tif and stats .md files are written.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_folder_path = Path(image_folder)
    if not image_folder_path.is_dir():
        raise FileNotFoundError(
            f"Image folder not found or not a directory: {image_folder}\n"
            "Ensure the path exists inside the container (e.g. mount the volume)."
        )

    image_folder_str = str(image_folder_path.resolve())
    print("Computing image bounds once for all images in image_folder...")
    bounds_dict = get_image_bounds(image_folder_str)
    print(f"Loaded bounds for {len(bounds_dict)} images.")

    geopackage_cache = {}

    for config_path in config_paths:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            base_config = json.load(f)

        print(f"Preloading model weights for config {config_path.name}...")
        base_config["model_states"] = model_utils.preload_model_states(
            base_config["saved_models"]
        )
        gp_path = base_config.get("geopackage")
        if gp_path is None:
            raise ValueError(f"Config {config_path} has no 'geopackage' key.")
        gp_path_str = str(Path(gp_path).resolve()) if isinstance(gp_path, str) else gp_path
        if gp_path_str in geopackage_cache:
            base_config["geopackage"] = geopackage_cache[gp_path_str]
            print("Using cached geopackage (same path as previous config).")
        else:
            print("Loading geopackage...")
            base_config["geopackage"] = gpd.read_file(gp_path)
            geopackage_cache[gp_path_str] = base_config["geopackage"]
        config_stem = config_path.stem

        for shape_path in shape_paths:
            shape_path = Path(shape_path)
            if not shape_path.exists():
                print(f"Shapefile not found: {shape_path}, skipping.")
                continue

            gdf = gpd.read_file(shape_path)
            shape_stem = shape_path.stem
            print(f"Shapefile {shape_path.name}: {len(gdf)} features.")

            for idx, row in gdf.iterrows():
                shape_id = normalize_tile_id_name(
                    str(row.get("id", row.get("tile_id", f"{idx}")))
                )
                print(
                    f"\n--- Config {config_stem} / Shape {shape_stem} / ID {shape_id} (index {idx}) ---"
                )

                parsed_json = deepcopy(base_config)
                initial_bounds = row.geometry.bounds
                parsed_json["bounds"] = initial_bounds
                image_folder_resolved = Path(image_folder).resolve()
                parsed_json["path_to_images"] = str(image_folder_resolved)
                # So the dataloader loads from image_folder (patch_dataset uses parent.parent / data_types / filename)
                n_data_types = max(1, len(parsed_json.get("data_types", [])))
                parsed_json["data_types"] = [[image_folder_resolved.name]] * n_data_types
                # Use smaller batch size to reduce GPU memory (e.g. when GPU is shared)
                parsed_json["batch_size"] = min(parsed_json.get("batch_size", 8), 4)

                image_paths, _ = filter_images_by_bounds(
                    path_to_images=parsed_json["path_to_images"],
                    bounds=parsed_json["bounds"],
                    bounds_dict=bounds_dict,
                )

                if not image_paths:
                    print(
                        f"No overlapping images found for shape ID {shape_id}. Skipping."
                    )
                    continue

                print(f"Found {len(image_paths)} overlapping images.")
                parsed_json["image_paths"] = image_paths

                t0 = time.perf_counter()
                final_probability_array, final_transform, dst_crs = (
                    process_images.process_images_from_dict(parsed_json)
                )
                inference_seconds = time.perf_counter() - t0

                unknown_border_size = parsed_json.get("unknown_boarder_size", 1.5)
                label_array = geopackage_to_label_v2.process_single_raster_labels(
                    gdf=parsed_json["geopackage"],
                    bounds=parsed_json["bounds"],
                    output_shape=final_probability_array.shape[-2:],
                    out_transform=final_transform,
                    unknown_border_size=unknown_border_size,
                    background_value=1,
                    ignore_value=LABEL_IGNORE_VALUE,
                    value_used_for_all_polygons=2,
                )

                label_path = output_path / f"{shape_stem}_{shape_id}.tif"
                if not label_path.exists():
                    save_predictions_data(
                        label_array.astype(np.uint8),
                        final_transform,
                        dst_crs,
                        str(label_path),
                    )
                    print(f"Saved label raster: {label_path}")
                else:
                    print(f"Reusing existing label raster: {label_path}")

                pred_im = final_probability_array.argmax(axis=0).astype(np.uint8)
                pred_path = output_path / f"{config_stem}_{shape_stem}_{shape_id}_pred_im.tif"
                save_predictions_data(
                    pred_im,
                    final_transform,
                    dst_crs,
                    str(pred_path),
                )
                print(f"Saved prediction raster: {pred_path}")

                # Error-type diff: 0=agree, 1=FP, 2=FN, 3=wrong class (see DIFF_* constants)
                diff_im = np.zeros_like(label_array, dtype=np.uint8)
                diff_im[label_array == pred_im] = DIFF_AGREE
                diff_im[(pred_im != 0) & (label_array == 0)] = DIFF_FP
                diff_im[(label_array != 0) & (pred_im == 0)] = DIFF_FN
                diff_im[
                    (label_array != 0) & (pred_im != 0) & (label_array != pred_im)
                ] = DIFF_WRONG_CLASS
                diff_path = output_path / (
                    f"{config_stem}_{shape_stem}_{shape_id}_label_pred_diff_im.tif"
                )
                save_predictions_data(
                    diff_im,
                    final_transform,
                    dst_crs,
                    str(diff_path),
                )
                print(
                    f"Saved label-pred diff (0=agree 1=FP 2=FN 3=wrong class): {diff_path}"
                )

                mask = label_array != LABEL_IGNORE_VALUE
                if mask.any():
                    label_flat = label_array[mask]
                    pred_flat = pred_im[mask]
                    IoU, Pixel_Accuracy, f1_score, precision, recall, confusion_matrix = (
                        get_classification_stats(label_flat, pred_flat)
                    )
                else:
                    IoU = Pixel_Accuracy = f1_score = precision = recall = 0.0
                    confusion_matrix = np.zeros((0, 0), dtype=np.int64)

                md = _format_stats_markdown(
                    IoU, Pixel_Accuracy, f1_score, precision, recall, confusion_matrix
                )
                md += "\n## Inference time\n\n"
                md += f"Ensemble (process_images): {inference_seconds:.2f} s ({inference_seconds / 60:.2f} min)\n"
                stats_path = output_path / f"{config_stem}_{shape_stem}_{shape_id}.md"
                with open(stats_path, "w") as f:
                    f.write(md)
                print(f"Saved classification stats: {stats_path}")


if __name__ == "__main__":
    print("#"*30)
    print("Example usage : ")
    print("python /home/projects/ML_geo_production/src/ML_geo_production/evaluate_models.py --config /mnt/T/mnt/config_files/bygnings_udpegning/evaluate_different_models/change_detection_5_models_2026_SOTA_subset_* --shape /mnt/T/mnt/trainingdata/bygningsudpegning/cph_validation_tiles/city.shp /mnt/T/mnt/trainingdata/bygningsudpegning/cph_validation_tiles/industri.shp /mnt/T/mnt/trainingdata/bygningsudpegning/cph_validation_tiles/parcellhuse.shp --image_folder /mnt/T/mnt/ML_input/building_change_detection_2026/rooftop_2025/10cmresampledto16cm/rooftop_rgb --output_folder /mnt/T/mnt/ML_output/building_change_detection_2026/evaluations")
    print("#"*30)
    parser = argparse.ArgumentParser(
        description="Evaluate models over areas: create label rasters and classification stats per (config, shape, feature). "
    )
    parser.add_argument(
        "--config",
        nargs="+",
        type=str,
        default=["config_files/change_detection.json"],
        help="One or more paths to JSON config files. Glob patterns supported (e.g. path/to/change_detection_5_models_2026_SOTA_* expands to all matching JSONs).",
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=str,
        required=True,
        help="One or more paths to shapefiles (.shp or .gpkg) defining areas.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing input images (overrides config path_to_images).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to folder where label .tif and stats .md files are written.",
    )
    args = parser.parse_args()

    config_paths = []
    for c in args.config:
        if "*" in c or "?" in c or "[" in c:
            config_paths.extend(glob.glob(c))
        else:
            config_paths.append(c)
    config_paths = sorted(config_paths)
    if not config_paths:
        print("No config files found (pattern matched zero files).", file=sys.stderr)
        sys.exit(1)

    run_evaluation(
        config_paths=config_paths,
        shape_paths=args.shape,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
    )
