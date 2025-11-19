import numpy as np
import os
from typing import Tuple
import rasterio
from shapely.geometry import shape
import geopandas as gpd
from pathlib import Path
import tempfile
import sys

from multi_channel_dataset_creation import geopackage_to_label_v2
from ML_geo_production import create_diff_im, copy_attributes, polygonize, get_pred_difference


def apply_thresholding(probs_array, diff_array, threshold=0.004):
    """
    Apply thresholding on a difference array based on values from a probability array.
    Modifies diff_array in place.
    """
    if probs_array.ndim != 3:
        sys.exit("Expected probs_array with 3 dimensions (C,H,W)")
    prob_channel = probs_array[2]
    mask = (diff_array == 1) & (prob_channel >= threshold)
    diff_array[mask] = 0
    return diff_array


def create_diff_polygons(probs, geopackage_data, transform, crs,bounds,
                         path_to_mapping="/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/roof_no_roof_mapping.txt",
                         create_new_mapping=False,
                         unknown_boarder_size=1.5,
                         extra_atributes=None):
    """
    Create polygons representing differences between predicted and ground truth labels.
    
    Parameters
    ----------
    probs : np.ndarray
        Probability array (C, H, W)
    geopackage_data : data loaded from geopackage
        Already loaded vector data
    transform : Affine
        Raster transform
    crs : CRS
        Coordinate reference system
    """

    if extra_atributes is None:
        extra_atributes = {"label_description": "label_description",
                           "model_description": "model_description"}

    pred_array= np.argmax(probs, axis=0).astype(np.uint8)




    # Create label image from geopackage
    label_array = geopackage_to_label_v2.process_single_raster_labels(
        gdf =geopackage_data,
        bounds= bounds, # (left, bottom, right, top)
        output_shape= probs.shape[-2:], # (height, width)
        out_transform= transform,
        unknown_border_size= unknown_boarder_size,
        background_value= 1,
        ignore_value= 0,
        value_used_for_all_polygons =2
    )


    # Create difference image
    diff_array = create_diff_im.create_diff_image(label_array, pred_array)

    # Compare predictions and labels
    print("% pixels differing from label:")
    print(get_pred_difference.compare_images(pred_array,label_array , ignore_index=0))

    # apply thresholding
    
    diff_array = apply_thresholding(probs, diff_array)

    # Polygonize
    diff_gdf = polygonize.polygonize_raster_data(diff_array, transform, crs, buffer_size=unknown_boarder_size)

    # Add attributes
    diff_gdf = copy_attributes.add_attributes(geopackage_data, diff_gdf, extra_atributes=extra_atributes)

    return diff_gdf


def process_single_raster_labels(
    gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float], # (left, bottom, right, top)
    output_shape: Tuple[int, int], # (height, width)
    out_transform: rasterio.transform.Affine,
    mean_res: float,
    attr_column: str,
    unknown_border_size: float,
    background_value: int,
    ignore_value: int
    ):



    label_array= geopackage_to_label_v2.process_single_raster_labels(
        gdf=geopackage_data,
        src=src,
        attr_column=extra_atributes,
        unknown_border_size=unknown_boarder_size,
        background_value= 1,
        ignore_value= 0
        )
    print("verify that values are 0 1 and 2 ")
    input(label_array.flatten().max()) 


    # Create difference array
    diff_array = create_diff_im.create_diff_image(label_array, pred_array)

    # Compare predictions and labels
    print("% pixels differing from label:")
    print(get_pred_difference.compare_images(pred_array, label_array, ignore_index=0))

    # apply thresholding
    diff_array = apply_thresholding(probs, diff_array)

    # Polygonize
    diff_gdf = polygonize.polygonize_raster_data(diff_array, transform, crs, buffer_size=unknown_boarder_size)

    # Add attributes
    diff_gdf = copy_attributes.add_attributes(loaded_local_geopackagedata, diff_gdf, extra_atributes=extra_atributes)

    return diff_gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create difference polygons from probability geotiff and geopackage")
    parser.add_argument("--probs_path", required=True, help="Path to probability GeoTIFF")
    parser.add_argument("--geopackage", required=True, help="Path to input geopackage")
    parser.add_argument("--polygon_output_folder", required=True, help="Path to output folder for polygons")
    parser.add_argument("--unknown_boarder_size", type=float, default=1.5)
    parser.add_argument("--create_new_mapping", action="store_true", help="Whether to create a new mapping")
    parser.add_argument("--path_to_mapping", type=str, default="/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/roof_no_roof_mapping.txt")

    args = parser.parse_args()

    os.makedirs(args.polygon_output_folder, exist_ok=True)

    # --- Disk access only happens here ---
    # Load raster
    with rasterio.open(args.probs_path) as src:
        probs = src.read()
        transform = src.transform
        crs = src.crs

    # Load geopackage
    geopackage_data = gpd.read_file(args.geopackage)

    # Create difference polygons in memory
    diff_gdf = create_diff_polygons(
        probs=probs,
        geopackage_data=geopackage_data,
        transform=transform,
        crs=crs,
        path_to_mapping=args.path_to_mapping,
        create_new_mapping=args.create_new_mapping,
        unknown_boarder_size=args.unknown_boarder_size,
    )

    # Save polygons to shapefile
    shapefile_path = os.path.join(args.polygon_output_folder, f"{Path(args.probs_path).stem}.shp")
    diff_gdf.to_file(shapefile_path)
    print(f"Saved polygons to {shapefile_path}")
