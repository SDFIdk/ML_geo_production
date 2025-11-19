import argparse
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import os
import glob
import pathlib

def polygonize_raster_data(image, transform, crs, buffer_size):
    """Polygonize a raster numpy array and return a GeoDataFrame."""
    mask = image != 0

    if buffer_size:
        print(f"Using buffer of size {buffer_size}")
        polygons = []
        for s, v in shapes(image, mask=mask, transform=transform):
            shrinked_grown = shape(s).buffer(-buffer_size)
            if shrinked_grown.is_valid and (not shrinked_grown.is_empty) and shrinked_grown.area > 0:
                polygons.append({'properties': {'value': v}, 'geometry': shrinked_grown.buffer(buffer_size)})
    else:
        print("No buffer")
        polygons = [
            {'properties': {'value': v}, 'geometry': shape(s)}
            for s, v in shapes(image, mask=mask, transform=transform)
        ]

    # Convert to GeoDataFrame
    if not polygons:
        print("No valid geometries found!")
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame.from_features(polygons)
        gdf = gdf.set_geometry("geometry")
        gdf = gdf[gdf.is_valid]
        valid_count = gdf.is_valid.sum()
        print(f"Number of valid geometries: {valid_count}")
        gdf.crs = crs

    return gdf


def polygonize_raster_file(input_raster, output_shapefile, buffer_size):
    """Open raster file, polygonize, and save shapefile."""
    with rasterio.open(input_raster) as src:
        image = src.read(1)
        gdf = polygonize_raster_data(image, src.transform, src.crs, buffer_size)

    pathlib.Path(output_shapefile).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shapefile)
    print(f"Polygonization complete, output saved to {output_shapefile}.")


def process_files(input_pattern, output_folder, buffer_size):
    input_files = glob.glob(input_pattern)
    if not input_files:
        print(f"No files found for pattern: {input_pattern}")
        return

    for input_file in input_files:
        base_filename = os.path.basename(input_file)
        output_shapefile = os.path.join(output_folder, os.path.splitext(base_filename)[0] + '.shp')
        if pathlib.Path(output_shapefile).exists():
            print(f"Skipping existing file: {output_shapefile}")
        else:
            print(f"Processing {input_file} -> {output_shapefile}")
            try:
                polygonize_raster_file(input_file, output_shapefile, buffer_size)
            except Exception as e:
                print("#" * 10)
                print(f"Failed with exception: {e}")
                print("#" * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polygonize GeoTIFF raster(s) to shapefile(s).")
    parser.add_argument('--input', type=str, help="Path to the input GeoTIFF raster file or pattern.")
    parser.add_argument('--output', type=str, help="Path to the output shapefile (.shp) or output folder if input is a pattern.")
    parser.add_argument('--buffer_size', type=float, default=1.5, help="Buffer size for shrinking/growing geometries.")

    args = parser.parse_args()

    if '*' in args.input or '?' in args.input:
        process_files(args.input, args.output, args.buffer_size)
    else:
        polygonize_raster_file(args.input, args.output, args.buffer_size)
