import argparse
import os
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape
import geopandas as gpd
import numpy as np

def sort_by_substring_first(strings, substring):
    # Sort with a key function that prioritizes strings containing the substring
    return sorted(strings, key=lambda x: (substring not in x, x))

def burn_in_unknown(folder_with_shape_files, folder_with_labels, output_folder, unknown_value=0):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each GeoTIFF in the folder_with_labels
    print(folder_with_labels)
    print(str(os.listdir(folder_with_labels)))
    shapefiles= [file for file in os.listdir(folder_with_shape_files) if ".shp" in file]
    for label_file in os.listdir(folder_with_labels):
        if label_file.endswith('.tif'):
            # Construct paths to the label GeoTIFF and corresponding shapefile
            label_path = os.path.join(folder_with_labels, label_file)
            #the automatically generated diff.shp files can be used to mask out bad labels but also masks out areas not coreclty classified by the model.
            #we should therfore prioritize using .shp files that have been cleaned up by the students preds_tile_remappad.shp--> geopackage --> tilename_verified_for_data_cleaning.shp
            all_shapefiles = [file for file in shapefiles if str(label_file).rstrip(".tif") in file]
            if len(all_shapefiles)==0:
                print("no .shp file for this label so we skip it")
                continue
            #if there exist a .shp file that contains teh corect tilename and also contains 'verified_for_data_cleaning' this .shp file will be used
            all_shapefiles = sort_by_substring_first(all_shapefiles,"verified_for_data_cleaning")

            shape_file_name = all_shapefiles[0]

            
            shape_path = os.path.join(folder_with_shape_files, shape_file_name)

            # Only proceed if the corresponding shapefile exists
            if os.path.exists(shape_path):
                # Open the shapefile and the GeoTIFF
                with rasterio.open(label_path) as src:
                    # Read image data
                    image_data = src.read(1)
                    out_meta = src.meta

                    # Load shapefile polygons
                    polygons = gpd.read_file(shape_path)
                    shapes = [shape(geom) for geom in polygons.geometry]

                    # Create a mask for the regions within the polygons
                    mask = geometry_mask(shapes, transform=src.transform, invert=True, out_shape=src.shape)

                    # Burn in the unknown_value for pixels within the mask
                    image_data[mask] = unknown_value

                    # Prepare output file path
                    output_file_path = os.path.join(output_folder, label_file)

                    # Save the modified GeoTIFF
                    out_meta.update(dtype=rasterio.uint8)  # Update metadata as necessary
                    with rasterio.open(output_file_path, 'w', **out_meta) as dst:
                        dst.write(image_data, 1)

                print(f"Processed and saved: {output_file_path}")
            else:
                print(f"Shapefile not found for {label_file}. Skipping.")

# Set up the command-line interface with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Burn a specified value into GeoTIFF pixels within polygons from shapefiles.")
    parser.add_argument("--folder_with_shape_files", default = "/mnt/T/mnt/trainingdata/bygningsudpegning/datacleaning/created_shapefiles/",type=str, help="Folder containing shapefiles")
    parser.add_argument("--folder_with_labels", default = "/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/labels/large_labels/",type=str, help="Folder containing GeoTIFF label files")
    parser.add_argument("--output_folder", default = "/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/labels/cleaned_large_labels/", type=str, help="Folder to save the modified GeoTIFF files")
    parser.add_argument("--unknown_value", type=int, default=0, help="Value to burn into pixels within polygons (default: 0)")

    args = parser.parse_args()

    burn_in_unknown(args.folder_with_shape_files, args.folder_with_labels, args.output_folder, args.unknown_value)

