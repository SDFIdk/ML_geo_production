import geopandas as gpd
from pathlib import Path
import argparse

def split_geopackage_by_tile(input_file, output_folder,layer_name,debug=False):
    # Convert input paths to Path objects
    input_path = Path(input_file)
    output_path = Path(output_folder)
    
    # Ensure the output folder exists (create if it doesn't)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Reading the GeoPackage file")
    if debug:
        print("in debug mode we onlyn read a small fraction of the data form the geopackage file")
        gdf = gpd.read_file(input_path,layer=layer_name).head(100)
        print(gdf)
    else:
        gdf = gpd.read_file(input_path,layer=layer_name)

    print("finnished reading geopackage file")
    
    print(" Filter the GeoDataFrame to include only rows where validup is True")
    filtered_gdf = gdf[(gdf['validup'] == True) | (gdf['validup'].isnull())]
    print("finished filtering away invalid change-marker polygons")

    if debug:
        print("checking out null values")
        null_gdf = gdf[(gdf['validup'].isnull())]

    
    # Get unique values from the 'tile' column
    unique_tiles = filtered_gdf['tile'].unique()
    print("nr of unique_tiles:"+str(len(unique_tiles)))
    
    # Loop through each unique tile and create a shapefile
    for tile in unique_tiles:
        # Filter the GeoDataFrame for the current tile
        filtered_tile_gdf = filtered_gdf[filtered_gdf['tile'] == tile]
        
        # Define the output path for this shapefile
        output_file = output_path / f"{tile}verified_for_data_cleaning.shp"
        
        # Save the filtered data as a shapefile
        filtered_tile_gdf.to_file(output_file, driver="ESRI Shapefile")
        
        print(f"Saved shapefile for tile '{tile}' to {output_file}")

        if debug:
            #in DEBUG also create unfiltered .shp  files and .shp files only showing polygons with null values
            # Filter the GeoDataFrame for the current tile
            unfiltered_tile_gdf = gdf[gdf['tile'] == tile]
            # Define the output path for this shapefile
            output_file = output_path / f"{tile}_unfiltered.shp"
            # Save the filtered data as a shapefile
            unfiltered_tile_gdf.to_file(output_file, driver="ESRI Shapefile")
            print(f"Saved unfiltered shapefile for tile '{tile}' to {output_file}")


            # Filter the GeoDataFrame for the current tile
            null_tile_gdf = null_gdf[null_gdf['tile'] == tile]
            # Define the output path for this shapefile
            output_file = output_path / f"{tile}_null.shp"
            # Save the filtered data as a shapefile
            null_tile_gdf.to_file(output_file, driver="ESRI Shapefile")
            print(f"Saved unfiltered shapefile for tile '{tile}' to {output_file}")

# Set up argparse for command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split GeoPackage by unique tile column and filter by validup=True.")
    parser.add_argument("--input_file", type=str, default = "/mnt/T/mnt/trainingdata/bygningsudpegning/datacleaning/plusminuspoly.gpkg" ,help="Path to the input GeoPackage file")
    parser.add_argument("--output_folder", type=str, default = "/mnt/T/mnt/trainingdata/bygningsudpegning/datacleaning/created_shapefiles/",help="Path to the output folder to save shapefiles")
    parser.add_argument("--debug", action="store_true", help="If set, only reading the first 100 rows from the geopackagefile")
    parser.add_argument("--layer_name", type=str, default = "plusminuspoly",help="the layer in the geopackage file to read ")
    args = parser.parse_args()

    # Call the function with arguments from argparse
    split_geopackage_by_tile(args.input_file, args.output_folder,args.layer_name,args.debug)
