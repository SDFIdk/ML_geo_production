import argparse
import subprocess
import json
import os
import glob
import shutil
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import laz_to_DSM_no_veg
import multiprocessing as mp

def copy_to_nas_func(source_laz, nas_folder):
    """Copy .laz file to the NAS folder."""
    if not os.path.exists(nas_folder):
        os.makedirs(nas_folder)
    destination = os.path.join(nas_folder, os.path.basename(source_laz))
    print(f"Copying {source_laz} to {destination}")
    shutil.copy2(source_laz, destination)
    return destination

def find_latest_laz_file(laz_folder, tile_id):
    """Find the latest .laz file in the correct subfolder for the given tile_id."""
    folder_name = f"10km_{tile_id.split('_')[1][0:3]}_{tile_id.split('_')[2][0:2]}"
    folder_path = os.path.join(laz_folder, folder_name)

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found")
        return None

    search_pattern = os.path.join(folder_path, f"*{tile_id}*.laz")
    laz_files = glob.glob(search_pattern)

    if not laz_files:
        print(f"No .laz files found for tile {tile_id}")
        return None

    laz_files.sort(key=os.path.getmtime, reverse=True)
    #remove eventuall copc files
    laz_files = [f for f in laz_files if "copc" not in f]
    return laz_files[0]

def process_tile(laz_file, output_folder, tile_id):
    """Process the .laz file to generate the filled TIFF output."""
    output_laz = os.path.join(output_folder, f"{tile_id}_ground.laz")
    output_tiff = os.path.join(output_folder, f"{tile_id}_dsm_no_veg.tif")
    filled_tiff_output = os.path.join(output_folder, f"{tile_id}_dsm_no_veg_filled.tif")
    if Path(filled_tiff_output).is_file():
        print("skipping creation of "+str(filled_tiff_output)+" since it already exists!")
    else:
        laz_to_DSM_no_veg.laz_to_DSM_no_veg(input_laz_file=laz_file,output_DSM_no_veg=filled_tiff_output,tmp_laz=output_laz,tmp_tif=output_tiff,skip_creation_if_DSM_no_veg_exists = True,delete_tmp_files=True)

        #run_pdal_translate(laz_file, output_laz)
        #run_pdal_pipeline(output_laz, output_tiff)
        #run_gdal_fillnodata(output_tiff, filled_tiff_output)

def helper_function(pair):
    (laz_file,dsm_folder) = pair
    tile_id = os.path.splitext(os.path.basename(laz_file))[0]
    print(f"Processing LAZ file from NAS: {laz_file}")
    process_tile(laz_file, dsm_folder, tile_id)
    print(f"Done Processing LAZ file from NAS: {laz_file}")

def process_all_nas_laz_files(nas_folder, dsm_folder,nr_of_processes):
    """Process all .laz files directly from the NAS folder."""
    laz_files = glob.glob(os.path.join(nas_folder, "*.laz"))
    if not laz_files:
        print(f"No .laz files found in the NAS folder: {nas_folder}")
        return
    print("nr of laz files to process :"+str(laz_files))
    # Create a pool of worker processes
    print("creating pool of : " +str(int(nr_of_processes))+ " nr of processes")
    pool = mp.Pool(processes=int(nr_of_processes))

    # Use pool.map to parallelize the loop
    pool.map(helper_function, [(laz_file,dsm_folder) for laz_file in laz_files]) 

    # Close and join the pool to free up resources
    pool.close()
    pool.join()


    #for laz_file in laz_files:
    #    tile_id = os.path.splitext(os.path.basename(laz_file))[0]
    #    print(f"Processing LAZ file from NAS: {laz_file}")
    #    process_tile(laz_file, dsm_folder, tile_id)

def main(laz_folder, dsm_folder, shapefile, last_finnished_tile, nas_folder, copy_to_nas, process_laz,nr_of_processes,tileid):
    """Main function to process all features in the shapefile."""
    if copy_to_nas:
        # Load the shapefile
        gdf = gpd.read_file(shapefile)
        nr_of_tiles = len(gdf)
        found_tile_to_skip = False

        for idx, feature in gdf.iterrows():
            tile_id = feature[tileid]  # Adjust this to match your shapefile's structure
            print(f"Processing tile {tile_id} ({idx+1} out of {nr_of_tiles})...")

            if tile_id == last_finnished_tile:
                found_tile_to_skip = True
                continue
            if last_finnished_tile and not found_tile_to_skip:
                continue

            laz_file = find_latest_laz_file(laz_folder, tile_id)

            if laz_file:
                laz_file_on_nas = copy_to_nas_func(laz_file, nas_folder)
                if process_laz:
                    process_tile(laz_file_on_nas, dsm_folder, tile_id)
            else:
                print(f"Could not find a .laz file for tile {tile_id}")
    else:
        # Process all LAZ files directly from NAS folder
        if process_laz:
            process_all_nas_laz_files(nas_folder, dsm_folder,nr_of_processes)
        else:
            print(f"Skipping processing as --process_laz is not provided.")

if __name__ == "__main__":
    example_usage = 'python create_LIDAR_no_veg.py --laz-folder F:\\GDB\\DHM\\Punktsky --1km2-DSM-no-veg-folder T:\\trainingdata\\bygningsudpegning\\DSM_no_veg --1km2-shapefile "T:\\config_files\\map_creation_bygningsudpegning\\shape_files\\1km2_tiles_verified.shp'
    print("#"*100)
    print("Example_usage:")
    print(example_usage)
    print("#"*100)

    parser = argparse.ArgumentParser(description="Process LiDAR LAZ files using a shapefile for tile IDs. files are located on frive , copied to nas and processed. its posible to skip the different steps by omitting things (no --copy-to-nas means no copying from f drive , no --process-laz means no processing of laz files to dsm )",formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    parser.add_argument("--laz-folder", default = None, help="Path to the folder containing LAZ files on F drive. ")
    parser.add_argument("--1km2-DSM-no-veg-folder", default = "/mnt/T/mnt/trainingdata/bygningsudpegning/DSM_no_veg", help="Path to the folder where output DSM TIFF files will be stored.")
    parser.add_argument("--1km2-shapefile", default = "/mnt/T/mnt/config_files/map_creation_bygningsudpegning/shape_files/1km2_tiles_verified.shp", help="Path to the shapefile with tile ID information.")
    parser.add_argument("--last_finnished_tile", type=str, default=None, help="Process will skip all tiles up to and including this tile.")
    parser.add_argument("--nas-folder", default = "/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/Rooftop_2024_GeoDanmark/laz_files", help="Path to the NAS folder where LAZ files should be copied or read from.")
    parser.add_argument("--copy-to-nas", action="store_true", help="If set, LAZ files will be copied to the NAS folder before processing.")
    parser.add_argument("--process-laz", action="store_true", help="If set, LAZ files will be processed.")
    parser.add_argument("--nr_of_processes",type = int, default = 10, help="how many proceses should be used for the processing of the laz files? best ITC pc can handle 20")
    parser.add_argument("--tileid", default = "KN1kmDK", help="name of the tile id e.g KN1kmDK or tileid")
    args = parser.parse_args()

    main(
        args.laz_folder,
        args.__dict__['1km2_DSM_no_veg_folder'],
        args.__dict__['1km2_shapefile'],
        args.__dict__['last_finnished_tile'],
        args.nas_folder,
        args.copy_to_nas,
        args.process_laz,
        args.nr_of_processes,
        args.tileid
    )
