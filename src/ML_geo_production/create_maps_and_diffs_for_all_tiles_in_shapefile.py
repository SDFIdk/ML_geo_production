import argparse
from processing_utils import save_output_data
import torch
import json
import os
import shutil
from pathlib import Path
import configparser
import fiona
from multi_channel_dataset_creation import create_txt_file_with_images_that_overlap_with_shapefile
from multi_channel_dataset_creation import copy_files_listed_in_txt_file
from multi_channel_dataset_creation import crop_to_shapefile
from geo_utils import get_image_bounds, open_shapefile, create_shapefile 
import geopandas as gpd
from shapely.geometry import box
import time
import process_images
import gc
import create_diff_polygons
from pathlib import Path, PureWindowsPath




#create a map and diff iamge for each tile in the shapefile
#data is asumed to be in: path_to_images
#croped data is saved in : --path_to_map_creation_output_folder
def main( shapefile_path,path_to_images,data_folders, channels,path_to_map_creation_output_folder,datatypes,skip,last_finnished_tile,start_with_tile,stop_after_tile,saved_models,model_names,means,stds,remove_tmp_files,batch_size,workers,args):
    main_start_time = time.time()
    # Open the shapefile using fiona
    with fiona.open(shapefile_path, 'r') as shapefile:
        # Access the CRS (Coordinate Reference System)
        crs = shapefile.crs
    print("cashing all image bounds")
    bounds_dict = get_image_bounds(path_to_images)
    print("cashing all image bounds DONE!"+ " took : "+str((time.time()-main_start_time)/60) + "minutes")

    # Open the verified shapefile
    tiles = open_shapefile(shapefile_path)
    #for each tile we need a folder and a crate.ini, infer.ini and merge.ini files triplet
    output_folders =[]
    ini_files = []

    # Loop through each 1km2 tile in the shapefile
    # create ini files for each 1km2 tile and put the folder path in a list
    found_tile_to_skip = False
    found_start_with_tile = False
    tile_ids = [tileid for (tileid,bounding_box) in tiles]
    middle =  tile_ids[int(len(tile_ids)/2)]
    print("nr of tiles: "+str(len(tile_ids)))
    print("middle tile :"+str(middle))
    tile_nr = 0
    if not "geopackage" in skip:
        print("reading geopackage..")
        reading_geopkg_start = time.time()
        layer = None # sometimes we only want to load a specific layer
        if layer:
            loaded_geopackage = gpd.read_file(args.geopackage,layer=layer)
        else:
            loaded_geopackage = gpd.read_file(args.geopackage)
        print("reading geopackage took: "+str(   (time.time()-reading_geopkg_start)/60 )+ " minutes")
    totall_nr_of_tiles= len(tiles)

    for (tileid,bounding_box) in tiles:
        tile_nr +=1
        print("tile nr :"+str(tile_nr))
        print("tiles left :"+str(totall_nr_of_tiles-tile_nr))
        if True: #try:
            tile_start_time = time.time()
            print("tileid:"+str(tileid))
            if tileid == start_with_tile:
                found_start_with_tile = True
                print("found start_with_tile")
            if tileid == last_finnished_tile:
                found_tile_to_skip = True
                print("found last tile to skip")
                continue
            if last_finnished_tile and not found_tile_to_skip:
                print("skipping this tile")
                continue
                print("tileid:"+str(tileid))
            if start_with_tile and not found_start_with_tile:
                print("skipping this tile")
                continue
                print("tileid:"+str(tileid))


            new_1km2_dataset = Path(path_to_map_creation_output_folder)/tileid
            new_1km2_dataset.mkdir(parents=True, exist_ok=True)
            # Create a new shapefile for the new 1km2 tile
            new_shape_file_path= Path(shapefile_path).parent/(tileid+".shp")
            create_shapefile(shapefile_path= new_shape_file_path,bounding_box=bounding_box,crs=crs)

            probs_path = str(new_1km2_dataset/(tileid +"_probs.tif"))
            final_probability_array, final_transform, dst_crs =process_images.process_images_in_folder(path_to_images=path_to_images ,data_folders=data_folders, channels=channels, shape_file=new_shape_file_path,  resolution=0.1,path_to_labels=None, remove_matching_label=False, patch_size=args.patch_size, overlap=args.overlap, batch_size=batch_size, num_workers=workers, saved_models=saved_models,model_names=model_names, n_classes=3, pixel_buffer=4,means=means,stds=stds,bounds_dict=bounds_dict)

            # Save the array to disk using the returned metadata (required for create_diff_polygons)
            save_output_data(final_probability_array, final_transform, dst_crs, probs_path)
            print(f"Saved output GeoTIFF to {probs_path}")

            if not "geopackage" in skip:
                #compare the predictions with the label from the geopackage, create polygons describing differences between the label and the predicitons
                diff_polygons_df = create_diff_polygons.create_diff_polygons(probs_path=probs_path,geopackage= loaded_geopackage,polygon_output_folder=new_1km2_dataset/"polygons",remove_tmp_files=remove_tmp_files,extra_atributes={"label_description":Path(args.geopackage).stem,"model_description": "_".join([Path(model_path).stem for model_path in saved_models])})
                # save polygons to disk
                diff_polygons_df.to_file(((new_1km2_dataset/"polygons")/tileid).with_suffix('.shp'))

            print("time spent on tile : "+str(tileid)+ " : "+str((time.time()-tile_start_time)/60) + " minutes")
            print("#"*40)
            if tileid == stop_after_tile:
                found_tile_to_skip = True
                print("finished stop_after_tile tile . now stopping")
                break
        else: #except Exception as e:
            print(f"An error occurred: {e}")
            with open("failures.txt", 'a') as file:
                # Write the text to the file
                try:
                    file.write(str(e) + '\n')  # Add a newline after the text for clarity
                except:
                    print("failed to log the error to file")
        gc.collect()
        torch.cuda.empty_cache()



def parse_list_of_lists(value):
    """
    Custom type function to parse a string into a list of lists of integers.

    Args:
        value (str): Input string representing a list of lists

    Returns:
        list: Parsed list of lists of integers
    """
    try:
        # Use json.loads to safely parse the string
        parsed = json.loads(value)

        # Validate that it's a list of lists of integers
        if not isinstance(parsed, list) or \
           not all(isinstance(sublist, list) and
                   all(isinstance(item, int) for item in sublist)
                   for sublist in parsed):
            raise argparse.ArgumentTypeError("Input must be a list of lists of integers")

        return parsed

    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON format")


# Argument parser to accept command-line arguments
if __name__ == "__main__":
    example_usage_2 = "python create_maps_and_diffs_for_all_tiles_in_shapefile.py --path_to_images /mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking_demo_komune_areas_20250205/data/rooftop_rgb/ --shapefile /mnt/T/mnt/trainingdata/bygningsudpegning/demo_komunes_areas_1_shapes/1km_6203_705.shp  --saved_model '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_50/models/andringsudpegning_1km2benchmark_iter_50_24.pth\"]' --channels '[[[0,1,2]]]' --data_types '[[\"rooftop_rgb\"]]' --geopackage /mnt/T/mnt/trainingdata/bygningsudpegning/bygninger20250130.gpkg --means '[[0.485, 0.456, 0.406]]' --stds '[[0.229, 0.224, 0.225]]' --path_to_map_creation_output_folder /mnt/T/mnt/random_files/output_iter_50_updated_pixel_buffer_version3 --batch_size 8"
    example_usage_3= "python create_maps_and_diffs_for_all_tiles_in_shapefile.py --path_to_images /mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking_demo_komune_areas_20250205/data/rooftop_rgb/ "\
                    "--shapefile /mnt/T/mnt/trainingdata/bygningsudpegning/demo_komunes_areas_1_shapes/demo_komunes_area_1_shape.shp "\
                    "--saved_model '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_53/models/andringsudpegning_1km2benchmark_iter_53_24.pth\",\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_52/models/andringsudpegning_1km2benchmark_iter_52_24.pth\"]'" \
                    " --channels '[[[0,1,2],[0]],[[0,1,2]]]' --data_types '[[\"rooftop_rgb\",\"rooftop_cir\"],[\"rooftop_rgb\"]]' --geopackage /mnt/T/mnt/trainingdata/bygningsudpegning/bygninger20250130.gpkg " \
                    " --means '[[0.485, 0.456, 0.406,0.40779021],[0.485, 0.456, 0.406]]' --stds '[[0.229, 0.224, 0.225,0.15176421],[0.229, 0.224, 0.225]]' --path_to_map_creation_output_folder /mnt/T/mnt/random_files/output_iter_52_53_fixed_patchextraction --remove_tmp_files --skip geopackage" 
    ensamble_example = "python create_maps_and_diffs_for_all_tiles_in_shapefile.py --shapefile /mnt/T/mnt/ML_input/building_change_detection_2025/20250610.shp --path_to_images /mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_200528/16_cm_20250610/rooftop_rgb/ --path_to_map_creation_output_folder  /mnt/T/mnt/ML_output/building_change_detection_2025/rooftop_200528_20250610_ensamble_version --saved_models '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_52/models/andringsudpegning_1km2benchmark_iter_52_24.pth\",\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_63/models/andringsudpegning_1km2benchmark_iter_63_24.pth\", \"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_62/models/andringsudpegning_1km2benchmark_iter_62_24.pth\"]' --means '[[0.485, 0.456, 0.406],[0.485, 0.456, 0.406],[0.485, 0.456, 0.406]]' --stds '[[0.229, 0.224, 0.225],[0.229, 0.224, 0.225],[0.229, 0.224, 0.225]]' --channels '[[[0,1,2]],[[0,1,2]],[[0,1,2]]]' --data_types '[[\"rooftop_rgb\"],[\"rooftop_rgb\"],[\"rooftop_rgb\"]]'  --model_names '[\"tf_efficientnetv2_l.in21k\",\"efficientnetv2_rw_m.agc_in1k\",\"resnet50\"]' --remove_tmp_files"

    exammple_usage_tested_and_working = "python create_maps_and_diffs_for_all_tiles_in_shapefile.py --shapefile /mnt/T/mnt/ML_input/building_change_detection_2025/hilleroed_tiles_05052025.shp --path_to_images /mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_200528/16cm_and_resampled_to_16cm/rooftop_rgb/ --path_to_map_creation_output_folder  /mnt/T/mnt/ML_output/building_change_detection_2025/rooftop_200528_part2_ensamble --saved_models '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_52/models/andringsudpegning_1km2benchmark_iter_52_24.pth\",\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_63/models/andringsudpegning_1km2benchmark_iter_63_24.pth\", \"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_62/models/andringsudpegning_1km2benchmark_iter_62_24.pth\"]' --means '[[0.485, 0.456, 0.406],[0.485, 0.456, 0.406],[0.485, 0.456, 0.406]]' --stds '[[0.229, 0.224, 0.225],[0.229, 0.224, 0.225],[0.229, 0.224, 0.225]]' --channels '[[[0,1,2]],[[0,1,2]],[[0,1,2]]]' --data_types '[[\"rooftop_rgb\"],[\"rooftop_rgb\"],[\"rooftop_rgb\"]]'  --model_names '[\"tf_efficientnetv2_l.in21k\",\"efficientnetv2_rw_m.agc_in1k\",\"resnet50\"]' --start_with_tile 1km_6203_696 --remove_tmp_files --batch_size 1 --workers 2"
    example_usage = "python create_maps_and_diffs_for_all_tiles_in_shapefile.py --shapefile /mnt/T/mnt/ML_input/building_change_detection_2025/hilleroed_tiles_05052025.shp --path_to_images /mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_200528/16cm_and_resampled_to_16cm/rooftop_rgb/ --path_to_map_creation_output_folder  /mnt/T/mnt/ML_output/building_change_detection_2025/rooftop_200528 --saved_models '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_52/models/andringsudpegning_1km2benchmark_iter_52_24.pth\"]' --means '[[0.485, 0.456, 0.406]]' --stds '[[0.229, 0.224, 0.225]]' --channels '[[[0,1,2]]]' --data_types '[[\"rooftop_rgb\"]]' --remove_tmp_files"
    print("example_usage")
    print(example_usage)
    parser = argparse.ArgumentParser(description="Process GeoTIFF and shapefile data based on ini templates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # This will show default values in the help message
    )
    # Adding required arguments
    parser.add_argument("--shapefile", type=str, default ="/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/validation_tile.shp", help = "Path to the shapefile: e.g /mnt/T/mnt/config_files/map_creation_bygningsudpegning/shape_files/1km2_tiles_verified.shp")
    #parser.add_argument("--path_to_images", type=str, default= "/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/Rooftop_2024_GeoDanmark/",help="Path to dataset that the uncroped images should be copied from")
    parser.add_argument("--path_to_images", type=str, default = "/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/validation_tile/rooftop_rgb",help="Path to dataset that the uncroped images should be copied from")
    parser.add_argument("--path_to_map_creation_output_folder", type=str, default= "/mnt/T/mnt/ML_output/building_change_detection_2025/tmpfolder/",help="Path to where the 1km2 datasets should be crated") 
    #parser.add_argument("--datatypes", default= ["rgb","cir","DSM","DTM"],help='default is ["rgb","cir","DSM","DTM"].')
    #parser.add_argument("--datatypes", default= ["rooftop_rgb","rooftop_cir","fast_trueorto","fast_trueorto_DSM","DSM_no_veg","lidar_dsm","lidar_dtm"],help='default is :%(default)s ')

    #parser.add_argument("--original_label_folder", type=str, default= "/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/Rooftop_2024_GeoDanmark_labels/large_labels",help="Path to the labels in the original dataset")

    #parser.add_argument("--original_label_folder", type=str, default= "/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/labels/large_labels/",help="Path to the labels in  the original dataset")

    parser.add_argument("--skip",help ="steps to be skipped e.g geopackage create_txt_file_with_overlapping_files copy_files crop_files create_dataset infer create_ini_files merge_inference_images",nargs ='+',default =[])
    parser.add_argument("--last_finnished_tile", type=str, default= None,help="process wil skip all tiles up to and including this tile")
    parser.add_argument("--start_with_tile", type=str, default= None,help="process wil skip all tiles up to (but not including) this tile")
    parser.add_argument("--stop_after_tile", type=str, default= None,help="process wil stop once this tile is finnished")
    parser.add_argument('--geopackage', type=str, default="/mnt/T/mnt/trainingdata/bygningsudpegning/bygninger20250326nofilter.gpkg",required=False, help='Path to the GeoPackage file')
    parser.add_argument('--unknown_boarder_size', type=float, default=1.5, help='how large baorder of "unkown"==0 values should there be around the areas defined by polygons? set to 0 to not have any boarder')
    parser.add_argument('--create_new_mapping', action='store_true', help='Whether to create a new mapping from the GeoPackage')
    parser.add_argument('--path_to_mapping', type=str, required=False,default = "/mnt/T/mnt/trainingdata/bygningsudpegning/iter_4/roof_no_roof_mapping.txt", help='Path to save or load the mapping file')

    #parser.add_argument('--path_to_model', type=str, default="/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_50/models/andringsudpegning_1km2benchmark_iter_50_24.pth",help ="path to model ")
    # Model parameters
    parser.add_argument("--saved_models", type=json.loads, default = '[\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_52/models/andringsudpegning_1km2benchmark_iter_52_24.pth\",\"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_63/models/andringsudpegning_1km2benchmark_iter_63_24.pth\", \"/mnt/T/mnt/logs_and_models/bygningsudpegning/andringsudpegning_1km2benchmark_iter_62/models/andringsudpegning_1km2benchmark_iter_62_24.pth\"]', help="JSON list of paths to the saved FastAI U-Net models (.pth files)")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of output classes for the model (same for all models)")
    parser.add_argument("--means", type=json.loads, default='[[0.485, 0.456, 0.406],[0.485, 0.456, 0.406],[0.485, 0.456, 0.406]]', help="JSON list of lists of lists for means, one set per model")
    parser.add_argument("--stds", type=json.loads, default='[[0.229, 0.224, 0.225],[0.229, 0.224, 0.225],[0.229, 0.224, 0.225]]', help="JSON list of lists of lists for stds, one set per model")
    parser.add_argument("--channels", type=json.loads, default = '[[[0,1,2]],[[0,1,2]],[[0,1,2]]]', help="JSON list of lists of lists for channels, one set per model")
    parser.add_argument("--data_types", type=json.loads, default = '[[\"rooftop_rgb\"],[\"rooftop_rgb\"],[\"rooftop_rgb\"]]', help="JSON list of lists for data folders, one set per model")
    parser.add_argument("--batch_size", type=int, default=8, help="batchsize")
    parser.add_argument("--patch_size", type=int, default=1000, help="patchsize")
    parser.add_argument("--workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--model_names", type=json.loads, default=["tf_efficientnetv2_l.in21k","efficientnetv2_rw_m.agc_in1k","resnet50"], help="model names")


    parser.add_argument('--path_to_codes', type=str, default="/mnt/T/mnt/trainingdata/bygningsudpegning/1km2data_for_benchmarking/codes.txt",help ="path to codes used to translate prediciton to class name")
    parser.add_argument('--overlap', type=int, default=40 , help = "how much overalp should there be between the splitted images?")
    #parser.add_argument('--apply_threshold', action="store_true",help = "use this flag in order to mask away all diff pixels with value == 1 (missing building) if uint8 building prob is >0)")
    #parser.add_argument('--prune_images', action="store_true",help = "use this flag in order to prune away images if posible")



    # Argument to provide a JSON file with all values used above.
    parser.add_argument("--json", type=str, help="Path to JSON file with arguments that override the defaults defined in the code above")

    # Parse only the --json argument first in order to overwrite the defaults
    partial_args, unknown = parser.parse_known_args()    
    if partial_args.json:
        # Load JSON
        with open(partial_args.json, "r") as f:
            json_args = json.load(f)
        
        # Convert JSON dict to argparse.Namespace
        args = argparse.Namespace(**json_args)
    else:
        # Parse from command line normally
        args = parser.parse_args()



    # Call the main function with parsed arguments
    main(shapefile_path= args.shapefile,path_to_images = args.path_to_images,data_folders=args.data_types, channels=args.channels,path_to_map_creation_output_folder = args.path_to_map_creation_output_folder ,datatypes=args.data_types ,skip = args.skip,last_finnished_tile = args.last_finnished_tile,start_with_tile=args.start_with_tile, stop_after_tile = args.stop_after_tile,saved_models =args.saved_models,model_names = args.model_names,means=args.means,stds=args.stds,remove_tmp_files=args.remove_tmp_files,batch_size =args.batch_size,workers=args.workers,args=args)
