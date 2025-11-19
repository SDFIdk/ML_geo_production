import os
import pathlib
import numpy as np
from osgeo import gdal
import argparse
import time
import rasterio
from PIL import Image
import shutil
import configparser
import sys
from types import SimpleNamespace
import random


def generate_random_filename_that_does_not_exist(folder_path,postfix=".tif"):
    """
    Generate a path to a file that does not exist
    the file is used for temporary storage and will be deleted after usage
    """
    name = ("".join([str(random.randint(0, 9)) for _i in range(10)])) + postfix
    random_file_name = pathlib.Path(folder_path) / name
    while random_file_name.is_file():
        name = ("".join([random.randint([0, 9]) for _i in range(10)])) + postfix
        random_file_name = pathlib.Path(folder_path) / name
    return str(random_file_name)

def get_patches_per_large_image(input_folder):
    """
    create a dictionary were the large image is key and value is  a dictionary containing a list of small images that are created from the large image, and a reg exp that gets you all those images if used with gdal


    :param input_folder:
    :return: {"large_image_name_1.tif":{"pattern":"path/to/match/*.tif","patches":["patch_1.tif","patch_2.tif",,,]},}
    """
    input_files = os.listdir(input_folder)
    #find all different image names (the name of the image before it was splitted up)
    images ={}
    for input_file in input_files:

        image_name =  pathlib.Path("_".join(input_file.split("_")[:-2]) + ".tif")
        patch_path = input_folder+"\\"+input_file

        if image_name in images:
            images[image_name]["patches"].append(patch_path)
        else:
            images[image_name]={"patches":[patch_path],"pattern":str(input_folder/image_name.with_suffix(''))+"*.tif"}


    return images

def combine_patches(pattern,output_file_path = r"C:\Users\B152325\Desktop\befæstelse_status_2023\\"):
    """
    create a mosaik by combining all patches to a single image
    """

    start_time = time.time()
    pathlib.Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    #we use a vrt file as a temporary step before turning it into a geotiff
    output_file_path_buldvrt = generate_random_filename_that_does_not_exist(folder_path=pathlib.Path(output_file_path).parent, postfix=".vrt")



    gdalbuildvrt = "gdalbuildvrt "
    gdalbuildvrt_process = gdalbuildvrt + output_file_path_buldvrt + " " + pattern

    print(gdalbuildvrt_process)
    return_value = os.system(gdalbuildvrt_process)
    if int(return_value) in [1,256]:
        sys.exit("\n "+gdalbuildvrt_process + "\n FAILED with status :"+str(return_value))



    end_time = time.time()
    print("end_time-start_time:"+str(end_time-start_time))
    print("done combining patches")


    print("turn the VRT file into a geotif")
    print("##running the following command ###")
    vrt_to_geotif = "gdal_translate -of GTiff "+output_file_path_buldvrt + " "+output_file_path
    print(vrt_to_geotif)
    vrt_to_tif_start_time = time.time()
    return_value = os.system(vrt_to_geotif)
    if int(return_value) in [1,256]:
        sys.exit("\n "+vrt_to_geotif + "\n FAILED with status :"+str(return_value))

    vrt_to_tif_end_time = time.time()
    print("##Done ###")
    print("vrt_to_tif took:"+str(vrt_to_tif_end_time-vrt_to_tif_start_time))




    #clean up
    pathlib.Path(output_file_path_buldvrt).unlink(missing_ok=True)







def save_preds(path_to_probs,path_to_preds=None):
    if not path_to_preds:
        path_to_preds = pathlib.Path(path_to_probs).with_name(pathlib.Path(path_to_probs).name.replace(".tif","_pred.tif"))

    with rasterio.open(path_to_probs) as src:
        # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
        #load the probs and do argmax in order to get predictions
        new_meta = src.meta.copy()
        new_xform = src.transform
        numpy_data = src.read()
        predictions = src.read().argmax(axis=0)

    new_meta["count"] = 1
    new_meta["dtype"] = np.uint8


    with rasterio.open(path_to_preds, "w", **new_meta) as dest:
        dest.write(np.expand_dims(predictions, axis=0))


def crop_image_to_shapefile_footprint(input_im_path,output_path,shape_file_path):
    """
    TODO: replace this functionality with in-memory croping! (saving the outut to a file is to slow! )


    :param input_im_path:
    :param output_im_path:
    :param shapefile_path:
    :return:
    """
    # Call process.
    call= "gdalwarp -cutline "+shape_file_path+" -crop_to_cutline "+str(input_im_path)+" "+output_path
    return_value = os.system(call)
    if int(return_value) in [1,256]:
        sys.exit("\n "+call + "\n FAILED with status :"+str(return_value))
    print("return_value:"+str(return_value))


def run_main_from_configfile(config_file):
    """
    converting the data that is stored in a config file to namespace notation (accessing entries with . notation)
    :param config_file: path to config file
    :return:
    """

    ini_parser = configparser.ConfigParser()
    #in order to keep the kapital letters in the variables in the config file we set the optionxform to str
    ini_parser.optionxform = str
    ini_parser.read(config_file)




    n = SimpleNamespace(**ini_parser["SETTINGS"])
    # convert 'False' and 'false' to the boolean False
    for key, value in n.__dict__.items():
        print(f"Key: {key}, Value: {value}")
        if value in ["False","false"]:
            n.__dict__[key] = False


    main(args=n)


def main(args):
    main_start_time = time.time()
    #make sure that output folders exist
    pathlib.Path(args.mosaicked_preds_folder).mkdir(parents=True, exist_ok=True)




    #lots of 1000x1000 croped probabilities-images are located in a folder
    #input_folder =r"T:\logs_and_models\befastelse\orthoimages_iteration_31\models\befaestelse_dataset_creation_test_2"

    #create a dictionary with the original image as key and a list of patches as value
    #e.g {"large_image_name_1.tif":["patch_1.tif","patch_2.tif",,,],}
    patches_and_patterns_dict = get_patches_per_large_image(args.input_preds)



    large_images=[]
    print("##################################################")
    print("This script does the following operation:")
    print("combine the ML output for all patches related to a certain image into an inference image of the same shape as the original large-image   (a mosaik -image)")

    print("##################################################")

    print("nr of large images to produce:"+str(len(patches_and_patterns_dict)))
    print("####")
    for i, large_image  in enumerate(patches_and_patterns_dict):
        print("working on image :"+str(i)+ " out of :"+str(len(patches_and_patterns_dict)))
        patches=patches_and_patterns_dict[large_image]["patches"]
        pattern=patches_and_patterns_dict[large_image]["pattern"]
        print("nr of patches to combine :"+str(len(patches)))

        print("pattern that match the patches we want to process:"+str(pattern))

        output_file_path= str(pathlib.Path(args.mosaicked_preds_folder)/  large_image )

        combine_patches(pattern=pattern,output_file_path=output_file_path)
        print("done merging patches to :"+output_file_path)
        if args.create_mosaic_pred_image:
            save_preds(output_file_path)


        large_images.append(output_file_path)

    #Croping is now done 'on the fly' during the merging in merge_probs.wiht_numpy.py
    """
    moved_croping_to_merge_with_numpy = True
    if moved_croping_to_merge_with_numpy:
        pass
    else:
        if args.shape_file:
            # crop the probability-images to fit the 1km2 tile
            crop_to_same_shape(image_folder=args.mosaicked_preds_folder,shape_file_path=args.shape_file,output_folder =args.Croped_mosaicked_preds_folder)
        else:
            #copy all files to the Croped_mosaicked_preds_folder instead
            source_folder = Path(args.mosaicked_preds_folder)
            destination_folder = Path(args.Croped_mosaicked_preds_folder)

            # Iterate over each file in the source folder
            for file in source_folder.iterdir():
                if file.is_file():
                    # Construct the destination path by joining the destination folder path with the file name
                    destination_path = destination_folder / file.name

                    # Copy the file to the destination folder
                    shutil.copy2(file, destination_path)
    """


    main_finnished_mosaik_time = time.time()


    main_finnished_prediction_image_time = time.time()
    print("########################DONE##################################################################################################")
    print("times:")
    print("totall : "+str(main_finnished_prediction_image_time-main_start_time))
    print("creating mosaik: "+str(main_finnished_mosaik_time-main_start_time))

    print("##############################################################################################################################")


if __name__ == "__main__":

    example_usage= r"python merge_all_images_probabilities.py -i path\to\folder\with\probs -m folder\to\save\mosaiked\probs\in"
    print("########################EXAMPLE USAGE########################")
    print(example_usage)
    print("#############################################################")


    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_preds", help="path/to/folder/with/probs  ",required=True)
    parser.add_argument("-m", "--mosaicked_preds_folder", help="path/to/folder/to/save/mosaicked/probs/in (crops from same images are recomined to a large image)",required=True)
    parser.add_argument("--create_pred_image",required=False, action='store_true',default =False)

    args = parser.parse_args()
    main(args)

