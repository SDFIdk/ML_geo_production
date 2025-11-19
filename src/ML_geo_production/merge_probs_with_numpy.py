import os
import pathlib
import numpy as np
import argparse
import time
import rasterio
from PIL import Image
from osgeo import ogr
from rasterio.crs import CRS
import configparser
import re
from types import SimpleNamespace
from ML_geo_production import merge_all_images_probabilities
import logging as log
import psutil
import random




def check_memory():
    # Get available memory in bytes
    available_memory = psutil.virtual_memory().available
    return available_memory

def filter_numbers_and_whitespaces(string):
    #Remove all text so we only get the coordinates left ('.' needs to be left as it is teh coma in case of coordinates in float format)
    filtered_string = re.sub(r'[^0-9\s.]', '', string)
    return filtered_string

#meter_per_pixel = 0.09900000000000001854)
def save_tiff(numpy_array,path,shape_file,meter_per_pixel):
    """
    Save an array as GEotif based on a shapefile for the area covered
    :param numpy_array:
    :param path:
    :param shape_file: e.g C:/Users/b199819/Desktop/imageshape.shp
    :return:
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    meter_per_pixel = float(meter_per_pixel)



    #extract the coordinates form the shapefile that defines the area we have made a map over
    #when creating a geotif we want the coordinates for the upper left corner.
    # the shapefile lists 5 points [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)] coresponding to lower left ,upper left ,upper right ,lower right and lower left
    #in order to get the coordinate for teh upper left we extract teh mimimum x and maximum y

    shapefile = ogr.Open(shape_file)
    layer = shapefile.GetLayer()
    #print("coordinates in shapefile :"+str([f.GetGeometryRef().ExportToWkt() for f in layer][0].replace("(","").replace(","," ").split()))

    #after splitting we get this format
    # ['POLYGON', '724000', '6175000', '724000', '6176000', '725000', '6176000', '725000', '6175000', '724000', '6175000))']
    #in order to get (x2,y2) we need to extract the item at position 3 and 4

    splitted_coordinates = [f.GetGeometryRef().ExportToWkt() for f in layer][0].replace("(","").replace(","," ").split()
    print("splitted_coordinates:"+str(splitted_coordinates))
    coordinate_list=filter_numbers_and_whitespaces([f.GetGeometryRef().ExportToWkt() for f in layer][0].replace(","," ")).split()
    #turn the coordinates from string format to float
    xs= [float(coordinate_list[i]) for i in range(0,len(coordinate_list),2)]
    ys = [float(coordinate_list[i]) for i in range(1, len(coordinate_list)-1, 2)]
    min_x = min(xs)
    max_y = max(ys)

    if len(numpy_array.shape)==2:
        #ad an extra dimension of there only are 2
        #rasterios write operation demans a 3dim array
        numpy_array= np.expand_dims(numpy_array,axis=0)

    number_of_bands, height, width = numpy_array.shape


    kwargs = {'driver': 'GTiff', 'dtype': numpy_array.dtype, 'nodata': None, 'width': width, 'height': height, 'count': number_of_bands, 'crs': CRS.from_epsg(25832), 'transform': rasterio.Affine(float(meter_per_pixel), 0.0, min_x, 0.0, -float(meter_per_pixel), max_y)}

    with rasterio.open(path, 'w', **kwargs) as dst:
        dst.write(numpy_array)
    print("saved file: "+str(path))




def save_probabilities(data,path,shape_file,meter_per_pixel):
    """
    Probabilities are saved in the format of the numpy arrray
    """

    save_tiff(data,path,shape_file=shape_file,meter_per_pixel=meter_per_pixel)


def normalize(arr):
    """
    making it easy to visulize channels by normalizing them
    """
    arr = arr - arr.min()
    arr = arr / arr.max()
    arr = np.array(arr * 255, dtype=np.uint8)
    return arr

def merge_with_numpy(image_folder,output_folder,save_probs,save_preds,output_probabilities_file_path,shape_file,meter_per_pixel):


    merge_with_numpy_start = time.time()


    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    image_paths = [pathlib.Path(image_folder)/im_name for im_name in os.listdir(image_folder) if ".tif" in im_name]
    print(image_paths)
    print("merging "+str(len(image_paths))+ " files with numpy")

    if len(image_paths) ==0:
        log.error('no files in  :'+str(image_folder) + " did the inference step fail?")


    for idx, im in enumerate(image_paths):
        print("opening image nr:"+str(idx)+" : "+str(im))

        try:
            #we need a temporary file to store some data in
            output_path = merge_all_images_probabilities.generate_random_filename_that_does_not_exist(folder_path = "./")


            #crop the prediction to the same shape as the shapefile
            #this crates a temporary image that in most cases is much larger than the input image although some parts of the input image might be croped away

            merge_all_images_probabilities.crop_image_to_shapefile_footprint(input_im_path=im,output_path=output_path,shape_file_path=shape_file)


            if idx == 0:
                #the first image

                #if there are probabilities to load , load them and add them to the new probabilities
                if output_probabilities_file_path and pathlib.Path(output_probabilities_file_path).is_file():
                    #add the loaded probabilites to the new probabilities
                    summed = np.array(rasterio.open(output_probabilities_file_path).read(),dtype=float)  + np.array(rasterio.open(output_path).read(),dtype=float)
                else:
                    #if there are no probabilities to load we use the new probabilities as thay are
                    summed = np.array(rasterio.open(output_path).read(),dtype=float) #np.zeros([11,10000,10000])
            else:
                #not the first image
                #add the new probabilities to the old probabilities
                summed = summed +np.array(rasterio.open(output_path).read(),dtype=float)

        except Exception as e:
            print("merge_with_numpy, failed to read :"+str(im))
            log.error('merge_with_numpy, failed to read :'+str(im)+' error message is '+str(e))
            log.error('available memmory is : '+str(check_memory()))
        finally:
            # delete the temporary file after usage
            with pathlib.Path(output_path) as file:
                file.unlink(missing_ok=True)

    #normlaize the probs to sum to 255 for each pixel (saved in uint8 format)
    #each multichannel probs image have shape [channels, h,w]
    #normalizing will make it posible to filter outut based in the absolute probabilites . thresholding only works if the range is fixed
    sum_of_probs_at_each_pixel=summed.sum(axis=0) 
    summed = ((summed*1.0) / sum_of_probs_at_each_pixel)*255
    summed = np.array(summed,dtype=np.uint8)


    merge_with_numpy_done_opening = time.time()
    if save_preds:
        output_pred_map_file= (pathlib.Path(output_folder)/ "preds"/ pathlib.Path(shape_file).name).with_suffix(".tif")
        save_tiff(summed.argmax(axis=0).astype(np.uint8),output_pred_map_file ,shape_file=shape_file,meter_per_pixel=meter_per_pixel)
    if save_probs:
        #saving the probs

        if not output_probabilities_file_path:
            #there is no filepath defined to save the probbilities to, so we make one
            output_probabilities_file_path = (pathlib.Path(output_folder) / "probs" / pathlib.Path(shape_file).name).with_suffix(".tif")


        save_probabilities(summed,output_probabilities_file_path,shape_file=shape_file,meter_per_pixel=meter_per_pixel)


        #if probabilities_format_to_save_in == "uint8":
        # saving the probs as uint8 (divide by 255 after loading the vlaues to get back to float format)
        #save_tiff((summed *255).astype(np.uint8),
        #          (pathlib.Path(output_folder) / ("uint8_probs" + pathlib.Path(shape_file).name)).with_suffix(".tif"),
        #          shape_file=shape_file)
        #else:


        #save_tiff(summed.astype(np.float32),output_probabilities_file_path,shape_file=shape_file)
        # saving the probs as uint8 (divide by 255 after loading the vlaues to get back to float format)
        #save_tiff((summed *255).astype(np.uint8),
        #          (pathlib.Path(output_folder) / ("uint8_probs" + pathlib.Path(shape_file).name)).with_suffix(".tif"),
        #          shape_file=shape_file)


    merge_with_numpy_end = time.time()

    print("loading all images took :"+str(merge_with_numpy_done_opening - merge_with_numpy_start))

    print("merge_with_numpy took "+str(merge_with_numpy_end - merge_with_numpy_start))
    return output_pred_map_file



def run_main_from_configfile(config_file):
    """
    :param config_file: path to config file
    :return:
    """
    ini_parser = configparser.ConfigParser()
    # in order to keep the kapital letters in the variables in the config file we set the optionxform to str
    ini_parser.optionxform = str
    ini_parser.read(config_file)

    n = SimpleNamespace(**ini_parser["SETTINGS"])
    # convert 'False' and 'false' to the boolean False
    for key, value in n.__dict__.items():
        print(f"Key: {key}, Value: {value}")
        if value in ["False", "false"]:
            n.__dict__[key] = False

    return main(args=n)


def main(args):
    return merge_with_numpy(shape_file=args.shape_file,image_folder=args.mosaicked_preds_folder,output_folder = args.output_merged_preds_folder,save_probs=args.save_result_probs,save_preds=args.save_result_preds,output_probabilities_file_path=args.output_probabilities_file_path,meter_per_pixel = args.meter_per_pixel)




if __name__ == "__main__":
    example_usage= r"python merge_probs_with_numpy.py -m T:\trainingdata\befastelse\1km2\mosaicked_preds_folder -o T:\trainingdata\befastelse\1km2\merged_with_numpy --Save_result_probs"
    print("########################EXAMPLE USAGE########################")
    print(example_usage)
    print("#############################################################")

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mosaicked_preds_folder", help="path/to/folder/to/save/mosaicked/probs/in ",required=True)
    parser.add_argument("-o", "--output_merged_preds_folder", help="path/to/folder/to/save/merged/probs/in",required=True)
    parser.add_argument("-s", "--shape_file", help="path/to/shapefile.sh",required=True)
    parser.add_argument('--save_result_probs', action='store_true',default=False)
    parser.add_argument("--meter_per_pixel",default=0.1)



    args = parser.parse_args()
    main(args)

