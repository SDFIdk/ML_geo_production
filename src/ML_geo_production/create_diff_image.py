import argparse
import rasterio
import numpy as np
from pathlib import Path
from PIL import Image
import configparser
from ML_geo_production import merge_probs_with_numpy
def apply_ignore_thresholded_red_areas(building_pred_above_threshold_image_mask,remapped_output_path,merge_inference_images_config ):
    #given a diff image . remove all red areas that overlap with the mask
    #diff image coding : 2 = new builing , 1 == missing building
    # Initialize the configparser object
    config = configparser.ConfigParser()
    # Read the .ini file
    config.read(merge_inference_images_config)
    assert config['SETTINGS']["save_result_probs"] == "true", "save_result_probs must be true in merge_config in order to be able createa a threshold mask"

    diff_image_np = np.array(Image.open(remapped_output_path))
    print("a")
    print(diff_image_np)
    #set areas with 1(missing buildign) in diff image and 1(building prob above threshold) in building_pred_above_threshold_image_mask to 0.
    areas_to_ignore_missing_buildings_in_np = np.array((((diff_image_np ==1) * (building_pred_above_threshold_image_mask  ==1))),dtype=np.uint8)
    areas_to_ignore_missing_buildings_in_bool =areas_to_ignore_missing_buildings_in_np ==1
    print("b")
    diff_image_np[areas_to_ignore_missing_buildings_in_bool] = 0
    merge_probs_with_numpy.save_tiff(diff_image_np,remapped_output_path,shape_file=config['SETTINGS']["shape_file"],meter_per_pixel=config['SETTINGS']["meter_per_pixel"])


def remap_diff_image_to_only_show_red_and_green(image_path, output_path,false_positive_roofs_set_to=2,false_negative_roof_set_to=1, unknown_labels_set_to = 0):
    #creating diff image where only to much detected roof or to little detected roofs is shown
    #data in range [10, 18] means model did not classify as any kind of roof when label says some kind of roof
    #data ending with 1 means model says some kind of roof when label dont say some kind of roof. (1 == label says background)
    #places where label says unkown shoud be treated separately (these are the values that ends with 0)
 
    # Open the GeoTIFF image using rasterio
    with rasterio.open(image_path) as src:
        # Read the image as a NumPy array
        img_array = src.read(1)  # Read the first band (grayscale)

        #areas with label == unknown
        unkown_label_mask = img_array % 10 == 0


        # MODEL did not find roof
        # Apply the first condition: Set values in range [10, 18] to 1
        false_negative_roof_mask = (img_array >= 10) & (img_array <= 18)


        # Model found non-existing roof
        # Apply the second condition: Set values that end with 1 to 2
        false_positive_roof_mask = img_array % 10 == 1



        img_array[false_negative_roof_mask] = false_negative_roof_set_to
        img_array[false_positive_roof_mask] = false_positive_roofs_set_to
        img_array[unkown_label_mask] = unknown_labels_set_to

        # Set all other values to 0
        img_array[(img_array != false_negative_roof_set_to) & (img_array != false_positive_roofs_set_to) & (img_array != unknown_labels_set_to)]   = 0

        # Define the metadata for the output file (same as input)
        metadata = src.meta.copy()

        # Update the metadata for the new file (if required)
        metadata.update({
            'dtype': 'uint8',  # Ensure that the data type is compatible with the new values
            'count': 1  # We're only working with one band
        })

        # Save the processed image as a new GeoTIFF
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(img_array, 1)  # Write the modified array to the first band


def get_single_tif_file(folder_path):
    # Convert the folder path to a Path object
    folder = Path(folder_path)
    
    # List all .tif files in the folder (excluding subfolders)
    tif_files = [file for file in folder.iterdir() if file.is_file() and file.suffix == ".tif"]
    
    # Check if there's exactly one .tif file
    if len(tif_files) == 1:
        return tif_files[0]  # Return the Path object of the single .tif file
    elif len(tif_files) == 0:
        return "No .tif files found in the folder."
    else:
        return "More than one .tif file found in the folder."

def main_from_ini(merge_inference_images_config):
    prediction_image_folder = get_pred_folder_from_ini_file(merge_inference_images_config)
    pred_im = get_single_tif_file(prediction_image_folder/"preds")
    label_im = get_single_tif_file(prediction_image_folder.parent/"label_1km2")
    output_diff_image = (prediction_image_folder.parent/"diff_image_1km2")/Path(pred_im).name
    remapped_output_path = main(pred_im,label_im,output_diff_image)
    return (remapped_output_path,label_im)

def get_pred_folder_from_ini_file(merge_inference_images_config):
    # Initialize the configparser object
    config = configparser.ConfigParser()
    
    # Read the .ini file
    config.read(merge_inference_images_config)
    
    # Extract the 'Output_merged_preds_folder' from the [SETTINGS] section
    output_folder = config['SETTINGS'].get('Output_merged_preds_folder', None)
    
    if output_folder:
        # Convert the output folder path to a Path object
        output_folder_path = Path(output_folder)
        
        return output_folder_path
    else:
        return None

def main(pred_im,label_im,output_diff_image):
    #


    # Ensure the output directory exists
    output_path = Path(output_diff_image)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Load the two images using the provided file paths
    with rasterio.open(pred_im) as src1:
        img1 = src1.read(1).astype(int)
        profile = src1.profile

    with rasterio.open(label_im) as src2:
        img2 = src2.read(1).astype(int)

    # find pixels that are the same
    correct = img1 == img2

    # Generate unique values for each pair (img1, img2)
    unique_values = img1 * 10 + img2
    unique_values[correct] = 0


    # Prepare the profile for the new image
    profile.update(dtype=rasterio.int8, count=1)  # Single-channel image with integer values

    # Save the unique values as a single-channel GeoTIFF to the specified output path
    with rasterio.open(output_diff_image, 'w', **profile) as dst:
        dst.write(unique_values, 1)  # Write to the first (and only) band



    #also make a copy of the diff image where only to much roof or to little roof is visible
    #the new iamge will show all non-background classifications as same classification  and not will not show pixels with label == unknown ,
    #OBS: if this is the only thing we are interested in we copuld spåeed up the process by simply creating two binary masks as step one and compare them directly instead!  
    remapped_output_path = output_path.with_name(output_path.stem + "_remapped.tif")
    remap_diff_image_to_only_show_red_and_green(output_path, remapped_output_path)
    return remapped_output_path




if __name__ == "__main__":
   # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate single-channel GeoTIFF from two input images.")
    parser.add_argument('--pred_im', type=str, required=True, help="Path to the prediction image (image1).")
    parser.add_argument('--label_im', type=str, required=True, help="Path to the label image (image2).")
    parser.add_argument('--output_diff_image', type=str, required=True, help="Path to save the output difference image.")

    args = parser.parse_args()



    main(pred_im=args.pred_im,label_im =args.label_im,output_diff_image=args.output_diff_image)
