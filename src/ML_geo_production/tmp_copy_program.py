import os
import shutil
import argparse

def copy_images(path_list, output_dir):
    """
    Copies images from a list of paths to a specified output directory.

    Args:
        path_list (list): A list of strings, where each string is the path to an image file.
        output_dir (str): The path to the directory where the images should be copied.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    print(f"Starting image copy process to: {output_dir}")

    for image_path in path_list:
        if not os.path.exists(image_path):
            print(f"Warning: Source file not found, skipping: {image_path}")
            continue

        try:
            # Get the filename from the path
            image_filename = os.path.basename(image_path)
            destination_path = os.path.join(output_dir, image_filename)

            # Handle potential duplicate filenames by appending a number
            base, extension = os.path.splitext(image_filename)
            counter = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(output_dir, f"{base}_{counter}{extension}")
                counter += 1

            shutil.copy2(image_path, destination_path) # copy2 attempts to preserve metadata
            print(f"Successfully copied: {image_path} to {destination_path}")

        except IOError as e:
            print(f"Error copying file {image_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images from a list of paths to an output directory.")
    parser.add_argument('--output_dir', required=True, help='The directory to copy the images to.')
    parser.add_argument('--path_list', default= ['/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1455.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1456.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1457.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1458.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1459.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1460.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1461.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1462.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_078_1463.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1942.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1943.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1944.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1945.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1946.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1947.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1948.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1949.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_081_1950.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1961.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1962.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1963.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1964.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1965.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1966.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1967.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1968.tif', '/mnt/T/mnt/ML_input/building_change_detection_2025/rooftop_rgb/O_roof_2025_84_41_082_1969.tif'])
    args = parser.parse_args()

    # Read the image paths from the provided file
    image_paths = args.path_list

    # Example usage:
    # Replace this with your actual list of image paths or load it from a file
    # path_list = [
    #     "/path/to/your/image1.jpg",
    #     "/another/path/to/image2.png",
    #     "/yet/another/image3.gif",
    # ]

    copy_images(image_paths, args.output_dir)
    print("Image copy process finished.")
