import argparse
import numpy as np
from PIL import Image
# Set the limit to a higher value to avoid Image complaining
# The number is the total pixel count (width * height).
Image.MAX_IMAGE_PIXELS = 10000000000 


def compare_images(pred_im, label_im, ignore_index):
    """
    Compares the label with the prediction, calculating accuracy only on the area
    that is not masked out by 'ignore_index' in the label or 'missing_prediction' 
    (pred_im == 0) in the prediction.

    Args:
        pred_im: numpy array of predicted image (grayscale)
        label_im: numpy array of labeled image (grayscale)
        ignore_index: pixel value to ignore in label image

    Returns:
        diff_percentage: percentage of differing pixels in the analyzed area
    """

    # --- Input Validation ---
    if not isinstance(pred_im, np.ndarray):
        raise TypeError("pred_im must be a numpy array")
    if not isinstance(label_im, np.ndarray):
        raise TypeError("label_im must be a numpy array")
    if pred_im.shape != label_im.shape:
        print("Images have different dimensions and cannot be compared.")
        return None

    # --- Mask Identification & Printout (Optimized) ---

    # 1. Identify missing predictions (pixels that should not be analyzed)
    missing_prediction = pred_im == 0
    nr_of_pixels_with_missing_prediction = np.sum(missing_prediction)
    print("nr_of_pixels with missing prediction: ")
    print(str(nr_of_pixels_with_missing_prediction))

    # 2. Identify ignore label pixels (CALCULATED ONLY ONCE)
    ignore_label_mask = label_im == ignore_index
    nr_of_ignore_pixels = np.sum(ignore_label_mask) # Derived from the mask
    print("nr_of_ignore_pixels: ")
    print(str(nr_of_ignore_pixels))

    # --- Analysis Mask Creation ---

    # Combine both conditions to create the analysis mask: True for pixels to be analyzed
    analysis_mask = np.logical_not(np.logical_or(missing_prediction, ignore_label_mask))

    # --- Calculation on Analyzed Area ---

    # Total pixels available for analysis
    total_analyzed_pixels = np.sum(analysis_mask)
    print("nr of total pixels in analyzed area: ")
    print(str(total_analyzed_pixels))

    # The total number of pixels in the image (unchanged)
    total_pixels_in_image = pred_im.size
    print("nr of total pixels in image: ")
    print(str(total_pixels_in_image))

    # Calculate differences *only* in the area defined by the analysis_mask
    diff_mask = pred_im != label_im
    nr_of_differing_pixels_in_analyzed_area = np.sum(np.logical_and(diff_mask, analysis_mask))

    print("nr of differing pixels in analyzed area: ")
    print(str(nr_of_differing_pixels_in_analyzed_area))

    if total_analyzed_pixels == 0:
        print("No pixels available for analysis after masking ignored/missing areas.")
        return 0.0

    # Calculate the accuracy (1 - error_rate)
    error_rate = nr_of_differing_pixels_in_analyzed_area / total_analyzed_pixels
    accuracy = 1.0 - error_rate
    diff_percentage = error_rate * 100

    print("#" * 20)
    print("Accuracy (1 == 100%) (only counting pixels in the analyzed area): ")
    print(f"{accuracy:.6f}")
    print("#" * 20)
    print(f"Percentage of pixels with different values (Error Rate): {diff_percentage:.2f}%")

    return diff_percentage

# Note: The rest of the script (load_image and main) remains the same.

def load_image(image_path):
    """
    Load an image from path and convert to grayscale numpy array.
    
    Args:
        image_path: path to the image file
    
    Returns:
        numpy array of the image in grayscale
    """
    return np.array(Image.open(image_path).convert("L"))


def main():
    parser = argparse.ArgumentParser(description="Compare pixel values of two images.")
    parser.add_argument("--pred_im", type=str, required=True, help="Path to the predicted image.")
    parser.add_argument("--label_im", type=str, required=True, help="Path to the labeled image.")
    parser.add_argument("--unkown_id", type=int, default=0, help="Ignore_index (default 0)")
    args = parser.parse_args()
    
    # Load images as numpy arrays
    pred_array = load_image(args.pred_im)
    label_array = load_image(args.label_im)
    
    # Call compare_images with numpy arrays
    compare_images(pred_array, label_array, args.unkown_id)


if __name__ == "__main__":
    main()
