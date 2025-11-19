import rasterio
import numpy as np
from typing import Optional, Union # Added for type hinting clarity

def create_diff_image(
    label_path_or_array: Union[str, np.ndarray], 
    pred_path_or_array: Union[str, np.ndarray], 
    diff_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a difference image comparing label and prediction data.
    
    The resulting diff image contains:
    - 1 where label = 2 and pred = 1 (False Negative of class 2)
    - 2 where label = 1 and pred = 2 (False Positive of class 2)
    - 0 elsewhere
    
    Parameters:
    -----------
    label_path_or_array : str or np.ndarray
        Path to the label geotiff file (values 0, 1, 2) OR the label data as a numpy array.
    pred_path_or_array : str or np.ndarray
        Path to the prediction geotiff file (values 1, 2) OR the prediction data as a numpy array.
    diff_path : Optional[str]
        Path where the output difference geotiff will be saved. 
        If None, the difference image is not saved to disk.
        
    Returns:
    --------
    np.ndarray
        The computed difference image data.
    """
    
    # 1. Handle Label Data Loading (Path or Array)
    if isinstance(label_path_or_array, np.ndarray):
        label_data = label_path_or_array
        # A placeholder profile is needed if we might save the output later
        profile = None
    else:
        label_path = label_path_or_array
        try:
            with rasterio.open(label_path) as label_src:
                label_data = label_src.read(1)  # Read the first band
                profile = label_src.profile.copy()  # Copy the metadata for output file
        except rasterio.RasterioIOError as e:
            print(f"Error reading label file {label_path}: {e}")
            raise

    # 2. Handle Prediction Data Loading (Path or Array)
    if isinstance(pred_path_or_array, np.ndarray):
        pred_data = pred_path_or_array
    else:
        pred_path = pred_path_or_array
        try:
            with rasterio.open(pred_path) as pred_src:
                pred_data = pred_src.read(1)  # Read the first band
            
            # If the label was an array, but pred was a path, we now have a profile 
            # from the pred file. We can use this for saving if needed.
            if profile is None:
                profile = pred_src.profile.copy()
                
        except rasterio.RasterioIOError as e:
            print(f"Error reading prediction file {pred_path}: {e}")
            raise
    
    # Check for shape mismatch before calculation
    if label_data.shape != pred_data.shape:
        raise ValueError(
            f"Input data shapes mismatch: Label {label_data.shape} vs Prediction {pred_data.shape}"
        )

    # 3. Create the Difference Image
    diff_data = np.zeros_like(label_data, dtype=np.uint8) # Explicitly set dtype for final result
    
    # Apply condition 1 (False Negative for class 2, or Error of Omission): 
    # Where label = 2 and pred = 1, set diff = 1
    diff_data[(label_data == 2) & (pred_data == 1)] = 1
    
    # Apply condition 2 (False Positive for class 2, or Error of Commission): 
    # Where label = 1 and pred = 2, set diff = 2
    diff_data[(label_data == 1) & (pred_data == 2)] = 2
    
    # 4. Write the difference image to file if diff_path is provided
    if diff_path is not None:
        if profile is None:
             raise ValueError(
                "Cannot save to disk: No metadata (profile) available. "
                "Both label and prediction inputs were NumPy arrays, and 'diff_path' was provided."
            )
        
        # Update the profile for the output file
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'nodata': 0, # Setting nodata to 0 is common for this type of mask
        })
        
        try:
            with rasterio.open(diff_path, 'w', **profile) as dst:
                dst.write(diff_data, 1)
        except rasterio.RasterioIOError as e:
            print(f"Error writing difference file {diff_path}: {e}")
            raise
        
    return diff_data
