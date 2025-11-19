# image_utils.py
import numpy as np
import rasterio
from rasterio.windows import Window
from fastai.vision.all import PILImage, PILMask

def load_central_window(fn, window_size=1000):
    """
    Uses rasterio to load only the central window from the image.
    
    Parameters:
    -----------
    fn : str
        Path to the image file
    window_size : int
        Size of the window to extract
        
    Returns:
    --------
    PILImage
        The extracted window as a FastAI PILImage
    """
    with rasterio.open(fn) as src:
        width, height = src.width, src.height
        # Compute the top-left corner of the central window.
        col_off = max((width - window_size) // 2, 0)
        row_off = max((height - window_size) // 2, 0)
        window = Window(col_off, row_off, window_size, window_size)
        # Read only the window from the file.
        arr = src.read(window=window)
        # Rearrange array dimensions: (channels, H, W) -> (H, W, channels)
        arr = np.moveaxis(arr, 0, -1)
        # If not already uint8, normalize and convert.
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            # Normalize to 0-255 based on the image's range.
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255
            arr = arr.astype(np.uint8)
    return PILImage.create(arr)


def load_dummy_mask(fn, window_size=1000):
    """
    Creates a dummy mask of zeros with dimensions window_size x window_size.
    
    Parameters:
    -----------
    fn : str
        Path to the image file (unused, kept for compatibility)
    window_size : int
        Size of the mask to create
        
    Returns:
    --------
    PILMask
        A dummy mask as a FastAI PILMask
    """
    mask_arr = np.zeros((window_size, window_size), dtype=np.uint8)
    return PILMask.create(mask_arr)
