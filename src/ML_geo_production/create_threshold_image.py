#!/usr/bin/env python3
"""
A script to convert a float16 GeoTIFF to a binary uint8 GeoTIFF based on a threshold value.
All pixels above the threshold will be set to 255, and all others to 0.
"""

import argparse
import numpy as np
import rasterio
from rasterio.enums import Resampling


def threshold_geotiff(input_path, output_path, threshold_value):
    """
    Applies a threshold to a GeoTIFF and creates a new binary uint8 GeoTIFF.
    
    Args:
        input_path (str): Path to the input GeoTIFF file
        output_path (str): Path to save the output GeoTIFF file
        threshold_value (float): Threshold value to apply
    """
    # Open the input GeoTIFF
    with rasterio.open(input_path) as src:
        # Read the data
        data = src.read(3)  # Assuming single band data
        
        # Apply threshold
        binary_data = np.zeros_like(data, dtype=np.uint8)
        binary_data[data > threshold_value] = 255
        
        # Prepare metadata for the output file
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            nodata=None
        )
        
        # Write the output GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(binary_data, 1)
            
    print(f"Created binary GeoTIFF at {output_path}")
    print(f"Pixels above {threshold_value} are set to 255, others to 0")


def main():
    parser = argparse.ArgumentParser(description='Convert a float16 GeoTIFF to a binary uint8 GeoTIFF based on a threshold.')
    parser.add_argument('input_tif', help='Path to the input GeoTIFF file')
    parser.add_argument('--threshold', type=float, required=True, 
                        help='Threshold value (e.g., 1.000e-21 or 0.001)')
    parser.add_argument('--output_tif', required=True, 
                        help='Path to save the output binary GeoTIFF file')
    
    args = parser.parse_args()
    print(args.threshold)
    print(args.threshold<0.0000001)


    
    threshold_geotiff(args.input_tif, args.output_tif, args.threshold)


if __name__ == "__main__":
    main()
