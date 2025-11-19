import argparse
import rasterio
import rasterio.features
import rasterio.mask
import fiona
import numpy as np
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from rasterio.transform import from_origin

def parse_args():
    parser = argparse.ArgumentParser(description="Count overlapping GeoTIFFs within a shapefile area.")
    parser.add_argument('--geotiff_paths', nargs='+', required=True, help='List of paths to GeoTIFF files.')
    parser.add_argument('--shape_file', required=True, help='Path to shapefile defining area of interest.')
    parser.add_argument('--resolution', type=float, default=0.1, help='Resolution of output raster.')
    parser.add_argument('--output_geotif', required=True, help='Path to output GeoTIFF file.')
    return parser.parse_args()

def get_shape_bounds(shape_file):
    with fiona.open(shape_file, 'r') as shapefile:
        shapes = [shape(feat['geometry']) for feat in shapefile]
        return unary_union(shapes).bounds, unary_union(shapes)

def rasterize_overlap_count(geotiff_paths, bounds, resolution, transform):
    """
    Create a raster array where each pixel value represents the number of GeoTIFFs
    that overlap that location.
    
    This function processes a list of GeoTIFF files and creates an output array
    covering the specified bounds. Each pixel in the output contains a count of
    how many input GeoTIFFs have valid data at that location.
    
    Parameters
    ----------
    geotiff_paths : list of str
        Paths to the GeoTIFF files to process.
    bounds : tuple of float
        Output extent as (minx, miny, maxx, maxy) in the coordinate system
        matching the transform parameter.
    resolution : float
        Pixel size in the same units as the bounds (e.g., degrees or meters).
    transform : affine.Affine
        Affine transformation matrix that maps pixel coordinates to geographic
        coordinates for the output array.
    
    Returns
    -------
    numpy.ndarray
        2D array of shape (height, width) with dtype uint8, where each pixel
        value is the count of overlapping GeoTIFFs (0-255).
    """
    # Calculate output array dimensions from bounds and resolution
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    
    # Initialize count array - uint8 limits max count to 255
    count_array = np.zeros((height, width), dtype=np.uint8)
    
    # Process each GeoTIFF
    for path in geotiff_paths:
        with rasterio.open(path) as src:
            # Create a box geometry from the GeoTIFF's bounds
            t_bounds = box(*src.bounds)
            
            # Generate a binary mask: True where GeoTIFF has data, False elsewhere
            # The mask is automatically clipped to the output extent defined by
            # the transform and out_shape parameters
            mask = rasterio.features.geometry_mask(
                [mapping(t_bounds)],
                transform=transform,
                out_shape=(height, width),
                invert=True  # True = inside geometry, False = outside
            )
            
            # Increment count for pixels covered by this GeoTIFF
            # Convert boolean mask to uint8 (True->1, False->0) before adding
            count_array += mask.astype(np.uint8)
    
    return count_array

def mask_and_save_raster(count_array, transform, shape_geom, output_path):
    out_meta = {
        'driver': 'GTiff',
        'height': count_array.shape[0],
        'width': count_array.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
    }

    # Mask raster by shape
    shapes = [mapping(shape_geom)]
    with rasterio.open('/vsimem/temp.tif', 'w', **out_meta) as tmp:
        tmp.write(count_array, 1)
    with rasterio.open('/vsimem/temp.tif') as tmp:
        out_image, out_transform = rasterio.mask.mask(tmp, shapes, crop=True, filled=True, nodata=0)
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_image)

def main():
    args = parse_args()
    bounds, shape_geom = get_shape_bounds(args.shape_file)
    minx, miny, maxx, maxy = bounds
    resolution = args.resolution
    transform = from_origin(minx, maxy, resolution, resolution)
    
    count_array = rasterize_overlap_count(args.geotiff_paths, shape_geom, bounds, resolution, transform)
    mask_and_save_raster(count_array, transform, shape_geom, args.output_geotif)

if __name__ == '__main__':
    main()
