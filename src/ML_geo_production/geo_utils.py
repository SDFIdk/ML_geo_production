# geo_utils.py
import os
from shapely.geometry import box
import math
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
import fiona

def normalize_tile_id_name(tile_id):
    """
    Tries to parse as a Windows path (handles '\') and falls back to
    the current OS's default Path (handles '/').
    """
    # Check if the path contains Windows separators or a drive letter
    if '\\' in tile_id or ':' in tile_id:
        # Use PureWindowsPath to correctly parse the Windows-style path
        path_obj = PureWindowsPath(tile_id)
    else:
        # Use default Path for Linux/Unix paths
        path_obj = Path(tile_id)

    # This will correctly handle the stem extraction for the final part of the path
    # regardless of whether the path was successfully parsed as Windows or Linux
    return path_obj.stem


def create_shapefile(shapefile_path,bounding_box,crs):
    bounding_box = box(*bounding_box)
    # Create a new GeoDataFrame with the bounding box as a single feature
    bounding_box_gdf = gpd.GeoDataFrame(geometry=[bounding_box], crs=crs)

    # Write the new GeoDataFrame to a new shapefile
    bounding_box_gdf.to_file(shapefile_path)

    print(f"Bounding box shapefile saved as {shapefile_path}")


def open_shapefile(shapefile_path):
    print(f"Opening shapefile: {shapefile_path}")
    # Open the shapefile using Fiona
    with fiona.open(shapefile_path, 'r') as shapefile:
        ids_and_bounding_boxes = []
        # Iterate over each feature in the shapefile
        for feature in shapefile:
            # Print available property names if 'tileid' is missing
            if 'tileid' not in feature['properties']:
                if 'df_name' not in feature['properties']:
                    #print("Available properties in feature:", str([key for key in feature['properties'].keys()]))
                    id_type = [key for key in feature['properties'].keys()][0] #take the first property
                else:
                    id_type = 'df_name'
            else:
                id_type = 'tileid'

            # Extract the bounding box from the feature's geometry
            geom = feature['geometry']
            bbox = fiona.bounds(geom)
            #print("bbox:"+str(bbox))
            #bounding_boxes[feature["tileid"]]=bbox
            try:
                name = str(feature['properties'][id_type])
                ids_and_bounding_boxes.append((name,bbox))
            except Exception as e:
                print(feature['properties'])
                print("was not able to read a tileid form the .shp file ")
                print("got this exception:"+str(e))
                print("using the .shp files name "+str(Path(shapefile_path).name.replace(".shp",""))+"instead")
                name = Path(shapefile_path).name.replace(".shp","")
                ids_and_bounding_boxes.append((name,bbox))
            #print("using name: "+str(name))

        #if a id is made up of a path, we change it to the filename
        return [(normalize_tile_id_name(id), bbox) for (id,bbox) in ids_and_bounding_boxes]

def get_image_bounds(path_to_images):
    bounds_dict = {}
    for img_path in sorted(os.listdir(path_to_images)):
        if img_path.endswith(".tif"):
            full_path = os.path.join(path_to_images, img_path)
            with rasterio.open(full_path) as src:
                bounds_dict[full_path] = src.bounds
    return bounds_dict

def filter_images_by_shapefile(path_to_images, shape_file,bounds_dict=None):
    """
    shape_file : str
    Path to the shapefile
    bounds_dict: dictionary
    preloaded bounds for the filepaths
    """
    # Load the shapefile and compute bounds
    gdf = gpd.read_file(shape_file)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    return filter_images_by_bounds(path_to_images, bounds,bounds_dict=None)

def filter_images_by_bounds(path_to_images, bounds,bounds_dict=None):
    """
    Filter images that overlap with the given shapefile.
    
    Parameters:
    -----------
    path_to_images : str
        Path to directory containing images
    bounds : ()
        bounding box describing the area we are interested in
    bounds_dict: dictionary
        preloaded bounds for the filepaths
        when this is present we only check the files in this dictionary instead of checking the files in the directory
    Returns:
    --------
    list
        List of paths to images that overlap with shapefile
    tuple
        Bounds of the shapefile (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    
    # Filter images that overlap with shapefile bounds
    valid_image_paths = []
    if bounds_dict:
        #check what files in the dictionary are overlapping with the .shp
        for full_path in bounds_dict:
            img_bounds = bounds_dict[full_path]
            if not (img_bounds.right < minx or img_bounds.left > maxx or
                img_bounds.top < miny or img_bounds.bottom > maxy):
                valid_image_paths.append(full_path)


    else:
        #check what files in the directory are overlapping with .shp
        for img_path in sorted(os.listdir(path_to_images)):
            print("img_path"+str(img_path))
            if img_path.endswith(".tif"):
                full_path = os.path.join(path_to_images, img_path)
                print("loading bounds from image instad of using cashed bounds... ")
                with rasterio.open(full_path) as src:
                    img_bounds = src.bounds
            if not (img_bounds.right < minx or img_bounds.left > maxx or
                    img_bounds.top < miny or img_bounds.bottom > maxy):
                    valid_image_paths.append(full_path)
    
    return valid_image_paths, bounds


def compute_target_dimensions(bounds, resolution, pixel_buffer=0):
    """
    Compute target dimensions for the output array.
    
    Parameters:
    -----------
    bounds : tuple
        (minx, miny, maxx, maxy) bounds
    resolution : float
        Resolution in map units per pixel
    pixel_buffer : float
        Buffer size in map units
        
    Returns:
    --------
    int
        Target width in pixels
    int
        Target height in pixels
    object
        Transform for the target array
    """
    minx, miny, maxx, maxy = bounds
    
    # Compute dimensions
    target_width = math.ceil((maxx - minx + 2 * pixel_buffer) / resolution)
    target_height = math.ceil((maxy - miny + 2 * pixel_buffer) / resolution)
    
    # Define transform
    transform = from_origin(minx - pixel_buffer, maxy + pixel_buffer, resolution, resolution)
    
    return target_width, target_height, transform
