import os
import shapefile  # pyshp
from osgeo import gdal
from shapely.geometry import box
from pyproj import CRS
import argparse

def crate_extents_shp(geotiffs, outputpath):
    # Create shapefile writer with polygon type
    shp_writer = shapefile.Writer(outputpath, shapeType=shapefile.POLYGON)
    shp_writer.field('filename', 'C')

    for path in geotiffs:
        ds = gdal.Open(path)
        if not ds:
            print(f"Could not open {path}")
            continue

        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize

        # Calculate bounds
        minx = gt[0]
        maxx = gt[0] + width * gt[1]
        miny = gt[3] + height * gt[5]
        maxy = gt[3]

        # Create rectangle polygon (clockwise)
        rect = [[
            [minx, miny],
            [minx, maxy],
            [maxx, maxy],
            [maxx, miny],
            [minx, miny]
        ]]
        shp_writer.poly(rect)
        shp_writer.record(filename=os.path.basename(path))

        ds = None  # close file

    shp_writer.close()

    # Create projection file (.prj) using EPSG code from the first raster
    srs = osr_from_gdal_ds(gdal.Open(geotiffs[0]))
    if srs:
        with open(f"{outputpath}.prj", "w") as prj_file:
            prj_file.write(srs.ExportToWkt())

def osr_from_gdal_ds(ds):
    """Extract spatial reference from a GDAL dataset and return as osr.SpatialReference"""
    from osgeo import osr
    proj = ds.GetProjection()
    if not proj:
        return None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    return srs

# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--geotiffs', nargs='+', required=True, help="List of GeoTIFF file paths")
    parser.add_argument('--outputpath', required=True, help="Path to output shapefile (without .shp)")
    args = parser.parse_args()

    crate_extents_shp(args.geotiffs, args.outputpath)
