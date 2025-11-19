import argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, mapping
import fiona
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Resample and georeference a non-georeferenced TIFF, and create footprint shapefile."
    )
    parser.add_argument("--input_tif", type=str, required=True, help="Path to input TIFF image")
    parser.add_argument("--input_resolution", type=float, default=0.25,
                        help="Input resolution in meters per pixel (default: 0.25, only used if not georeferenced)")
    parser.add_argument("--output_resolution", type=float, default=0.1,
                        help="Output resolution in meters per pixel (default: 0.1)")
    parser.add_argument("--CRS", type=str, default="EPSG:25832",
                        help="Output CRS (default: EPSG:25832, only used if input not georeferenced)")
    parser.add_argument("--output_tif", type=str, default="output/output_resampled.tif",
                        help="Path to output GeoTIFF")
    parser.add_argument("--output_shp", type=str, default="output/output_footprint.shp",
                        help="Path to output footprint shapefile")

    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.output_tif).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_shp).parent.mkdir(parents=True, exist_ok=True)

    # Open the input image and check if it's georeferenced
    with rasterio.open(args.input_tif) as src:
        data = src.read()
        count, height, width = data.shape
        
        # Check if input is already georeferenced
        is_georeferenced = src.crs is not None and src.transform != rasterio.Affine.identity()
        
        if is_georeferenced:
            print(f"✓ Input is already georeferenced")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            transform_in = src.transform
            crs_in = src.crs
            # Calculate input resolution from transform
            input_res = abs(src.transform[0])
            print(f"  Input resolution: {input_res:.4f} m/px")
        else:
            print(f"⚠ Input is NOT georeferenced, using provided parameters")
            print(f"  Assuming resolution: {args.input_resolution} m/px")
            print(f"  Assuming CRS: {args.CRS}")
            transform_in = from_origin(0, 0, args.input_resolution, args.input_resolution)
            crs_in = args.CRS
            input_res = args.input_resolution

    # Compute output dimensions
    scale = input_res / args.output_resolution
    out_height = int(height * scale)
    out_width = int(width * scale)

    # Define output transform
    if is_georeferenced:
        # Use the same origin as input, but with new resolution
        origin_x = transform_in.c
        origin_y = transform_in.f
        transform_out = from_origin(origin_x, origin_y, args.output_resolution, args.output_resolution)
        crs_out = crs_in  # Keep the same CRS
    else:
        transform_out = from_origin(0, 0, args.output_resolution, args.output_resolution)
        crs_out = args.CRS

    # Prepare output array
    dst_data = np.empty((count, out_height, out_width), dtype=data.dtype)

    # Reproject & resample
    for b in range(count):
        reproject(
            source=data[b],
            destination=dst_data[b],
            src_transform=transform_in,
            src_crs=crs_in,
            dst_transform=transform_out,
            dst_crs=crs_out,
            resampling=Resampling.bilinear
        )

    # Save as GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "count": count,
        "dtype": dst_data.dtype,
        "crs": crs_out,
        "transform": transform_out
    }

    with rasterio.open(args.output_tif, "w", **profile) as dst:
        dst.write(dst_data)

    print(f"✅ Resampled GeoTIFF saved to {args.output_tif}")
    print(f"   Size: {out_width} x {out_height} px, Resolution: {args.output_resolution} m/px")
    print(f"   CRS: {crs_out}")

    # ---- Create footprint shapefile ----
    bounds = array_bounds(out_height, out_width, transform_out)  # (miny, maxy, minx, maxx)
    miny, maxy, minx, maxx = bounds
    footprint = box(minx, miny, maxx, maxy)

    schema = {"geometry": "Polygon", "properties": {}}

    with fiona.open(
        args.output_shp,
        "w",
        driver="ESRI Shapefile",
        crs=rasterio.crs.CRS.from_string(str(crs_out)).to_dict(),
        schema=schema
    ) as shp:
        shp.write({"geometry": mapping(footprint), "properties": {}})

    print(f"✅ Footprint shapefile saved to {args.output_shp}")


if __name__ == "__main__":
    main()
