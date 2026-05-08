import argparse
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import os
import glob
import pathlib


def _geom_valid_non_empty_positive_area(geom) -> bool:
    return geom.is_valid and not geom.is_empty and geom.area > 0


def _write_debug_shapefile(features, crs, output_path):
    if not features:
        return
    gdf = gpd.GeoDataFrame.from_features(features)
    gdf = gdf.set_geometry("geometry")
    gdf = gdf[gdf.is_valid]
    if gdf.empty:
        return
    gdf.crs = crs
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path)
    print(f"Saved polygon debug layer to {output_path}")


def polygonize_raster_data(
    image,
    transform,
    crs,
    buffer_size,
    *,
    save_original_polygons=False,
    save_buffer_in_polygons=False,
    save_buffer_in_buffer_out_polygons=False,
    polygon_debug_output_folder=None,
):
    """Polygonize a raster numpy array and return a GeoDataFrame."""
    want_save = (
        save_original_polygons
        or save_buffer_in_polygons
        or save_buffer_in_buffer_out_polygons
    )
    if want_save and not polygon_debug_output_folder:
        raise ValueError(
            "polygon_debug_output_folder is required when any save_original_polygons, "
            "save_buffer_in_polygons, or save_buffer_in_buffer_out_polygons is True."
        )

    out_dir = pathlib.Path(polygon_debug_output_folder) if want_save else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    mask = image != 0
    pre_buffer_features = []
    buffer_in_features = []
    buffer_in_out_features = []

    if buffer_size:
        print(f"Using buffer of size {buffer_size}")
        polygons = []
        for s, v in shapes(image, mask=mask, transform=transform):
            geom0 = shape(s)
            if (
                save_original_polygons
                and _geom_valid_non_empty_positive_area(geom0)
            ):
                pre_buffer_features.append(
                    {"properties": {"value": v}, "geometry": geom0}
                )

            shrinked_grown = geom0.buffer(-buffer_size)
            if _geom_valid_non_empty_positive_area(shrinked_grown):
                if save_buffer_in_polygons:
                    buffer_in_features.append(
                        {"properties": {"value": v}, "geometry": shrinked_grown}
                    )
                grown = shrinked_grown.buffer(buffer_size)
                polygons.append(
                    {"properties": {"value": v}, "geometry": grown}
                )
                if save_buffer_in_buffer_out_polygons:
                    buffer_in_out_features.append(
                        {"properties": {"value": v}, "geometry": grown}
                    )
    else:
        print("No buffer")
        if save_buffer_in_polygons or save_buffer_in_buffer_out_polygons:
            print(
                "save_buffer_in_polygons / save_buffer_in_buffer_out_polygons: "
                "no buffer stage; skipping those writes"
            )
        polygons = []
        for s, v in shapes(image, mask=mask, transform=transform):
            geom = shape(s)
            polygons.append({"properties": {"value": v}, "geometry": geom})
            if (
                save_original_polygons
                and _geom_valid_non_empty_positive_area(geom)
            ):
                pre_buffer_features.append(
                    {"properties": {"value": v}, "geometry": geom}
                )

    if out_dir is not None:
        if save_original_polygons:
            _write_debug_shapefile(
                pre_buffer_features,
                crs,
                out_dir / "valid_pre_buffer_polyogons.shp",
            )
        if buffer_size and save_buffer_in_polygons:
            _write_debug_shapefile(
                buffer_in_features,
                crs,
                out_dir / "valid_buffer_in_polygons.shp",
            )
        if buffer_size and save_buffer_in_buffer_out_polygons:
            _write_debug_shapefile(
                buffer_in_out_features,
                crs,
                out_dir / "valid_buffer_in_buffer_out_polyogons.shp",
            )

    # Convert to GeoDataFrame
    if not polygons:
        print("No valid geometries found!")
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame.from_features(polygons)
        gdf = gdf.set_geometry("geometry")
        gdf = gdf[gdf.is_valid]
        valid_count = gdf.is_valid.sum()
        print(f"Number of valid geometries: {valid_count}")
        gdf.crs = crs

    return gdf


def polygonize_raster_file(input_raster, output_shapefile, buffer_size):
    """Open raster file, polygonize, and save shapefile."""
    with rasterio.open(input_raster) as src:
        image = src.read(1)
        gdf = polygonize_raster_data(image, src.transform, src.crs, buffer_size)

    pathlib.Path(output_shapefile).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shapefile)
    print(f"Polygonization complete, output saved to {output_shapefile}.")


def process_files(input_pattern, output_folder, buffer_size):
    input_files = glob.glob(input_pattern)
    if not input_files:
        print(f"No files found for pattern: {input_pattern}")
        return

    for input_file in input_files:
        base_filename = os.path.basename(input_file)
        output_shapefile = os.path.join(output_folder, os.path.splitext(base_filename)[0] + '.shp')
        if pathlib.Path(output_shapefile).exists():
            print(f"Skipping existing file: {output_shapefile}")
        else:
            print(f"Processing {input_file} -> {output_shapefile}")
            try:
                polygonize_raster_file(input_file, output_shapefile, buffer_size)
            except Exception as e:
                print("#" * 10)
                print(f"Failed with exception: {e}")
                print("#" * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polygonize GeoTIFF raster(s) to shapefile(s).")
    parser.add_argument('--input', type=str, help="Path to the input GeoTIFF raster file or pattern.")
    parser.add_argument('--output', type=str, help="Path to the output shapefile (.shp) or output folder if input is a pattern.")
    parser.add_argument('--buffer_size', type=float, default=1.5, help="Buffer size for shrinking/growing geometries.")

    args = parser.parse_args()

    if '*' in args.input or '?' in args.input:
        process_files(args.input, args.output, args.buffer_size)
    else:
        polygonize_raster_file(args.input, args.output, args.buffer_size)
