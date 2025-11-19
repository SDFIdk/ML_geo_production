import geopandas as gpd
import numpy as np
import argparse
import os

def add_attributes(loaded_geopackage, shp_gdf, 
                   atributes_to_copy=['STATUS','GEOMETRISTATUS','ID_LOKALID','REGISTRERINGFRA'], 
                   extra_atributes={"label_description": "2025"}):
    """
    Copies the attributes in atributes_to_copy from a loaded geopackage to polygons in a shapefile GeoDataFrame
    based on spatial overlap. For each polygon in the shapefile, the function finds overlapping
    polygons in the geopackage and copies attributes from the polygon with the largest overlap.
    Also adds some extra attributes in extra_atributes (e.g., label_description).

    Parameters
    ----------
    loaded_geopackage : GeoDataFrame
        Already loaded geopackage with STATUS and GEOMETRISTATUS attributes.
    shp_gdf : GeoDataFrame
        Already loaded shapefile GeoDataFrame that will receive the attributes.
    atributes_to_copy : list of str
        Names of attributes to copy from the geopackage.
    extra_atributes : dict
        Additional attributes to add to the shapefile.

    Returns
    -------
    GeoDataFrame
        The updated shapefile data with added attributes.
    """
    if 'STATUS' not in loaded_geopackage.columns or 'GEOMETRISTATUS' not in loaded_geopackage.columns:
        raise ValueError("The geopackage must contain STATUS and GEOMETRISTATUS attributes")

    # Make sure the attributes to be copied exist in the shapefile GeoDataFrame
    for key in atributes_to_copy:
        if key not in shp_gdf.columns:
            shp_gdf[key] = None

    # Add extra attributes
    for key, value in extra_atributes.items():
        shp_gdf[key] = value

    # Align CRS
    if loaded_geopackage.crs != shp_gdf.crs:
        loaded_geopackage = loaded_geopackage.to_crs(shp_gdf.crs)

    # Perform spatial overlap and copy attributes
    for idx, shp_poly in shp_gdf.iterrows():
        overlaps = loaded_geopackage[loaded_geopackage.geometry.intersects(shp_poly.geometry)]
        if overlaps.empty:
            continue

        intersection_areas = []
        for _, gpkg_poly in overlaps.iterrows():
            try:
                intersection = shp_poly.geometry.intersection(gpkg_poly.geometry)
                intersection_areas.append(intersection.area)
            except Exception as e:
                print(f"Error calculating intersection: {e}")
                intersection_areas.append(0)

        if intersection_areas and max(intersection_areas) > 0:
            max_idx = np.argmax(intersection_areas)
            for key in atributes_to_copy:
                shp_gdf.at[idx, key] = overlaps.iloc[max_idx][key]

    # Add area in square meters rounded to 2 decimals
    shp_gdf["Areal"] = shp_gdf.geometry.area.round(2)

    return shp_gdf


def add_attributes_to_shapefile(loaded_geopackage, shapefile,
                                atributes_to_copy=['STATUS','GEOMETRISTATUS','ID_LOKALID','REGISTRERINGFRA'],
                                extra_atributes={"label_description": "2025"}):
    """
    Loads a shapefile, adds attributes from a loaded geopackage, and saves it back to disk.
    """
    shp_gdf = gpd.read_file(shapefile)
    shp_gdf = add_attributes(loaded_geopackage, shp_gdf,
                             atributes_to_copy=atributes_to_copy,
                             extra_atributes=extra_atributes)
    shp_gdf.to_file(shapefile)
    return shp_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add attributes from a geopackage to a shapefile.")
    parser.add_argument("--geopackage", type=str, required=True, help="Path to the input geopackage")
    parser.add_argument("--shapefile", type=str, required=True, help="Path to the input shapefile")
    parser.add_argument("--label_description", type=str, default="2025", help="label_description to add as an attribute")

    args = parser.parse_args()

    loaded_geopackage = gpd.read_file(args.geopackage)
    add_attributes_to_shapefile(loaded_geopackage, args.shapefile,
                                extra_atributes={"label_description": args.label_description})
