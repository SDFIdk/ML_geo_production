import argparse
import subprocess
import json
import os
import glob
import shutil
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import random
def generate_random_filename_that_does_not_exist(folder_path,postfix=".tif"):
    """
    Generate a path to a file that does not exist
    the file is used for temporary storage and will be deleted after usage
    """
    name = ("".join([str(random.randint(0, 9)) for _i in range(10)])) + postfix
    random_file_name = Path(folder_path) / name
    while random_file_name.is_file():
        name = ("".join([random.randint([0, 9]) for _i in range(10)])) + postfix
        random_file_name = Path(folder_path) / name
    return str(random_file_name)


def run_pdal_translate(input_laz, output_laz):
    if Path(output_laz).is_file():
        print("skipping creation of "+str(output_laz)+" since it already exists!")
        return "allready exists"

    """Run PDAL translate command using a constructed pipeline from a dictionary."""
    pdal_translate_dict = {
        "pipeline": [
            {
                "type": "filters.assign",
                "assignment": "Classification[3:5]=7"
            },
            {
                "type": "filters.assign",
                "assignment": "Classification[6:6]=2"
            },
            {
                "type": "filters.assign",
                "assignment": "Classification[8:9]=2"
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"
            }
        ]
    }

    pdal_translate_pipeline_name =  generate_random_filename_that_does_not_exist(folder_path = "./",postfix=".json")

    with open(pdal_translate_pipeline_name, "w") as f:
        json.dump(pdal_translate_dict, f)

    translate_command = [
        "pdal", "translate",
        "--input", input_laz,
        "--output", output_laz,
        "--json", pdal_translate_pipeline_name
    ]
    print(f"Running PDAL translate: {' '.join(translate_command)}")
    subprocess.run(translate_command, check=True)
    #delete the temporary pipeline file after usage
    Path(pdal_translate_pipeline_name).unlink()

def run_pdal_pipeline(ground_laz, output_tif):
    """Run PDAL pipeline command using a constructed pipeline from a dictionary."""
    pdal_pipeline_dict = {
        "pipeline": [
            ground_laz,
            {
                "filename": output_tif,
                "gdaldriver": "GTiff",
                "output_type": "max",
                "resolution": "0.1",
                "type": "writers.gdal"
            }
        ]
    }
    pdal_pipeline_name =  generate_random_filename_that_does_not_exist(folder_path = "./",postfix=".json")

    with open(pdal_pipeline_name, "w") as f:
        json.dump(pdal_pipeline_dict, f)

    pipeline_command = ["pdal", "pipeline", pdal_pipeline_name]
    print(f"Running PDAL pipeline: {' '.join(pipeline_command)}")
    subprocess.run(pipeline_command, check=True)

def run_gdal_fillnodata(input_tif, output_tif, max_distance=1000, smoothing_iterations=0):
    """Run GDAL fillnodata command."""
    gdal_command = f"gdal_fillnodata -md {max_distance} -si {smoothing_iterations} {input_tif} {output_tif}"
    print(f"Running GDAL fillnodata: {gdal_command}")
    subprocess.run(gdal_command, shell=True, check=True)


def laz_to_DSM_no_veg(input_laz_file,output_DSM_no_veg,tmp_laz=None,tmp_tif=None,skip_creation_if_DSM_no_veg_exists = True,delete_tmp_files=True):

    """Process the .laz file to generate the filled TIFF output."""
    if not tmp_laz:
        tmp_laz = output_DSM_no_veg.replace(".tif","tmp.laz")
    if not tmp_tif:
        tmp_tif = output_DSM_no_veg.replace(".tif","tmp.tif")
    if skip_creation_if_DSM_no_veg_exists and Path(output_DSM_no_veg).is_file():
        print("skipping creation of "+str(output_DSM_no_veg)+" since it already exists!")
    else:
        run_pdal_translate(input_laz_file,tmp_laz)
        run_pdal_pipeline(tmp_laz, tmp_tif)
        run_gdal_fillnodata(tmp_tif,output_DSM_no_veg)
    if delete_tmp_files:
        for tmp_file in [tmp_tif,tmp_laz]:
            Path(tmp_file).unlink()


if __name__ == "__main__":
    example_usage = 'python laz_to_DSM_no_veg.py -h'
    print("#"*100)
    print("Example_usage:")
    print(example_usage)
    print("#"*100)

    parser = argparse.ArgumentParser(description="create DSM_no_veg geotiff files",formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    parser.add_argument("--laz_file", help="Path to  LAZ file ",required=True)
    parser.add_argument("--DSM_no_veg_tif",  help="Path to the DSM_no_veg.tif file to create",required=True)

    args = parser.parse_args()


    laz_to_DSM_no_veg(input_laz_file=args.laz_file,output_DSM_no_veg=args.DSM_no_veg_tif)
