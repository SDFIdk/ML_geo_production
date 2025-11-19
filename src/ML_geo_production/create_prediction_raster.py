import argparse
from pathlib import Path
import geopandas as gpd
import json
from ML_geo_production import create_diff_polygons, model_utils, process_images, processing_utils
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ensemble model inference using a JSON configuration file and direct image paths."
    )
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON configuration file.")
    return parser.parse_args()



def init():
    """Initialization: loads JSON config and preloads model weights"""
    args = parse_arguments()

    with open(args.json, 'r') as f:
        parsed_json = json.load(f)

    print("Preloading all model weights into main memory...")
    parsed_json["model_states"] = model_utils.preload_model_states(parsed_json["saved_models"])

    print("Initialization complete.")
    return parsed_json


def main():
    #Preload all model weights into main memory and store weights in parsed_json["model_states"]
    parsed_json = init()

    # predict using ML model ensamble
    final_probability_array, final_transform, dst_crs = process_images.process_images_from_dict(parsed_json)

    #save probabilities to disk ?
    # not nececeary for creating prediction raster but convinient for debugging
    if parsed_json.get("save_probs"):
        # Save the array to disk
        processing_utils.save_output_data(final_probability_array, final_transform, dst_crs, parsed_json["probs_path"])
        print(f"Saved output probs GeoTIFF to {parsed_json['probs_path']}")
    #save predictions to disk ?
    #when creating prediction raster the value for 'save_preds' in the json should always be true
    assert(parsed_json.get("save_preds"))
    if parsed_json.get("save_preds"):
        processing_utils.save_predictions_data(final_probability_array.argmax(axis=0), final_transform, dst_crs, parsed_json["preds_path"])
        print(f"Saved output predictions GeoTIFF to {parsed_json['preds_path']}")




if __name__ == "__main__":
    main()
