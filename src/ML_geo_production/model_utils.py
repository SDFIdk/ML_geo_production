#model_utils.py
import sys
from pathlib import Path
import torch
from fastai.vision.all import *
from ML_geo_production.image_utils import load_central_window, load_dummy_mask
from wwf.vision.timm import *
import time

from typing import List, Dict, Any, Union

def preload_model_states(model_paths: List[str]) -> List[Union[Dict[str, Any], Any]]:
    """
    Loads model state dictionaries from a list of file paths into main memory.
    
    This function uses memory-mapping (mmap=True) to potentially speed up 
    the disk read operation by leveraging the OS's Page Cache.

    Parameters:
    -----------
    model_paths : List[str]
        A list of full file paths to the model weight files (.pth).

    Returns:
    --------
    List[Union[Dict[str, Any], Any]]
        A list of the loaded state dictionaries (model weights).
    """
    
    preloaded_states = []
    
    print("--- Starting Model State Pre-loading ---")
    
    for i, path in enumerate(model_paths):
        print(f"Loading state {i+1}/{len(model_paths)} from disk: '{path}'...")
        
        loading_state_start = time.time()
        
        try:
            # Load the state dictionary using mmap=True for efficient disk access.
            # We map it to 'cpu' as it only needs to be in main memory (RAM) 
            # at this stage, not yet GPU memory (VRAM).
            state = torch.load(
                path, 
                map_location='cpu', 
                weights_only=False, 
                mmap=True
            )
            preloaded_states.append(state)
            
            load_time = time.time() - loading_state_start
            print(f"  -> Load successful. Took: {load_time:.4f} seconds.")
            
        except Exception as e:
            print(f"  -> ERROR loading state from {path}: {e}")
            # Optionally raise the error or continue, depending on your error handling needs
            
    print("--- Finished Model State Pre-loading ---")
    
    return preloaded_states


def create_dummy_dls(folder, size=224, bs=1, n_classes=3):
    """
    Creates a dummy DataLoaders from .tif images in the given folder.
    A dummy mask (all zeros) is provided for each image.
    This version uses rasterio to load only the central 1000 x 1000 pixels.
    
    Parameters:
    -----------
    folder : str
        Path to the folder containing .tif images
    size : int
        Size to resize images to
    bs : int
        Batch size
    n_classes : int
        Number of classes
        
    Returns:
    --------
    DataLoaders
        FastAI DataLoaders object
    """
    print("Creating dummy_dls")
    folder = Path(folder)
    files = list(folder.glob('*.tif'))
    max_files = 2
    files = files[:max_files]  # Limit to first 'max_files' images

    if not files:
        sys.exit(f"No .tif image files found in folder: {folder}")

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    print("Creating Datablock")

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=list(range(n_classes)))),
        get_items=lambda folder: list(Path(folder).glob('*.tif')),
        splitter=FuncSplitter(lambda o: False),  # All items go to training set.
        get_x=lambda o: load_central_window(o, window_size=1000),
        get_y=lambda o: load_dummy_mask(o, window_size=1000),
        batch_tfms=[Normalize.from_stats(means, stds)]
    )

    return dblock.dataloaders(folder, bs=bs, size=size)


def load_unet_from_state(model_state, model_name, input_folder, n_classes=3, n_in=3, device="cuda"):
    """
    Loads a saved FastAI U-Net model from a pre-loaded state dictionary 
    and transfers it to the specified device (e.g., GPU).

    Parameters:
    -----------
    model_state : dict
        The pre-loaded state dictionary (weights) of the model.
        This dict comes from an earlier torch.load() call (e.g., state = torch.load('model.pth')).
    model_name : str
        The name of the backbone architecture (e.g., "resnet34").
    input_folder : str
        Path to folder with sample images (used to create the dummy DataLoader).
    n_classes : int
        Number of output classes.
    n_in : int
        Number of input channels.
    device : str
        Device to load the model on ("cuda" or "cpu").

    Returns:
    --------
    Learner
        FastAI Learner object with loaded model.
    """
    
    # 1. Create Learner (Architecture Definition)
    dls = create_dummy_dls(input_folder)
    print("Creating learner architecture")
    
    if model_name == "resnet34":
        learn = unet_learner(dls, resnet34, n_out=n_classes, pretrained=False)
    elif model_name == "resnet50":
        learn = unet_learner(dls, resnet50, n_out=n_classes, pretrained=False)
    else:
        learn = load_saved_timm_unet(dls, model_name, n_classes=n_classes, n_in=n_in)

    # 2. Load Weights (The Fast Step)
    print("Loading state dictionary into model")
    loading_state_start = time.time()
    
    # Extract the actual model state if it's nested in the FastAI dict structure
    state = model_state
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    
    # Load the weights into the model
    learn.model.load_state_dict(state)
    
    # NOTE: The weights are now in main memory, attached to the model object.
    
    print("loading state dictionary took : " + str(time.time() - loading_state_start))
    
    # 3. Move Model to Device (GPU)
    print(f"Transferring model to {device}")
    learn.model.to(device)
    learn.model.eval()
    
    print("Returning learner")
    return learn

def load_saved_timm_unet(dls,model_name,bottleneck="conv",n_classes=2,n_in=3):
    a_loss_func= CrossEntropyLossFlat(axis=1,ignore_index=0)
    learn = timm_unet_learner(dls, model_name, bottleneck=bottleneck ,pretrained=False ,n_in=n_in,n_out=n_classes)
    return learn 
