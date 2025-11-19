import torch
import rasterio
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window
from rasterio.transform import from_origin
from torch.utils.data._utils.collate import default_collate
from affine import Affine

def save_input_data(single_image, transform, crs, filename):
    """Save image data as a GeoTIFF with geospatial metadata."""
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=single_image.shape[1],
        width=single_image.shape[2],
        count=single_image.shape[0],
        dtype=single_image.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(single_image)

class LargeImageDataset(Dataset):
    def __init__(self, image_paths, label_paths=None, patch_size=1000, overlap=40):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.image_metadata = self._compute_image_metadata()
        self.patches = self._compute_patches()
    
    def _compute_image_metadata(self):
        metadata = []
        with rasterio.Env():
            for path in self.image_paths:
                with rasterio.open(path) as src:
                    metadata.append({
                        "width": src.width,
                        "height": src.height,
                        "transform": src.transform,
                        "crs": src.crs,
                        "filename": os.path.basename(path)
                    })
        return metadata
    
    def _compute_patches(self):
        patches = []
        for img_idx, meta in enumerate(self.image_metadata):
            width, height = meta["width"], meta["height"]
            for y in range(0, height, self.stride):
                if y + self.patch_size > height:
                    y = height - self.patch_size
                for x in range(0, width, self.stride):
                    if x + self.patch_size > width:
                        x = width - self.patch_size
                    patches.append((img_idx, x, y))
        return patches
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        img_idx, x, y = self.patches[index]
        meta = self.image_metadata[img_idx]
        
        with rasterio.open(self.image_paths[img_idx]) as src:
            window = Window(x, y, self.patch_size, self.patch_size)
            image_patch = src.read(window=window, boundless=True, fill_value=0).astype(np.uint8)  # Convert to uint8
        
        # Corrected transform calculation preserving rotation/shear
        new_transform = meta["transform"] * Affine.translation(x, y)
    
        data = {
            "image": torch.tensor(image_patch, dtype=torch.uint8),  # Store as uint8
            "transform": new_transform,  # Updated transform
            "crs": meta["crs"],
            "filename": meta["filename"],
            "x": x,
            "y": y
        }
        
        if self.label_paths:
            with rasterio.open(self.label_paths[img_idx]) as src:
                label_patch = src.read(window=window, boundless=True, fill_value=0).astype(np.uint8)  # Convert to uint8
            data["label"] = torch.tensor(label_patch, dtype=torch.uint8)  # Store as uint8
        
        return data




def custom_collate_fn(batch):
    batch_dict = {}
    
    for key in batch[0]:  # Get all keys from the first batch item
        if isinstance(batch[0][key], torch.Tensor) or isinstance(batch[0][key], np.ndarray):
            batch_dict[key] = default_collate([item[key] for item in batch])
        else:
            batch_dict[key] = [item[key] for item in batch]  # Store as a list
    
    return batch_dict

def get_dataloader(image_paths, label_paths=None, batch_size=8, num_workers=4):
    dataset = LargeImageDataset(image_paths, label_paths)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(args.path_to_images) if f.endswith(".tif")])
    image_paths = [os.path.join(args.path_to_images, f) for f in image_files]
    label_paths = None
    
    if args.path_to_labels:
        label_files = sorted([f for f in os.listdir(args.path_to_labels) if f.endswith(".tif")])
        label_paths = [os.path.join(args.path_to_labels, f) for f in label_files]
        
        if args.remove_matching_label:
            matched_files = set(image_files) & set(label_files)
            image_paths = [os.path.join(args.path_to_images, f) for f in matched_files]
            label_paths = [os.path.join(args.path_to_labels, f) for f in matched_files]
    
    dataloader = get_dataloader(image_paths, label_paths)
    
    for batch_idx, batch in enumerate(dataloader):
        for i in range(batch["image"].shape[0]):
            base_name = os.path.splitext(batch["filename"][i])[0]
            output_filename = os.path.join(
                args.output_folder,
                f"{base_name}_patchy_{batch['y'][i]}_patchx_{batch['x'][i]}.tif"
            )
            save_input_data(
                batch["image"][i].numpy(),
                batch["transform"][i],
                batch["crs"][i],
                output_filename
            )
        print("Saved batch", batch_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on large images with optional labels")
    parser.add_argument("--path_to_images", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--path_to_labels", type=str, default=None, help="Path to directory containing labels (optional)")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to directory where output patches will be saved")
    parser.add_argument("--remove_matching_label", action="store_true", help="Remove images without matching labels and vice versa")
    args = parser.parse_args()
    
    main(args)
