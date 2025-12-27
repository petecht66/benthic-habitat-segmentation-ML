import os
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create Habitat dataset class
# attempts to address the three main problems in the assignment prompt
# contains LLM-generated code

class HabitatDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode='patch', patch_size=256, 
                 context_width=None, samples_per_image=50, transform=None,
                 band_stats=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.patch_size = patch_size
        self.context_width = context_width if context_width else patch_size
        self.samples_per_image = samples_per_image
        self.transform = transform
        self.band_stats = band_stats
        
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        
        # Compute padding for context
        self.pad_width = (self.context_width - patch_size) // 2
        
        # Build index for patch/tile mode
        if mode in ['patch', 'tile']:
            self.samples = self._build_sample_index()
        
        # Compute normalization stats if not provided
        if self.band_stats is None:
            print("Computing band statistics for normalization...")
            self.band_stats = self._compute_band_stats()
    
    def _build_sample_index(self):
        """Build index of all patches/tiles for efficient sampling."""
        samples = []
        
        for img_name in self.images:
            img_path = os.path.join(self.img_dir, img_name)
            
            with rio.open(img_path) as src:
                height, width = src.height, src.width
            
            if self.mode == 'patch':
                # Random patches - store image name and sample count
                for _ in range(self.samples_per_image):
                    samples.append({
                        'img_name': img_name,
                        'height': height,
                        'width': width,
                        'random': True
                    })
            
            elif self.mode == 'tile':
                # Non-overlapping tiles
                for i in range(0, height - self.patch_size + 1, self.patch_size):
                    for j in range(0, width - self.patch_size + 1, self.patch_size):
                        samples.append({
                            'img_name': img_name,
                            'i': i,
                            'j': j,
                            'random': False
                        })
        
        return samples
    
    def _compute_band_stats(self):
        """Compute mean and std for each spectral band across dataset."""
        n_samples = min(len(self.images), 10)  # Sample up to 10 images
        band_values = []
        
        for img_name in self.images[:n_samples]:
            img_path = os.path.join(self.img_dir, img_name)
            with rio.open(img_path) as src:
                img = src.read().astype('float32')
                # Reshape to (bands, n_pixels)
                img_flat = img.reshape(img.shape[0], -1)
                band_values.append(img_flat)
        
        # Concatenate all samples
        all_bands = np.concatenate(band_values, axis=1)
        
        # Compute statistics per band
        mean = np.mean(all_bands, axis=1)
        std = np.std(all_bands, axis=1)
        
        print(f"Band means: {mean}")
        print(f"Band stds: {std}")
        
        return {'mean': mean, 'std': std}
    
    def _normalize(self, img):
        """Normalize image using precomputed statistics."""
        mean = self.band_stats['mean'].reshape(-1, 1, 1)
        std = self.band_stats['std'].reshape(-1, 1, 1)
        return (img - mean) / (std + 1e-8)
    
    def __len__(self):
        if self.mode in ['patch', 'tile']:
            return len(self.samples)
        else:
            return len(self.images)
    
    def __getitem__(self, idx):
        if self.mode in ['patch', 'tile']:
            return self._get_patch_item(idx)
        else:
            return self._get_resize_item(idx)
    
    def _get_patch_item(self, idx):
        """Get a patch or tile from an image."""
        sample = self.samples[idx]
        img_name = sample['img_name']
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load images
        with rio.open(img_path) as src:
            img = src.read().astype('float32')
        
        with rio.open(mask_path) as src:
            mask = src.read(1).astype('int64')
        
        # Pad image for context if needed
        if self.pad_width > 0:
            img = np.pad(img, pad_width=((0, 0), (self.pad_width, self.pad_width), 
                                        (self.pad_width, self.pad_width)), mode='edge')
        
        # Extract patch
        if sample['random']:
            # Random location
            max_i = mask.shape[0] - self.patch_size
            max_j = mask.shape[1] - self.patch_size
            i = np.random.randint(0, max_i + 1)
            j = np.random.randint(0, max_j + 1)
        else:
            # Fixed tile location
            i, j = sample['i'], sample['j']
        
        # Extract patches (account for padding in image)
        img_patch = img[:, i:i+self.context_width, j:j+self.context_width]
        mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
        
        # Normalize
        img_patch = self._normalize(img_patch)
        
        # Transpose to (H, W, C) for augmentation
        img_patch = np.transpose(img_patch, (1, 2, 0))
        
        # Apply transforms
        if self.transform:
            aug = self.transform(image=img_patch, mask=mask_patch)
            img_patch, mask_patch = aug['image'], aug['mask']
        
        return img_patch, mask_patch
    
    def _get_resize_item(self, idx):
        """Get a resized full image."""
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load images
        with rio.open(img_path) as src:
            img = src.read().astype('float32')
        
        with rio.open(mask_path) as src:
            mask = src.read(1).astype('int64')
        
        # Normalize
        img = self._normalize(img)
        
        # Transpose to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        
        # Apply transforms (which should include resizing)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        
        return img, mask
    
    @staticmethod
    def get_class_weights(device='cpu'):
        # Return predefined class weights for handling class imbalance.
        import torch
        
        class_weights = torch.tensor([
            1/0.035, 1/0.0306, 1/0.0765, 1/0.1191,
            1/0.086, 1/0.3789, 1/0.2741
        ], dtype=torch.float32).to(device)
        
        return class_weights


def prepare_dataset(img_dir, mask_dir, mode='patch', patch_size=256, 
                   context_width=None, samples_per_image=50, transform=None):
    
     # Convenience function to prepare dataset with all preprocessing.
    dataset = HabitatDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        mode=mode,
        patch_size=patch_size,
        context_width=context_width,
        samples_per_image=samples_per_image,
        transform=transform
    )
    
    print(f"\nDataset prepared:")
    print(f"  Mode: {mode}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Patch size: {patch_size}")
    print(f"  Context width: {dataset.context_width}")
    
    return dataset


# Example usage:
if __name__ == "__main__":
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # bring in my raw images and annotations
    IMG_DIR = r"C:\Users\Peter Chapman\Personal_Research_Projects\benthic-habitat-segmentation-ML\Project 4 Dataset\Project 4 Dataset\raw_labeled_data\images"
    MASK_DIR = r"C:\Users\Peter Chapman\Personal_Research_Projects\benthic-habitat-segmentation-ML\Project 4 Dataset\Project 4 Dataset\raw_labeled_data\annotations"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define augmentation pipeline through transformations
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ])
    
    # Create dataset with patch mode
    dataset = prepare_dataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        mode='patch',
        patch_size=256,
        context_width=320,
        samples_per_image=50,
        transform=train_transform
    )
    
    # Get class weights for weighted loss
    class_weights = HabitatDataset.get_class_weights(device)
    print(f"\nClass weights on {device}:")
    print(class_weights)
    print(f"\nUse in loss function:")
    print(f"criterion = nn.CrossEntropyLoss(weight=class_weights)")