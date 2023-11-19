import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

class CarvanaDatasetLoaded(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, val_ratio=None, train=True, device="cpu"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # preload images
        image_names = os.listdir(image_dir)
        if val_ratio is not None:
            split_idx = int(len(image_names) * (1.0 - val_ratio))
            if train:
                image_names = image_names[:split_idx]
            else:
                image_names = image_names[split_idx:]

        self.raw_images = []
        print("Loading images...")
        loop = tqdm(image_names, total=len(image_names), leave=False)
        for file_name in loop:
            image = Image.open(os.path.join(image_dir, file_name)).convert("RGB")
            self.raw_images.append(torch.tensor(np.array(image), dtype=torch.float32))
        self.raw_images = torch.stack(self.raw_images).to(device)
        self.images = torch.empty_like(self.raw_images).to(device)

        print("Loading masks...") 
        # preload masks
        self.raw_masks = []
        loop = tqdm(image_names, total=len(image_names), leave=False)
        for file_name in loop:
            file_name = file_name.replace(".jpg", "_mask.gif")
            mask = Image.open(os.path.join(mask_dir, file_name)).convert("L")
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            mask[mask==255.0] = 1.0
            self.raw_masks.append(mask)
        self.raw_masks = torch.stack(self.raw_masks).to(device)
        self.masks = torch.empty_like(self.raw_masks).to(device)
    
    def apply_transform(self, image, mask, batch_size=500):
        if self.image_transform is None and self.mask_transform is None:
            raise Warning("No transforms were specified")

        self.transformed_images, self.transformed_masks = [], []

        low = 0
        high = batch_size
        while low < len(self.images):
            if high > len(self.images):
                high = len(self.images)
            if self.image_transform is not None:
                self.images[low:high] = self.image_transform(self.raw_images[low:high])
            if self.mask_transform is not None:
                self.masks[low:high] = self.mask_transform(self.raw_masks[low:high])
            low += batch_size
            high += batch_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx] if self.image_transform is not None else self.raw_images[idx]
        mask = self.masks[idx] if self.mask_transform is not None else self.raw_masks[idx]
        return image, mask

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, val_ratio=None, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        split_idx = int(len(self.images) * (1.0 - val_ratio))
        if train:
            self.images = self.images[:split_idx]
        else:
            self.images = self.images[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask