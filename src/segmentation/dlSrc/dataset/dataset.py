from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .transforms import get_preprocessing


class TrainDataset(Dataset):
    def __init__(self, df, cfg, transform=None, cache=True):
        self.data_dir = Path(cfg.data.path)
        self.repeat_dataset = cfg.data.repeat_dataset

        self.transform = transform
        self.preprocessing = get_preprocessing(size=cfg.data.width)
        self.img_paths = df['img_path'].values
        self.mask_paths = df['mask_path'].values

        self.cache = dict()

    def __getitem__(self, idx):
        idx = idx % len(self.img_paths)
        img_path = self.data_dir / self.img_paths[idx]
        mask_path = self.data_dir / self.mask_paths[idx]

        if img_path not in self.cache:
            img = nib.load(img_path).get_data()
            mask = nib.load(mask_path).get_data()
            self.cache[img_path] = img
            self.cache[mask_path] = mask
        img = self.cache[img_path]
        mask = self.cache[mask_path]

        # Get random 3-channel image and corresponding mask
        i = np.random.randint(img.shape[2]-2) + 1
        img = img[:, :, i-1:i+2]
        mask = mask[:, :, i]

        if self.transform:
            sample = self.transform(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        sample = self.preprocessing(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']

        return img.float(), mask.float()
    
    def __len__(self):
        return len(self.img_paths) * (self.repeat_dataset + 1)


class ValidDataset(Dataset):
    def __init__(self, df, cfg):
        self.data_dir = Path(cfg.data.path)
        self.preprocessing = get_preprocessing(size=cfg.data.width)

        img_paths = df['img_path'].values
        mask_paths = df['mask_path'].values
        self.images, self.masks, self.ranges = [], [], []
        cum_idx = 0

        for ip, mp in zip(img_paths, mask_paths):
            img = nib.load(self.data_dir / ip).get_data()
            mask = nib.load(self.data_dir / mp).get_data()

            not_empty_slices = mask.sum(axis=0).sum(axis=0).nonzero()
            img = img[:, :, not_empty_slices].squeeze(2)
            mask = mask[:, :, not_empty_slices].squeeze(2)

            z_len = mask.shape[2] - 2
            r = (cum_idx, cum_idx + z_len - 1)
            cum_idx += z_len
            self.ranges.append(r)
            self.images.append(img)
            self.masks.append(mask)


    def __getitem__(self, idx):
        img_idx, slice_idx = self.get_indexes(idx)
        img = self.images[img_idx][:, :, slice_idx-1:slice_idx+2]
        mask = self.masks[img_idx][:, :, slice_idx]

        sample = self.preprocessing(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']

        return img.float(), mask.float()

    def get_indexes(self, idx):
        for img_idx, (l, r) in enumerate(self.ranges):
            if idx > r:
                continue
            else:
                slice_idx = idx - l
                break
        slice_idx = slice_idx + 1
        return img_idx, slice_idx
    
    def __len__(self):
        return self.ranges[-1][1] + 1