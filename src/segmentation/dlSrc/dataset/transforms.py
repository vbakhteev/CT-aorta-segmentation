from functools import partial

import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_augmentations():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.RandomCrop(300, 300, p=0.25),
    ])


def get_preprocessing(size, mean=(0., 0., 0.), std=(1., 1., 1.)):
    preprocessing_fn = partial(
        normalize_img, mean=mean, std=std,
    )

    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Resize(size, size),
        ToTensorV2(),
    ]
    return albu.Compose(transform)

def normalize_img(x, mean=None, std=None, **kwargs):
    x = x / 3200.
    if mean is not None:
        x = x - np.array(mean)
    if std is not None:
        x = x / np.array(std)
    return x