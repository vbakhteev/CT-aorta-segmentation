import warnings
from pathlib import Path
from typing import Union

import cv2
import torch
import torchio
import numpy as np
import scipy.ndimage

from ..utils import load_scan, get_pixels_hu, get_spacing, load_slice


class Patient:
    """Class for storing 3D CT snapshot and mask of patient 
    """

    def __init__(self, snapshot: np.array, spacing: tuple = (1, 1, 1), mask: np.array = None):
        """
        snapshot: np.array(z, y, x) - 3D image of patient
        spacing: tuple with voxel spacing
        mask: np.array optional segmentation mask
        """
        self.snapshot = snapshot
        self.spacing = spacing
        self.mask = mask


    @classmethod
    def from_path(cls, path: Union[str, Path], mask_path: Union[str, Path] = None):
        """Loads DICOM file by given directory path
        """
        slices = load_scan(path)
        spacing = get_spacing(slices[0])
        snapshot = get_pixels_hu(slices)

        mask = None                 # TODO load mask from given_path

        return cls(snapshot, spacing=spacing, mask=mask)


    @classmethod
    def from_torchio(cls, sample: dict):
        """Creates Patient object from torchio subject instance
        """
        snapshot = sample['snapshot']['data'][0].numpy().transpose()
        slice_path = next(Path(sample['snapshot']['path']).iterdir())
        spacing = get_spacing(load_slice(slice_path))
        mask = sample.get('mask', None)
        if mask is not None:
            mask = mask['data'][0].numpy().transpose()

        return cls(snapshot, spacing=spacing, mask=mask)
    

    @property
    def shape(self):
        """Returns a shape of Patient's snapshot
        """
        return self.snapshot.shape


    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask: np.array):
        assert (mask is None) or (mask.ndim == 3 and mask.shape == self.snapshot.shape)
        self._mask = mask

    def __getitem__(self, key):
        return self.snapshot[key]

    def __setitem__(self, key, value):
        self.snapshot[key] = value


    def resample(self, new_spacing=[1, 1, 1]):
        """Resamples 3D such that is has spacing of one voxel as new_spacing
        """
        resize_factor = self.spacing / new_spacing
        new_real_shape = self.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.shape

        self.spacing = self.spacing / real_resize_factor
        self.snapshot = scipy.ndimage.interpolation.zoom(self.snapshot, real_resize_factor, order=1)