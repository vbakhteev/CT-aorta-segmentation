import warnings
from pathlib import Path

import cv2
import torch
import torchio
import numpy as np
import scipy.ndimage

from ..utils import load_scan, get_pixels_hu, get_spacing, load_slice


class Patient:
    """Class for storing 3D CT snapshot and mask of patient 
    """

    def __init__(self, snapshot, spacing=(1, 1, 1), mask=None):
        """
        snapshot: np.array(z, y, x) - 3D image of patient
        spacing: tuple with voxel spacing
        mask: np.array optional segmentation mask
        """
        self.snapshot = snapshot
        self.spacing = spacing
        self.mask = mask


    @classmethod
    def from_path(cls, path: str, mask=None):
        slices = load_scan(path)
        spacing = get_spacing(slices[0])
        snapshot = get_pixels_hu(slices)

        return cls(snapshot, spacing=spacing, mask=mask)


    @classmethod
    def from_torchio(cls, sample):
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
    def mask(self, mask):
        assert (mask is None) or (mask.ndim == 3 and mask.shape == self.snapshot.shape)
        self._mask = mask


    def resample(self, new_spacing=[1, 1, 1]):
        """Resamples 3D such that is has spacing of one voxel as new_spacing
        """
        resize_factor = self.spacing / new_spacing
        new_real_shape = self.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.shape

        self.spacing = self.spacing / real_resize_factor
        self.snapshot = scipy.ndimage.interpolation.zoom(self.snapshot, real_resize_factor, order=1)


class Image(torchio.Image):
    r"""Class to store information about an image.
    """
    def __init__(self, name: str, path: str, type_: str):
        self.name = name
        self.path = self._parse_path(path)
        self.type = type_

    def _parse_path(self, path) -> Path:
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        return path

    def load(self, check_nans: bool = True):
        r"""Load the image from disk.
        Args:
            check_nans: If ``True``, issues a warning if NaNs are found
                in the image
        Returns:
            Tuple containing a 4D data tensor of size
        : str:`(1, D_{in}, H_{in}, W_{in})`
            and a 2D 4x4 affine matrix
        """
        tensor = torch.from_numpy(
            get_pixels_hu(load_scan(self.path)).transpose()
        )
        affine = np.zeros((4, 4)) # TODO change affine to appropriate values
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine