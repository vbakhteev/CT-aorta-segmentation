import cv2
import numpy as np
import scipy.ndimage

from ..utils import load_scan, get_pixels_hu


class Patient:
    """Class for storing 3D CT snapshot and mask of patient 
    """

    def __init__(self, path: str, snapshot_mask=None):
        slices = load_scan(path)
        slice_thickness = float(slices[0].SliceThickness)
        xy_spacing = [float(ps) for ps in slices[0].PixelSpacing]

        self.spacing = np.array([slice_thickness] + xy_spacing)
        self.snapshot = get_pixels_hu(slices)
        self.snapshot_mask = snapshot_mask
    

    @property
    def shape(self):
        """Returns a shape of Patient's snapshot
        """
        return self.snapshot.shape


    @property
    def snapshot_mask(self):
        return self._snapshot_mask

    @snapshot_mask.setter
    def snapshot_mask(self, snapshot_mask):
        assert (snapshot_mask is None) or (snapshot_mask.ndim == 3 and snapshot_mask.shape == self.snapshot.shape)
        self._snapshot_mask = snapshot_mask


    def resample(self, new_spacing=[1, 1, 1]):
        """Resamples 3D such that is has spacing of one voxel as new_spacing
        """
        resize_factor = self.spacing / new_spacing
        new_real_shape = self.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.shape

        self.spacing = self.spacing / real_resize_factor
        self.snapshot = scipy.ndimage.interpolation.zoom(self.snapshot, real_resize_factor, order=1)