import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from .utils import load_scan, get_pixels_hu


class Patient:
    """Class for 3D CT image of human and his visualisation
    """

    def __init__(self, path: str):
        self._slices = load_scan(path)
        self._image = get_pixels_hu(self._slices)

    @property
    def spacing(self):
        """Returns np.array(z, x, y): spacing of one voxel in mm
        """
        slice_thickness = float(self._slices[0].SliceThickness)
        xy_spacing = [float(ps) for ps in self._slices[0].PixelSpacing]
        spacing = [slice_thickness] + xy_spacing
        return np.array(spacing)

    @property
    def shape(self):
        """Returns a shape of 3D image
        """
        return self._image.shape

    def resample(self, new_spacing=[1, 1, 1]):
        """Resamples 3D such that is has spacing of one voxel as new_spacing
        """
        resize_factor = self.spacing / new_spacing
        new_real_shape = self.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.shape
        new_spacing = self.spacing / real_resize_factor

        self._image = scipy.ndimage.interpolation.zoom(self._image, real_resize_factor, order=1)

        for s in self._slices:
            s.SliceThickness = str(new_spacing[0])
            s.PixelSpacing = [str(s) for s in new_spacing[1:]]

    def horizontal_plot(self, z: int, vmin=-1024, vmax=500):
        img = self._image[z]
        z, y, x = self.spacing
        aspect = y / x
        self._single_plot(img, aspect=aspect, vmin=vmin, vmax=vmax)

    def frontal_plot(self, y: int, vmin=-1024, vmax=500):
        img = self._image[:, y, :].T
        z, y, x = self.spacing
        aspect = x / z
        self._single_plot(img, aspect=aspect, figsize=(20, 5), vmin=vmin, vmax=vmax)

    def longitudinal_plot(self, x: int, vmin=-1024, vmax=500):
        img = self._image[:, :, x].T
        z, y, x = self.spacing
        aspect = y / z
        self._single_plot(img, aspect=aspect, figsize=(20, 5), vmin=vmin, vmax=vmax)

    def _single_plot(self, img, aspect=1.0, figsize=(10, 10), vmin=-1024, vmax=500):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(img, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)
        ax.set_aspect(aspect)
        plt.plot()
