import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from .utils import load_scan, get_pixels_hu, make_mesh, plotly_3d


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
        self._single_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)

    def longitudinal_plot(self, x: int, vmin=-1024, vmax=500):
        img = self._image[:, :, x].T
        z, y, x = self.spacing
        aspect = y / z
        self._single_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)

    def plot_3d(self, threshold=300):
        v, f = make_mesh(self._image, threshold, 5)
        plotly_3d(v, f)

    def _single_plot(self, img, aspect=1.0, figsize=None, vmin=-1024, vmax=500):
        figsize = (0.01 * img.shape[1], 0.01 * img.shape[0]) if figsize is None else figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(img, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)
        ax.set_aspect(aspect)
        plt.plot()

    def aorta_mask_slice_cv(self, img_index, min_area=60, threshold=275):
        """
        min_area: minimum area for contour to check if it is aorta, to filter out to small contours
        threshold: binary threshold, all that bigger substituted to 255, all that smaller to 0
        """
        ret, img_n = cv2.threshold(self._image[img_index], threshold, 255, cv2.THRESH_BINARY)
        img_n = cv2.normalize(src=img_n, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cnt, _ = cv2.findContours(img_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # to manage nested contours
        for c in cnt:
            cv2.drawContours(img_n, [c], 0, 255, -1)
        cnt, _ = cv2.findContours(img_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # here assumption is made that aorta is a contour which is the closest to circle
        # not always true, but in most cases works
        cnt_areas = []
        for c in cnt:
            contour_area = cv2.contourArea(c)
            if contour_area > min_area:
                _, radius = cv2.minEnclosingCircle(c)
                radius = int(radius)
                circle_area = np.pi * radius ** 2
                # check how close contour is to circle
                a = round(circle_area / contour_area, 1)
                cnt_areas.append((c, contour_area, a))

        img_n = np.zeros(img_n.shape)
        if len(cnt_areas) != 0:
            cnt_areas = sorted(cnt_areas, key=lambda x: x[2])
            cv2.drawContours(img_n, [cnt_areas[0][0]], 0, 1, cv2.FILLED)
        return img_n

    def aorta_mask_scan_cv(self):
        #TODO memory efficient version

        mask_img = []
        for i, img in enumerate(self._image):
            mask_img.append(self.aorta_mask_slice_cv(i))
        mask_img = np.array(mask_img)
        return mask_img

    def plot_cutted_by_longitude(self, x, left, right, vmin=-1024, vmax=500):
        img = self._image[:, :, x].T[:, left:right]
        self._single_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)

    def cut_by_longitude(self, left, right, vmin=-1024, vmax=500):
        self._image = self._image[left:right, :, :]
