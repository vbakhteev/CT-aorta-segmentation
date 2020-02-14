import cv2
import numpy as np
import scipy.ndimage

from .plots import make_mesh, plotly_3d, single_2d_plot
from ..data_structs import Patient


class PatientSnapshot(Patient):
    """Class for visualisation of Patient
    """

    def horizontal_plot(self, z: int, vmin=-1024, vmax=500):
        img = self.snapshot[z]
        z, y, x = self.spacing
        aspect = y / x
        single_2d_plot(img, aspect=aspect, vmin=vmin, vmax=vmax)


    def frontal_plot(self, y: int, vmin=-1024, vmax=500):
        img = self.snapshot[:, y, :].T
        z, y, x = self.spacing
        aspect = x / z
        single_2d_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)


    def longitudinal_plot(self, x: int, vmin=-1024, vmax=500):
        img = self.snapshot[:, :, x].T
        z, y, x = self.spacing
        aspect = y / z
        single_2d_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)


    def plot_3d(self, threshold=300):
        v, f = make_mesh(self.snapshot, threshold, 5)
        plotly_3d(v, f)


    def plot_cutted_by_longitude(self, x, left, right, vmin=-1024, vmax=500):
        img = self.snapshot[:, :, x].T[:, left:right]
        single_2d_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)
