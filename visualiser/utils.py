from pathlib import Path

import pydicom
import numpy as np

__all__ = ['load_scan', 'get_pixels_hu']


def load_scan(path):
    path = Path(path)
    slices = [pydicom.read_file(str(s)) for s in path.iterdir()]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    #     image[image == -2000] = 0

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)
