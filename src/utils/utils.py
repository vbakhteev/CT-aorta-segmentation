from pathlib import Path
from collections import Counter

import pydicom
import numpy as np


def load_scan(path, filter_slices=False):
    path = Path(path)
    slices = [pydicom.read_file(str(s)) for s in path.iterdir()]
    if filter_slices:
        slices = filter_bad_slices(slices)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def filter_bad_slices(slices):
    spacings = Counter([s.PixelSpacing[0] for s in slices])
    common_spacing = spacings.most_common(1)[0][0]
    cleaned_slices = list(filter(lambda s: s.PixelSpacing[0]==common_spacing, slices))

    return cleaned_slices