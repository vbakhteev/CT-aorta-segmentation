from pathlib import Path
from collections import Counter

import cv2
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


def remove_legs(snapshot):
    """Takes 3d image of human and removes legs.
    Legs located close to begining of array, head close to the end.

    snapshot: np.array(z, y, x)

    returns: np.array(new_z, y, x)
    """
    z, y, x = snapshot.shape
    low_border, up_border = 0, z-1
    border = z//2

    while True:
        if are_legs(snapshot[border]):
            low_border = border
            border = (border + up_border) // 2
        else:
            up_border = border
            border = (border + low_border) // 2

        if up_border - low_border < 4:
            break
    
    return snapshot[border:]
    

def are_legs(img, threshold=-100, min_area=500):
    _, x = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    x = x.astype(np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((4,4),np.uint8))
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((8,8),np.uint8))
    cnt, _ = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_objects = 0
    for c in cnt:
        contour_area = cv2.contourArea(c)

        if contour_area > min_area:
            _, radius = cv2.minEnclosingCircle(c)
            radius = int(radius)
            circle_area = np.pi * radius ** 2
            # check how close contour is to circle
            a = round(circle_area / contour_area, 1)

            if a < 3:
                num_objects += 1

    return num_objects == 2