from typing import List, Tuple

import cv2
import json
import numpy as np


class SliceMaskPoints:
    """Represents contours of mask in one slice
    """
    def __init__(self, contours: List[np.array], size: Tuple[int, int]):
        """contours - list of mask's contours
        size - 2d shape of mask
        """
        self.contours = contours
        self.size = size

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            metadata = json.loads(f.read())
        
        contours = [
            np.array(p['points']['exterior'])
            for p in metadata['objects']
            if p['classTitle']=='Aorta'
        ]
        size = metadata['size']
        size = (size['height'], size['width'])

        return cls(contours, size)

    @classmethod
    def from_array(cls, mask):
        contours, size = mask_to_points(mask)
        return cls(contours, size)

    @property
    def mask(self):
        mask = np.zeros(self.size, dtype=np.uint8)
        mask = cv2.drawContours(mask, self.contours, -1, (255), 1)
        return mask

    @mask.setter
    def mask(self, mask):
        self.contours, self.size = mask_to_points(mask)


def mask_to_points(mask):
    assert mask.ndim == 2
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    size = tuple(mask.shape)

    return contours, size


class SliceMask:
    """Mask of one slice
    """
    pass