from typing import Tuple
import morphsnakes as ms
import numpy as np


def morph_snakes_segmentation(snapshot,
                              start_point: Tuple[int, int],
                              start_radius: int,
                              num_iterations: int,
                              balloon=0,
                              z=None):
    """
    Now only works with 2D image, thus always set z for now
    # TODO: do for 3D snapshot
    """
    # assert snapshot.ndim == 3

    img = snapshot
    # get the gaussian gradient for higher contrast
    gimg = ms.inverse_gaussian_gradient(img, alpha=5.0, sigma=5.0)

    # level_set is the starting
    level_set = ms.circle_level_set(img.shape, start_point, start_radius)
    mask = ms.morphological_geodesic_active_contour(gimg.astype('float64'), num_iterations, level_set, balloon=balloon)
    # repeat the 2D mask so that it is the same size as the image
    mask = np.repeat(mask[np.newaxis, :, :], snapshot.shape[0], axis=0)

    return mask
