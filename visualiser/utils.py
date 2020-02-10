from pathlib import Path
from collections import Counter

import pydicom
import numpy as np
from skimage import measure
from plotly.offline import iplot
from plotly import figure_factory as FF

__all__ = ['load_scan', 'get_pixels_hu']


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


def make_mesh(image, threshold=-300, step_size=1):
    '''
    :param threshold: to choose tissue u whant to visualize
    (-1000 - air
    -100 -50 - fat
    +30 +70 - blood
    +10 +40 - muscle
    +40 +60 - liver
    > +700 - bone)
    :param step_size: bigger number worse detalization but faster
    '''
    p = image.transpose(2, 1, 0)

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
    aspectratio = dict(
        x=1,
        y=1,
        z=2,
    )

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            show_colorbar=False,
                            aspectratio=aspectratio,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)