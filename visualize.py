import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as FF
from plotly.graph_objs import *

data_path = ''
output_path = working_path = ''


def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for i, s in enumerate(os.listdir(path))]
    slices.sort(key=lambda x: int(x.SliceLocation))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]])

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


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

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)


def plt_3d(verts, faces):
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()


id = 0

## load scan
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
# np.save(output_path + "fullimages_%d.npy" % (id), imgs)

# file_path_before_resample = output_path + "fullimages_%d.npy" % id
# imgs_to_process = np.load(file_path_before_resample).astype(np.float64)

## histogram
# plt.hist(imgs_to_process.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()

## resampling
# print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs, patient, [1, 1, 1])
# np.save(output_path + "fullimages_resampled_%d.npy" % (id), imgs_after_resamp)
# print("Shape after resampling\t", imgs_after_resamp.shape)

# file_path_resampled = output_path + "fullimages_resampled_%d.npy" % id
# imgs_after_resamp = np.load(file_path_resampled).astype(np.float64)

v, f = make_mesh(imgs_after_resamp, 300, 5)
plotly_3d(v, f)
