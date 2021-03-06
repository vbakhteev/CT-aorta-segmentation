import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from plotly.offline import iplot
from plotly import figure_factory as FF


def single_2d_plot(img, mask=None, aspect=1.0, figsize=None, vmin=-1024, vmax=500):
    if figsize is None:
        figsize = (0.015 * img.shape[1], 0.015 * img.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.imshow(img, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)
    ax.set_aspect(aspect)

    if mask is not None:
        mask = np.ma.masked_where(mask == 0, mask)
        plt.imshow(mask, alpha=0.4, cmap='viridis')

    plt.plot()


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


def plotly_3d(verts, faces, mask, aspectratio=dict(x=1, y=1, z=1)):
    # plot using trisurf
    x, y, z = zip(*verts)

    colormap = ['rgb(236, 236, 212)', (1, 0.65, 0.12)]

    def mask_code(x, y, z):
        x, y, z = int(x), int(y), int(z)
        return mask[z, y, x]

    fig = FF.create_trisurf(
        x=x, y=y, z=z,
        color_func=mask_code if mask is not None else None,
        plot_edges=False,
        show_colorbar=True,
        aspectratio=aspectratio,
        colormap=colormap,
        simplices=faces,
        backgroundcolor='rgb(64, 64, 64)',
        title="Interactive Visualization",
    )
    iplot(fig)
