import matplotlib.pyplot as plt
from skimage import measure
from plotly.offline import iplot
from plotly import figure_factory as FF


def single_2d_plot(img, aspect=1.0, figsize=None, vmin=-1024, vmax=500):
    if figsize is None:
        figsize = (0.01 * img.shape[1], 0.01 * img.shape[0])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.imshow(img, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)
    ax.set_aspect(aspect)
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


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
    aspectratio = dict(
        x=1, y=1, z=3,
    )

    fig = FF.create_trisurf(
        x=x, y=y, z=z,
        plot_edges=False,
        show_colorbar=False,
        aspectratio=aspectratio,
        colormap=colormap,
        simplices=faces,
        backgroundcolor='rgb(64, 64, 64)',
        title="Interactive Visualization",
    )
    iplot(fig)