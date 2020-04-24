import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from .plots import make_mesh, plotly_3d, single_2d_plot
from ..data_structs import Patient
from plotly.offline import iplot


class PatientSnapshot(Patient):
    """Class for visualisation of Patient
    """

    def horizontal_plot(self, z: int, vmin=-1024, vmax=500, plot_mask=True):
        img = self.snapshot[z]
        mask = self.mask[z] if (plot_mask and self.mask is not None) else None

        z, y, x = self.spacing
        aspect = y / x
        single_2d_plot(img, mask, aspect=aspect, vmin=vmin, vmax=vmax)

    def frontal_plot(self, y: int, vmin=-1024, vmax=500, plot_mask=False):
        img = self.snapshot[:, y, :].T
        mask = self.mask[:, y, :].T if (plot_mask and self.mask is not None) else None

        z, y, x = self.spacing
        aspect = x / z
        single_2d_plot(img, mask, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin,
                       vmax=vmax)

    def longitudinal_plot(self, x: int, vmin=-1024, vmax=500, plot_mask=False):
        img = self.snapshot[:, :, x].T
        mask = self.mask[:, :, x].T if (plot_mask and self.mask is not None) else None

        z, y, x = self.spacing
        aspect = y / z
        single_2d_plot(img, mask, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin,
                       vmax=vmax)

    def diagonal_plot(self, offset, r: Rotation, draw_3d=True, slice_size=300):
        """
        :param offset: point through which the cutting plane passes
        :param r: scipy Rotation object
        :param draw_3d: Draw 3d plot where the slice coordinates are more readable
        :return:
        """
        axes = [range(-slice_size, slice_size) for _ in range(2)]
        shape = [len(r) for r in axes]
        grid = np.meshgrid(*axes, indexing='xy')
        grid = np.stack([*grid, np.ones(shape)], axis=-1).reshape(-1, 3)
        grid_rotated = r.apply(grid) + offset
        grid_rotated = grid_rotated[..., ::-1].T

        img = scipy.ndimage.map_coordinates(self.snapshot, grid_rotated, order=0).reshape(shape)
        single_2d_plot(img)

        def draw_cut_on3d(x, y, z, colors):
            v, f = make_mesh(self.snapshot, 350, 2)

            window_upper = 1000
            window_lower = 0

            #     trying to map density to opacity/color
            def intensity_func(x, y, z):
                x = x.astype(int)
                y = y.astype(int)
                z = z.astype(int)
                res = self.snapshot[z, y, x]
                res[np.where(res < window_lower)] = window_lower
                res[np.where(res > window_upper)] = window_upper
                return res

            def plotly_triangular_mesh(vertices, faces, intensities=intensity_func, opacity=1, colorscale="Viridis",
                                       showscale=False, reversescale=False, plot_edges=False):

                x, y, z = vertices.T
                I, J, K = faces.T

                # setting intensity func
                if hasattr(intensities, '__call__'):
                    intensity = intensities(x, y, z)  # the intensities are computed here via the passed function,
                    # that returns a list of vertices intensities
                elif isinstance(intensities, (list, np.ndarray)):
                    intensity = intensities  # intensities are given in a list
                else:
                    raise ValueError("intensities can be either a function or a list, np.array")

                mesh = dict(type='mesh3d',
                            x=x,
                            y=y,
                            z=z,
                            colorscale=colorscale,
                            opacity=opacity,
                            reversescale=reversescale,
                            intensity=intensity,
                            i=I,
                            j=J,
                            k=K,
                            name='',
                            showscale=showscale
                            )

                if showscale is True:
                    mesh.update(colorbar=dict(thickness=20, ticklen=4, len=0.75))

                if plot_edges is False:  # the triangle sides are not plotted
                    return [mesh]

            pl_BrBG = 'Viridis'

            import plotly.graph_objects as go

            axis = dict(showbackground=True,
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="rgb(255, 255, 255)",
                        zerolinecolor="rgb(255, 255, 255)")

            noaxis = dict(visible=False)

            layout = dict(
                title='Isosurface in volumetric data',
                font=dict(family='Balto'),
                showlegend=False,
                width=800,
                height=800,
                scene=dict(xaxis=axis,
                           yaxis=axis,
                           zaxis=axis,
                           aspectratio=dict(x=1,
                                            y=1,
                                            z=1)
                           )
            )

            fig = go.Figure()
            data = plotly_triangular_mesh(v, f, opacity=0.8, colorscale=pl_BrBG, showscale=True)
            fig.add_trace(data[0])
            fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color=colors, colorscale='Hot'))
            fig.layout = layout
            iplot(fig, filename='multiple volumes')

        def get_cut_coords(grid_rotated):
            print(type(grid_rotated))
            x = [0, 0, 0, 0, 512, 512, 512, 512]
            y = [0, 0, 512, 512, 0, 0, 512, 512]
            z = [0, 537, 0, 537, 0, 537, 0, 537]

            grid_rotated = grid_rotated[..., ::-1].T
            grid_rotated = grid_rotated[..., ::-1].T
            grid_rotated = grid_rotated[..., ::-1].T
            grid_rotated = grid_rotated[np.arange(0, grid_rotated.shape[0], 500)]

            x = np.concatenate((x, grid_rotated[:, 0]))
            y = np.concatenate((y, grid_rotated[:, 1]))
            z = np.concatenate((z, grid_rotated[:, 2]))

            colors = np.ones(8)
            colors = np.concatenate((colors, np.zeros(grid_rotated.shape[0])))
            return x, y, z, colors

        if draw_3d:
            x, y, z, colors = get_cut_coords(grid_rotated)
            draw_cut_on3d(x, y, z, colors)

    def plot_3d(self, threshold=300, steps=5, aspectratio=dict(x=1, y=1, z=1)):
        v, f = make_mesh(self.snapshot, threshold, steps)
        plotly_3d(v, f, mask=self.mask, aspectratio=aspectratio)

    def plot_cutted_by_longitude(self, x, left, right, vmin=-1024, vmax=500):
        img = self.snapshot[:, :, x].T[:, left:right]
        single_2d_plot(img, aspect='auto', figsize=(0.01 * img.shape[1], 0.01 * img.shape[0]), vmin=vmin, vmax=vmax)
