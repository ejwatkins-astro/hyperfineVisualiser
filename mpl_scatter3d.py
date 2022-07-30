# -*- coding: utf-8 -*-
"""
Old code for 3d plotting and contours.
"""

import numpy as np
import matplotib.pyplot as plt
from matplotlib import colors
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import colorcet as cc

import conversion_utilities as util
import ncomponent_data as ncomp
import config
import io

try:
    from astroContours.convert import maskContours, contourPolygons, levelContours
except ImportError:
    pass

class Scatter_3d(ncomp.Dataframe_mutlicomponent):
    """
    Class for plotting Velocity data using matplotlib
    """

    def plot_3d_scatter(self, use_world_coords=True, convert_world=False, zlims=None, **kwargs):

        """
        Given a datamap, converts the information into a 3d scatter plot.
        """

        self.use_world_coords = use_world_coords
        self.convert_world = convert_world
        self.zlims = zlims

        self.fig = plt.figure('3d scatter')
        self.ax = self.fig.add_subplot(projection='3d')

        cmap = kwargs.pop('cmap', cc.m_glasbey)
        cmap = discrete_cmap(cmap, 1, self.maximum_components)
        alpha = kwargs.pop('alpha',0.01)

        x, y = self._plot_coords_and_labels()
        self.set_plot_limits(zlims=zlims)

        sc = self.ax.scatter(xs=x, ys=y, zs=self.cube_z_values, c=self.z_positions, zdir='z', cmap=cmap, alpha=alpha,
                   **kwargs)

        #Second scatter helps show the data
        self.ax.scatter(xs=x, ys=y, zs=self.cube_z_values, zdir='z', color='k', s=1, marker='.', alpha=alpha*2 if alpha <= 0.5 else 1)

        return self.ax

    def _plot_coords_and_labels(self):

        if self.use_world_coords and self.header is not None:
            if self.convert_world:
                x, y = self.get_world_positions()[2:]
                self.label_plot(coord_type=config.coord_conversion[self.world_coord_type])
            else:
                x, y = self.get_world_positions()[:2]
                self.label_plot(coord_type=self.world_coord_type)

        else:
            x, y = self.x_positions, self.y_positions
            self.label_plot(coord_type='pixel')

        return x, y

    def label_plot(self, coord_type, zlabel='Velocity [km/s]'):

        labels = coord_types[coord_type]
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(zlabel)

    def set_plot_limits(self, zlims=None):

        xlims, ylims, size_x, size_y = self.get_plot_limits()

        self.ax.set_xlim(*xlims)
        self.ax.set_ylim(*ylims)

        if zlims is None:
            self.zlims = [np.min(self.cube_z_values), np.max(self.cube_z_values)]
        self.ax.set_zlim(*self.zlims)

    def get_plot_limits(self):

        header = self.header
        if self.use_world_coords and header is not None:
            if self.convert_world:
                new_world = config.coord_conversion[self.world_coord_type]
                header = convert_header_to_new_world(self.component_cube[0], header)

            xlims, ylims = get_world_data_lims(header)
            size_x, size_y = header['NAXIS1'], header['NAXIS2']
        else:
            size_x, size_y = header['NAXIS1'], header['NAXIS2']
            xlims = [0, size_x-1]
            ylims = [0, size_y-1]
        # size_x, size_y = self.size_x, self.size_y

        return xlims, ylims, size_x, size_y

    # @functools.lru_cache
    def reproject_to_new_world(self, data_2d):
        header = self.header
        if self.use_world_coords and header is not None:
            if self.convert_world:
                new_world = config.coord_conversion[self.world_coord_type]
                data_2d, header = util.convert_world_system_of_image(data_2d, header)
        return data_2d, header

    def _prep_contour_params(self, data_2d):

        data_rpj, header_rpj = self.reproject_to_new_world(data_2d)

        size = np.shape(data_rpj)

        x = np.linspace(*self.ax.get_xlim(), size[1])
        y = np.linspace(*self.ax.get_ylim(), size[0])

        X,Y = np.meshgrid(x, y)

        return X, Y, data_rpj

    def _params_for_contour_method(self, data, levels, **kwargs):

        import importlib
        astroContours_spec = importlib.util.find_spec("astroContours")
        astroContours_exists = astroContours_spec is not None

        if astroContours_exists:
            out = levelContours.Patch_astroContours(data=data, levels=levels, use_world_coords=self.use_world_coords, header=self.header, convert_world=self.convert_world, **kwargs).get_contours()[0]
            # out =
            return out, astroContours_exists
        else:
            out = self._prep_contour_params(data)
            return out, astroContours_exists

    def plot_imshow(self, data_2d, **kwargs):
        """
        Plot 2d data onto a 3d axis
        """
        if isinstance(data_2d, str):
            hdu_i = kwargs.pop('hdu_i', None)
            data_2d = io.Loader.lazy_load(data_2d, hdu_i=hdu_i)

        levels = kwargs.pop('levels', 500)

        X, Y, data_2d = self._prep_contour_params(data_2d)

        contf = self.ax.contourf(X, Y, data_2d, zdir='z', levels=500, offset=self.ax.get_zlim[0], origin='lower', **kwargs)
        #Z = np.ones_like(X) * self.ax.get_zlim[0]
        # cmap = kwargs.pop('cmap', plt.cm.viridis)
        # self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap(data_2d), shade=False)

    def plot_contours(self, data_2d, **kwargs):

        #plotting contours at a guess of good positions
        levels = kwargs.pop('levels', np.nanpercentile(data_2d, [5, 90, 99]))

        #data can just be a file name. Needs to be a 2d array like
        if isinstance(data_2d, str):
            hdu_i = kwargs.pop('hdu_i', None)
            data_2d = io.Loader.lazy_load(data_2d, hdu_i=hdu_i)

        out, use_astroContours = self._params_for_contour_method(data_2d, levels, **kwargs)

        if use_astroContours:
            self._plot_from_astroContour(out, levels, **kwargs)

        else:
            cont = self.ax.contour(out[0], out[1], out[2], zdir='z', levels=levels, offset=self.ax.get_zlim()[0], origin='lower',  **kwargs)

    def _plot_from_astroContour(self, contours, levels, **kwargs):

        vmin  = kwargs.pop('vmin', np.min(levels))
        vmax  = kwargs.pop('vmax', np.max(levels))

        cmap  = kwargs.pop('cmap', plt.cm.viridis)
        color = kwargs.pop('color', None)

        #These if statments colours the contours using a cmap
        #if a single colour isn't given
        if len(levels) == 1 :
            if color is None:
                color = ['k'] * len(contours)
            else:
                color = [color] * len(contours)
        elif color is None:
            color = [ levels[i] for i in range(len(levels)) for j in range(len(contours[i]))]
            color = levelContours.get_colours_from_cmap(color, vmin, vmax, cmap)

            #contours has a list of all the contours per level. i.e contours[1]
            #contains all the contours bounding the second level. So need to flatten
            contours = mf.flatten_list(contours)
        else:
            contours = mf.flatten_list(contours)
            color = [color] * len(contours)

        #world are x,y whereas pixel are y,x
        if self.use_world_coords:
            [self.ax.plot(xs=contours[i][:,0], ys=contours[i][:,1], zs=[self.ax.get_zlim()[0]] * len(contours[i]), zdir='z', color=color[i], **kwargs) for i in range(len(contours))]
        else:
            [self.ax.plot(xs=contours[i][:,1], ys=contours[i][:,0], zs=[self.ax.get_zlim()[0]] * len(contours[i]), zdir='z', color=color[i], **kwargs) for i in range(len(contours))]
