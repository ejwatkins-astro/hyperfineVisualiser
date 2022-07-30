# -*- coding: utf-8 -*-
"""
Program for getting all the data files, their axis values and world
projections in one dictionary.
TODO: classes do slightly different things to load. Might want to homogenise
"""

import numpy as np
import functools
import pandas as pd

from . import config
from . import conversion_utilities as util
from . import io
from . import make_maps

#Assuming 2d images are their own induvidual file
class Data_2d:
    """
    Given a list of filenames, will load all the data and place
    them into a dictionary called `all_data`.
    All data must have the same projection. Class requires a header.
    If no header is given, the first filemust be a fits file containing
    the header information.
    Each 2d data needs a label to identify what data it is.

    TODO: Have slightly reorganised but needs a little more. Should make needing
        the header optional, if only care about x,y.
    TODO: Class has to be initalised using files. This makes adding maps
        to this dictionary weird. This class was wrote assuming the viewed maps
        would be mom0 maps, which would always be saved.  So class is initalised
        with the filenames only. Need to change this so that the class
        can be initalised from files or from a given 2d array and header.
        For now, to add a non saved map, initalise the class from a file, and add
        the map to the data dictionary `all_data` using method
        `add_new_map_to_all_all
    """

    def __init__(self, filenames, labels, header=None, reproject_all=False):

        if isinstance(filenames, str):
            self.filenames = [filenames]
        else:
            self.filenames = filenames

        self.header = header
        self.reproject_all = reproject_all

        if header is None:
            self.header = self._get_header_from_first_file()
        else:
            self.header = header

        self.labels = labels

        data = self.load_files()

        self.all_data = {}
        self.world_coord_type = util.get_world_type(self.header)
        self.set_data_dict(data, self.world_coord_type, self.header)

        #reproject the 2d data
        if self.reproject_all:
            self.reproject_all_2d_maps()

        self.set_data_dict(data, 'pixel', None)

    def _get_header_from_first_file(self):
        """
        Gets and sets the header, which is needed for world coordinates

        Raises
        ------
        TypeError
            Raises a type error if the first file is not a fits file with a header

        Returns
        -------
        header: astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference.

        """
        ld = io.Loader(self.filenames[0])
        if ld.method != 'from_fits_file':
            raise TypeError('If no header is given, first file must be a fits file with a header')
        else:
            header = ld.lazy_load()[1]
            return header

    def load_files(self):
        """
        Gets the data from each filename

        Returns
        -------
        data : dict
            Dictionary of the data, acessed via the label given.

        """

        data = {}
        for i in range(len(self.filenames)):
            data[self.labels[i]] = list(io.Loader(self.filenames[i], header=self.header).lazy_load())

            if isinstance(data[self.labels[i]][0], dict):

                data[self.labels[i]][0] = data[self.labels[i]][0][self.labels[i]]

        return data

    def get_world_axis_arrays(self, header):
        """
        Gets the x and y values of the 2d data in world coordinates.
        This uses only a simple scheme for getting the world axis
        values based of the header reference values

        Parameters
        ----------
        header : astropy.io.Header.header
            Dictionary-like object containing the world coordinate reference
            values

        Returns
        -------
        world_x_values : 1darray
            The coordinate values of the x axis in world coordinates
        world_y_values : 1darray
            The coordinate values of the y axis in world coordinates

        """
        world_x_values = util.get_axis_values_from_fits(header, 1)
        world_y_values = util.get_axis_values_from_fits(header, 2)

        return world_x_values, world_y_values

    def get_pixel_axis_arrays(self):
        """
        Gets the values of the x and y axis in pixel coordinates

        Returns
        -------
        x_values : 1darray
            The coordinate values of the x axis in pixel coordinates
        y_values : 1darray
            The coordinate values of the y axis in pixel coordinates

        """

        data_2d = self.all_data[self.world_coord_type]['data'][self.labels[0]][0]

        ylen, xlen = np.shape(data_2d)
        x_values = np.arange(0, xlen, 1)
        y_values = np.arange(0, ylen, 1)

        return x_values, y_values

    def set_data_dict(self, data, world_coord_type, header):
        """
        Stores `data` which is a dictionary of each data map loaded in with
        its header, into a larger dictionary to access the data in different
        world frames, along with the axis values in that coordinates system.

        For example:
            mom0_data = all_data['fk5']['mom0']['data'][0] #is the data
        in fk5.
        header_fk5 = all_data['fk5']['mom0']['data'][1] #header in fk5
        ra_values =  all_data['fk5']['x'] # x axis values in fk5
        y_values =  all_data['pixel']['y']
        mom1_data_gal = all_data['galactic']['mom1']['data'][0] #data in galactic
        glon_values =  all_data['galactic']['x']


        Parameters
        ----------
        data : dict
            Dictionary where each key is a data map and its corresponding header.
        world_coord_type : str
            The world system the data is in. Can be `fk5`, `galactic` or `pixel`

        Returns
        -------
        None.

        """
        if world_coord_type == 'fk5':
            self.ra_axis_values, self.dec_axis_values = self.get_world_axis_arrays(header)
            self.all_data['fk5'] = {'data': data, 'x':self.ra_axis_values, 'y':self.dec_axis_values}

        elif world_coord_type == 'galactic':
            self.glon_axis_values, self.glat_axis_values = self.get_world_axis_arrays(header)
            self.all_data['galactic'] = {'data': data, 'x':self.glon_axis_values, 'y':self.glat_axis_values}

        else:
            x, y = self.get_pixel_axis_arrays()
            self.all_data['pixel'] = {'data': data, 'x':x, 'y':y}

    def add_new_map(self, data, label):
        """
        If there is a 2d map that is generated that you do not have saved to file
        this method allows you to add that map to the main dictionary.

        Parameters
        ----------
        data : 2darray
            Data to be added to the dictionary `self.all_data`.
        label : str
            The label of the data to access the map

        Returns
        -------
        None.

        """

        self.all_data[self.world_coord_type]['data'][label] = [data, self.header]
        self.all_data['pixel']['data'][label] = [data, self.header]

        self.labels.append(label)

        if self.reproject_all:
            new_world_type = config.coord_conversion[self.world_coord_type]
            data_rpj = reproject_to_new_world(data, self.header_rpj)

            self.all_data[new_world_type]['data'][label] = [data_rpj, self.header_rpj]

    def reproject_all_2d_maps(self):
        """
        Reprojects all data into a new world project

        Returns
        -------
        data_rpj : dict
            Dictionary where each key is a data map and its corresponding header.

        """
        data_rpj = {}


        for label in self.labels:
            data_rpj[label] = reproject_to_new_world(*self.all_data[self.world_coord_type]['data'][label])
        self.header_rpj = data_rpj[self.labels[0]][1]

        self.set_data_dict(data_rpj, config.coord_conversion[self.world_coord_type], self.header_rpj)

        return data_rpj

class Data_3d:

    """
    Class for loading in and storing the 3d data, such as the alma cube, and model
    cube. Model cube can be a 4d hypercube with the errors, a list of cubes
    containing the model and errors, or just the model cube. Model cube, and its
    errors are optional.
    If error cubes are given, data structure assumes first ([0]) is the model,
    second ([1] is the lower error and third ([2]) is the upper error. Can
    change this order using the `model_labels` varable, but must keep
    the labels as `fit`, `lower`, `upper`.


    TODO: Add functionality for loading induvidual component models into
    this class. Could technically accept this, but would be per model rather
    than allowing access to every component.
    """

    def __init__(self, data_cube, header, model_data_cubes=None, data_is_freq=False, rest_freq=None, model_labels=['fit', 'upper', 'lower'], in_kelvin=True, model_in_kelvin=True):


        self.header = header
        self.data_is_freq = data_is_freq
        self.rest_freq = rest_freq
        self.model_labels = model_labels

        self.in_kelvin = in_kelvin
        self.model_in_kelvin = model_in_kelvin

        if self.in_kelvin:
            self.data_cube = data_cube
        else:
            self.data_cube = self.convert_to_kelvin(data_cube)

        if model_data_cubes is not None and self.model_in_kelvin:
            self.model_data_cubes = model_data_cubes
        else:
            self.model_data_cubes = self.convert_to_kelvin(model_data_cubes)





        self.all_data_3d = self.get_dictionary()

    @classmethod
    def load_data(cls, cube_filename, model_cube_filename=None, header=None, data_is_freq=False, rest_freq=None, model_labels=['fit', 'lower', 'upper'], in_kelvin=True, model_in_kelvin=True, **kwargs):
        """
        Class method for loading the data from files

        Parameters
        ----------

        cube_filename : str
            filepath to datacube
        model_cube_filename : str or list, optional
            filepath to fitted model. Fitted data model. The default is None.
        header : astropy.io.Header.header, optional
            Dictionary-like object containing the world coordinate reference
            values. The default is None.
        data_is_freq : bool, optional
            Set True if the datacube is in units of frequency. The default is None.
        rest_freq : float, optional
            If data is in frequency and header does not contain the rest frequency
            set the rest frequency here (in Hz).
        model_labels : list, optional
            labels and order of the modelled cube values.
            The default is ['fit', 'lower', 'upper'].
        in_kelvin : bool
            Bool for if cube is in kelvin. If not, it is converted. Default is True
        model_in_kelvin : bool
            Bool for if model is in kelvin. If not, it is converted. Default is True
        **kwargs : dict
            extra paramters needed to load in data. If data has an extension,
            set `hdu_i` to the extension name/number.

        Returns
        -------
        cls: object
            initalises class

        """

        #if loading from a fits file
        data_cube, header = io.Loader(cube_filename, header=header, **kwargs).lazy_load() #header, maximum_components
        data_cube = data_cube.byteswap().newbyteorder()
        model_data_cube = []
        if model_cube_filename is not None:
            #a list of file names indicates other model params
            if isinstance(model_cube_filename, str):
                model_cube_filename = [model_cube_filename]
            for file in model_cube_filename:
                model_data_cube.append(io.Loader(file, header=header, **kwargs).lazy_load()[0])

            if len(np.shape(model_data_cube)) == 5:
                model_data_cube = model_data_cube[0]

            return cls(data_cube=data_cube, header=header, model_data_cubes=model_data_cube, data_is_freq=data_is_freq, rest_freq=rest_freq, model_labels=model_labels, in_kelvin=in_kelvin, model_in_kelvin=model_in_kelvin)

    def get_velocity_values(self):
        """
        Gets the velocity information using the header reference values

        Raises
        ------
        KeyError
            f rest frequeny is not provided and data is in frequency
            will raise an if rest frequeny is not given in header

        Returns
        -------
        velocity_values : 1darray
            The values of the velocity axis

        """

        velocity_values = util.get_axis_values_from_fits(self.header, 3)

        if self.data_is_freq:
            if self.rest_freq is None:
                frequency_keys = ['RESTFRQ', 'RESTFREQ']
                for key in frequency_keys:
                    self.rest_freq = self.header[key] if key in self.header.keys() and self.rest_freq is None else None

                if self.rest_freq is None:
                    raise KeyError('Could not find `RESTFRQ` in `header`. Need'\
                                   ' to provide rest frequency in Hz to convert' \
                                   ' from frequency into velocity [km/s]')
            velocity_values = util.frequency2velocity(velocity_values)

        return velocity_values / 1000 #km/s

    def get_dictionary(self):
        """
        Creates a dictionary containing the data and axis information for the 3d cube
        and any fitted model provided

        Returns
        -------
        data_3d : dict
            Dictionary containing the data cube and axis information for the 3d cube
        and any fitted model provided.

        """

        data_3d = {'data':self.data_cube,
                   'velocity':self.get_velocity_values()}

        if self.model_data_cubes is not None:
            if len(np.shape(self.model_data_cubes)) == 3:
                self.model_data_cubes = [self.model_data_cubes]

            for i in range(len(self.model_data_cubes)):
                data_3d[self.model_labels[i]] = self.model_data_cubes[i]
        return data_3d

    def convert_to_kelvin(self, data):
        # Convert to K, not Jy/beam
        freq_kws = ['RESTFRQ', 'RESTFREQ']

        rest_freq_found = False

        for freq_kw in freq_kws:
            if freq_kw in self.header.keys():
                rest_freq_ghz = self.header[freq_kw] / 1e9
                rest_freq_found = True
                break

        if not rest_freq_found:
            raise Warning('Rest freq not found in header keywords')

        bmaj, bmin = self.header['BMAJ'] * 3600, self.header['BMIN'] * 3600

        data = 1.222e6 * data / (rest_freq_ghz ** 2 * bmaj * bmin)

        return data

class Mutlicomponent:
    """
    Class for getting the flattened positions of a n component
    fitted modelled. Assumes the data is a cube, where the
    x and y coordinates coorsponded to the positions from the
    original data, and the z direction is the value of the fit
    for each component. So `data[0, y, x]` is the fitted parameter
    of the first component.

    To get the world coordinates, either load from a fits file
    or give the header.

    If many components were fitted, can limit the components that
    will be considered by setting `maximum_components`. Assumes
    the first component is the first fitted.

    Pixels with no fit need to be set to a single value that can be
    ignored. Default is `numpy.nan`
    """

    def __init__(self, component_cube, header=None, maximum_components=None, no_data_value=np.nan):

        self.header = header
        self.no_data_value = no_data_value
        self.ax = None

        if len(np.shape(component_cube)) == 2:
            component_cube = component_cube[None,:,:]

        #Might only want to view the largest components.
        if maximum_components is None:
            maximum_components = np.shape(component_cube)[0]

        self.maximum_components = maximum_components

        self.component_cube = component_cube

        # positions of the fits in x, y. z = n-1 component number
        if np.isnan(no_data_value):
            self.z_positions, self.y_positions, self.x_positions = np.where(~np.isnan(self.component_cube))
        else:
            self.z_positions, self.y_positions, self.x_positions = np.where(self.component_cube != self.no_data_value)
        self.cube_z_values = self.component_cube[self.z_positions, self.y_positions, self.x_positions]

        #If no header is given, will only have x, y positions
        #With the header, will calculate the world positions
        if self.header is not None:
            self.world_coord_type = util.get_world_type(self.header)

        self.df = self.df()


    @classmethod
    def load_data(cls, filename, header=None, maximum_components=None, key_prefix=None, no_data_value=np.nan, **kwargs):
        """
        If the data needs to be loaded, can initalise the class
        using filenames

        Parameters
        ----------
        filename : str
            file containing fitted components.
        header : astropy.io.Header.header or None, optional
            header of the data. The default is None.
        maximum_components : int, optional
            Assuming 0 is the first fitted component, `maximum_components`
            limits the amount of components you want to load. This
            is more relvent if the n comp fits have been run through
            a clustering algorithm such as `acorns`, which will
            cluster up the points into connected parts, can
            can have 100's of "clusters". The default is None.
        key_prefix : str, optional
            If the data is saved in a dictionary, with each component
            labeled *1, *2 or *1*, *2*, need to specify the key needed
            to access the data as something like `v_%d_err' or `v_%d`
        no_data_value : optional
            Indicates whih pixels have no fit and are to be ignored.
            The default is np.nan.
        **kwargs : dict
            Optional parameters that might be needed to load in the data

        Raises
        ------
        AssertionError
            If the loaded data is a dictionary, AssertionError will be
            raised if `key_prefix` is not set.

        Returns
        -------
        cls
            Initalises the object.

        """

        #if loading from a fits file, output contains the header
        data, header = io.Loader(filename, header=header, **kwargs).lazy_load()

        if isinstance(data, dict):
            if key_prefix is None:
                raise AssertionError('`key_prefix` needs to be set if loading a dictionary')

            data = make_cube_from_dict_maps(dict_of_maps=data,
                                            key_prefix=key_prefix,
                                            amount=maximum_components)

        return cls(data, header, maximum_components, no_data_value)

    def df(self):
        """
        Collects all the flattened arrays into a panda's data frame. By
        convention, a dataframe is abbreviated as df. As dataframe
        is a mix of a dictionary and an object. Can access data both
        in a dictionary-like way (i.e., `ra_data = df['ra_data'])
        or and object-like way (i.e., `ra_data = df.ra_data)

        Returns
        -------
        df : panda.DataFrame
            A panda dataframe containing all the flattened arrays. Note, the
            `component` in `df` are the z positions + 1. This is so that the first
            component has a value of 1, the 2nd a value of 2 etc.

        """
        if self.header is not None:
            ra, dec, glon, glat = self.get_world_positions()

            df = pd.DataFrame(dict(RA=ra, DEC=dec, Velocity=self.cube_z_values, component=self.z_positions + 1, x=self.x_positions, y=self.y_positions, GLON=glon, GLAT=glat))

        else:
            df = pd.DataFrame(dict(Velocity=self.cube_z_values, component=self.z_positions + 1, x=self.x_positions, y=self.y_positions))

        return df

    @functools.lru_cache
    def get_world_positions(self):
        """
        If `header` is given, the world coordinates of the data are
        calculated both in Equatorial and in Galactic.
        TODO: Messy method. Need to split up a bit

        Returns
        -------
        ra_positions : 1darray
            RA positions of all components
        dec_positions : 1darray
            DEC positions of all components
        glon_positions : 1darray
            GLON positions of all components
        glat_positions : 1darray
            GLAT positions of all components

        """

        if self.world_coord_type == 'fk5':
            ra_positions, dec_positions = util.get_pixel2world(self.x_positions, self.y_positions, self.header)
            glon_positions, glat_positions = util.get_world2world(ra_positions, dec_positions, self.world_coord_type, config.coord_conversion[self.world_coord_type])


        if self.world_coord_type == 'galactic':
            glon_positions, glat_positions = util.get_pixel2world(self.x_positions, self.y_positions, self.header)
            ra_positions, dec_positions = util.get_world2world(glon_positions, glat_positions, self.world_coord_type, coord_conversion[self.world_coord_type])

        return ra_positions, dec_positions, glon_positions, glat_positions

    def get_flat2d_component_map(self):
        """
        Generates a 2d map showing the number of components fitted per pixel

        Returns
        -------
        flat_ncomp_map: 2darray
            Map showing the number of components fitted per pixel.

        """
        positions = [self.z_positions, self.y_positions, self.x_positions]
        flat_ncomp_map = make_maps.flat_ncomp_map(self.component_cube, positions)

        return flat_ncomp_map


# class HyperfineModels:
#     def __init__(self, fit_params, tex_label, tau_label, peak_v label, sigma_label, err_labels=None):


#         pass

#     def load_data(filenames, tex_label, tau_label, peak_v label, sigma_label, err_labels=None):

#         fit_params = {}
#         if isinstance(filenames, str):
#             filenames = [filenames]

#         for i in range(len(filenames)):
#             data, header = io.Loader(filename, header=header, **kwargs).lazy_load()

#         if isinstance(data, dict) and len(filenames) > 1:
#             fit_params



def all_fit_param_cubes(dict_of_maps, key_prefixes, err_suffix=None, max_comp=None):

    if isinstance(dict_of_maps, str):
        dict_of_maps = io.Loader(dict_of_maps).lazy_load()[0]

    fit_parma_cubes = {}

    for key in key_prefixes:

        fit_cubes = []
        cube = make_cube_from_dict_maps(dict_of_maps, key, max_comp)
        if err_suffix is not None:
            lower = cube - make_cube_from_dict_maps(dict_of_maps, key+err_suffix[0], max_comp)
            upper = cube + make_cube_from_dict_maps(dict_of_maps, key+err_suffix[1], max_comp)
            fit_parma_cubes[key[:-3]] = np.array([cube, lower, upper])
        else:
            fit_parma_cubes[key[:-3]] = np.array([cube])

    return fit_parma_cubes



def make_cube_from_dict_maps(dict_of_maps, key_prefix, amount=None):
    """
    Converts dictionary 2d data from a fit into a cube

    Parameters
    ----------
    dict_of_maps : dict
        Dictionary containing the data
    key_prefix : str
        key for accessing the data. Different components must be labeled
        with intergers for each fit (i.e., `velo_1`, `velo_2`, `velo_1_err`).
        The key to access these then must have a `%d` where the number is
        (i.e., `velo_%d`, `velo_%d_err`)
    amount : int, optional
        Number of fitted components. If None, will try and get all the
        components.The default is None.

    Returns
    -------
    cube : 3darray
        Contains the fitted parameters at each z slice, where data in y, x is
        the fitted value at that position.

    """

    if amount is None:
        amount = 5000

    cube = []
    try:
        for i in range(amount):
            key = key_prefix % i
            data_2d = dict_of_maps[key]
            cube.append(data_2d)
    except KeyError:
        cube = np.array(cube)
        return cube

    cube = np.array(cube)
    return cube



def reproject_to_new_world(data_2d, header):
    """
    Wrapper to reproject data to new world coordinate system


    Parameters
    ----------
    data_2d : 2darray
        Data to be reprojected
    header : astropy.io.Header.header
        Dictionary-like object containing the world coordinate reference
        values


    Returns
    -------
    data_2d_rpj : 2darray
        Reprojected data

    header : astropy.io.Header.header
        Dictionary-like object containing the world coordinate reference
        values in the reprojected frame

    """
    data_2d_rpj, new_header = util.convert_world_system_of_image(data_2d, header)

    return data_2d_rpj, new_header

if __name__ == "__main__":
    pass