# -*- coding: utf-8 -*-
"""
Template for running the visualiser. For options and help,
on the functions and classes enter help(class_name) or help(function_name)
for further documentation.
"""

from nfitVisualiser import get_data, dash_visualise
from nfitVisualiser import conversion_utilities as util
import numpy as np
from astropy.io import fits

from nfitVisualiser import make_maps


#setting paths
DRIVE_PATH =
PROJECT_PATH = DRIVE_PATH + 'andy_alma_test\\sdc24p489\\'

alma_setup = '12m+7m+tp'

component_filename = PROJECT_PATH + 'SDC24p489-SPW07-N2Hp-uvb-merged-prim_maps.pkl'

data_2d_paths = [component_filename]
data_2d_labels = ['chisq_red']

data_3d_path = PROJECT_PATH + 'SDC24p489-SPW07-N2Hp-uvb-merged-prim.fits'
data_3d_fit_upper_lower_path = PROJECT_PATH + 'SDC24p489-SPW07-N2Hp-uvb-merged-prim_fit_cube.npy'

#if you get an error about little endian, or big endian, or something like this,
# for the data causing it, add .byteswap().newbyteorder() to the data and
# it will fix it

#component data has no header so need to load that separatly
with fits.open(data_3d_path) as hdu:
    header = hdu[0].header

header = util.remove_axis_from_fits_header(header, 4)
header = util.remove_axis_from_fits_header(header, 3)

key_prefix = 'v_%d'
max_components = None
zlims=None # can set the range of velocities to initalise at

#initalise the objects and load in the data =======================================

#Fitted parameters for 3d scatter. Loading from pickle so need header
print('Loading component fits')
data_obj = get_data.Mutlicomponent.load_data(
    filename=component_filename,
    header=header,
    maximum_components=max_components,
    key_prefix=key_prefix,
    no_data_value=np.nan
)
#2d map showing the number of components fitted per pixel
# ncomp_2d = data_obj.get_flat2d_component_map()


#first file contains header information so do not need to give header info,
#if no files have header infomation, give it below. If one file contains
#header information, list it first in the `data_2d_paths` list

#2d data maps to view
print('Loading and reprojecting 2d data maps')
data_2d_obj = get_data.Data_2d(
    filenames=data_2d_paths,
    labels=data_2d_labels,
    header=header,
    reproject_all=True
)


#actual data cube and the model spectrum (with upper and lower error bounds)
print('Loading in data cube and model spectra')
real_data_obj = get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=data_3d_fit_upper_lower_path,
    header=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'lower', 'upper'],
    in_kelvin=False
)

#The function only works for if the models params are labelled in a dictionary
#with a label for each component, and the label names assend as
#v_1, v_2 v_3 etc and errors are v_1_err_down, v_1_err_up etc
fit_lower_upper = get_data.all_fit_param_cubes(component_filename, ['sigma_%d', 'tau_%d', 'tex_%d', 'v_%d'], err_suffix=['_err_down', '_err_up'])


#initalise the visualiser parameters
print('Initalising visualiser')
vis = dash_visualise.Visualiser(
    data_obj=data_obj,
    data_2d_obj=data_2d_obj,
    real_data_obj=real_data_obj,
    single_fits = fit_lower_upper,
    zlims=zlims
)

# #run the visuliser: copy the address into the url. Works with chrome. Hope it
# #works for other browsers...
vis.run_visualiser()
