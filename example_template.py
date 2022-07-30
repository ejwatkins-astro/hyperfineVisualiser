# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:19:31 2021

@author: Liz_J
"""

import nfitVisuliser as nfit
import numpy as np
from astropy.io import fits


#setting paths
DRIVE_PATH = 'C:\\Users\\Liz_J\\Documents\\'
PROJECT_PATH = DRIVE_PATH + 'alma-mine-full_N2Hp\\v1p2\\'

alma_setup = '12m+7m+tp'

component_filename = PROJECT_PATH + 'fitted\\g316_75_%s_maps.pkl' % (alma_setup) #'maps.pkl'

data_2d_paths = [PROJECT_PATH + 'derived\\g316_75_%s_n2hp10_strict_mom0.fits' % (alma_setup), PROJECT_PATH + component_filename]
data_2d_labels = ['mom0', 'chisq_red']

data_3d_path = PROJECT_PATH + 'postprocess\\g316_75\\g316_75_12m+7m+tp_n2hp10_pbcorr_trimmed_k.fits'
data_3d_fit_upper_lower_path = PROJECT_PATH + 'fitted\\g316_75_12m+7m+tp_fit_cube.npy'

#if you get an error about little endian, or big endian, something like this,
# for the data causing it, add .byteswap().newbyteorder() to the data and
# it will fix it

#component data has no header so need to load that separatly
with fits.open(data_2d_paths[0])[0] as hdulist:
    header = hdu.header

key_prefix = 'v_%d'
max_components = 7
zlims=None # can set the range of velocities to initalise at

#initalise the objects and load in the data =============================

#Fitted parameters for 3d scatter
data_obj = nfit.get_data.Mutlicomponent.load_data(
    filename=component_filename,
    header=header,
    maximum_components=maximum_components,
    key_prefix=key_prefix,
    no_data_value=np.nan
)

#first file contains header information so do not need to give header info

#2d data maps to view
data_2d_obj = nfits.get_data.Data_2d(
    filenames=data_2d_paths,
    labels=data_2d_labels,
    header=None,
    reproject_all=True
)

#actual data cube and the model spectrum (with upper and lower error bounds)
real_data_obj = nfits.get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=data_3d_fit_upper_lower_path,
    header=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'upper', 'lower']
)

#initalise the visuliser parameters
vis = nfit.dash_visulise(
    data_obj=data_obj,
    data_2d_obj=data_2d_obj,
    real_data_obj=real_data_obj,
    available_maps=data_2d_labels,
    zlims=zlims)

#run the visuliser
vis.run_visuliser()