# -*- coding: utf-8 -*-
"""
sdc326
Template for running the visualiser. For options and help,
on the functions and classes enter help(class_name) or help(function_name)
for further documentation and options for on loading in the data

NOTES ON VISUALISER:
>Hover over point in 3d scatter to see the spectrum there
>Hold right click to pan about
>Hold left click to rotate
>For more complex rotations when viewing in 3d, click the "orbit" button
>To save one image, click the "save to png button"

>Clicking on a a point on the 3d scatter zooms into the 2d map at that position.
 This zoom on click can be toggled off on the visualiser
 NB: When viewed in world coords, the click to zoom on the 3d scatter is bugged.
 It will sometimes not be able to show the zoomed in position. But, if you use
 pixel coordinates, no buggy zoom occurs. Zoom-in is disabled for reprojections


>If the error _hash appears. Just wait a couple of seconds or refresh page

> 2d map showing the number of components fitted per pixel is automatically
made and loaded into visualser.
"""

from nfitVisualiser import get_data, dash_visualise
from nfitVisualiser import conversion_utilities as util
import numpy as np
from astropy.io import fits


#setting paths =====================================================================
DRIVE_PATH =
PROJECT_PATH = DRIVE_PATH + 'andy_alma_test\\sdc326\\'

alma_setup = 'TP+7m+12m'

component_filename = PROJECT_PATH + 'SDC326_n2h+_%s.image.pbcor_maps.pkl' % alma_setup

# If one 2d data file has header, list it first in 'data_2d_paths`
data_2d_paths = [component_filename]
data_2d_labels = ['chisq_red']

data_3d_path = PROJECT_PATH + 'SDC326_n2h+_%s.image.pbcor.fits' % alma_setup
data_3d_fit_upper_lower_path = PROJECT_PATH + 'SDC326_n2h+_%s.image.pbcor_fit_cube.npy' % alma_setup


#Fit params and the 2d data maps are not saved with headers so am loading it here
with fits.open(data_3d_path) as hdu:
    header = hdu[0].header

#Remove 3d info from header
header = util.remove_axis_from_fits_header(header, 4)
header = util.remove_axis_from_fits_header(header, 3)

key_prefix     = 'v_%d'     #Set to None if not needed, or exlcude from function call
max_components = None       #Set to None if not needed, or exlcude from function call
velo_start     = [-50, -30] #Set to None if not wanted, or exlcude from function call
start_spectrum = None       #Give [x, y] to initalise with a specific spectrum

#initalise the objects and load in the data =======================================

#Fit params for 3d scatter. Loading from pickle so need to manually provide header
print('Loading component fits')
data_obj = get_data.Mutlicomponent.load_data(
    filename=component_filename,
    header=header,
    maximum_components=max_components,  #Optional
    key_prefix=key_prefix,              #Optional
    no_data_value=np.nan                #Value where no fit was made. Default: nan
)

#2d data maps to view
#The 2d files have header information, so need to manually provide it
print('Loading and reprojecting 2d data maps')
data_2d_obj = get_data.Data_2d(
    filenames=data_2d_paths,
    labels=data_2d_labels,
    header=header,
    reproject_all=True #False means 2d maps will not be reprojected to galactic
)

#actual data cube and the model spectrum (with upper and lower error bounds)
print('Loading in data cube and model spectra')
real_data_obj = get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=data_3d_fit_upper_lower_path,
    header=None,                                      #The data cube will have a header
    data_is_freq=False,                               #Set True if data is in frequency
    rest_freq=None,                                   #Give rest frequency if header does not contain it
    model_labels=['fit', 'lower', 'upper'],           #Order of models in 4d hypercube
    in_kelvin=False                                   #Set False if data in in Jy/beam
)

#The function only works for if the models params are labelled in a dictionary
#with a label for each component, and the label names accend as
#v_1, v_2 v_3 etc and errors are v_1_err_down, v_1_err_up etc << dont use these errors
single_model_fits = get_data.all_fit_param_cubes(component_filename, ['sigma_%d', 'tau_%d', 'tex_%d', 'v_%d'])


##initalise the visualiser parameters
print('Initalising visualiser')
vis = dash_visualise.Visualiser(
    data_obj=data_obj,             #peak velocities, xy, RA-DEC, GLON-GLAT coords
    data_2d_obj=data_2d_obj,       #2d data maps
    real_data_obj=real_data_obj,   #datacube
    single_fits=single_model_fits, #datacube model
    zlims=velo_start,
    inital_view_spectrum=start_spectrum
)

# #run the visuliser: copy the address into the url. Works with chrome. Hope it
# #works for other browsers...
# vis.run_visualiser()
