# -*- coding: utf-8 -*-
"""
Some functions for converting between units.
TODO: write function documentation
"""

import numpy as np
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import reproject as rpj
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd

from . import config

def remove_axis_from_fits_header(header, axis=3): #fits_header3D_to_fits_header2d
    """
    This function takes in a astropy fits header and removes
    dimention keys for the axis entered in the function. If no axis is
    entered, the 3rd axis is assumed the one that needs to be removed
    Returns the adjusted header
    """

    for i in range(len(header)-1,-1,-1):
        if str(axis) in list(header.keys())[i]:
            del header[i]
    header['NAXIS'] = axis-1
    try:
        del header['WCSAXES']
    except KeyError:
        pass

    return header

def convert_pixel2world(paired_pixel_coordinates, header):
    """
    Load the WCS information from a fits header, and use it
    to convert world coordinates to pixel coordinates.
    """
    w = wcs.WCS(header)
    paired_world_coordinates= w.all_pix2world(paired_pixel_coordinates, 0)

    return paired_world_coordinates


def get_pixel2world(x_position, y_position, header):
    """
    Preps two arrays/lists of pixel positions, converts them
    and then outputs them back as two arrays in world coordinates
    """

    xy_position = np.column_stack((x_position, y_position))
    xy_position_world = convert_pixel2world(xy_position, header)

    x_position_world = xy_position_world[:,0]
    y_position_world = xy_position_world[:,1]

    return x_position_world, y_position_world

def convert_between_worlds(world_coordinates, current_world, new_world):
    """
    This method converts the world co-ordinate types from equatorial to
    galactic

    """

    current_world_x, current_world_y = world_coordinates[:,0], world_coordinates[:,1]

    current_sky_coords = SkyCoord(current_world_x*u.degree, current_world_y*u.degree, frame=current_world)
    new_sky_coords = current_sky_coords.transform_to(new_world).to_string()


    new_world_x, new_world_y = map(list, zip(*[coords.split(' ') for coords in new_sky_coords]))
    new_world_coordinates = np.column_stack((new_world_x, new_world_y)).astype(float)

    return new_world_coordinates

def get_world2world(x_position, y_position, current_world, new_world):
    """
    Preps two arrays/lists of world positions, converts them to the new
    """

    xy_position = np.column_stack((x_position, y_position))
    xy_position_world = convert_between_worlds(xy_position, current_world, new_world)

    x_position_world = xy_position_world[:,0]
    y_position_world = xy_position_world[:,1]

    return x_position_world, y_position_world

def get_world_type(header):
    try:
        current_world = header['RADESYS']
    except KeyError: # will guess from the CTYPE using `world_types` dictionary
        ctype = header['CTYPE1']
        for key in config.world_types.keys():
            if key in ctype:
                current_world = config.world_types[key]
                break
            else:
                raise KeyError('World system could not be identified')
    return current_world.lower()

def convert_header_to_new_world(data, header):

    current_world = get_world_type(header)
    new_world = config.coord_conversion[current_world]

    wcs_out, shape_out = find_optimal_celestial_wcs([(data, header)], frame=new_world)

    new_header_wcs = wcs_out.to_header()

    new_header = header.copy()
    for key in new_header_wcs:
        new_header[key] = new_header_wcs[key]

    new_header['NAXIS2'], new_header['NAXIS1'] = shape_out

    return new_header

def convert_world_system_of_image(image, header):

    new_header = convert_header_to_new_world(image, header)

    new_image = rpj.reproject_interp((image, header), new_header)[0]

    return new_image, new_header

def get_axis_values_from_fits(header, axis_dimension=1):
    """
    Gets the physical units of an axis from a fits header reference values
    can be merged with extend in an imshow if wcs plotting cannot be used
    """
    ax_No = str(axis_dimension)

    fits_info = ['CRPIX', 'CRVAL', 'CDELT', 'NAXIS']

    try:
        projection_info = [header[info + ax_No] for info in fits_info]
    except KeyError:
        fits_info[2] = 'CD' + ax_No + '_'
        projection_info = [header[info+ax_No] for info in fits_info]

    return get_values_for_axis_using_fits_info(*projection_info)

def get_values_for_axis_using_fits_info(index_of_reference_pixel, value_at_reference_pixel,\
                                        size_of_each_pixel, length_of_axis):
    """
    ok, so in fits headers, the index starts at 1, therefore I need to take 1
    away for python indexing
    """
    index_of_reference_pixel -= 1 # python indexing starts at 0
    value_at_start_pixel = value_at_reference_pixel - index_of_reference_pixel * size_of_each_pixel

    values_of_axis = size_of_each_pixel * np.arange(0, length_of_axis, 1) + value_at_start_pixel

    if len(values_of_axis) != length_of_axis:
        raise ValueError('The length of the new axis values does not equal '\
                         'the given length of the axis. New axis length = %d'\
                         ' given axis length =%d' %(len(values_of_axis),length_of_axis))
    return values_of_axis

def frequency2velocity(rest_frequency, frequency_values):
    """
    Converts frequency to velocity given the rest frequency
    """

    from astropy import units as u # Move to top
    rest_frequency = rest_frequency * u.Hz
    freq_to_vel = u.doppler_radio(rest_frequency)
    velocity_values = (frequency_values * u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)[:].value

    return velocity_values

if __name__ == "__main__":
    pass