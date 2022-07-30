# -*- coding: utf-8 -*-
"""
Codes for generating maps from n-component fitted data
"""

import numpy as np
import copy

#Relatve velocity positions and heights of N2H+
v_lines = {
    'n2hp10_01': -7.9930,
    # 'J1-0_02': -7.9930,
    # 'J1-0_03': -7.9930,
    'n2hp10_04': -0.6112,
    # 'J1-0_05': -0.6112,
    # 'J1-0_06': -0.6112,
    'n2hp10_07': 0.0000,
    'n2hp10_08': 0.9533,
    # 'J1-0_09': 0.9533,
    'n2hp10_10': 5.5371,
    # 'J1-0_11': 5.5371,
    # 'J1-0_12': 5.5371,
    'n2hp10_13': 5.9704,
    # 'J1-0_14': 5.9704,
    'n2hp10_15': 6.9238,
}
strength_lines = {
    'n2hp10_01': 0.025957 + 0.065372 + 0.019779,
    # 'J1-0_02': 0.065372,
    # 'J1-0_03': 0.019779,
    'n2hp10_04': 0.004376 + 0.034890 + 0.071844,
    # 'J1-0_05': 0.034890,
    # 'J1-0_06': 0.071844,
    'n2hp10_07': 0.259259,
    'n2hp10_08': 0.156480 + 0.028705,
    # 'J1-0_09': 0.028705,
    'n2hp10_10': 0.041361 + 0.013309 + 0.056442,
    # 'J1-0_11': 0.013309,
    # 'J1-0_12': 0.056442,
    'n2hp10_13': 0.156482 + 0.028705,
    # 'J1-0_14': 0.028705,
    'n2hp10_15': 0.037038,
}

v_line_centre_array = np.array([v_lines[line_name] for line_name in v_lines.keys()])

def components_mask(component_cube, positions=None, no_data_value=np.nan):
    """
    Takes a cube where the z direction is the fitted parameter per found component
    and calculates the number of components present for each y,x position

    Parameters
    ----------
    component_cube : 3darray
        3darray containing the fitted parameter for each component found at each
        y,x position.
    positions : 3xn array like, optional
        The z y x postions of where a fit was made. The default is None.
    no_data_value:
        The value of pixels that do not contain a fit

    Returns
    -------
    mask_component_cube : 3darray
        3d mask where a component was found

    """
    mask_component_cube = copy.copy(component_cube)

    if positions is None:
        if np.isnan(no_data_value):
            z, y, x = np.where(~np.isnan(component_cube))
        else:
            z, y, x = np.where(component_cube != no_data_value)
    else:
        z, y, x = positions

    mask_component_cube[z, y, x] = 1

    return mask_component_cube

def flat_ncomp_map(component_cube, positions=None, no_data_value=np.nan):
    """
    Parameters
    ----------
    component_cube : 3darray
        3darray containing the fitted parameter for each component found at each
        y,x position.
    positions : 3xn array like, optional
        The z y x postions of where a fit was made. The default is None.
    no_data_value:
        The value of pixels that do not contain a fit

    Returns
    -------
    flat_components : 2darray
        2d map showing the number of components at each y,x location..

    """
    mask_component_cube = components_mask(component_cube, positions, no_data_value)
    flat_components = np.nansum(mask_component_cube, axis=0)

    return flat_components


def gaussian(x, amp, centre, width):
    """
    Author: Tom williams
    Evaluate a Gaussian on a 1D grid.
    Calculates a Gaussian, using :math:`f(x) = A \exp[-0.5(x - \mu)^2/\sigma^2]`.
    Args:
        x (np.ndarray): Grid to calculate Gaussian on.
        amp (float or np.ndarray): Height(s) of curve peak(s), :math:`A`.
        centre (float or np.ndarray): Peak centre(s), :math:`\mu`.
        width (float or np.ndarray): Standard deviation(s), :math:`\sigma`.
    Returns:
        np.ndarray: Gaussian model array
    """

    y = amp * np.exp(- (x - centre) ** 2 / (2 * width ** 2))
    return y

def hyperfine_structure_all(vel, t_ex, tau, v_centre, line_width, return_hyperfine_components=False, logged_tau=False):
    """
    Author: Tom williams, modified here for vectorisation
    Create hyperfine intensity profile.
    Takes line strengths and relative velocity centres, along with excitation temperature and optical depth to produce
    a hyperfine intensity profile.
    Args:
        vel (np.ndarray): Array of velocities to calculate line intensity at (km/s).
        t_ex (float): Excitation temperature (K).
        tau (float): Total optical depth of the line.
        v_centre (float): Central velocity of strongest component (km/s).
        line_width (float): Width of components (assumed to be the same for each hyperfine component; km/s).
        return_hyperfine_components (bool): Return the intensity for each hyperfine component. Defaults to False.
    Returns:
        Intensity for each individual hyperfine component (if `return_hyperfine_components` is True), and the total
            intensity for all components
    """
    T_BACKGROUND = 2.73
    if logged_tau:
        tau = np.exp(tau)
    strength = np.array([tau * strength_lines[line_name] for line_name in v_lines.keys()])
    #If I had
    tau_components = gaussian(vel[:,None,None], strength, v_line_centre_array[:,None] + v_centre,
                              line_width)

    total_tau = np.nansum(tau_components, axis=1)
    intensity_total = (1 - np.exp(-total_tau)) * (t_ex - T_BACKGROUND)

    if not return_hyperfine_components:
        return intensity_total

    else:
        intensity_components = (1 - np.exp(-tau_components)) * (t_ex - T_BACKGROUND)

        return intensity_components, intensity_total


if __name__ == "__main__":
    pass

