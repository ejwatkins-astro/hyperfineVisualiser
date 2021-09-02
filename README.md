# hyperfineVisualiser

Template for running the visualiser. For options and help,
on the functions and classes enter help(class_name) or help(function_name)
for further documentation.

Provide the path for the fitted velocity peak components : `component_filename`
Should either be stored as a cube, where the z direction is the component number
and y,x are the positions, or can be stored as a dictionary with labels:
    `some_string_%d` where %d is an interger representing the component number


If loading from a saved dictionary, the key/label must be given in the format
`some_string_%d`. For example (using the class method load.data):

data_obj = get_data.Mutlicomponent.load_data(
    filename=component_filename,
    header=header, #provide header if given file has no header info
    maximum_components=7, #optional
    key_prefix=velocity_%d,
    no_data_value=np.nan #optional
    )


Can also load in yourself and just provide the cube
directly to component class

component_filename = path_to_data

data_obj = get_data.Mutlicomponent(
    component_cube=some_component_cube,
    header=header,
    maximum_components=max_components,
    no_data_value=np.nan
    )

-------------------------------------

For the 2d data shown in the visualiser, this must be loaded
from files : `data_2d_paths`= [file1, file2]
For each file give them a label. If the loaded file is a dictionary
the label needs to be the key that accesses the map:

If no header is given, the first file must be a fitsfile with
the header infomation.

If wanted, the 2d data will be reprojected to a different celestial frame. So if
the data is in RA, DEC, it will be reprojected to GLON GLAT

Example:

data_2d_paths= [filepath_with_header, file_is_dict]
data_2d_labels = ['Mom0', 'a_dict_key']

data_2d_obj = get_data.Data_2d(
    filenames=data_2d_paths,
    labels=data_2d_labels,
    header=None, #optional, but must be given if files have no headers
    reproject_all=True
)

get_data.Data_2d has to be initalised using files. This makes adding maps
to the class dictionary weird. This class was wrote assuming the viewed maps
would be mom0 maps etc, which would always be saved.  So class is initalised
with the filenames only.For now, to add a non saved map, initalise the class
from a file, and add the map to the data dictionary `all_data` using method
`add_new_map_to_all_all
For example:

ncomp_2d = data_obj.get_flat2d_component_map() #get_data.Mutlicomponent method
data_2d_obj.add_new_map(data=ncomp_2d, label='ncomp')


---------------------------------------

To load in the data cube, can load from file or give a cube to the object
Example from file:
data_3d_path = your_data_cube

real_data_obj = get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=None,
    header=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'lower', 'upper'],
    in_kelvin=True #True if in Kelvin, False if in Jy/beam
)

Example as cube:

datacube, header = load in

real_data_obj = get_data.Data_3d(
    data_cube=datacube,
    header=header,
    model_data_cubes=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'lower', 'upper'],
    in_kelvin=True, #True if in Kelvin, False if in Jy/beam
    model_in_kelvin=True
)

To show the fitted model cube, either give a file name if loading from
`cls.load_data` or give the cube in a simular way as for the data_cube
Example:

real_data_obj = get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=data_3d_fit_upper_lower_path,
    header=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'lower', 'upper'],
    in_kelvin=False
    model_in_kelvin=False
)
setting `in_kelvin` and `model_in_kelvin` means the data is in Jy/beam and
we need to convert it to kelvin

If the model cube has errors, either enter the data as a 4d hypercube 
i.e., fit_lower_upper = np.array([model_cube, lower_cube, upper_cube])
or give a single file with a 4d hyper cube, or list the filenames for
the three model fits 
Example of from file:

real_data_obj = get_data.Data_3d.load_data(
    cube_filename=data_3d_path,
    model_cube_filename=[model_filepath, lower_filepath, upper_filepath]
    header=None,
    data_is_freq=False,
    rest_freq=None,
    model_labels=['fit', 'lower', 'upper'], #change the order here to match your data 
    in_kelvin=False
)

If the data is in frequency, `set data_is_freq` to True to convert to velocity.
Will use the header to get convert to velocity. If this can't be done, give
the rest frequency using the variable `rest_freq`


If you want to see the paramers for each induvidual fittted component,
need to make a dictionary containing the parameter map in shape
(1, velocity length, ncomp) where 1 if just the model, 3 if the lower
and upper error bounds are included.

To run the visualiser, enter all the objects initalised from the files into
the class `dash_visualise.Visualiser`

Example:



So a dict of models['tex'][0][5][125][128] means:
'tex' = the excitation temperature
[0] is the best fit, (so [1] is the values of the lower bound, [2] upper)
[5] = the 5 velocity component found along the line of sight
125 the y coordinate
128 the x coordinate

No classes currently exist to support different this load and prep, so
if you want this to show on the plot, you have to set it up yourself
for the visualiser and enter it using the variable `single` fits.

vis = dash_visualise.Visualiser(
    data_obj=data_obj,
    data_2d_obj=data_2d_obj,
    real_data_obj=real_data_obj,
    single_fits = None,
    zlims=None #will default to the max velocitties when inialising
)

with induvidual components shown on spetrum plot:
zlims = [-45, -35]

vis = dash_visualise.Visualiser(
    data_obj=data_obj,
    data_2d_obj=data_2d_obj,
    real_data_obj=real_data_obj,
    single_fits = None,
    zlims=zlims
)

#run the visuliser: copy the address into the url. Works with chrome. Hope it
#works for other browsers...
vis.run_visualiser()
