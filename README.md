# hyperfineVisualiser


For options and help,on the functions and classes enter help(class_name) 
or help(function_name) for further documentation. <br />
Required packages: <br />
* dash
* plotly
* numpy
* reproject
* astropy
* pandas
* pickle
* json

NB, I use every option. I have not tested if code/visualiser works if some optional settings are not used.

<h2>TLDR</h2>
Open up example_tempate.py <br />
Set your file paths between the equal signs <br />
    Get file containing velocity peak per component. Will plot in 3d. Call it 'component_filename' <br />
    Get some 2d maps to look at a map. Call it data_2d_paths=['file1', 'file2', 'file3' .. etc] and label them data_2d_labels=['label1', 'label2', 'label3'] <br />
    Get the actual data cube to look at the actual spectrum. Call it `data_3d_path` <br />

Chuck these into the classes:
> data_obj = get_data.Mutlicomponent : component_filename <br />
> data_2d_obj = get_data.Data_2d : ['file1', 'file2', 'file3'], ['label1', 'label2', 'label3'] <br />
> real_data_obj =  get_data.Data_3d <br />

Chuck classes into visualiser and run
vis = dash_visualise.Visualiser(data_obj, data_2d_obj, real_data_obj)
vis.run_visualiser()

EXTRAS:
    Show total model spectra. Add this to `real_data_obj`
    Show induvidual model spectra. Set it up, add to `dash_visualise.Visualiser`

----------------------------------------------------------------------------

Provide the path for the fitted velocity peak components : `component_filename`
Should either be stored as a cube, where the z direction is the component number
and y,x are the positions, or can be stored as a dictionary with labels:
    `some_string_%d` where %d is an interger representing the component number

If loading from a saved dictionary, the key/label must be given in the format
`some_string_%d`. For example (using the class method load.data):

> data_obj = get_data.Mutlicomponent.load_data( <br />
> filename=component_filename, <br />
> header=header, #provide header if needed <br />
> maximum_components=7, #optional <br />
> **key_prefix=velocity_%d**, <br />
> no_data_value=np.nan #optional <br />
> )

Can also load in yourself and just provide the cube directly to component class

component_filename = path_to_data

> data_obj = get_data.Mutlicomponent( <br />
> component_cube=some_component_cube, <br />
> header=header, <br />
> maximum_components=max_components, <br />
> no_data_value=np.nan <br />
> )

---------------------------------------------------------------------


For the 2d data shown in the visualiser, this must be loaded
from files : `data_2d_paths`= [file1, file2]
For each file give them a label. If the loaded file is a dictionary
the label needs to be the key that accesses the map:

If no header is given, the first file must be a fitsfile with
the header infomation.

If wanted, the 2d data will be reprojected to a different celestial frame. So if
the data is in RA, DEC, it will be reprojected to GLON GLAT. Example:

data_2d_paths= [filepath_with_header, file_is_dict]
data_2d_labels = ['Mom0', 'a_dict_key']

> data_2d_obj = get_data.Data_2d( <br />
> filenames=data_2d_paths, <br />
> labels=data_2d_labels, <br />
> header=None, #optional, but must be given if files have no headers <br />
> reproject_all=True <br />
> )

get_data.Data_2d has to be initalised using files. This makes adding maps
to the class dictionary weird. This class was wrote assuming the viewed maps
would be mom0 maps etc, which would always be saved.  So class is initalised
with the filenames only.For now, to add a non saved map, initalise the class
from a file, and add the map to the data dictionary `all_data` using method
`add_new_map_to_all_all. For example:

#get_data.Mutlicomponent method
ncomp_2d = data_obj.get_flat2d_component_map() 

data_2d_obj.add_new_map(data=ncomp_2d, label='ncomp')


--------------------------------------------------------------------


To load in the data cube, can load from file or give a cube to the object. Example from file:
data_3d_path = your_data_cube

real_data_obj = get_data.Data_3d.load_data( <br />
> cube_filename=data_3d_path, <br />
> model_cube_filename=None, <br />
> header=None, <br />
> data_is_freq=False, <br />
> rest_freq=None, <br />
> model_labels=['fit', 'lower', 'upper'], <br />
> in_kelvin=True #True if in Kelvin, False if in Jy/beam <br />
> )

Example as cube:

datacube, header = load in

> real_data_obj = get_data.Data_3d(<br />
> data_cube=datacube,<br />
> header=header,<br />
> model_data_cubes=None,<br />
> data_is_freq=False,<br />
> rest_freq=None,<br />
> model_labels=['fit', 'lower', 'upper'], <br />
> in_kelvin=True, #True if in Kelvin, False if in Jy/beam <br />
> model_in_kelvin=True <br />
> )

To show the fitted model cube, either give a file name if loading from
`cls.load_data` or give the cube in a simular way as for the data_cube. Example:

> real_data_obj = get_data.Data_3d.load_data(<br />
> cube_filename=data_3d_path,<br />
> model_cube_filename=data_3d_fit_upper_lower_path,<br />
> header=None,<br />
> data_is_freq=False,<br />
> rest_freq=None,<br />
> model_labels=['fit', 'lower', 'upper'],<br />
> in_kelvin=False<br />
> model_in_kelvin=False<br />
> )

Setting `in_kelvin` and `model_in_kelvin` means the data is in Jy/beam 

If the model cube has errors, either enter the data as a 4d hypercube 
i.e., fit_lower_upper = np.array([model_cube, lower_cube, upper_cube])
or give a single file with a 4d hyper cube, or list the filenames for
the three model fits. Example of from file:

> real_data_obj = get_data.Data_3d.load_data(<br />
> cube_filename=data_3d_path,<br />
> model_cube_filename=[model_filepath, lower_filepath, upper_filepath]<br />
> header=None,<br />
> data_is_freq=False,<br />
> rest_freq=None,<br />
> model_labels=['fit', 'lower', 'upper'], #change the order here to match your data <br />
> in_kelvin=False<br />
> )

If the data is in frequency, `set data_is_freq` to True to convert to velocity.
Will use the header to get convert to velocity. If this can't be done, give
the rest frequency using the variable `rest_freq`


If you want to see the paramers for each induvidual fittted component,
need to make a dictionary containing the parameter map in shape
(1, velocity length, ncomp) where 1 if just the model, 3 if the lower
and upper error bounds are included.

To run the visualiser, enter all the objects initalised from the files into
the class `dash_visualise.Visualiser`. Example:



So a dict of models['tex'][0][5][125][128] means: <br />
'tex' = the excitation temperature <br />
[0] is the best fit, (so [1] is the values of the lower bound, [2] upper) <br />
[5] = the 5 velocity component found along the line of sight <br />
125 the y coordinate <br />
128 the x coordinate

No classes currently exist to support different this load and prep, so
if you want this to show on the plot, you have to set it up yourself
for the visualiser and enter it using the variable `single_fits`.

> vis = dash_visualise.Visualiser(<br />
> data_obj=data_obj, <br />
> data_2d_obj=data_2d_obj, <br />
> real_data_obj=real_data_obj,<br />
> single_fits = single_fits,<br />
> zlims=None #will default to the max velocitties when inialising <br />
> )

run the visuliser: copy the address printed into the url. Works with chrome. Hope it
works for other browsers...
vis.run_visualiser()
