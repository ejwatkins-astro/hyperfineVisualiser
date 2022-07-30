# -*- coding: utf-8 -*-
"""
Program to initialise and run the visualiser using the web based modules,
plotly and dash
"""

import numpy as np
import math
import copy

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import callback_context as ctx
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from flask_caching import Cache

from . import config
# from . import plotly_functions as plotly_func
from . import conversion_utilities as util
from . import make_maps



# Dictionary for modifying how the spectrum plots
spectrum_params = {
    'data':{
        'name':'Data',
        'mode':'lines',
        'showlegend':False
    },
    'fit':{
        'name':'Fit',
        'mode':'lines',
        'showlegend':False,
        'line':dict(color='rgba(255, 0, 0, 0.8)', width=3)#'rgb(31, 119, 180)')
    },
    'upper':{
        'name':'Upper bound',
        'mode':'lines',
        'showlegend':False,
        'line':dict(width=0),
        'marker':dict(color="#444")
    },
    'lower':{
        'name':'Lower bound',
        'mode':'lines',
        'showlegend':False,
        'line':dict(width=0),
        'marker':dict(color="#444"),
        'fillcolor':'rgba(68, 68, 68, 0.3)',
        'fill':'tonexty',
        }
}

class Visualiser:
    """
    Visualiser for fitted velocity data.
    """
    def __init__(self, data_obj, data_2d_obj, real_data_obj, single_fits=None, zlims=None, inital_view_spectrum=None, logged_tau=False):

        self.data_obj = data_obj
        self.data_2d_obj = data_2d_obj
        self.single_fits = single_fits
        self.inital_view_spectrum = inital_view_spectrum
        self.logged_tau = logged_tau

        #2d map showing the number of components fitted per pixel
        self.ncomp_2d = self.data_obj.get_flat2d_component_map()
        self.data_2d_obj.add_new_map(self.ncomp_2d, 'ncomp')


        self.real_data_obj = real_data_obj
        self.df = self.data_obj.df
        self.header = self.data_obj.header
        self.available_maps = self.data_2d_obj.labels

        self.all_data_3d = copy.copy(self.real_data_obj.all_data_3d)
        self.velocity_values = self.all_data_3d.pop('velocity')

        self.all_data_2d = self.data_2d_obj.all_data

        self.world_coord_type = util.get_world_type(self.header)

        if zlims is None:
            self.zlims = self.get_zlims()
        else:
            self.zlims = [rounddown(zlims[0], 1), roundup(zlims[1], 1)]

    def get_zlims(self):
        """
        Finds suitable starting velocities for 3d scatter plot

        Returns
        -------
        [zmin, zmax]: list
            Suitable lower and upper bounds for the velocity

        """
        zmin = rounddown(self.df.Velocity.min(), 1)
        zmax = roundup(self.df.Velocity.max(), 1)

        return [zmin, zmax]

    def get_velo_slider_values(self):
        """
        Gets values for the velocity slider. Rounds to the biggest whole 10

        Returns
        -------
        velo_slider : dict
            Parameters needed to initalise and set the velocity slider.

        """
        min_velo, max_velo, velocity_slider = velocity_silder_values(self.velocity_values, 10)

        velo_slider = {
            'id':'velocity-chooser',
            'min':min_velo,
            'max':max_velo,
            'step':1,
            'marks':velocity_slider,
            'value': self.zlims
            }

        return velo_slider

    def get_colour_slider_values(self):
        """
        Gets the inital values of the slider for he first 2d map lited

        Returns
        -------
        colour_slider : dict
            Parameters needed to initalise and set the colour scale slider
            for the 2d image

        """

        starting_data_2d = self.all_data_2d[self.world_coord_type]['data'][self.available_maps[0]][0]
        min_val, max_val, data_2d_slider = data2d_silder_values(starting_data_2d)

        colour_slider = {
            'id':'2d-colour-range',
            'min':min_val,
            'max':max_val,
            'step':0.1,
            'marks':data_2d_slider,
            'value':[min_val, max_val]
            }

        return colour_slider

    def get_random_coords(self):
        """
        Gets random starting values for the spectum being shown

        Returns
        -------
        [rand_x, rand_y]
            The random x and y positions (in pixel coordinates) of a random pixel

        """
        rand_int = np.random.randint(len(self.df.x))

        rand_x = self.df.x[rand_int]
        rand_y = self.df.y[rand_int]

        return [rand_x, rand_y]

    def create_spectrum(self, spectra_dict, title, single_plot_param=None, single_comp_specta=None):
        """
        Creates each spectrum to be plotted

        Parameters
        ----------
        spectra_dict : dict
            Dictionary containing the visual parameters of the plot.
        title : str
            String which annotates the plot with.

        Returns
        -------
        fig : plolty fig
            Plotly fig object that is passed to the app for plotting

        """
        scs = [go.Scatter(
                x=self.velocity_values,
                y=spectra_dict[key],
                **spectrum_params[key]
            ) for key in self.all_data_3d.keys()]

        if self.single_fits is not None:

            scs1 = [go.Scatter(
                x=self.velocity_values,
                y=spectra_dict[key],
                hoverinfo='none',
                **single_plot_param[key]
            ) for key in single_comp_specta.keys()]

            scs = flatten_list([scs, scs1])

        fig = go.Figure(scs)

        fig.update_layout(
            xaxis_title='Velocity [km/s]',
            yaxis_title='Temperature [K]',
            hovermode="x",
            height=300,
            margin={'l': 20, 'b': 30, 'r': 10, 't': 10}
        )

        fig.update_traces(selector=dict(name="Data"), line=dict(color='rgb(0, 0, 0)', shape='hvh'))


        fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                           xref='paper', yref='paper', showarrow=False, align='left',
                           text=title)

        # fig.update_layout(height=300, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
        # fig.show() config={'toImageButtonOptions': {'width': None,'height': None}}

        return fig

    def initalise_app(self):
        """
        Sets up the app window, component locations and components to create.
        Each html.Div indicates what is contained with that component space

        Returns
        -------
        app : Dash app object
            The Webapp.

        """
        if self.inital_view_spectrum is None:
            rand_xy = self.get_random_coords()
        else:
            rand_xy = self.inital_view_spectrum

        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.Div([

                html.Div([

                    html.Label('Velocity components'),
                    dcc.Checklist(
                        id='velo-comp-checkbox',
                        options=[{'label': i, 'value': str(i)} for i in range(1, self.data_obj.maximum_components+1)],
                        value=['1']
                    ),

                    html.Label('Coordinates'),
                    dcc.RadioItems(
                        id='coord-chooser',
                        options=[{'label': key.capitalize(), 'value': key} for key in config.coord_types.keys()],
                        value=self.world_coord_type,
                        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                    ),

                    html.Label('Velocity range'),
                    dcc.RangeSlider(
                        allowCross=False,
                        **self.get_velo_slider_values()
                    ),

                    html.Label('Plot transparency'),
                    dcc.Input(
                        id='transparency',
                        type='number',
                        value=0.02,
                        step=0.01,
                        style={"margin-top": "18px"}
                )
            ], style={'width': '30%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Data maps'),
                    dcc.Dropdown(
                        id='2d-data_selector',
                        options=[{'label': i, 'value': i} for i in self.available_maps],
                        value=self.available_maps[0],
                    ),

                    html.Label('Colour range'),
                    dcc.RangeSlider(allowCross=False,
                                    **self.get_colour_slider_values()
                    ),

                    dcc.RadioItems(
                        id='zoom-chooser',
                        options=[{'label': 'Zoom on click', 'value': 'True'}, {'label': 'Staic full', 'value': 'False'}],
                        value='True',
                        labelStyle={'display': 'inline-block', 'marginTop': '8px'}
                        )
                ],  style={'width': '30%', 'float': 'right', 'display': 'inline-block'})
            ], style={'padding': '15px'}
            ),

            html.Div([
                dcc.Graph(
                    id='3d-scatter-plot',
                    hoverData={'points':[{'customdata': rand_xy}]},
                    clickData={'points':[{'customdata': rand_xy}]},
                    config={
                        'toImageButtonOptions':{
                            'filename':'3d-scatter',
                            'scale':10
                            }
                        }
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

            html.Div([
                dcc.Graph(
                    id='2d-data',
                    hoverData={'points':[{'customdata': rand_xy}]},
                    #clickData={'points':[{'customdata': rand_xy}]}
                    ),

                dcc.Graph(id='spectrum',
                          config={
                              'toImageButtonOptions':{
                                  'filename':'spectrum',
                                  'scale':10
                                  }
                              }
                          )

            ], style={'display': 'inline-block', 'float':'right', 'width': '49%'})
        ])

        return app

    def initalise_visualiser(self):
        """
        Adds all the components to the app and updates them

        Returns
        -------
        app : Dash app object
            The Webapp.

        """
        app = self.initalise_app()
        cache = Cache(app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'cache-directory'
        })

        df = self.df

        @cache.memoize(timeout=60)
        def get_scatter_points(df, velo_checklist):

            dff = df[np.isin(df.component.astype(str), velo_checklist)]

            return dff

        @app.callback(
            Output('3d-scatter-plot', 'figure'),
            [Input('velo-comp-checkbox', 'value'),
             Input('coord-chooser', 'value'),
             Input('velocity-chooser', 'value'),
             Input('transparency', 'value')]
            )
        def update_scatter(velo_checklist, world_system, zlims, alpha):
            """
            Updates the 3d scatterplot with the callbacks/options
            selected on the web app

            Parameters
            ----------
            velo_checklist : list of str
                List containing which velocity components will be shown.
            world_system : str
                The world coodinate system to view the 3d plot in.
            zlims : list
                The minimum and maximum velocities to view on the 3d plot.
            alpha : float
                The opacity of the points on the 3d plot.

            Returns
            -------
            None.

            """
            coords = config.coord_types[world_system]

            xlims, ylims = fix_aspect_ratio_plotly_px(df, coords)

            dff = get_scatter_points(df, velo_checklist) #df[np.isin(df.component.astype(str), velo_checklist)]

            fig = px.scatter_3d(dff, x=dff[coords[0]], y=dff[coords[1]], z=dff.Velocity,
                    color=dff.component.astype(str), #hover_data=['component'],
                    height=800, width=800, opacity=alpha, hover_data=['x', 'y'],
                    custom_data=[dff.x, dff.y, dff.component],
                    range_x=xlims, range_y=ylims, range_z=zlims,
                    color_discrete_sequence=[plotly_colors[int(n)-1] for n in velo_checklist])

            #custom data allows us to update spectum on hover
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Component: %{customdata[2]}",
                    coords[0] +": %{x}",
                    coords[1] +": %{y}",
                    "Velocity [km/s]: %{z}",
                    "x: %{customdata[0]}",
                    "y: %{customdata[1]}",
                ])
        )
            fig.update_yaxes(matches=None)
            fig.update_xaxes(matches=None)

            fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                              hovermode='closest',
                              yaxis=dict(scaleanchor='x'),
                              scene=dict(zaxis=dict(range=zlims))
                              ) # transition_duration=500,

            fig['layout']['uirevision'] = world_system

            return fig

        @app.callback(
            dash.dependencies.Output('2d-data', 'figure'),
            [Input('2d-data_selector', 'value'),
             Input('2d-colour-range', 'value'),
             Input('3d-scatter-plot', 'clickData'),
             Input('zoom-chooser', 'value'),
             Input('coord-chooser', 'value')]
            )


        def update_2d_image(map_type, colour_range, clickData, zoom, world_system):
            """
            Updates which 2d image is being shown on the webapp

            Parameters
            ----------
            map_type : str
                The map to be plotted.
            colour_range : list
                The scale range of colour bar.
            clickData : dict
                Positional information of the clicked data from the 3d scatter plot.
            zoom : str of a bool
                Sets the 2d plot so that it updates position on click or shows
                full map.
            world_system : str
                The world projection to show the map in.

            Returns
            -------
            figh : plotly fig
                The figure object to plot

            """

            other_world = config.coord_conversion[world_system]

            #If we did not reproject the 2d data, do not allow for the rpj option
            if not self.data_2d_obj.reproject_all and world_system !=self.world_coord_type:
                world_system = config.coord_conversion[world_system]


            data_2d = self.all_data_2d[world_system]['data'][map_type][0]
            x = self.all_data_2d[world_system]['x']
            y = self.all_data_2d[world_system]['y']

        #right now, cannot zoom in a reprojected frame
            if zoom == 'True' and world_system != config.coord_conversion[self.world_coord_type]:
                yl, yu, xl, xu = data_zoom(clickData, data_2d)
                data_2d = data_2d[yl:yu, xl:xu]
                data_shape = np.shape(data_2d)
                xx, yy = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))
                yy += yl
                xx += xl
            else:
                yl, yu, xl, xu = [None] * 4
                data_shape = np.shape(data_2d)
                xx, yy = np.meshgrid(np.arange(data_shape[1]), np.arange(data_shape[0]))

            coords = config.coord_types[world_system]

            customdata = np.stack((xx, yy), axis=-1)

            hovertemplate = "<br>".join([
                coords[0] +": %{x}",
                coords[1] +": %{y}",
                "x: %{customdata[0]}",
                "y: %{customdata[1]}",
                "Colour: %{z}<extra></extra>"
            ])

            figh = px.imshow(data_2d, origin='lower', aspect='equal',
                              zmin=colour_range[0], zmax=colour_range[1],
                              template="simple_white", x=x[xl:xu], y=y[yl:yu],
                              labels=dict(
                                  x=coords[0],
                                  y=coords[1]
                                  ),
                              )


            figh.update(data=[{'customdata': customdata,
                                'hovertemplate': hovertemplate}])

            figh.update_traces(customdata=customdata, hovertemplate=hovertemplate)#,text=customdata)


            figh.update_layout(height=500, width=500, margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
                               xaxis={'title': coords[0]}, yaxis={'title': coords[1]})

            #only reverse x axis if using world coordinates
            if world_system != 'pixel':
                figh['layout']['xaxis']['autorange'] = "reversed"

            return figh


        @app.callback(
            Output('spectrum', 'figure'),
            [Input('3d-scatter-plot', 'hoverData'),
             Input('velo-comp-checkbox', 'value'),
             Input('2d-data', 'clickData'),
             Input('2d-data_selector', 'value'),
             Input('coord-chooser', 'value')]
            )
        def update_spectrum(hoverData, velo_checklist, clickData, map_type, world_system):
            """
            Updates the spectrum being shown

            Parameters
            ----------
            hoverData : dict
                Contains the positional information of the
                hovered data from the 3d plot.
            velo_checklist : list of str
                List containing which velocity components will be shown.

            Returns
            -------
            figh  : plotly fig
                The figure object to plot

            """

            event_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if event_id == '3d-scatter-plot' or clickData is None:
                x_index = hoverData['points'][0]['customdata'][0]
                y_index = hoverData['points'][0]['customdata'][1]
                print(y_index, x_index)

            elif event_id == '2d-data':
                data_2d = self.all_data_2d[world_system]['data'][map_type][0]
                y_index, x_index = np.where(data_2d == clickData['points'][0]['z'])
                y_index = y_index[0]
                x_index = x_index[0]
                print(y_index, x_index)

            spetra_values = {}
            for key in self.all_data_3d.keys():
                spetra_values[key] = self.all_data_3d[key][:, y_index, x_index]

            if self.single_fits is not None:
                single_comp_specta = get_model_for_each_found_component(self.real_data_obj.all_data_3d['velocity'], self.single_fits, y_index, x_index, logged_tau=self.logged_tau)
                single_plot_param = get_single_model_spectrum_params(velo_checklist, self.data_obj.maximum_components)

                spetra_values = {**spetra_values, **single_comp_specta}
            else:
                single_plot_param = None

            number_components = self.ncomp_2d[y_index, x_index]

            title = 'x: %d<br>y: %d<br>Components: %d' % (x_index, y_index, number_components)

            return self.create_spectrum(spetra_values, title, single_plot_param, single_comp_specta)

        @app.callback(
            [Output(component_id='2d-colour-range', component_property='min'),
             Output(component_id='2d-colour-range', component_property='max'),
             Output(component_id='2d-colour-range', component_property='marks'),
             Output(component_id='2d-colour-range', component_property='value')],
            Input(component_id='2d-data_selector', component_property='value'),
                       )
        def update_slider(map_type):
            """
            Updates the colourbar slider to match the 2d data being shown

            Parameters
            ----------
            map_type : str
                Which map is selected.

            Returns
            -------
            min_val : int
                Minimum value of the slider.
            max_val : int
                Maximum value of the slider.
            data_2d_slider : dict
                Info about the labels and label positions on the slider.
            [min_val, max_val] : list
                The inital start values of the slider.

            """

            min_val, max_val, data_2d_slider = data2d_silder_values(self.all_data_2d[self.world_coord_type]['data'][map_type][0])

            return min_val, max_val, data_2d_slider, [min_val, max_val]

        return app

    def run_visualiser(self, **kwargs):
        """
        Runs the app

        Returns
        -------
        None.

        """

        app = self.initalise_visualiser()
        app.run_server(debug=False)


def roundup(x, sf=10):
    """
    Rounds a number up to the nearest significant figure (Ceiling)

    Parameters
    ----------
    x : float
        The value to round.
    sf : float, optional
        Which degree to round to. 10 means it will round up the nearest 10.
        The default is 10.

    Returns
    -------
    x_rounded
        New value rounded up to nearest significant figure

    """
    return int(math.ceil(x / sf)) * sf

def rounddown(x, sf=10):
    """
    Rounds a number down to the nearest significant figure (floor)

    Parameters
    ----------
    x : float
        The value to round.
    sf : float, optional
        Which degree to round to. 10 means it will round up the nearest 10.
        The default is 10.

    Returns
    -------
    x_rounded
        New value rounded up to nearest significant figure


    """
    return int(math.floor(x / sf)) * sf

def velocity_silder_values(velocity_values, steps):
    """
    Calculates the velocity slider appearence (size, starting value
    and number of ticks to show)

    Parameters
    ----------
    velocity_values : 1darray
        The entire velocity range of the data
    steps : int
        Number of ticks to show

    Returns
    -------
    min_velo : int
        Minimum value of the slider
    max_velo : int
        Maximum value of the slider.
    velocity_slider : dict
        Info about the labels and label positions on the slider.

    """

    min_velo = rounddown(np.min(velocity_values))
    max_velo = roundup(np.max(velocity_values))

    velocity_slider = {}
    for i in range(min_velo, max_velo+1, steps):
        velocity_slider[i] = '%d km/s' % i
    return min_velo, max_velo, velocity_slider

def data2d_silder_values(data2d):
    """


    Parameters
    ----------
    data2d : 2darray
        The currently shown map

    Returns
    -------
    min_val : int
        Minimum value of the slider.
    max_val : int
        Maximum value of the slider.
    data_2d_slider : dict
        Info about the labels and label positions on the slider.

    """
    min_val = rounddown(np.nanmin(data2d), 1)
    max_val = roundup(np.nanmax(data2d), 1)
    mid_val = np.nanmean([min_val, max_val])
    vals = [min_val, mid_val, max_val]

    data_2d_slider = {}
    for val in vals:
        data_2d_slider[val] = '%.2f' % val
    return min_val, max_val, data_2d_slider

def fix_aspect_ratio_plotly_px(df, coord_type):
    """
    Forces the aspect ratio to be equal
    Non xy coordinates have weird aspect ratios. If I used plotly go I have more
    contol and can fix this at the cost of longer code. Or I can manually
    specifiy the axes range to be equal in plotly px.


    Parameters
    ----------
    df : panda.Dataframe
        dataframe containing the coordinate information
    coord_type : str
        String indicating the units from the coordinate system being used

    Returns
    -------
    xlims: size 2 list
        Contains the lower and upper bounds of the x axis in world coords
    ylims: size 2 list
        Contains the lower and upper bounds of the y axis in world coords

    """

    if coord_type[0] != 'x':
        x_values = df[coord_type[0]]
        y_values = df[coord_type[1]]

        xsize = x_values.max() - x_values.min()
        ysize = y_values.max() - y_values.min()

        half_x = xsize/2
        half_y = ysize/2

        x_middle = x_values.min() + half_x
        y_middle = y_values.min() + half_y

        if xsize > ysize:
            half = half_x
        else:
            half = half_y

        y_lower = y_middle - half
        y_upper = y_middle + half

        #left, right reversed
        x_lower = x_middle + half
        x_upper = x_middle - half

        return [x_lower, x_upper], [y_lower, y_upper]
    else:
        return None, None

def data_zoom(clickData, data_2d, zoom=75):
    """
    From a clicked position in the three scatter plot, zooms the
    2d image to that location
    TODO: Messy function, neaten up at some point

    Parameters
    ----------
    clickData : dict
        dictionary containing the clicked position
    data_2d : 2d array
        2d data
    zoom : int, optional
        the number of pixels to zoom to either side of the centre
        value. The default is 75.

    Returns
    -------
    y_lower : int
        lower limit on y zoom.
    y_upper : int
        upper limit on y zoom.
    x_lower : int
        lower limit on x zoom.
    x_upper : int
        upper limit on x zoom.

    """

    ysize, xsize = np.shape(data_2d)

    x_index = clickData['points'][0]['customdata'][0]
    y_index = clickData['points'][0]['customdata'][1]



    x_lower = x_index - zoom #if (x_index >= zoom) else 0
    y_lower = y_index - zoom #if (y_index >= zoom) else 0


    x_upper = x_index + zoom #if ((x_index + zoom) < xsize) else xsize - 1
    y_upper = y_index + zoom #if ((y_index + zoom) < ysize) else ysize - 1

    if x_lower < 0:
        x_lower = 0

    if y_lower < 0:
        y_lower = 0

    if x_upper >= xsize:
        x_upper = xsize - 1

    if y_upper >= ysize:
        y_upper = ysize - 1

    return y_lower, y_upper, x_lower, x_upper

plotly_colors_hex = px.colors.qualitative.Plotly
plotly_colors = []
for h in plotly_colors_hex:
    h = h.lstrip('#')
    plotly_colors.append('rgb'+ str(tuple(int(h[i:i+2], 16) for i in (0, 2, 4))))

def get_single_model_spectrum_params(velo_checklist, maximum_components):
    """
    For the selected velocity components, sets the visual plotting
    parameters for the induvidual spectum. If selected, the component
    is plotted in the same colour as shown on the 3d plot. If not
    selected, a low alpha/transpaency spectrum is plotted.

    Parameters
    ----------
    velo_checklist : list
        The velocity components selected.
    maximum_components : int
        The maximum amount of velocity components present.

    Returns
    -------
    spectrum_param_ncomp : dict
        Dictionary containing the visual .


    """
    spectrum_param_ncomp = {}
    for i in range(maximum_components):
        opacity_line = 0.1
        opacity_fill = 0.05
        color = dict(color='black')
        fillcolor = 'rgba(68, 68, 68, 0.05)'
        comp = str(i+1)

        if str(i+1) in velo_checklist:
            opacity_line = 0.5
            opacity_fill = 0.1
            fillcolor = plotly_colors[i%len(plotly_colors)].replace(')', ', %.4f)' % opacity_fill).replace('rgb','rgba')#'rgba(68, 68, 68, 0.1)'
            color = dict(color=plotly_colors[i%len(plotly_colors)])

        spectrum_param_ncomp[comp] = {
            'mode':'lines',
            'showlegend':False,
            'line': color,
            'opacity': opacity_line
        }

        spectrum_param_ncomp[comp+'u'] = {
            'mode':'lines',
            'showlegend':False,
            'line':dict(width=0),
            'marker':dict(color="#444"),
            'opacity':opacity_fill
        }

        spectrum_param_ncomp[comp+'l'] = {
            'mode':'lines',
            'showlegend':False,
            'line':dict(width=0),
            'marker':dict(color="#444"),
            'fillcolor':fillcolor,
            'fill':'tonexty',

        }
    #labels = flatten_list(labels)
    return spectrum_param_ncomp#, labels

def get_model_for_each_found_component(velocity, models, y, x, tex_label='tex', tau_label='tau', peak_v_label='v', sigma_label='sigma', logged_tau=False): # logged_tau=False
    """
    From a dictionary containing the hyperfine fit parameters, outputs
    the model specturm for each component at the given y,x coordinates

    Parameters
    ----------
    velocity : 1darray
        The velocity range.
    models : dict
        Each key the fitted paramter and the value is a 3d cube where the z axis
        are the fitted values per component in y,x
    y : int
        y position to make model from.
    x : int
        x position to make model from.
    tex_label : str, optional
        Label/key to access the fitted excitation temperature. The default is 'tex'.
    tau_label : str, optional
        Label/key to access the fitted opacity. The default is 'tau'.
    peak_v_label : str, optional
        Label/key to access the fitted central peak. The default is 'v'.
    sigma_label : str, optional
        Label/key to access the fitted dispersion. The default is 'sigma'.

    Returns
    -------
    as_dict : dict
        Dictionary where each number is the model for that component. The `l` and
        `u` suffix on the key represents the lower and upper error bound if given

    """

    single_model = [make_maps.hyperfine_structure_all(velocity, #real_data_obj.all_data_3d['velocity'],
                                               models[tex_label][i][:, y, x],
                                               models[tau_label][i][:, y, x],
                                               models[peak_v_label][i][:, y, x],
                                               models[sigma_label][i][:, y, x],
                                               return_hyperfine_components=False,
                                               logged_tau=logged_tau) for i in range(len(models[tex_label]))]

    single_model = np.array(single_model)

    as_dict = {}
    suffix = ['', 'l', 'u']
    for i in range(np.shape(single_model)[-1]):
        modal_spectra = single_model[:,:,i]

        for j in range(np.shape(modal_spectra)[0]):
            as_dict[str(i+1)+suffix[j]] = modal_spectra[j]


    return as_dict

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

if __name__ == "__main__":
    pass
# .quantity.value.byteswap().newbyteorder()