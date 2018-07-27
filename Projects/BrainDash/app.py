# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import dash_colorscales as dcs
import numpy as np
import json
import RAM_helpers
from ptsa.data.readers.tal import TalReader
from ptsa.data.readers.index import JsonIndexReader
import nibabel as nib
from textwrap import dedent as d
import pandas as pd
from pandas.io.json import json_normalize

app = dash.Dash(__name__)
server = app.server

DEFAULT_COLORSCALE = [[0, 'rgb(12,51,131)'], [0.25, 'rgb(10,136,186)'],\
        [0.5, 'rgb(242,211,56)'], [0.75, 'rgb(242,143,56)'], \
        [1, 'rgb(217,30,30)']]

DEFAULT_COLORSCALE_NO_INDEX = [ea[1] for ea in DEFAULT_COLORSCALE]


reader = JsonIndexReader('/protocols/r1.json')

def to_dict(arr):
    if arr.ndim == 0:
        return {}

    arr_as_dict = []
    names_without_remove = [name for name in arr.dtype.names if name != '_remove']
    for x in arr:
        if (not '_remove' in x.dtype.names) or (not x['_remove']):
            entry = {}
            for name in names_without_remove:
                entry[name] = x[name]
            arr_as_dict.append(entry)

    if len(arr_as_dict) == 0:
        return arr_as_dict

    recarray_keys = []
    array_keys = []
    for key, value in arr_as_dict[0].items():
        if isinstance(value, (np.ndarray, np.record)) and value.dtype.names:
            recarray_keys.append(key)
        elif isinstance(value, (np.ndarray)):
            array_keys.append(key)

    for key in recarray_keys:
        for entry in arr_as_dict:
            if entry[key].size > 1:
                entry[key] = to_dict(entry[key])
            else:
                entry[key] = dict(zip(entry[key].dtype.names, entry[key]))

    for key in array_keys:
        for entry in arr_as_dict:
            entry[key] = list(entry[key])


    return arr_as_dict

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        else:
            return super(MyEncoder, self).default(obj)

def to_json(arr, fp=None):
    if fp:
        json.dump(to_dict(arr), fp, cls=MyEncoder, indent=2, sort_keys=True)
    else:
        return json.dumps(to_dict(arr), cls=MyEncoder, indent=2, sort_keys=True)

def load_brain_data():
    """
    Loads brain surface. Currently only the average surface, not subject specific.

    """

    # load brain geometry
    # l_coords, l_faces = nib.freesurfer.read_geometry('/Users/jmiller/data/eeg/freesurfer/subjects/%s/surf/lh.pial' % subject)
    # r_coords, r_faces = nib.freesurfer.read_geometry('/Users/jmiller/data/eeg/freesurfer/subjects/%s/surf/rh.pial' % subject)
    l_coords, l_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/lh.pial' % 'average')
    r_coords, r_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/rh.pial' % 'average')

    # load elec info
    # f_path_mono = reader.aggregate_values('contacts', subject=self.subject, montage=self.montage)
    # self.elec_info_mono = TalReader(filename=list(f_path)[0], struct_type='bi' if self.bipolar else 'mono').read()

    # f_path_bipol = reader.aggregate_values('pairs', subject=self.subject, montage=self.montage)
    # self.elec_info_bipol = TalReader(filename=list(f_path)[0], struct_type='bi' if self.bipolar else 'mono').read()
    return l_coords, l_faces, r_coords, r_faces

def load_subj_elec_info(subject, montage):
    """
    Loads electrode information for a given subejct and montage.
    :param subject:
    :param montage:
    :return:
    """

    elec_info_mono = RAM_helpers.load_tal(subject, montage, bipol=False)
    elec_info_bipolar = RAM_helpers.load_tal(subject, montage, bipol=True)
    return elec_info_mono, elec_info_bipolar


def plot_brain_mesh(l_coords, l_faces, r_coords, r_faces, colorscale="Viridis", flatshading=False):
    """

    :param l_coords:
    :param l_faces:
    :param r_coords:
    :param r_faces:
    :param colorscale:
    :param flatshading:
    :return:
    """

    mesh_l = dict(
        type='mesh3d',
        x=l_coords[:, 0],
        y=l_coords[:, 1],
        z=l_coords[:, 2],
        i=l_faces[:, 0],
        j=l_faces[:, 1],
        k=l_faces[:, 2],
        colorscale=colorscale,
        # intensity= intensities,
        flatshading=flatshading,
        color='gray',
        # opacity=0.80,
        name='l_hemi',
    )

    mesh_l.update(lighting=dict(ambient=0.18,
                                diffuse=1,
                                fresnel=0.1,
                                specular=1,
                                roughness=0.1,
                                facenormalsepsilon=1e-6,
                                vertexnormalsepsilon=1e-12))

    mesh_l.update(lightposition=dict(x=100,
                                     y=200,
                                     z=0))

    mesh_r = dict(
        type='mesh3d',
        x=r_coords[:, 0],
        y=r_coords[:, 1],
        z=r_coords[:, 2],
        i=r_faces[:, 0],
        j=r_faces[:, 1],
        k=r_faces[:, 2],
        colorscale=colorscale,
        # intensity=intensities,
        flatshading=flatshading,
        color='gray',
        # opacity=0.80,
        name='r_hemi',
    )

    mesh_r.update(lighting=dict(ambient=0.18,
                                diffuse=1,
                                fresnel=0.1,
                                specular=1,
                                roughness=0.1,
                                facenormalsepsilon=1e-6,
                                vertexnormalsepsilon=1e-12))

    mesh_r.update(lightposition=dict(x=100,
                                     y=200,
                                     z=0))
    return [mesh_l, mesh_r]



# on startup, load average brain and show it
l_coords, l_faces, r_coords, r_faces = load_brain_data()
meshes = plot_brain_mesh(l_coords, l_faces, r_coords, r_faces, colorscale="Viridis", flatshading=False)

# load list of subjects and populate dropdown
subjs = RAM_helpers.get_subjs_and_montages('RAM_FR1')
subject_dict_list = [{'label': x[0]+'_'+x[1], 'value': x[0]+'_'+x[1]} for x in subjs]


axis_template = dict(
    showbackground=True,
    backgroundcolor="rgb(10, 10,10)",
    gridcolor="rgb(255, 255, 255)",
    type='linear',
    zerolinecolor="rgb(255, 255, 255)")


plot_layout = dict(
         title = '',
         margin=dict(t=0,b=0,l=0,r=0),
         font=dict(size=12, color='white'),
         width=700,
         height=700,
         showlegend=False,
         plot_bgcolor='black',
         paper_bgcolor='black',
         scene=dict(xaxis=axis_template,
                    yaxis=axis_template,
                    zaxis=axis_template,
                    # aspectratio=dict(x=1, y=1, z=.7),
                    aspectmode='data',
                    # camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=0.08, y=2.2, z=0.08),
                    ),
                    annotations=[]
                )
        )

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'padding': '10px',
        'marginBottom': '20px'
    },
    'graph': {
        'userSelect': 'none',
        'margin': 'auto'
    },
}

'''
~~~~~~~~~~~~~~~~
~~ APP LAYOUT ~~
~~~~~~~~~~~~~~~~
'''

app.layout = html.Div(children=[
    dcc.Dropdown(id='subject_drop',
        options=subject_dict_list,
        placeholder="Select a subject",
    ),

    html.Div(children=[
        html.Div([dcc.Graph(
            id='brain-graph',
            # figure={
            #     'data': meshes,
            #     'layout': plot_layout,
            # },
            # config={'editable': True, 'scrollZoom': False},
            # style=styles['graph']
    )], style={'display': 'inline-block'}),
        html.Div([dcc.Graph(id='elec-graph')],
                 style={'display': 'inline-block'})],
    style={'width': '100%', 'display': 'inline-block'}),

    dt.DataTable(id='elec_table',
        rows=[{}], # initialise the rows
        row_selectable=True,
        editable=False,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
    ),

    # Hidden div to store subject specific info
    html.Div(id='intermediate-value', style={'display': 'none'}),
], style={'margin': '0 auto'})


@app.callback(Output('intermediate-value', 'children'), [Input('subject_drop', 'value')])
def subj_elec_info_to_json(value):
    if value is not None:
        val_split = value.split('_')
        subject = val_split[0]
        montage = val_split[1]
        elec_info_mono, elec_info_bipolar = load_subj_elec_info(subject, montage)

        df_elec_info_mono = json_normalize(to_dict(elec_info_mono))
        df_elec_info_mono['ref'] = ['mono'] * df_elec_info_mono.shape[0]

        df_elec_info_bipolar = json_normalize(to_dict(elec_info_bipolar))
        df_elec_info_bipolar['ref'] = ['bipolar'] * df_elec_info_bipolar.shape[0]

        df_elec_info = pd.concat([df_elec_info_mono, df_elec_info_bipolar])
        return df_elec_info.reset_index().to_json(orient='split')

# 2. Update rows in a callback
@app.callback(Output('brain-graph', 'figure'), [Input('elec_table', 'rows')])
def update_brain_fig(elec_info):
    """
    For user selections, return the relevant table
    """
    # if jsonified_cleaned_data is not None:
    # elec_info = pd.read_json(jsonified_cleaned_data, orient='split').to_dict('records')
    print(type(elec_info))
    x,y,z = np.stack([x['xyz_avg'] for x in elec_info]).T
    names = [x['tag_name'] for x in elec_info]
    print('here1')
    # print(x)

    data = [dict(
        x=x,
        y=y,
        z=z,
        mode='markers',
        type='scatter3d',
        text=names,
        marker=dict(
            size=18,
            opacity=0.8
        )
    )]

    figure = {
                 'data': data,
                 'layout': plot_layout,
             }

    return figure


# 2. Update rows in a callback
@app.callback(Output('elec_table', 'rows'), [Input('intermediate-value', 'children')])
def update_datatable(jsonified_cleaned_data):
    """
    For user selections, return the relevant table
    """
    # if jsonified_cleaned_data is not None:
    return pd.read_json(jsonified_cleaned_data, orient='split').to_dict('records')
    # return dff.to_dict('records')
    # if user_selection == 'Summary':
    #     return DATA.to_dict('records')
    # else:
    #     return SOMEOTHERDATA.to_dict('records')

# app.css.append_css({'external_url': 'https://codepen.io/plotly/pen/YeqjLb.css'})

# @app.callback(
#     Output('brain-graph', 'figure'),
#     [Input('brain-graph', 'clickData'),
    # Input('radio-options', 'value'),
    # Input('colorscale-picker', 'colorscale')],
    # [State('brain-graph', 'figure')])
# def add_marker(clickData, val, colorscale, figure):
#
#     if figure['data'][0]['name'] != val:
#         if val == 'human':
#             pts, tri=read_mniobj("realct.obj")
#             intensities=np.loadtxt('realct.txt')
#             figure['data']=plotly_triangular_mesh(pts, tri, intensities,
#                                         colorscale=DEFAULT_COLORSCALE, flatshading=False,
#                                         showscale=False, reversescale=False, plot_edges=False)
#         elif val == 'human_atlas':
#             pts, tri=read_mniobj("surf_reg_model_both.obj")
#             intensities=np.loadtxt('aal_atlas.txt')
#             figure['data']=plotly_triangular_mesh(pts, tri, intensities,
#                                         colorscale=DEFAULT_COLORSCALE, flatshading=False,
#                                         showscale=False, reversescale=False, plot_edges=False)
#         elif val == 'mouse':
#             pts, tri=read_mniobj("mouse_surf.obj")
#             intensities=np.loadtxt('mouse_map.txt')
#             figure['data']=plotly_triangular_mesh(pts, tri, intensities,
#                                         colorscale=DEFAULT_COLORSCALE, flatshading=False,
#                                         showscale=False, reversescale=False, plot_edges=False)
#             pts, tri=read_mniobj("mouse_brain_outline.obj")
#             outer_mesh = plotly_triangular_mesh(pts, tri)[0]
#             outer_mesh['opacity'] = 0.5
#             outer_mesh['colorscale'] = 'Greys'
#             figure['data'].append(outer_mesh)
#         figure['data'][0]['name'] = val
#
#     elif clickData != None:
#         if 'points' in clickData:
#             marker = dict(
#                 x = [clickData['points'][0]['x']],
#                 y = [clickData['points'][0]['y']],
#                 z = [clickData['points'][0]['z']],
#                 mode = 'markers',
#                 marker = dict(size=15, line=dict(width=3)),
#                 name = 'Marker',
#                 type = 'scatter3d',
#                 text = ['Click point to remove annotation']
#             )
#             anno = dict(
#                 x = clickData['points'][0]['x'],
#                 y = clickData['points'][0]['y'],
#                 z = clickData['points'][0]['z'],
#                 font = dict(color = 'black'),
#                 bgcolor = 'white',
#                 borderpad = 5,
#                 bordercolor = 'black',
#                 borderwidth = 1,
#                 captureevents = True,
#                 ay = -50,
#                 arrowcolor = 'white',
#                 arrowwidth = 2,
#                 arrowhead = 0,
#                 text = 'Click here to annotate<br>(Click point to remove)',
#             )
#             if len(figure['data']) > 1:
#                 same_point_found = False
#                 for i, pt in enumerate(figure['data']):
#                     if pt['x'] == marker['x'] and pt['y'] == marker['y'] and pt['z'] == marker['z']:
#                         ANNO_TRACE_INDEX_OFFSET = 1
#                         if val == 'mouse':
#                             ANNO_TRACE_INDEX_OFFSET = 2
#                         figure['data'].pop(i)
#                         print('DEL. MARKER', i, figure['layout']['scene']['annotations'])
#                         if len(figure['layout']['scene']['annotations']) >= (i-ANNO_TRACE_INDEX_OFFSET):
#                             try:
#                                 figure['layout']['scene']['annotations'].pop(i-ANNO_TRACE_INDEX_OFFSET)
#                             except:
#                                 pass
#                         same_point_found = True
#                         break
#                 if same_point_found == False:
#                     figure['data'].append(marker)
#                     figure['layout']['scene']['annotations'].append(anno)
#             else:
#                 figure['data'].append(marker)
#                 figure['layout']['scene']['annotations'].append(anno)
#
#     cs = []
#     for i, rgb in enumerate(colorscale):
#         cs.append([i/(len(colorscale)-1), rgb])
#     figure['data'][0]['colorscale'] = cs
#
#     return figure
#
# @app.callback(
#     Output('click-data', 'children'),
#     [Input('brain-graph', 'clickData')])
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=4)
#
# @app.callback(
#     Output('relayout-data', 'children'),
#     [Input('brain-graph', 'relayoutData')])
# def display_click_data(relayoutData):
#     return json.dumps(relayoutData, indent=4)

if __name__ == '__main__':
    app.run_server(debug=True)