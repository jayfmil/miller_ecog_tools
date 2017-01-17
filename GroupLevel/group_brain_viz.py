import matplotlib
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import nibabel as nib
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plotly.offline.init_notebook_mode()


def plot_elec_coverage(subject_objs, radius=12.5):
    """
    Plots brain, shaded number of subjects contributing to a given region.

    subject_objs: the subject_objs attribute of a group analysis.
          radius: radius of sphere used to determine if an electrode is close to a vertex
    """

    # load average brain pial surface mesh
    l_coords, l_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/average/surf/lh.pial')
    r_coords, r_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/average/surf/rh.pial')

    # initialize array to count whether a given face is near an electrode for each subject
    l_face_count = np.zeros((l_faces.shape[0], len(subject_objs)))
    r_face_count = np.zeros((r_faces.shape[0], len(subject_objs)))

    # loop over each subject.
    for i, subj in enumerate(subject_objs):
        print('%s: finding valid faces.' % subj.subj)
        for elec in subj.elec_xyz_avg:
            l_face_count[np.any((np.linalg.norm(l_coords - elec, axis=1) < radius)[l_faces], axis=1), i] = 1
            r_face_count[np.any((np.linalg.norm(r_coords - elec, axis=1) < radius)[r_faces], axis=1), i] = 1

    # what is the maximum number of subjects who contribute to a face. Used to normlized colors
    max_count = np.max([l_face_count.sum(axis=1).max(), r_face_count.sum(axis=1).max()])

    # get the matplotlib colormap 'hot' and transform into rgb
    cmap = matplotlib_to_plotly('hot')

    # map counts of each hemisphere to rgb
    l_colors = val_to_color(cmap, l_face_count.sum(axis=1), 0, max_count)
    r_colors = val_to_color(cmap, r_face_count.sum(axis=1), 0, max_count)

    # create left hemi mesh
    data = go.Data([
        go.Mesh3d(
            x=l_coords[:, 0],
            y=l_coords[:, 1],
            z=l_coords[:, 2],
            i=l_faces[:, 0],
            j=l_faces[:, 1],
            k=l_faces[:, 2],
            name='l_brain',
            showscale=False,
            facecolor=l_colors,
            color='gray',
            lightposition=dict(x=-1000, y=1000, z=0),
            lighting=dict(ambient=.4, fresnel=0),
        ),

        # create right hemi mesh
        go.Mesh3d(
            x=r_coords[:, 0],
            y=r_coords[:, 1],
            z=r_coords[:, 2],
            i=r_faces[:, 0],
            j=r_faces[:, 1],
            k=r_faces[:, 2],
            name='r_brain',
            showscale=False,
            facecolor=r_colors,
            color='gray',
            lightposition=dict(x=-1000, y=1000, z=0),
            lighting=dict(ambient=.4, fresnel=0),
        )
    ])

    # this layout turns off grid and axes labels
    layout = go.Layout(scene=go.Scene(
        xaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False),
        yaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False),
        zaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-2, y=0, z=0))
    ))

    # plot it
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def matplotlib_to_plotly(cmap_name):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    h = 1.0 / (255 - 1)
    pl_colorscale = []

    for k in range(255):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append('rgb' + str((C[0], C[1], C[2])))
    return pl_colorscale

def val_to_color(cmap, vals, min_val, max_val):
    if isinstance(vals, list):
        vals = np.array(vals)

    color_range = float(max_val - min_val)
    norm_colors = (vals - min_val) / color_range
    norm_colors = np.round(norm_colors * (len(cmap) - 1))
    norm_colors[norm_colors < 0] = 0
    norm_colors[norm_colors > (len(cmap) - 1)] = len(cmap) - 1
    return [cmap[int(c)] for c in norm_colors]