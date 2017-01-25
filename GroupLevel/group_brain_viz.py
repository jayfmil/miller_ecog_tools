import matplotlib
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import nibabel as nib
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.stats import ttest_1samp
plotly.offline.init_notebook_mode()
import pdb

import platform
basedir = ''
if platform.system() == 'Darwin':
    basedir = '/Users/jmiller'


def load_brain_mesh(subj='average'):

    # load average brain pial surface mesh
    l_coords, l_faces = nib.freesurfer.read_geometry(basedir+'/data/eeg/freesurfer/subjects/%s/surf/lh.pial' % subj)
    r_coords, r_faces = nib.freesurfer.read_geometry(basedir+'/data/eeg/freesurfer/subjects/%s/surf/rh.pial' % subj)
    return l_coords, l_faces, r_coords, r_faces


def get_elec_coverage_faces(subject_objs, radius=12.5):

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # initialize array to count whether a given face is near an electrode for each subject
    l_face_count = np.zeros((l_faces.shape[0], len(subject_objs)))
    r_face_count = np.zeros((r_faces.shape[0], len(subject_objs)))

    # loop over each subject.
    for i, subj in enumerate(subject_objs):
        print('%s: finding valid faces.' % subj.subj)
        for elec in subj.elec_xyz_avg:
            l_face_count[np.any((np.linalg.norm(l_coords - elec, axis=1) < radius)[l_faces], axis=1), i] = 1
            r_face_count[np.any((np.linalg.norm(r_coords - elec, axis=1) < radius)[r_faces], axis=1), i] = 1

    return l_face_count, r_face_count


def get_elec_coverage_verts(subject_objs, radius=12.5):

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # initialize array to count whether a given face is near an electrode for each subject
    l_vert_count = np.zeros((l_coords.shape[0], len(subject_objs)))
    r_vert_count = np.zeros((r_coords.shape[0], len(subject_objs)))

    # loop over each subject.
    for i, subj in enumerate(subject_objs):
        print('%s: finding valid vertices.' % subj.subj)
        for elec in subj.elec_xyz_avg:
            l_vert_count[np.linalg.norm(l_coords - elec, axis=1) < radius, i] = 1
            r_vert_count[np.linalg.norm(r_coords - elec, axis=1) < radius, i] = 1

    return l_vert_count, r_vert_count


def plot_elec_coverage(subject_objs, radius=12.5):
    """
    Plots brain, shaded number of subjects contributing to a given region.

    subject_objs: the subject_objs attribute of a group analysis.
          radius: radius of sphere used to determine if an electrode is close to a vertex
    """

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # faces x subjects count
    l_face_count, r_face_count = get_elec_coverage_faces(subject_objs, radius)

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


def get_elec_ts_faces(subject_objs, elec_inds, radius=12.5, res_key='ts'):

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # initialize array to count whether a given face is near an electrode for each subject
    l_face_mean = np.zeros((l_faces.shape[0], len(subject_objs)))
    l_face_mean[:] = np.nan
    r_face_mean = np.zeros((r_faces.shape[0], len(subject_objs)))
    r_face_mean[:] = np.nan

    # loop over each subject.
    for i, subj in enumerate(subject_objs):

        # mean the res within the freq band given
        mean_val = subj.res[res_key][elec_inds]
        if mean_val.ndim > 1:
            mean_val = np.mean(mean_val, axis=0)

        l_subj_faces = np.zeros((l_faces.shape[0], subj.elec_xyz_avg.shape[0]))
        l_subj_faces[:] = np.nan
        r_subj_faces = np.zeros((r_faces.shape[0], subj.elec_xyz_avg.shape[0]))
        r_subj_faces[:] = np.nan

        print('%s: finding valid faces.' % subj.subj)
        for e_num, (elec, val) in enumerate(zip(subj.elec_xyz_avg, mean_val)):

            l_subj_faces[np.any((np.linalg.norm(l_coords - elec, axis=1) < radius)[l_faces], axis=1), e_num] = val
            r_subj_faces[np.any((np.linalg.norm(r_coords - elec, axis=1) < radius)[r_faces], axis=1), e_num] = val

        l_face_mean[:, i] = np.nanmean(l_subj_faces, axis=1)
        r_face_mean[:, i] = np.nanmean(r_subj_faces, axis=1)
    return l_face_mean, r_face_mean


def get_elec_ts_verts(subject_objs, elec_inds, radius=12.5, res_key='ts'):

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # initialize array to count whether a given face is near an electrode for each subject
    l_vert_mean = np.zeros((l_coords.shape[0], len(subject_objs)))
    l_vert_mean[:] = np.nan
    r_vert_mean = np.zeros((r_coords.shape[0], len(subject_objs)))
    r_vert_mean[:] = np.nan

    # loop over each subject.
    for i, subj in enumerate(subject_objs):

        # mean the res within the freq band given
        mean_val = subj.res[res_key][elec_inds]
        if mean_val.ndim > 1:
            mean_val = np.mean(mean_val, axis=0)

        l_subj_verts = np.zeros((l_coords.shape[0], subj.elec_xyz_avg.shape[0]))
        l_subj_verts[:] = np.nan
        r_subj_verts = np.zeros((r_coords.shape[0], subj.elec_xyz_avg.shape[0]))
        r_subj_verts[:] = np.nan

        print('%s: finding valid vertices.' % subj.subj)
        for e_num, (elec, val) in enumerate(zip(subj.elec_xyz_avg, mean_val)):

            l_subj_verts[np.linalg.norm(l_coords - elec, axis=1) < radius, e_num] = val
            r_subj_verts[np.linalg.norm(r_coords - elec, axis=1) < radius, e_num] = val

        l_vert_mean[:, i] = np.nanmean(l_subj_verts, axis=1)
        r_vert_mean[:, i] = np.nanmean(r_subj_verts, axis=1)
    return l_vert_mean, r_vert_mean


def plot_elec_ts(subject_objs, elec_inds, radius=12.5, res_key='ts'):
    """
    Plots brain, shaded number of subjects contributing to a given region.

    subject_objs: the subject_objs attribute of a group analysis.
          radius: radius of sphere used to determine if an electrode is close to a vertex
    """

    # load average brain pial surface mesh
    l_coords, l_faces, r_coords, r_faces = load_brain_mesh('average')

    # faces x subjects mean
    l_face_mean, r_face_mean = get_elec_ts_faces(subject_objs, elec_inds, radius, res_key)

    clim = np.max([np.nanmax(np.abs(np.nanmean(l_face_mean, axis=1))), np.nanmax(np.abs(np.nanmean(r_face_mean, axis=1)))])
    clim = 1

    l_ts, l_ps = ttest_1samp(l_face_mean, 0, axis=1, nan_policy='omit')
    r_ts, r_ps = ttest_1samp(r_face_mean, 0, axis=1, nan_policy='omit')

    # get the matplotlib colormap 'hot' and transform into rgb
    cmap = matplotlib_to_plotly('RdBu_r')

    # map counts of each hemisphere to rgb
    l_means = np.nanmean(l_face_mean, axis=1)
    l_exc = np.sum(~np.isnan(l_face_mean), axis=1) <= 5
    l_inds = ~np.isnan(l_means)

    r_means = np.nanmean(r_face_mean, axis=1)
    r_exc = np.sum(~np.isnan(r_face_mean), axis=1) <= 5
    r_inds = ~np.isnan(r_means)

    l_colors = np.chararray(l_means.shape, itemsize=20)
    l_colors[l_inds] = val_to_color(cmap, l_means[l_inds], -clim, clim)
    l_colors[l_ps > .05] = 'rgb(127, 127, 127)'
    # l_colors[~l_inds] = 'rgb(0, 0, 0)'
    l_colors[l_exc] = 'rgb(0, 0, 0)'

    r_colors = np.chararray(r_means.shape, itemsize=20)
    r_colors[r_inds] = val_to_color(cmap, r_means[r_inds], -clim, clim)
    r_colors[r_ps > .05] = 'rgb(127, 127, 127)'
    # r_colors[~r_inds] = 'rgb(0, 0, 0)'
    r_colors[r_exc] = 'rgb(0, 0, 0)'

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
            lightposition=dict(x=1000, y=1000, z=0),
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
            eye=dict(x=-1.35, y=0, z=0))
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