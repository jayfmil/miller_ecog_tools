from GroupLevel.Analyses import group_oscillation_cluster
from SubjectLevel.Analyses import subject_oscillation_cluster, subject_SME
from GroupLevel import group_brain_viz
from tqdm import tqdm
from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp
from scipy.stats import ttest_ind, pearsonr, ttest_1samp

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib

import ipyvolume.pylab as p3
import ipyvolume as ipv

import numpy as np
import ram_data_helpers
import pycircstat
import RAM_helpers
import numexpr

import matplotlib.cm as cmx
import matplotlib.colors as clrs

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(style='ticks', palette='Set2')
sns.despine()
sns.set_context("talk", font_scale=1.4)


def aggregate_cluster_corrs(subj_obs, min_elecs=5):
    all_elec_freqs = []
    all_elec_freqs_norm = []
    all_elec_ts_peak = []
    all_elec_ts_low = []
    all_regions = []
    all_cluster_freqs = []

    all_rhos_peak = []
    all_rhos_low = []
    all_regions_for_rhos = []
    all_cluster_freqs_for_rhos = []

    all_subjs = []

    for subj in subj_objs:
        res = subj.res
        lfa_inds = (subj.freqs >= 1) & (subj.freqs <= 10)
        lfa_ts = res['sme_low_freqs']['ts'][lfa_inds].mean(axis=0)

        for cluster_freq in res['clusters']:
            clusters = res['clusters'][cluster_freq]
            num_clusters = len(res['clusters'][cluster_freq]['elec_freqs'])
            for cluster_num in range(num_clusters):
                if len(res['clusters'][cluster_freq]['elec_freqs'][cluster_num]) > min_elecs:
                    elec_freqs = res['clusters'][cluster_freq]['elec_freqs'][cluster_num]
                    elec_freqs_norm = (elec_freqs - np.min(elec_freqs)) / np.ptp(elec_freqs)
                    elec_ts_peak = res['clusters'][cluster_freq]['elec_ts'][cluster_num]
                    elec_ts_low = lfa_ts[clusters['elecs'][cluster_num]]
                    this_region = res['clusters'][cluster_freq]['cluster_region'][cluster_num]
                    this_cluster_freq = res['clusters'][cluster_freq]['mean_freqs'][cluster_num]

                    all_elec_freqs.append(elec_freqs)
                    all_elec_freqs_norm.append(elec_freqs_norm)
                    all_elec_ts_peak.append(elec_ts_peak)
                    all_elec_ts_low.append(elec_ts_low)
                    all_regions.append([this_region] * len(elec_freqs))
                    all_cluster_freqs.append([this_cluster_freq] * len(elec_freqs))
                    all_rhos_peak.append(pearsonr(elec_ts_peak, elec_freqs)[0])
                    all_rhos_low.append(pearsonr(elec_ts_low, elec_freqs)[0])
                    all_regions_for_rhos.append(this_region)
                    all_cluster_freqs_for_rhos.append(this_cluster_freq)
                    all_subjs.append(subj.subj)

    return all_elec_freqs, all_elec_freqs_norm, all_elec_ts_peak, all_elec_ts_low, all_regions, all_cluster_freqs, all_rhos_peak, all_rhos_low, all_regions_for_rhos, all_cluster_freqs_for_rhos, all_subjs


from ptsa.data.readers.tal import TalReader
from ptsa.data.readers.index import JsonIndexReader

import ipyvolume.pylab as p3
import ipyvolume as ipv
import matplotlib.cm as cmx
import matplotlib.colors as clrs

reader = JsonIndexReader('/protocols/r1.json')


def plot_RAM_brain(subj, montage, plot_elecs=True, bipolar=True, opacity=.85, use_average_surface=False,
                   plot_morph=True):
    """
    Plots a brain using ipyvolume. Either plots the average brain surface
    or the subject specific surface.

    Parameters
    ----------
                   subj - subject code to plot
                montage - montage number of subject (usually '0')
             plot_elecs - whether to plot the electrodes
                bipolar - if plotting electrodes, whether to plot bipolor or monopolar locations
                opacity - transparency of brain surface
    use_average_surface - whether to plot the average brain of the subject specific brain
             plot_morph - whether to color the sulci and gyri

    Returns
    -------
        fig - ipyvolume figure
    """

    # load brain geometry
    if use_average_surface:
        subj_name = 'average'
    else:
        subj_name = subj

    l_coords, l_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/lh.pial' % subj_name)
    r_coords, r_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/rh.pial' % subj_name)

    # if plot_morph, load the murch data to make the brain colors
    if plot_morph:
        l_curv = nib.freesurfer.read_morph_data('/data/eeg/freesurfer/subjects/%s/surf/lh.curv' % subj_name)
        l_curv = (l_curv > 0).astype(float)
        l_curv = (l_curv - 0.5) / 3 + 0.5
        l_curv = l_curv[:, np.newaxis] * [1, 1, 1]
        l_curv = np.concatenate([l_curv, np.ones((l_curv.shape[0], 1)) * opacity], -1)

        r_curv = nib.freesurfer.read_morph_data('/data/eeg/freesurfer/subjects/%s/surf/rh.curv' % subj_name)
        r_curv = (r_curv > 0).astype(float)
        r_curv = (r_curv - 0.5) / 3 + 0.5
        r_curv = r_curv[:, np.newaxis] * [1, 1, 1]
        r_curv = np.concatenate([r_curv, np.ones((r_curv.shape[0], 1)) * opacity], -1)
    else:
        l_curv = r_curv = [.6, .6, .6, opacity]

    # create figure
    fig = p3.figure(width=800, height=800, lighting=True)

    # plot left hemi
    brain_l = p3.plot_trisurf(l_coords[:, 0], l_coords[:, 1], l_coords[:, 2], triangles=l_faces, color=l_curv)
    brain_l.material.transparent = True

    # plot right hemi
    brain_r = p3.plot_trisurf(r_coords[:, 0], r_coords[:, 1], r_coords[:, 2], triangles=r_faces, color=r_curv)
    brain_r.material.transparent = True

    # i like to turn off the axes lines
    ipv.style.box_off()
    ipv.style.axes_off()
    ipv.style.background_color('black')
    ipv.squarelim()
    ipv.show()

    # load json database of file locations if we are plotting electrodes
    # and then load electrode locations
    if plot_elecs:
        f_path = reader.aggregate_values('pairs' if bipolar else 'contacts', subject=subj, montage=montage)
        elec_info = TalReader(filename=list(f_path)[0], struct_type='bi' if bipolar else 'mono').read()

        if use_average_surface:
            xyz = np.array([[elec['avg.dural']['x'], elec['avg']['y'], elec['avg']['z']] for elec in elec_info.atlases])
        else:
            xyz = np.array([[elec['ind.dural']['x'], elec['ind']['y'], elec['ind']['z']] for elec in elec_info.atlases])
        points = p3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], size=2.5, marker="sphere", color=[.55, .05, .1])
        points.geo = 'circle_2d'
    return fig


def colorize_elecs(brain_fig, vals, cmap='RdBu_r', vmin=None, vmax=None):
    num_elecs = brain_fig.scatters[0].x.shape[0]

    # default min and max are symmetric around zero, and the magnitude is the max value
    max_val = np.max(np.abs(vals))
    if vmin is None:
        vmin = -max_val
    if vmax is None:
        vmax = max_val

        # get colors
    cm = plt.get_cmap(cmap)
    cNorm = clrs.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    colors = scalarmap.to_rgba(vals)[:, :3]

    # set colors on electrodes
    brain_fig.scatters[0].color = colors
    return brain_fig


def plot_RAM_elec_by_tag(brain_fig, subj, montage, channel_tags, bipolar=True, use_average_surface=False):
    # load electrode information
    f_path = reader.aggregate_values('pairs' if bipolar else 'contacts', subject=subj, montage=montage)
    elec_info = TalReader(filename=list(f_path)[0], struct_type='bi' if bipolar else 'mono').read()

    # find specific tag
    elecs_to_plot = np.in1d(elec_info.tagName, channel_tags)

    if use_average_surface:
        xyz = np.array([[elec['avg.dural']['x'], elec['avg']['y'], elec['avg']['z']] for elec in elec_info.atlases])
    else:
        xyz = np.array([[elec['ind.dural']['x'], elec['ind']['y'], elec['ind']['z']] for elec in elec_info.atlases])

    xyz = xyz[elecs_to_plot]
    print(xyz)
    points = p3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], size=2.5, marker="sphere", color=[.55, .05, .1])
    points.geo = 'circle_2d'
    return points