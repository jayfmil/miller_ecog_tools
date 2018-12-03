import os
import re
import numpy as np
import pycircstat
import numexpr
import pandas as pd

from scipy.signal import hilbert
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# bunch of matplotlib stuff
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import colorcet as cc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for brain plotting
import nilearn.plotting as ni_plot

from miller_ecog_tools.SubjectLevel.par_funcs import par_find_peaks_by_chan, my_local_max
from ptsa.data.filters import MorletWaveletFilter
from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_ram_eeg_data import SubjectRamEEGData


class SubjectTravelingWaveAnalysis(SubjectAnalysisBase, SubjectRamEEGData):
    """
    Subclass of SubjectAnalysis and SubjectRamEEGData.

    Meant to be run as a SubjectAnalysisPipeline with SubjectOscillationClusterAnalysis as the first step.
    Uses the results of SubjectOscillationClusterAnalysis to compute traveling wave statistics on the clusters.
    """

    # when band passing the EEG, don't allow the frequency range to go below this value
    LOWER_MIN_FREQ = 0.5

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectTravelingWaveAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'trav_waves.p'

        # when computing bandpass, plus/minus this number of frequencies
        self.hilbert_half_range = 1.5

        # time period on which to compute cluster statistics
        self.cluster_stat_start_time = 0
        self.cluster_stat_end_time = 1600

        # optional recall_filter_func. When present, will run SME at the bandpass frequency
        self.recall_filter_func = None

        # regions within with which to average phase over electrodes for saving to res
        self.rois = [('Frontal', 'left'),
                     ('Frontal', 'right'),
                     ('Hipp', 'both'),
                     ('Hipp', 'left'),
                     ('Hipp', 'right')]

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__ + '_res')

    def analysis(self):
        """
        For each cluster in res['clusters']:

        1.

        """

        # make sure we have data
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
            return

        # we must have 'clusters' in self.res
        if 'clusters' in self.res:
            self.res['traveling_waves'] = {}

            # get cluster names from dataframe columns
            cluster_names = list(filter(re.compile('cluster[0-9]+').match, self.res['clusters'].columns))

            # get circular-linear regression parameters
            theta_r, params = self.compute_grid_parameters()

            # compute cluster stats for each cluster
            for this_cluster_name in cluster_names:
                cluster_res = {}

                # get the names of the channels in this cluster
                cluster_elecs = self.res['clusters'][self.res['clusters'][this_cluster_name].notna()]['label']

                # for the channels in this cluster, bandpass and then hilbert to get the phase info
                phase_data, power_data, cluster_mean_freq = self.compute_hilbert_for_cluster(this_cluster_name)

                # reduce to only time inverval of interest
                time_inds = (phase_data.time >= self.cluster_stat_start_time) & (
                        phase_data.time <= self.cluster_stat_end_time)
                phase_data = phase_data[:, :, time_inds]

                # get electrode coordinates in 2d
                norm_coords = self.compute_2d_elec_coords(this_cluster_name)

                # run the cluster stats for time-averaged data
                mean_rel_phase = pycircstat.mean(phase_data.data, axis=2)
                mean_cluster_wave_ang, mean_cluster_wave_freq, mean_cluster_r2_adj = \
                    circ_lin_regress(mean_rel_phase.T, norm_coords, theta_r, params)
                cluster_res['mean_cluster_wave_ang'] = mean_cluster_wave_ang
                cluster_res['mean_cluster_wave_freq'] = mean_cluster_wave_freq
                cluster_res['mean_cluster_r2_adj'] = mean_cluster_r2_adj

                # and run it for each time point
                num_times = phase_data.shape[-1]
                data_as_list = zip(phase_data.T, [norm_coords] * num_times, [theta_r] * num_times, [params] * num_times)
                res_as_list = Parallel(n_jobs=12, verbose=5)(delayed(circ_lin_regress)(x[0].data, x[1], x[2], x[3])
                                                             for x in data_as_list)
                cluster_res['cluster_wave_ang'] = np.stack([x[0] for x in res_as_list], axis=0).astype('float32')
                cluster_res['cluster_wave_freq'] = np.stack([x[1] for x in res_as_list], axis=0).astype('float32')
                cluster_res['cluster_r2_adj'] = np.stack([x[2] for x in res_as_list], axis=0).astype('float32')
                cluster_res['mean_freq'] = cluster_mean_freq
                cluster_res['channels'] = cluster_elecs.values
                cluster_res['time'] = phase_data.time.data
                cluster_res['phase_data'] = pycircstat.mean(phase_data, axis=1).astype('float32')

                # finally, compute the subsequent memory effect
                if hasattr(self, 'recall_filter_func') and callable(self.recall_filter_func):
                    recalled = self.recall_filter_func(self.subject_data)
                    delta_z, ts, ps = self.compute_sme_for_cluster(power_data)
                    cluster_res['sme_t'] = ts
                    cluster_res['sme_z'] = delta_z
                    cluster_res['ps'] = ps
                    cluster_res['phase_data_recalled'] = pycircstat.mean(phase_data[:, recalled], axis=1).astype(
                        'float32')
                    cluster_res['phase_data_not_recalled'] = pycircstat.mean(phase_data[:, ~recalled], axis=1).astype(
                        'float32')
                    cluster_res['recalled'] = recalled

                # finally finally, bin phase by roi
                cluster_res['phase_by_roi'] = self.bin_phase_by_region(phase_data, this_cluster_name)
                self.res['traveling_waves'][this_cluster_name] = cluster_res

        else:
            print('{}: self.res must have a clusters entry before running.'.format(self.subject))
            return

    def compute_grid_parameters(self):
        """
        Angle and phase offsets over which to compute the traveling wave statistics. Consider making these
        modifiable.
        """

        thetas = np.radians(np.arange(0, 356, 5))
        rs = np.radians(np.arange(0, 18.1, .5))
        theta_r = np.stack([(x, y) for x in thetas for y in rs])
        params = np.stack([theta_r[:, 1] * np.cos(theta_r[:, 0]), theta_r[:, 1] * np.sin(theta_r[:, 0])], -1)
        return theta_r, params

    def compute_hilbert_for_cluster(self, this_cluster_name):

        # first, get the eeg for just channels in cluster
        cluster_rows = self.res['clusters'][this_cluster_name].notna()
        cluster_elec_labels = self.res['clusters'][cluster_rows]['label']
        cluster_eeg = self.subject_data[:, np.in1d(self.subject_data.channel, cluster_elec_labels)]

        # bandpass eeg at the mean frequency, making sure the lower frequency isn't too low
        cluster_mean_freq = self.res['clusters'][cluster_rows][this_cluster_name].mean()
        cluster_freq_range = [cluster_mean_freq - self.hilbert_half_range, cluster_mean_freq + self.hilbert_half_range]
        if cluster_freq_range[0] < SubjectTravelingWaveAnalysis.LOWER_MIN_FREQ:
            cluster_freq_range[0] = SubjectTravelingWaveAnalysis.LOWER_MIN_FREQ
        filtered_eeg = RAM_helpers.band_pass_eeg(cluster_eeg, cluster_freq_range)
        filtered_eeg = filtered_eeg.transpose('channel', 'event', 'time')

        # run the hilbert transform
        complex_hilbert_res = hilbert(filtered_eeg.data, N=filtered_eeg.shape[-1], axis=-1)

        # compute the phase of the filtered eeg
        phase_data = filtered_eeg.copy()
        phase_data.data = np.unwrap(np.angle(complex_hilbert_res))

        # compute the power
        power_data = filtered_eeg.copy()
        power_data.data = np.abs(complex_hilbert_res) ** 2

        # compute mean phase and phase difference between ref phase and each electrode phase
        ref_phase = pycircstat.mean(phase_data.data, axis=0)
        phase_data.data = pycircstat.cdiff(phase_data.data, ref_phase)
        return phase_data, power_data, cluster_mean_freq

    def compute_2d_elec_coords(self, this_cluster_name):

        # compute PCA of 3d electrode coords to get 2d coords
        cluster_rows = self.res['clusters'][this_cluster_name].notna()
        xyz = self.res['clusters'][cluster_rows][['x', 'y', 'z']].values
        xyz -= np.mean(xyz, axis=0)
        pca = PCA(n_components=3)
        norm_coords = pca.fit_transform(xyz)[:, :2]
        return norm_coords

    def compute_sme_for_cluster(self, power_data):
        # zscore the data by session
        z_data = RAM_helpers.zscore_by_session(power_data.transpose('event', 'channel', 'time'))

        # compare the recalled and not recalled items
        recalled = self.recall_filter_func(self.subject_data)
        delta_z = np.nanmean(z_data[recalled], axis=0) - np.nanmean(z_data[~recalled], axis=0)
        ts, ps, = ttest_ind(z_data[recalled], z_data[~recalled])
        return delta_z, ts, ps

    def bin_phase_by_region(self, phase_data, this_cluster_name):
        """
        Bin the channel x event by time phase data into rois x event x time. Means over all electrodes in a given
        region of interest.
        """

        cluster_rows = self.res['clusters'][this_cluster_name].notna()
        cluster_region_df = self.get_electrode_roi_by_hemi()[cluster_rows]

        mean_phase_data = {}
        for this_roi in self.rois:
            if this_roi[1] == 'both':
                cluster_elecs = cluster_region_df[cluster_rows].region == this_roi[0]
            else:
                cluster_elecs = (cluster_region_df.region == this_roi[0]) & (cluster_region_df.hemi == this_roi[1])
            if cluster_elecs.any():

                mean_phase_data[this_roi[1]+'-'+this_roi[0]] = pycircstat.mean(phase_data[cluster_elecs.values], axis=0)
        return mean_phase_data

    def get_electrode_roi_by_hemi(self):

        if 'stein.region' in self.elec_info:
            region_key1 = 'stein.region'
        elif 'locTag' in self.elec_info:
            region_key1 = 'locTag'
        else:
            region_key1 = ''

        if 'ind.region' in self.elec_info:
            region_key2 = 'ind.region'
        else:
            region_key2 = 'indivSurf.anatRegion'

        hemi_key = 'ind.x' if 'ind.x' in self.elec_info else 'indivSurf.x'
        if self.elec_info[hemi_key].iloc[0] == 'NaN':
            hemi_key = 'tal.x'
        regions = self.bin_electrodes_by_region(elec_column1=region_key1 if region_key1 else region_key2,
                                                elec_column2=region_key2,
                                                x_coord_column=hemi_key)
        regions['merged_col'] = regions['hemi'] + '-' + regions['region']
        return regions

    def plot_cluster_stats(self, cluster_name):
        """
        Multi panel plot showing:

        1. brain map of electrodes in the cluster, color-coded by phase
        2. brain map of electrodes in the cluster, color-coded by subsequent memory effect
        3. timecourse of resultant vector length of wave direction, averaged across trials
        4. timecourse of r^2, averaged across trials
        5. polor plot of left-frontal and hippocampal electrode phases
        6/7. same as 5, but for recalled and not recalled items only

        Data are taken from the timepoint with the highest r-square value.

        Figure creation code is so ugly sorry.
        """

        ############################
        # GET TIME TIME FOR X-AXIS #
        ############################
        time_axis = self.res['traveling_waves'][cluster_name]['time']

        #####################
        # SET UP THE FIGURE #
        #####################
        gs = gridspec.GridSpec(5, 3)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[2, :])
        ax3 = plt.subplot(gs[3, :])
        ax4 = plt.subplot(gs[4, 0], projection='polar')
        ax6 = plt.subplot(gs[4, 1], projection='polar')
        ax5 = plt.subplot(gs[1, :])

        # some figure parameters
        fig = plt.gcf()
        fig.set_size_inches(15, 25)
        mpl.rcParams['xtick.labelsize'] = 18
        mpl.rcParams['ytick.labelsize'] = 18

        #############################################
        # INFO ABOUT THE ELECTRODES IN THIS CLUSTER #
        #############################################
        cluster_rows = self.res['clusters'][cluster_name].notna()
        regions_all = self.get_electrode_roi_by_hemi()
        regions = regions_all[cluster_rows]['merged_col'].unique()
        regions_str = ', '.join(regions)
        xyz = self.res['clusters'][cluster_rows][['x', 'y', 'z']].values

        ###############################
        # ROW 1: electrodes and phase #
        ###############################
        mean_r2 = np.nanmean(self.res['traveling_waves'][cluster_name]['cluster_r2_adj'], axis=1)
        argmax_r2 = np.argmax(mean_r2)
        phases = self.res['traveling_waves'][cluster_name]['phase_data'][:, argmax_r2]
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        phases *= 180. / np.pi
        phases -= phases.min() - 1
        colors = np.stack([[0., 0., 0., 0.]] * len(phases))
        cm = clrs.LinearSegmentedColormap.from_list('cm', cc.cyclic_mrybm_35_75_c68_s25)
        cNorm = clrs.Normalize(vmin=0, vmax=359.99)
        colors[~np.isnan(phases)] = cmx.ScalarMappable(norm=cNorm, cmap=cm).to_rgba(phases[~np.isnan(phases)])
        ni_plot.plot_connectome(np.eye(xyz.shape[0]), xyz,
                                node_kwargs={'alpha': 0.7, 'edgecolors': None},
                                node_size=45, node_color=colors, display_mode='lzr',
                                axes=ax1)
        mean_freq = self.res['traveling_waves'][cluster_name]['mean_freq']
        plt.suptitle('{0} ({1:.2f} Hz): {2}'.format(self.subject, mean_freq, regions_str), y=.9)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='6%', pad=15)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm,
                                        norm=cNorm,
                                        orientation='vertical', ticks=[0, 90, 180, 270], )
        cb1.ax.tick_params(labelsize=14)
        for label in cb1.ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + mpl.transforms.ScaledTranslation(0.15, 0, fig.dpi_scale_trans))

        ##################################################
        # ROW 2: electrodes and subsequent memory effect #
        ##################################################
        sme = self.res['traveling_waves'][cluster_name]['sme_t'][:, argmax_r2]
        colors = np.stack([[0., 0., 0., 0.]] * len(sme))
        cm = plt.get_cmap('RdBu_r')
        clim = np.max(np.abs(sme))
        cNorm = clrs.Normalize(vmin=-clim, vmax=clim)
        colors[~np.isnan(sme)] = cmx.ScalarMappable(norm=cNorm, cmap=cm).to_rgba(sme[~np.isnan(sme)])
        ni_plot.plot_connectome(np.eye(xyz.shape[0]), xyz,
                                node_kwargs={'alpha': 0.7, 'edgecolors': None},
                                node_size=45, node_color=colors, display_mode='lzr',
                                axes=ax5)
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', size='6%', pad=15)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap='RdBu_r',
                                        norm=cNorm,
                                        orientation='vertical')
        cb2.ax.tick_params(labelsize=14)
        for label in cb2.ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + mpl.transforms.ScaledTranslation(0.15, 0, fig.dpi_scale_trans))

        ############################
        # ROW 3: timecourse of RVL #
        ############################
        rvl = pycircstat.resultant_vector_length(self.res['traveling_waves'][cluster_name]['cluster_wave_ang'], axis=1)
        ax2.plot(time_axis, rvl, lw=2)
        ax2.set_ylabel('RVL', fontsize=20)

        ##################################
        # ROW 4: timecourse of r-squared #
        ##################################
        ax3.plot(time_axis, np.nanmean(self.res['traveling_waves'][cluster_name]['cluster_r2_adj'], axis=1), lw=2)
        ax3.set_xlabel('Time (ms)', fontsize=20)
        ax3.set_ylabel('mean($R^{2}$)', fontsize=20)

        ############################
        # ROW 5a: phase polar plots #
        ############################
        phases = np.deg2rad(phases)
        cluster_regions = regions_all[cluster_rows]['merged_col']
        phases_left_front = phases[cluster_regions == 'left-Frontal']
        phases_hipp = phases[(cluster_regions == 'left-Hipp') | (cluster_regions == 'right-Hipp')]
        phases_other = phases[~cluster_regions.isin(['left-Frontal', 'left-Hipp', 'right-Hipp'])]

        for this_phase in phases_left_front:
            ax4.plot([this_phase, this_phase], [0, 1], lw=3, c='#67a9cf', alpha=.5)
        for this_phase in phases_hipp:
            ax4.plot([this_phase, this_phase], [0, 1], lw=3, c='#ef8a62', alpha=.5)
        for this_phase in phases_other:
            ax4.plot([this_phase, this_phase], [0, .7], lw=2, c='k', alpha=.4, zorder=-1)
        ax4.grid()
        for r in np.linspace(0, 2 * np.pi, 5)[:-1]:
            ax4.plot([r, r], [0, 1.3], lw=1, c=[.7, .7, .7], zorder=-2)
        ax4.spines['polar'].set_visible(False)
        ax4.set_ylim(0, 1.3)
        ax4.set_yticklabels([])
        ax4.set_aspect('equal', 'box')
        red_patch = mpatches.Patch(color='#67a9cf', label='L. Frontal')
        blue_patch = mpatches.Patch(color='#ef8a62', label='Hipp')
        _ = ax4.legend(handles=[red_patch, blue_patch], loc='lower left', bbox_to_anchor=(0.9, 0.9),
                       frameon=False, fontsize=16)

        ################################################
        # ROW 5b: phase polar plots for recalled items #
        ################################################
        phases = np.deg2rad(phases)
        cluster_regions = regions_all[cluster_rows]['merged_col']
        phases_left_front = phases[cluster_regions == 'left-Frontal']
        phases_hipp = phases[(cluster_regions == 'left-Hipp') | (cluster_regions == 'right-Hipp')]
        phases_other = phases[~cluster_regions.isin(['left-Frontal', 'left-Hipp', 'right-Hipp'])]

        for this_phase in phases_left_front:
            ax4.plot([this_phase, this_phase], [0, 1], lw=3, c='#67a9cf', alpha=.5)
        for this_phase in phases_hipp:
            ax4.plot([this_phase, this_phase], [0, 1], lw=3, c='#ef8a62', alpha=.5)
        for this_phase in phases_other:
            ax4.plot([this_phase, this_phase], [0, .7], lw=2, c='k', alpha=.4, zorder=-1)
        ax4.grid()
        for r in np.linspace(0, 2 * np.pi, 5)[:-1]:
            ax4.plot([r, r], [0, 1.3], lw=1, c=[.7, .7, .7], zorder=-2)
        ax4.spines['polar'].set_visible(False)
        ax4.set_ylim(0, 1.3)
        ax4.set_yticklabels([])
        ax4.set_aspect('equal', 'box')
        red_patch = mpatches.Patch(color='#67a9cf', label='L. Frontal')
        blue_patch = mpatches.Patch(color='#ef8a62', label='Hipp')
        _ = ax4.legend(handles=[red_patch, blue_patch], loc='lower left', bbox_to_anchor=(0.9, 0.9),
                       frameon=False, fontsize=16)

        plt.subplots_adjust(hspace=.5)
        return fig

def circ_lin_regress(phases, coords, theta_r, params):
    """

    """

    n = phases.shape[1]
    pos_x = np.expand_dims(coords[:, 0], 1)
    pos_y = np.expand_dims(coords[:, 1], 1)

    # compute predicted phases for angle and phase offset
    x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y

    # Compute resultant vector length. This is faster than calling pycircstat.resultant_vector_length
    x1 = numexpr.evaluate('sum(cos(x) / n, axis=1)')
    x1 = numexpr.evaluate('x1 ** 2')
    x2 = numexpr.evaluate('sum(sin(x) / n, axis=1)')
    x2 = numexpr.evaluate('x2 ** 2')
    Rs = numexpr.evaluate('-sqrt(x1 + x2)')

    # for each time and event, find the parameters with the smallest -R
    min_vals = theta_r[np.argmin(Rs, axis=1)]

    sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
    offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
                      np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)

    # compute circular correlation coefficient between actual phases and predicited phases
    circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)

    # compute adjusted r square
    r2_adj = circ_corr_coef ** 2

    wave_ang = min_vals[:, 0]
    wave_freq = min_vals[:, 1]

    return wave_ang, wave_freq, r2_adj

