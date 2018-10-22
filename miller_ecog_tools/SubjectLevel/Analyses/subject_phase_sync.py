"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency, compare correctly and incorrectly recalled
items using a t-test.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pycircstat
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, ttest_ind
from scipy.signal import hilbert
from itertools import combinations

from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_events_data import SubjectEventsRAMData


class SubjectPhaseSyncAnalysis(SubjectAnalysisBase, SubjectEventsRAMData):
    """
    Subclass of SubjectAnalysis and SubjectEventsRAMData

    The user must define the .recall_filter_func attribute of this class. This should be a function that, given a set
    of events, returns a boolean array of recalled (True) and not recalled (False) items.
    """

    res_str_tmp = 'elec_cluster_%d_mm_%d_elec_min_%s_elec_type_%s_sep_hemis_%.2f_cluster_range.p'
    attrs_in_res_str = ['elec_types_allowed', 'min_elec_dist', 'min_num_elecs', 'separate_hemis', 'cluster_freq_range']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectPhaseSyncAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = SubjectPhaseSyncAnalysis.res_str_tmp

        # The SME analysis is a contract between two conditions (recalled and not recalled items). Set
        # recall_filter_func to be a function that takes in events and returns bool of recalled items
        self.recall_filter_func = None

        # a list of lists defining ROIs. Each sublist will be treated as a single ROI. Append `left-` and `right-` to
        # each label you input
        self.roi_list = [['left-IFG'], ['left-Hipp', 'right-Hipp']]

        self.start_time = -500
        self.end_time = 1500
        self.wave_num = 5
        self.buf_ms = 2000
        self.noise_freq = [58., 62.]
        self.resample_freq = 250.
        self.hilbert_band_pass_range = [1, 4]
        self.log_power = True

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """

        """

        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s SME: please provide a .recall_filter_func function.' % self.subject)
        recalled = self.recall_filter_func(self.subject_data)

        # filter to electrodes in ROIs. First get broad electrode region labels
        region_df = self.bin_electrodes_by_region()
        region_df['merged_col'] = region_df['hemi'] + '-' + region_df['region']

        # make sure we have electrodes in each unique region
        for roi in self.roi_list:
            for label in roi:
                if ~np.any(region_df.merged_col == label):
                    print('{}: no {} electrodes, cannot compute synchrony.'.format(self.subject, label))
                    return

        # then filter into just to ROIs defined above
        elecs_to_use = region_df.merged_col.isin([item for sublist in self.roi_list for item in sublist])
        elec_scheme = self.elec_info.copy(deep=True)
        elec_scheme['ROI'] = region_df.merged_col[elecs_to_use]
        elec_scheme = elec_scheme[elecs_to_use].reset_index()

        # load eeg with pass band
        phase_data = RAM_helpers.load_eeg(self.subject_data,
                                          self.start_time,
                                          self.end_time,
                                          buf_ms=self.buf_ms,
                                          elec_scheme=elec_scheme,
                                          noise_freq=self.noise_freq,
                                          resample_freq=self.resample_freq,
                                          pass_band=self.hilbert_band_pass_range)

        # get phase at each timepoint
        phase_data.data = np.angle(hilbert(phase_data, N=phase_data.shape[-1], axis=-1))

        # remove the buffer
        phase_data = phase_data.remove_buffer(self.buf_ms / 1000.)

        # so now we have event x elec x time phase values. What to do?
        # define the pairs
        elec_label_pairs = []
        elec_region_pairs = []
        elec_pair_pvals = []
        elec_pair_zs = []
        elec_pair_pvals_rec = []
        elec_pair_zs_rec = []
        elec_pair_pvals_nrec = []
        elec_pair_zs_nrec = []

        # loop over each pair of ROIs
        for region_pair in combinations(self.roi_list, 2):
            elecs_region_1 = np.where(elec_scheme.ROI.isin(region_pair[0]))[0]
            elecs_region_2 = np.where(elec_scheme.ROI.isin(region_pair[1]))[0]

            # loop over all pairs of electrodes in the ROIs
            for elec_1 in elecs_region_1:
                for elec_2 in elecs_region_2:
                    elec_label_pairs.append([elec_scheme.iloc[elec_1].label, elec_scheme.iloc[elec_2].label])
                    elec_region_pairs.append(region_pair)

                    # and take the difference in phase values for this electrode pair
                    elec_pair_diff = pycircstat.cdiff(phase_data[:, elec_1], phase_data[:, elec_2])

                    # compute rayleigh on the phase difference
                    elec_pair_pval, elec_pair_z = pycircstat.rayleigh(elec_pair_diff, axis=0)
                    elec_pair_pvals.append(elec_pair_pval)
                    elec_pair_zs.append(elec_pair_z)

                    # also compute for recalled and not recalled items
                    elec_pair_pval_rec, elec_pair_z_rec = pycircstat.rayleigh(elec_pair_diff[recalled], axis=0)
                    elec_pair_pvals_rec.append(elec_pair_pval_rec)
                    elec_pair_zs_rec.append(elec_pair_z_rec)

                    elec_pair_pval_nrec, elec_pair_z_nrec = pycircstat.rayleigh(elec_pair_diff[~recalled], axis=0)
                    elec_pair_pvals_nrec.append(elec_pair_pval_nrec)
                    elec_pair_zs_nrec.append(elec_pair_z_nrec)

                    # do some shuffling here. Probably pull this whole section out into different function




    def bin_eloctrodes_into_rois(self):
        """

        Returns
        -------

        """

        # figure out the column names to use. Can very depending on where the electrode info came from
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

        # hardcoding this dictionary mapping electrode labels to regions
        roi_dict = {'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                             'Right CA3', 'Right DG', 'Right Sub'],
                    'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC'],
                    'IFG': ['parsopercularis', 'parsorbitalis', 'parstriangularis'],
                    'MFG': ['caudalmiddlefrontal', 'rostralmiddlefrontal'],
                    'SFG': ['superiorfrontal'],
                    'Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal'],
                    'Parietal': ['inferiorparietal', 'supramarginal', 'superiorparietal', 'precuneus'],
                    'Occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']}

        regions = self.bin_electrodes_by_region(elec_column1=region_key1 if region_key1 else region_key2,
                                                elec_column2=region_key2,
                                                x_coord_column=hemi_key,
                                                roi_dict=roi_dict)
        return regions

    def plot_timecourse(self, elec_label='', region_column='', loc_tag_column='',
                        freq_bins=[[1, 4], [4, 10], [10, 14], [16, 26], [28, 44], [46, 100]]):

        # get the index into the data for this electrode
        elec_ind = self.subject_data.channel == elec_label
        if ~np.any(elec_ind):
            print('%s: must enter a valid electrode label, as found in self.subject_data.channel' % self.subject)
            return

        max_val = 0
        x = self.subject_data.time.data
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                fig, axs = plt.subplots(len(freq_bins), 1, sharex=True, sharey=True)
                for f, ax in enumerate(axs):
                    freq_inds = (self.freqs >= freq_bins[f][0]) & (self.freqs <= freq_bins[f][1])

                    elec_data = np.squeeze(np.mean(self.res['ts'][freq_inds][:, elec_ind], axis=0))
                    max_val = np.max([max_val, np.max(np.abs(elec_data))])

                    ax.plot(x, elec_data, lw=3)
                    ax.plot(x, [0] * len(x), '-k', lw=1)

        ax.set_ylim((-max_val - .25, max_val + .25))
        ax.set_ylabel('T-stat')
        ax.set_xlabel('Time (s)')
        fig.set_size_inches(12, 10)

        # get some localization info for the title
        elec_info_chan = self.elec_info[self.elec_info.label == elec_label]
        title_str = ''
        for col in [region_column, loc_tag_column]:
            if col:
                title_str += ' ' + str(elec_info_chan[col].values) + ' '

        _ = axs[0].set_title('%s - %s' % (self.subject, elec_label) + title_str)

    def plot_spectra_average(self, elec_label='', region_column='', loc_tag_column='', time_range=None):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for recalled (red) and not-recalled (blue) items. Bottom panel is t-stat at each frequency comparing the
        recalled and not recalled distributions, with shaded areas indicating p<.05.

        elec_label: electrode label that you wish to plot.
        region_column: column of the elec_info dataframe that will be used to label the plot
        loc_tag_column: another of the elec_info dataframe that will be used to label the plot
        time_range: if data were computed with a time axis, time start and stop to mean in between
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subject)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subject)
            return

        # get the index into the data for this electrode
        elec_ind = self.subject_data.channel == elec_label
        if ~np.any(elec_ind):
            print('%s: must enter a valid electrode label, as found in self.subject_data.channel' % self.subject)
            return

        # normalize spectra
        recalled = self.res['recalled']
        p_spect = deepcopy(self.subject_data.data)

        # mean over time
        if self.subject_data.ndim == 4:
            if time_range is not None:
                time_inds = (self.subject_data.time >= time_range[0]) & (self.subject_data.time <= time_range[1])
            else:
                time_inds = np.ones(self.subject_data.shape[3]).astype(bool)
            p_spect = self.subject_data[:, :, :, time_inds].mean(axis=3)

        # normalize
        p_spect = self.normalize_spectra(p_spect)

        # create axis
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

                # will plot in log space
                x = np.log10(self.subject_data.frequency)

                ###############
                ## Top panel ##
                ###############
                # recalled mean and err
                rec_mean = np.mean(p_spect[recalled, :, elec_ind], axis=0)
                rec_sem = sem(p_spect[recalled, :, elec_ind], axis=0)
                ax1.plot(x, rec_mean, c='#8c564b', label='Good Memory', linewidth=2)
                ax1.fill_between(x, rec_mean + rec_sem, rec_mean - rec_sem, color='#8c564b', alpha=.5)

                # not recalled mean and err
                nrec_mean = np.mean(p_spect[~recalled, :, elec_ind], axis=0)
                nrec_sem = sem(p_spect[~recalled, :, elec_ind], axis=0)
                ax1.plot(x, nrec_mean, color='#1f77b4', label='Bad Memory', linewidth=2)
                ax1.fill_between(x, nrec_mean + nrec_sem, nrec_mean - nrec_sem, color='#1f77b4', alpha=.5)

                # y labels and y ticks
                ax1.set_ylabel('Normalized log(power)')
                ax1.yaxis.label.set_fontsize(24)
                ax1.yaxis.set_ticks([-2, -1, 0, 1, 2])
                # ax1.set_ylim([-2, 2])

                # make legend
                l = ax1.legend()
                frame = l.get_frame()
                frame.set_facecolor('w')
                for legobj in l.legendHandles:
                    legobj.set_linewidth(5)

                ##################
                ## Bottom panel ##
                ##################
                y = np.squeeze(self.res['ts'][:, elec_ind])
                p = np.squeeze(self.res['ps'][:, elec_ind])
                ax2.plot(x, y, '-k', linewidth=4)
                ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
                ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=1)
                ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y > 0), facecolor='#8c564b', edgecolor='#8c564b')
                ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y < 0), facecolor='#1f77b4', edgecolor='#1f77b4')
                ax2.set_ylabel('t-stat')
                plt.xlabel('Frequency', fontsize=24)
                ax2.yaxis.label.set_fontsize(24)
                # ax2.yaxis.set_ticks([-2, 0, 2])

                # put powers of two on the x-axis for both panels
                new_x = self.compute_pow_two_series()
                ax2.xaxis.set_ticks(np.log10(new_x))
                ax2.xaxis.set_ticklabels(new_x, rotation=0)
                ax1.xaxis.set_ticks(np.log10(new_x))
                ax1.xaxis.set_ticklabels('')

                # get some localization info for the title
                elec_info_chan = self.elec_info[self.elec_info.label == elec_label]
                title_str = ''
                for col in [region_column, loc_tag_column]:
                    if col:
                        title_str += ' ' + str(elec_info_chan[col].values) + ' '

                _ = ax1.set_title('%s - %s' % (self.subject, elec_label) + title_str)
                plt.gcf().set_size_inches(12, 9)

        return plt.gcf()

    def plot_elec_heat_map(self, sortby_column1='', sortby_column2=''):
        """
        Frequency by electrode SME visualization.

        Plots a channel x frequency visualization of the subject's data
        sortby_column1: if given, will sort the data by this column and then plot
        sortby_column2: secondary column for sorting
        """

        # group the electrodes by region, if we have the info
        do_region = True
        if sortby_column1 and sortby_column2:
            regions = self.elec_info[sortby_column1].fillna(self.elec_info[sortby_column2]).fillna(value='')
            elec_order = np.argsort(regions)
            groups = np.unique(regions)
        elif sortby_column1:
            regions = self.elec_info[sortby_column1].fillna(value='')
            elec_order = np.argsort(regions)
            groups = np.unique(regions)
        else:
            elec_order = np.arange(self.elec_info.shape[0])
            do_region = False

        # make dataframe of results for easier plotting
        df = pd.DataFrame(self.res['ts'], index=self.freqs,
                          columns=self.subject_data.channel)
        df = df.T.iloc[elec_order].T

        # make figure. Add axes for colorbar
        with mpl.rc_context({'ytick.labelsize': 14,
                             'xtick.labelsize': 14,
                             'axes.labelsize': 20}):
            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.1)

            # plot heatmap
            plt.gcf().set_size_inches(18, 12)
            clim = np.max(np.abs(self.res['ts']))
            sns.heatmap(df, cmap='RdBu_r', linewidths=.5,
                        yticklabels=df.index.values.round(2), ax=ax,
                        cbar_ax=cax, vmin=-clim, vmax=clim, cbar_kws={'label': 't-stat'})
            ax.set_xlabel('Channel', fontsize=24)
            ax.set_ylabel('Frequency (Hz)', fontsize=24)
            ax.invert_yaxis()

            # if plotting region info
            if do_region:
                ax2 = divider.append_axes('top', size='3%', pad=0)
                for i, this_group in enumerate(groups):
                    x = np.where(regions[elec_order] == this_group)[0]
                    ax2.plot([x[0] + .5, x[-1] + .5], [0, 0], '-', color=[.7, .7, .7])
                    if len(x) > 1:
                        if ' ' in this_group:
                            this_group = this_group.split()[0]+' '+''.join([x[0].upper() for x in this_group.split()[1:]])
                        else:
                            this_group = this_group[:12] + '.'
                        plt.text(np.mean([x[0] + .5, x[-1] + .5]), 0.05, this_group,
                                 fontsize=14,
                                 horizontalalignment='center',
                                 verticalalignment='bottom', rotation=90)
                ax2.set_xlim(ax.get_xlim())
                ax2.set_yticks([])
                ax2.set_xticks([])
                ax2.axis('off')

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.event.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.event.data['session'] == sess)
            m = np.mean(X[sess_event_mask], axis=1)
            m = np.mean(m, axis=0)
            s = np.std(X[sess_event_mask], axis=1)
            s = np.mean(s, axis=0)
            X[sess_event_mask] = (X[sess_event_mask] - m) / s
        return X

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))