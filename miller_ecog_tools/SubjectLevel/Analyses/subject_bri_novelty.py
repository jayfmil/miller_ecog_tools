"""

"""
import os
import pycircstat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
import h5py

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import signal
from scipy.stats import zscore, ttest_ind, sem
from scipy.signal import hilbert
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData
from miller_ecog_tools.Utils import neurtex_bri_helpers as bri

# figure out the number of cores available for a parallel pool. Will use half
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()


class SubjectNoveltyAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """

    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectNoveltyAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # this needs to be an event-locked analyses
        self.do_event_locked = True

        # frequencies at which to compute power using wavelets
        self.power_freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # also compute power/phase at frequencies in specific bands using hilbert transform, if desired
        self.hilbert_bands = np.array([[1, 4], [4, 10]])

        # how much time (in s) to remove from each end of the data after wavelet convolution
        self.buffer = 1.5

        # settings for guassian kernel used to smooth spike trains
        # enter .kern_width in milliseconds
        self.kern_width = 150
        self.kern_sd = 10

        # window to use when computing spike phase
        self.phase_bin_start = 0.0
        self.phase_bin_stop = 1.0

        # do we want to only include neurons and trials where the neuron actual modulates its firing rate in response
        # to the item coming on the screen
        self.z_responsive_thresh = 3

        # set to True to only include hits and correct rejections
        # self.only_correct_items = False

        # string to use when saving results files
        self.res_str = 'novelty.hdf5'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def load_res_data(self):
        """
        Load results if they exist and modify self.res to hold them.
        """
        if self.res_save_file is None:
            self._make_res_dir()

        if os.path.exists(self.res_save_file):
            print('%s: loading results.' % self.subject)
            self.res = h5py.File(self.res_save_file, 'r')
        else:
            print('%s: No results to load.' % self.subject)

    def save_res_data(self):
        """

        """
        pass

    def unload_res_data(self):
        """
        Load results if they exist and modify self.res to hold them.
        """
        self.res.close()

    def analysis(self):
        """
        For each session, channel
        """

        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
            return

        # create the file
        res_file = h5py.File(self.res_save_file, 'w')

        # arguments to pass into the event filtering function
        event_filters = {'all_events': {'only_correct': False},
                         'all_events_correct': {'only_correct': True},
                         'resp_events': {'filter_events': 'one', 'do_inverse': False, 'only_correct': False},
                         'resp_events_only_correct': {'filter_events': 'one', 'do_inverse': False, 'only_correct': True},
                         'resp_items': {'filter_events': 'both', 'do_inverse': False, 'only_correct': False},
                         'resp_items_only_correct': {'filter_events': 'both', 'do_inverse': False, 'only_correct': True},
                         'resp_events_inv': {'filter_events': 'one', 'do_inverse': True, 'only_correct': False},
                         'resp_items_inv': {'filter_events': 'both', 'do_inverse': True, 'only_correct': False}}

        # open a parallel pool using joblib
        with Parallel(n_jobs=int(NUM_CORES/2) if NUM_CORES != 1 else 1,  verbose=5) as parallel:

            # loop over sessions
            for session_name, session_grp in self.subject_data.items():
                print('{} processing.'.format(session_grp.name))

                # and channels
                for channel_num, channel_grp in tqdm(session_grp.items()):
                    res_channel_grp = res_file.create_group(channel_grp.name)

                    # self.res[channel_grp.name] = {}
                    # self.res[channel_grp.name]['firing_rates'] = {}

                    # load behavioral events
                    events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')
                    events['item_name'] = events.name.apply(lambda x: x.split('_')[1])

                    # load eeg for this channel
                    eeg_channel = self._create_eeg_timeseries(channel_grp, events)

                    # length of buffer in samples. Used below for extracting smoothed spikes
                    samples = int(np.ceil(float(eeg_channel['samplerate']) * self.buffer))

                    # next we want to compute the power at all the frequencies in self.power_freqs and at
                    # all the timepoints in eeg. This function save the results to file and returns phase data
                    # for later use.
                    # Compute for wavelet frequencies
                    phase_data = run_novelty_effect(eeg_channel, self.power_freqs, self.buffer, res_channel_grp,
                                                    parallel, '_wavelet', save_to_file=False)

                    # and for hilbert bands
                    phase_data_hilbert = run_novelty_effect(eeg_channel, self.hilbert_bands, self.buffer,
                                                            res_channel_grp, parallel, '_hilbert', save_to_file=False)

                    # also store region and hemisphere for easy reference
                    res_channel_grp.attrs['region'] = eeg_channel.event.data['region'][0]
                    res_channel_grp.attrs['hemi'] = eeg_channel.event.data['hemi'][0]

                    # and clusters
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        res_cluster_grp = res_channel_grp.create_group(clust_str)

                        # find number of spikes at each timepoint and the time in samples when each occurred
                        spike_counts, spike_rel_times = self._create_spiking_counts(cluster_grp, events,
                                                                                    eeg_channel.shape[1])

                        # smooth the spike train. Also remove the buffer
                        kern_width_samples = int(eeg_channel.samplerate.data / (1000 / self.kern_width))
                        kern = signal.gaussian(kern_width_samples, self.kern_sd)
                        kern /= kern.sum()
                        smoothed_spike_counts = np.stack([signal.convolve(x, kern, mode='same')[samples:-samples]
                                                          for x in spike_counts * eeg_channel.samplerate.data], 0)
                        smoothed_spike_counts = self._create_spike_timeseries(smoothed_spike_counts,
                                                                              eeg_channel.time.data[
                                                                              samples:-samples],
                                                                              channel_grp.attrs['samplerate'],
                                                                              events)

                        # get the phases at which the spikes occurred and bin into novel and repeated items
                        # 1. for each freq in power_freqs
                        spike_phases = _compute_spike_phase_by_freq(np.array(spike_rel_times),
                                                                    self.phase_bin_start,
                                                                    self.phase_bin_stop,
                                                                    phase_data,
                                                                    events)

                        # 2: for each hilbert band
                        spike_phases_hilbert = _compute_spike_phase_by_freq(np.array(spike_rel_times),
                                                                            self.phase_bin_start,
                                                                            self.phase_bin_stop,
                                                                            phase_data_hilbert,
                                                                            events)

                        # and finally loop over event conditions
                        for this_event_cond, event_filter_kwargs in event_filters.items():

                            # figure out which events to use
                            if 'all_events' in this_event_cond:
                                events_to_keep = np.array([True] * events.shape[0])
                            else:
                                events_to_keep = self._filter_to_event_condition(eeg_channel, spike_counts, events,
                                                                                 **event_filter_kwargs)

                            if event_filter_kwargs['only_correct']:
                                events_to_keep = self._filter_to_correct_items(events, events_to_keep)

                            # do the same computations on the wavelet derived spikes and hilbert
                            for phase_data_list in zip([spike_phases, spike_phases_hilbert],
                                                             ['_wavelet', '_hilbert'],
                                                             [self.power_freqs, self.hilbert_bands]):

                                event_filter_grp = res_cluster_grp.create_group(this_event_cond+phase_data_list[1])
                                do_compute_mem_effects = run_phase_stats(phase_data_list[0], events, events_to_keep,
                                                                         event_filter_grp)

                                # also compute the power effects for these filtered event conditions
                                if do_compute_mem_effects:
                                    _ = run_novelty_effect(eeg_channel[events_to_keep], phase_data_list[2], self.buffer,
                                                           event_filter_grp, parallel, '',
                                                           save_to_file=True)

                                # finally, compute stats based on normalizing from the pre-stimulus interval
                                spike_res_zs = compute_novelty_stats_without_contrast(smoothed_spike_counts[events_to_keep])
                                event_filter_grp.create_dataset('zdata_novel_mean_' + this_event_cond,
                                                                data=spike_res_zs[0])
                                event_filter_grp.create_dataset('zdata_repeated_mean_' + this_event_cond,
                                                                data=spike_res_zs[1])
                                event_filter_grp.create_dataset('zdata_novel_sem_' + this_event_cond,
                                                                data=spike_res_zs[2])
                                event_filter_grp.create_dataset('zdata_repeated_sem_' + this_event_cond,
                                                                data=spike_res_zs[3])
                                event_filter_grp.create_dataset('zdata_ts_' + this_event_cond,
                                                                data=spike_res_zs[4])
                                event_filter_grp.create_dataset('zdata_ps_' + this_event_cond,
                                                                data=spike_res_zs[5])
        res_file.close()
        self.res = h5py.File(self.res_save_file, 'r')

    def _filter_to_correct_items(self, events, to_keep_bool):

        # get boolean of correct responses
        novel_items = events['isFirst'].values
        pressed_old_key = events['oldKey'].values
        hits = pressed_old_key & ~novel_items
        misses = ~pressed_old_key & ~novel_items
        correct = hits | misses
        return to_keep_bool & correct

    def _filter_to_event_condition(self, eeg_channel, spike_counts, events, filter_events='', do_inverse=False,
                                   only_correct=False):

        # normalize the presentation interval based on the mean and standard deviation of a pre-stim interval
        baseline_bool = (eeg_channel.time.data > -1) & (eeg_channel.time.data < -.2)
        baseline_spiking = np.sum(spike_counts[:, baseline_bool], axis=1) / .8
        baseline_mean = np.mean(baseline_spiking)
        baseline_std = np.std(baseline_spiking)

        # get the firing rate of the presentation interval now and zscore it
        presentation_bool = (eeg_channel.time.data > 0) & (eeg_channel.time.data <= 1)
        presentation_spiking = np.sum(spike_counts[:, presentation_bool], axis=1) / 1.
        z_firing = (presentation_spiking - baseline_mean) * baseline_std

        # Here, keep all events where the firing is above our threshold
        if filter_events == 'one':
            responsive_items = np.unique(events['item_name'][z_firing > self.z_responsive_thresh])

            # make sure no "filler" items are present
            responsive_items = np.array([s for s in responsive_items if 'filler' not in s])

        # Here, only keep events if both presentations of the item are above threshold
        else:
            responsive_items_all = events['item_name'][z_firing > self.z_responsive_thresh]
            responsive_items = []
            for this_item in responsive_items_all:
                if np.sum(responsive_items_all == this_item) == 2:
                    responsive_items.append(this_item)
            responsive_items = np.unique(responsive_items)

        # make a boolean of the items to keep
        to_keep_bool = np.in1d(events['item_name'], responsive_items)
        if do_inverse:
            to_keep_bool = ~to_keep_bool

        return to_keep_bool

    def _compute_item_pair_diff(self, smoothed_spike_counts):
        data = smoothed_spike_counts[~((smoothed_spike_counts.event.data['isFirst']) & (smoothed_spike_counts.event.data['lag'] == 0))]
        item_names = data.event.data['item_name']

        novel_rep_diffs = []
        mean_item_frs = []
        novel_mean = []
        rep_mean = []

        for this_item in np.unique(item_names):
            data_item = data[item_names == this_item]
            if data_item.shape[0] == 2:
                novel_data_item = data_item[data_item.event.data['isFirst']].values
                rep_data_item = data_item[~data_item.event.data['isFirst']].values
                diff_due_to_cond = novel_data_item - rep_data_item
                novel_rep_diffs.append(diff_due_to_cond)
                novel_mean.append(novel_data_item)
                rep_mean.append(rep_data_item)
                mean_item_frs.append(np.mean(data_item.data))

        novel_mean = np.squeeze(np.stack(novel_mean))
        novel_sem = sem(novel_mean, axis=0)
        novel_trial_means = np.mean(novel_mean, axis=1)
        novel_mean = np.mean(novel_mean, axis=0)

        rep_mean = np.squeeze(np.stack(rep_mean))
        rep_sem = sem(rep_mean, axis=0)
        rep_trial_means = np.mean(rep_mean, axis=1)
        rep_mean = np.mean(rep_mean, axis=0)

        return np.squeeze(np.stack(novel_rep_diffs)), np.stack(mean_item_frs), novel_mean, rep_mean, novel_sem, \
               rep_sem, novel_trial_means, rep_trial_means

    def _create_spiking_counts(self, cluster_grp, events, n):
        spike_counts = []
        spike_ts = []

        # loop over each event
        for index, e in events.iterrows():
            # load the spike times for this cluster
            spike_times = np.array(cluster_grp[str(index)])

            # interpolate the timestamps for this event
            start = e.stTime + self.start_ms * 1000
            stop = e.stTime + self.stop_ms * 1000
            timestamps = np.linspace(start, stop, n, endpoint=True)

            # find the closest timestamp to each spike (technically, the closest timestamp following a spike, but I
            # think this level of accuracy is fine). This is the searchsorted command. Then count the number of spikes
            # that occurred at each timepoint with histogram
            spike_bins = np.searchsorted(timestamps, spike_times)
            bin_counts, _ = np.histogram(spike_bins, np.arange(len(timestamps) + 1))
            spike_counts.append(bin_counts)
            spike_ts.append(spike_bins)

        return np.stack(spike_counts, 0), spike_ts

    def _create_eeg_timeseries(self, grp, events):
        data = np.array(grp['ev_eeg'])
        time = grp.attrs['time']
        channel = grp.attrs['channel']
        sr = grp.attrs['samplerate']

        # create an TimeSeries object (in order to make use of their wavelet calculation)
        dims = ('event', 'time', 'channel')
        coords = {'event': events[events.columns[events.columns != 'index']].to_records(),
                  'time': time,
                  'channel': [channel]}

        return TimeSeries.create(data, samplerate=sr, dims=dims, coords=coords)

    def _create_spike_timeseries(self, spike_data, time, sr, events):
        # create an TimeSeries object
        dims = ('event', 'time')
        coords = {'event': events[events.columns[events.columns != 'index']].to_records(),
                  'time': time}
        return TimeSeries.create(spike_data, samplerate=sr, dims=dims, coords=coords)

    def aggregate_ensemble_phases_by_condition(self):

        # labels of each band
        bands = np.array(['{}-{}'.format(*x) for x in self.hilbert_bands])

        # will hold long dataframes, one for novel and one for repeated spiking phases
        novel_dfs = []
        rep_dfs = []

        # loop over each channel
        for k, v in self.res.items():

            # hemi and region is the same for all clusters on this channel
            hemi = v['hemi']
            region = v['region']

            # loop over each cluster in channel
            for k_clust, v_clust in v['firing_rates'].items():
                novel_phases = v_clust['novel_phases_hilbert']
                rep_phases = v_clust['rep_phases_hilbert']

                # make a dataframe for novel and for repeated spiking phases
                for i, data in enumerate([novel_phases, rep_phases]):
                    df = pd.DataFrame(data=data.T)
                    df['hemi'] = hemi
                    df['region'] = region
                    df['bands'] = bands
                    df['label'] = k + '-' + k_clust
                    df = df.melt(id_vars=['label', 'hemi', 'region', 'bands'], var_name='spike', value_name='phase')

                    # and store it
                    if i == 0:
                        novel_dfs.append(df)
                    else:
                        rep_dfs.append(df)

        # combine into one larger dataframe for each conditon
        novel_dfs = pd.concat(novel_dfs).reset_index(drop=True)
        rep_dfs = pd.concat(rep_dfs).reset_index(drop=True)

        return novel_dfs, rep_dfs

    def plot_channel_res(self, channel_str, savedir=None, do_t_not_z=False):
        """
        Plot time x freq heatmap, firing rates, and phase results for a given channel.
        """

        # results for this channel only
        channel_res = self.res[channel_str]

        # pull out the specific results
        if do_t_not_z:
            lfp_data = channel_res['delta_t']
            spike_data_key = 'delta_spike_t'
            cbar_label = 't-stat'
        else:
            lfp_data = channel_res['delta_z']
            spike_data_key = 'delta_spike_z'
            cbar_label = 'z-score'

        time = lfp_data.columns.values
        clim = np.max(np.abs(lfp_data.values))
        hemi = channel_res['hemi']
        region = channel_res['region']

        # how many units were recorded on this channel
        cluster_keys = channel_res['firing_rates']
        num_clusters = len(cluster_keys)

        with plt.style.context('seaborn-white'):
            with mpl.rc_context({'ytick.labelsize': 22,
                                 'xtick.labelsize': 22}):

                # make the initial figure
                # top left, heatmap
                ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=3, rowspan=3)

                # below heatmap, up to 3 cluster firing rates
                ax2 = plt.subplot2grid((6, 6), (3, 0), colspan=3)
                ax2.axis('off')
                ax3 = plt.subplot2grid((6, 6), (4, 0), colspan=3)
                ax3.axis('off')
                ax4 = plt.subplot2grid((6, 6), (5, 0), colspan=3)
                ax4.axis('off')

                # to the right of heatmap, up to 3 phase by freq
                ax5 = plt.subplot2grid((6, 6), (0, 3), rowspan=3)
                ax5.axis('off')
                ax6 = plt.subplot2grid((6, 6), (0, 4), rowspan=3)
                ax6.axis('off')
                ax7 = plt.subplot2grid((6, 6), (0, 5), rowspan=3)
                ax7.axis('off')
                fig = plt.gcf()
                fig.set_size_inches(30, 20)

                # make heatmap
                im = ax1.imshow(lfp_data.values,
                                aspect='auto', interpolation='bicubic', cmap='RdBu_r', vmax=clim, vmin=-clim)
                ax1.invert_yaxis()

                # set the x values to be specific timepoints
                x_vals = np.array([-500, -250, 0, 250, 500, 750, 1000]) / 1000
                new_xticks = np.round(np.interp(x_vals, time, np.arange(len(time))))
                ax1.set_xticks(new_xticks)
                ax1.set_xticklabels([x for x in x_vals], fontsize=22)
                ax1.set_xlabel('Time (s)', fontsize=24)

                # now the y
                new_y = np.interp(np.log10(np.power(2, range(7))), np.log10(self.power_freqs),
                                  np.arange(len(self.power_freqs)))
                ax1.set_yticks(new_y)
                ax1.set_yticklabels(np.power(2, range(7)), fontsize=20)
                ax1.set_ylabel('Frequency (Hz)', fontsize=24)

                # add colorbar
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                fig.colorbar(im, cax=cax, orientation='vertical')

                # add a title
                title_str = '{} - {} {}'.format(channel_str, hemi, region)
                ax1.set_title(title_str, fontsize=20)

                # firing rate plots
                for i, this_cluster in enumerate(zip([ax2, ax3, ax4], list(cluster_keys))):
                    this_cluster_ax = this_cluster[0]
                    this_cluster_ax.axis('on')
                    divider = make_axes_locatable(this_cluster_ax)
                    dummy_ax = divider.append_axes('right', size='5%', pad=0.1)
                    dummy_ax.axis('off')

                    this_cluster_data = channel_res['firing_rates'][this_cluster[1]][spike_data_key]

                    zdata_novel = channel_res['firing_rates'][this_cluster[1]]['zdata_novel_mean']
                    zdata_novel_sem = channel_res['firing_rates'][this_cluster[1]]['zdata_novel_sem']
                    zdata_repeated = channel_res['firing_rates'][this_cluster[1]]['zdata_repeated_mean']
                    zdata_repeated_sem = channel_res['firing_rates'][this_cluster[1]]['zdata_repeated_sem']
                    zdata_ps = channel_res['firing_rates'][this_cluster[1]]['zdata_ps']

                    novel_c = [0.6922722029988465, 0.0922722029988466, 0.1677047289504037]
                    rep_c = [0.023913879277201077, 0.19653979238754324, 0.3919261822376009]
                    this_cluster_ax.plot(this_cluster_data.columns.values,
                                         zdata_novel, lw=3, label='Novel', c=novel_c)
                    this_cluster_ax.fill_between(this_cluster_data.columns.values,
                                                 zdata_novel - zdata_novel_sem,
                                                 zdata_novel + zdata_novel_sem, alpha=.6, color=novel_c)

                    this_cluster_ax.plot(this_cluster_data.columns.values,
                                         zdata_repeated, lw=3, label='Repeated', c=rep_c)
                    this_cluster_ax.fill_between(this_cluster_data.columns.values,
                                                 zdata_repeated - zdata_repeated_sem,
                                                 zdata_repeated + zdata_repeated_sem, alpha=.6, color=rep_c)
                    this_cluster_ax.legend(loc='best')

                    x = np.array([-500, -250, 0, 250, 500, 750, 1000]) / 1000
                    this_cluster_ax.set_xticks(x)
                    this_cluster_ax.set_xticklabels(['{0:.2}'.format(xstr) for xstr in x])
                    this_cluster_ax.set_ylabel('Z(Firing Rate)', fontsize=22)
                    this_cluster_ax.plot([-.5, 1], [0, 0], '--k', zorder=-2, lw=1.5, c=[.7, .7, .7])
                    this_cluster_ax.set_xlim(-.5, 1)
                    this_cluster_ax.set_title(this_cluster[1], fontsize=16)
                    if (i + 1) == num_clusters:
                        this_cluster_ax.set_xlabel('Time (s)', fontsize=22)

                # phase_plots
                for i, this_cluster in enumerate(zip([ax5, ax6, ax7], list(cluster_keys))):
                    this_cluster_ax_left = this_cluster[0]
                    this_cluster_ax_left.axis('on')

                    z_novel = channel_res['firing_rates'][this_cluster[1]]['z_novel']
                    z_rep = channel_res['firing_rates'][this_cluster[1]]['z_rep']
                    z_delta = z_novel - z_rep
                    this_cluster_ax_left.plot(z_delta,
                                              np.log10(self.power_freqs), '-k', lw=3)
                    yticks = np.power(2, range(1, 7))
                    this_cluster_ax_left.set_yticks(np.log10(yticks))
                    this_cluster_ax_left.set_yticklabels(yticks)
                    this_cluster_ax_left.set_ylim(np.log10(1), np.log10(100))
                    this_cluster_ax_left.set_xlabel(r'$\Delta$(Z)', fontsize=22)
                    this_cluster_ax_left.plot([0, 0], this_cluster_ax_left.get_ylim(), '--k', zorder=-2, lw=1.5,
                                              c=[.7, .7, .7])
                    xlim = np.max(np.abs(this_cluster_ax_left.get_xlim()))
                    this_cluster_ax_left.set_xlim(-xlim, xlim)
                    this_cluster_ax_left.set_xticks([-xlim, xlim])

                    divider = make_axes_locatable(this_cluster_ax_left)
                    this_cluster_ax_right = divider.append_axes('right', size='95%', pad=0.05)
                    data = -np.log10(channel_res['firing_rates'][this_cluster[1]]['med_pvals'])
                    this_cluster_ax_right.plot(data,
                                               np.log10(self.power_freqs), '-k', lw=3)
                    yticks = np.power(2, range(1, 7))
                    this_cluster_ax_right.set_yticks(np.log10(yticks))
                    this_cluster_ax_right.set_yticklabels([])
                    this_cluster_ax_right.set_ylim(np.log10(1), np.log10(100))
                    this_cluster_ax_right.xaxis.set_label_position("top")
                    this_cluster_ax_right.xaxis.tick_top()

                    this_cluster_ax_right.set_xlabel('-log(p)', fontsize=22)
                    this_cluster_ax_right.plot([-np.log10(0.05), -np.log10(0.05)], this_cluster_ax_right.get_ylim(),
                                               '--', zorder=-2, lw=1.5, c=[.4, .0, .0])
                    this_cluster_ax_right.set_title(this_cluster[1], color='k', rotation=-90, x=1.2, y=0.55,
                                                    fontsize=20)

                plt.subplots_adjust(wspace=0.8, hspace=1.)
                #             plt.tight_layout()
                if savedir is not None:
                    fname = '{}_{}_time_x_freq_grid.pdf'.format(self.subject, channel_str.replace('/', '-'))
                    fname = os.path.join(savedir, fname)
                    fig.savefig(fname, bbox_inches='tight')
                    return fname


def compute_hilbert_at_single_band(eeg, freq_band, buffer_len):

    # band pass eeg
    # makes sure to pass in a list not an array because wtf PTSA
    band_eeg = bri.band_pass_eeg(eeg, freq_band.tolist() if isinstance(freq_band, np.ndarray) else freq_band).squeeze()

    # run hilbert to get the complexed valued result
    complex_hilbert_res = hilbert(band_eeg.data, N=band_eeg.shape[-1], axis=-1)

    # get phase at each timepoint
    phase_data = band_eeg.copy()
    phase_data.data = np.angle(complex_hilbert_res)
    phase_data = phase_data.remove_buffer(buffer_len)
    phase_data.coords['frequency'] = np.mean(freq_band)

    # and power
    power_data = band_eeg.copy()
    power_data.data = np.abs(complex_hilbert_res) ** 2
    power_data = power_data.remove_buffer(buffer_len)
    power_data.coords['frequency'] = np.mean(freq_band)

    return power_data, phase_data


def compute_wavelet_at_single_freq(eeg, freq, buffer_len):

    # compute phase
    data = MorletWaveletFilter(eeg,
                               np.array([freq]),
                               output=['power', 'phase'],
                               width=5,
                               cpus=12,
                               verbose=False).filter()

    # remove the buffer from each end
    data = data.remove_buffer(buffer_len)
    return data.squeeze()


def _compute_spike_phase_by_freq(spike_rel_times, phase_bin_start, phase_bin_stop, phase_data, events):

    # only will count samples the occurred within window defined by phase_bin_start and _stop
    valid_samps = np.where((phase_data.time > phase_bin_start) & (phase_data.time < phase_bin_stop))[0]

    # throw out novel items that were never repeated
    good_events = events[~((events['isFirst']) & (events['lag'] == 0))].index.values

    # will grow as we iterate over spikes in each condition.
    phases = []
    for (index, e), spikes, phase_data_event in zip(events.iterrows(), spike_rel_times, phase_data):
        phases_event = []
        if index in good_events:
            if len(spikes) > 0:
                valid_spikes = spikes[np.in1d(spikes, valid_samps)]
                if len(valid_spikes) > 0:
                    phases_event = phase_data_event[valid_spikes].data

        phases.append(phases_event)

    # will be number of spikes x frequencies
    #     if len(novel_phases) > 0:
    #         novel_phases = np.vstack(novel_phases)
    #     else:
    #         novel_phases = np.array(novel_phases)
    #     if len(rep_phases) > 0:
    #         rep_phases = np.vstack(rep_phases)
    #     else:
    #         rep_phases = np.array(rep_phases)

    return np.array(phases)


def run_phase_stats(spike_phases, events, events_to_keep, event_filter_grp):

    novel_phases = np.array([])
    rep_phases = np.array([])

    # get the novel and repeated spike phases for this event condition. Some events have no
    # spikes, so filter those out
    spike_phase_cond = spike_phases[events_to_keep]
    if np.any(events[events_to_keep].isFirst):
        novel_phases = spike_phase_cond[events[events_to_keep].isFirst]
        novel_phases = novel_phases[np.array([len(x) > 0 for x in novel_phases])]
    if novel_phases.shape[0] == 0:
        novel_phases = []
    else:
        novel_phases = np.vstack(novel_phases)

    if np.any(~events[events_to_keep].isFirst):
        rep_phases = spike_phase_cond[~events[events_to_keep].isFirst]
        rep_phases = rep_phases[np.array([len(x) > 0 for x in rep_phases])]
    if rep_phases.shape[0] == 0:
        rep_phases = []
    else:
        rep_phases = np.vstack(rep_phases)

    if (len(novel_phases) > 0) & (len(rep_phases) > 0):
        p_novel, z_novel, p_rep, z_rep, ww_pvals, ww_fstat, med_pvals, med_stat, p_kuiper, \
        stat_kuiper = _compute_novel_rep_spike_stats(novel_phases, rep_phases)
    else:
        p_novel = z_novel = p_rep = z_rep = ww_pvals = ww_fstat = med_pvals \
            = med_stat = p_kuiper = stat_kuiper = np.array([np.nan])

    event_filter_grp.create_dataset('p_novel', data=p_novel)
    event_filter_grp.create_dataset('z_novel', data=z_novel)
    event_filter_grp.create_dataset('p_rep', data=p_rep)
    event_filter_grp.create_dataset('z_rep', data=z_rep)
    event_filter_grp.create_dataset('ww_pvals', data=ww_pvals)
    event_filter_grp.create_dataset('ww_fstat', data=ww_fstat)
    event_filter_grp.create_dataset('med_stat', data=med_stat)
    event_filter_grp.create_dataset('p_kuiper', data=p_kuiper)
    event_filter_grp.create_dataset('stat_kuiper', data=stat_kuiper)

    event_filter_grp.create_dataset('rep_phases', data=rep_phases)
    event_filter_grp.create_dataset('novel_phases', data=novel_phases)

    return (len(novel_phases) > 0) & (len(rep_phases) > 0)


def _compute_novel_rep_spike_stats(novel_phases, rep_phases):

    # compute rayleigh test for each condition
    p_novel, z_novel = pycircstat.rayleigh(novel_phases, axis=0)
    p_rep, z_rep = pycircstat.rayleigh(rep_phases, axis=0)

    # test whether the means are different
    ww_pvals, ww_tables = pycircstat.watson_williams(novel_phases, rep_phases, axis=0)
    ww_fstat = np.array([x.loc['Columns'].F for x in ww_tables])

    # test whether the medians are different
    med_pvals, med_stat = pycircstat.cmtest(novel_phases, rep_phases, axis=0)

    # finall run kuiper test for difference in mean and/or dispersion
    p_kuiper, stat_kuiper = pycircstat.kuiper(novel_phases, rep_phases, axis=0)

    return p_novel, z_novel, p_rep, z_rep, ww_pvals, ww_fstat, med_pvals, med_stat, p_kuiper, stat_kuiper


def run_novelty_effect(eeg_channel, power_freqs, buffer, grp, parallel=None, key_suffix='', save_to_file=False):

    f = compute_lfp_novelty_effect

    if parallel is None:
        memory_effect_channel = []
        for freq in power_freqs:
            memory_effect_channel.append(f(eeg_channel, freq, buffer))
    else:
        memory_effect_channel = parallel((delayed(f)(eeg_channel, freq, buffer) for freq in power_freqs))
    phase_data = xarray.concat([x[4] for x in memory_effect_channel], dim='frequency').transpose('event', 'time', 'frequency')

    if save_to_file:
        fname = grp.file.filename
        pd.concat([x[0] for x in memory_effect_channel]).to_hdf(fname, grp.name + '/delta_z'+key_suffix)
        pd.concat([x[1] for x in memory_effect_channel]).to_hdf(fname, grp.name + '/delta_t'+key_suffix)
        pd.concat([x[2] for x in memory_effect_channel]).to_hdf(fname, grp.name + '/delta_z_lag'+key_suffix)
        pd.concat([x[3] for x in memory_effect_channel]).to_hdf(fname, grp.name + '/delta_t_lag'+key_suffix)
    return phase_data


def compute_novelty_stats(data_timeseries):
    def compute_z_diff_lag(df):
        novel = df[df.isFirst]
        repeated = df[~df.isFirst]
        cols = df.columns[~np.in1d(df.columns, ['lag', 'isFirst'])]
        return novel[cols].mean() - repeated[cols].mean()

    def compute_t_stat_lag(df):
        novel = df[df.isFirst]
        repeated = df[~df.isFirst]
        cols = df.columns[~np.in1d(df.columns, ['lag', 'isFirst'])]
        ts, ps = ttest_ind(novel[cols], repeated[cols], axis=0)
        return pd.Series(ts.data, index=cols)

    # remove the filler novel items (they were never repeated)
    data = data_timeseries[~((data_timeseries.event.data['isFirst']) & (data_timeseries.event.data['lag'] == 0))]

    # then zscore across events
    zdata = zscore(data, axis=0)

    # split events into conditions of novel and repeated items
    novel_items = data.event.data['isFirst']

    # compute mean difference in zpower at each timepoint
    zpower_diff = zdata[novel_items].mean(axis=0) - zdata[~novel_items].mean(axis=0)
    df_zpower_diff = pd.DataFrame(pd.Series(zpower_diff, index=data.time)).T

    # also compute t-stat at each timepoint
    ts, ps = ttest_ind(zdata[novel_items], zdata[~novel_items], axis=0)
    df_tstat_diff = pd.DataFrame(pd.Series(ts, index=data.time)).T

    # create dataframe of results for easier manipulation for computing difference by lag
    df = pd.DataFrame(data=zdata, columns=data.time)
    df['lag'] = data.event.data['lag']
    df['isFirst'] = novel_items

    df_lag_zpower_diff = df.groupby(['lag']).apply(compute_z_diff_lag)
    df_lag_tstat_diff = df.groupby(['lag']).apply(compute_t_stat_lag)
    return df_zpower_diff, df_tstat_diff, df_lag_zpower_diff, df_lag_tstat_diff


def compute_lfp_novelty_effect(eeg, freq, buffer_len):

    # compute the power first
    if isinstance(freq, float):
        power_data, phase_data = compute_wavelet_at_single_freq(eeg, freq, buffer_len)
    else:
        power_data, phase_data = compute_hilbert_at_single_band(eeg, freq, buffer_len)
        freq = np.mean(freq)

    # compute the novelty statistics
    df_zpower_diff, df_tstat_diff, df_lag_zpower_diff, df_lag_tstat_diff = compute_novelty_stats(power_data)

    # add the current frequency to the dataframe index
    df_zpower_diff.set_index(pd.Series(freq), inplace=True)
    df_zpower_diff.index.rename('frequency', inplace=True)
    df_tstat_diff.set_index(pd.Series(freq), inplace=True)
    df_tstat_diff.index.rename('frequency', inplace=True)

    n_rows = df_lag_tstat_diff.shape[0]
    index = pd.MultiIndex.from_arrays([df_lag_zpower_diff.index, np.array([freq] * n_rows)], names=['lag', 'frequency'])
    df_lag_zpower_diff.index = index
    index = pd.MultiIndex.from_arrays([df_lag_tstat_diff.index, np.array([freq] * n_rows)], names=['lag', 'frequency'])
    df_lag_tstat_diff.index = index
    return df_zpower_diff, df_tstat_diff, df_lag_zpower_diff, df_lag_tstat_diff, phase_data


def compute_novelty_stats_without_contrast(data_timeseries, baseline_bool=None):

    # remove the filler novel items (they were never repeated)
    data = data_timeseries[~((data_timeseries.event.data['isFirst']) & (data_timeseries.event.data['lag'] == 0))]

    # determine the mean and std of the baseline period for normalization
    # if baseline bool is not given, use all timepoints before 0
    if baseline_bool is None:
        baseline_bool = data.time.values < 0
    baseline_data = data[:, baseline_bool].mean(dim='time')
    m = np.mean(baseline_data)
    s = np.std(baseline_data)

    # compute the zscored data
    zdata = (data - m) / s

    # pull out the data for each condition
    novel_items = data.event.data['isFirst']
    zdata_novel = zdata[novel_items]
    zdata_repeated = zdata[~novel_items]

    # run stats at each timepoint
    ts, ps = ttest_ind(zdata_novel, zdata_repeated, axis=0)

    # return the statistics and the mean of each condition
    zdata_novel_mean = np.mean(zdata_novel, axis=0)
    zdata_novel_sem = sem(zdata_novel, axis=0)
    zdata_repeated_mean = np.mean(zdata_repeated, axis=0)
    zdata_repeated_sem = sem(zdata_repeated, axis=0)

    return zdata_novel_mean, zdata_repeated_mean, zdata_novel_sem, zdata_repeated_sem, ts, ps













































