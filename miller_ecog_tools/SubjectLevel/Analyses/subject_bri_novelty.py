"""

"""
import os
import pycircstat
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import signal
from scipy.stats import zscore, ttest_ind, sem
from scipy.signal import hilbert
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter

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

        # string to use when saving results files
        self.res_str = 'novelty.p'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        For each session, channel
        """

        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # open a parallel pool using joblib
        with Parallel(n_jobs=int(NUM_CORES/2) if NUM_CORES != 1 else 1,  verbose=5) as parallel:

            # loop over sessions
            for session_name, session_grp in self.subject_data.items():
                print('{} processing.'.format(session_grp.name))

                # and channels
                for channel_num, channel_grp in tqdm(session_grp.items()):
                    self.res[channel_grp.name] = {}
                    self.res[channel_grp.name]['firing_rates'] = {}

                    # load behavioral events
                    events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')

                    # load eeg for this channel
                    eeg_channel = self._create_eeg_timeseries(channel_grp, events)

                    # length of buffer in samples. Used below for extracting smoothed spikes
                    samples = int(np.ceil(float(eeg_channel['samplerate']) * self.buffer))

                    # next we want to compute the power at all the frequencies in self.power_freqs and at all the
                    # timepoints in eeg. Can easily take up a lot of memory. So we will process one frequency at a time.
                    # This function computes power and compares the novel and repeated conditions
                    f = compute_lfp_novelty_effect
                    memory_effect_channel = parallel((delayed(f)(eeg_channel, freq, self.buffer)
                                                      for freq in self.power_freqs))
                    phase_data = xarray.concat([x[4] for x in memory_effect_channel],
                                               dim='frequency').transpose('event', 'time', 'frequency')

                    self.res[channel_grp.name]['delta_z'] = pd.concat([x[0] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_t'] = pd.concat([x[1] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_z_lag'] = pd.concat([x[2] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_t_lag'] = pd.concat([x[3] for x in memory_effect_channel])

                    # also compute power and phase in specific hilbert bands
                    if self.hilbert_bands is not None:
                        memory_effect_channel_hilbert = [f(eeg_channel, freq_band, self.buffer)
                                                         for freq_band in self.hilbert_bands]
                        self.res[channel_grp.name]['delta_z_hilbert'] = pd.concat([x[0] for x in memory_effect_channel_hilbert])
                        self.res[channel_grp.name]['delta_t_hilbert'] = pd.concat([x[1] for x in memory_effect_channel_hilbert])
                        self.res[channel_grp.name]['delta_z_lag_hilbert'] = pd.concat([x[2] for x in memory_effect_channel_hilbert])
                        self.res[channel_grp.name]['delta_t_lag_hilbert'] = pd.concat([x[3] for x in memory_effect_channel_hilbert])

                        phase_data_hilbert = xarray.concat([x[4] for x in memory_effect_channel_hilbert],
                                                           dim='frequency').transpose('event', 'time', 'frequency')

                    # also store region and hemisphere for easy reference. and time
                    self.res[channel_grp.name]['region'] = eeg_channel.event.data['region'][0]
                    self.res[channel_grp.name]['hemi'] = eeg_channel.event.data['hemi'][0]

                    # for each cluster in the channel, compute smoothed firing rate
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        self.res[channel_grp.name]['firing_rates'][clust_str] = {}

                        # compute number of spikes at each timepoint and the time in samples when each occurred
                        spike_counts, spike_rel_times = self._create_spiking_counts(cluster_grp, events,
                                                                                    eeg_channel.shape[1])

                        # compute the phase of each spike at each frequency using the already computed phase data
                        # for this channel. Perform rayleigh test and other stats at each frequency
                        novel_phases, rep_phases = _compute_spike_phase_by_freq(spike_rel_times,
                                                                                self.phase_bin_start,
                                                                                self.phase_bin_stop,
                                                                                phase_data,
                                                                                events)
                        p_novel, z_novel, p_rep, z_rep, ww_pvals, ww_fstat, med_pvals, med_stat, p_kuiper, \
                            stat_kuiper = _copmute_novel_rep_spike_stats(novel_phases, rep_phases)
                        self.res[channel_grp.name]['firing_rates'][clust_str]['p_novel'] = p_novel
                        self.res[channel_grp.name]['firing_rates'][clust_str]['z_novel'] = z_novel
                        self.res[channel_grp.name]['firing_rates'][clust_str]['p_rep'] = p_rep
                        self.res[channel_grp.name]['firing_rates'][clust_str]['z_rep'] = z_rep
                        self.res[channel_grp.name]['firing_rates'][clust_str]['ww_pvals'] = ww_pvals
                        self.res[channel_grp.name]['firing_rates'][clust_str]['ww_fstat'] = ww_fstat
                        self.res[channel_grp.name]['firing_rates'][clust_str]['med_pvals'] = med_pvals
                        self.res[channel_grp.name]['firing_rates'][clust_str]['med_stat'] = med_stat
                        self.res[channel_grp.name]['firing_rates'][clust_str]['p_kuiper'] = p_kuiper
                        self.res[channel_grp.name]['firing_rates'][clust_str]['stat_kuiper'] = stat_kuiper

                        # also compute novel and repeated phases for each band in hilbert phases
                        if self.hilbert_bands is not None:
                            novel_phases_hilbert, rep_phases_hilbert = _compute_spike_phase_by_freq(spike_rel_times,
                                                                                                    self.phase_bin_start,
                                                                                                    self.phase_bin_stop,
                                                                                                    phase_data_hilbert,
                                                                                                    events)
                            self.res[channel_grp.name]['firing_rates'][clust_str]['novel_phases_hilbert'] = novel_phases_hilbert
                            self.res[channel_grp.name]['firing_rates'][clust_str]['rep_phases_hilbert'] = rep_phases_hilbert

                        # smooth the spike train. Also remove the buffer
                        kern_width_samples = int(eeg_channel.samplerate.data / (1000/self.kern_width))
                        kern = signal.gaussian(kern_width_samples, self.kern_sd)
                        kern /= kern.sum()
                        smoothed_spike_counts = np.stack([signal.convolve(x, kern, mode='same')[samples:-samples]
                                                          for x in spike_counts], 0)

                        # compute stats on novel and repeated items for the smoothed spike counts
                        smoothed_spike_counts = self._create_spike_timeseries(smoothed_spike_counts,
                                                                              eeg_channel.time.data[samples:-samples],
                                                                              channel_grp.attrs['samplerate'],
                                                                              events)

                        spike_res = compute_novelty_stats(smoothed_spike_counts)
                        self.res[channel_grp.name]['firing_rates'][clust_str]['delta_spike_z'] = spike_res[0]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['delta_spike_t'] = spike_res[1]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['delta_spike_z_lag'] = spike_res[2]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['delta_spike_t_lag'] = spike_res[3]

                        # finally, compute stats based on normalizing from the pre-stimulus interval
                        spike_res_zs = compute_novelty_stats_without_contrast(smoothed_spike_counts)
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_novel_mean'] = spike_res_zs[0]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_repeated_mean'] = spike_res_zs[1]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_novel_sem'] = spike_res_zs[2]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_repeated_sem'] = spike_res_zs[3]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_ts'] = spike_res_zs[4]
                        self.res[channel_grp.name]['firing_rates'][clust_str]['zdata_ps'] = spike_res_zs[5]

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


def compute_hilbert_at_single_band(eeg, freq_band, buffer_len):

    # band pass eeg
    # makes sure to pass in a list not an array because wtf PTSA
    band_eeg = bri.band_pass_eeg(eeg, freq_band.tolist() if isinstance(freq_band, np.ndarray) else freq_band)

    # run hilbert to get the complexed valued result
    complex_hilbert_res = hilbert(band_eeg.data, N=band_eeg.shape[-1], axis=-1)

    # get phase at each timepoint
    phase_data = band_eeg.copy()
    phase_data.data = np.unwrap(np.angle(complex_hilbert_res))
    phase_data = phase_data.squeeze()
    phase_data = phase_data.remove_buffer(buffer_len)
    phase_data.coords['frequency'] = np.mean(freq_band)

    # and power
    power_data = band_eeg.copy()
    power_data.data = np.abs(complex_hilbert_res) ** 2
    power_data = power_data.squeeze()
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
    novel_phases = []
    rep_phases = []
    for (index, e), spikes, phase_data_event in zip(events.iterrows(), spike_rel_times, phase_data):
        if index in good_events:
            is_novel_event = e.isFirst
            if len(spikes) > 0:
                valid_spikes = spikes[np.in1d(spikes, valid_samps)]
                if len(valid_spikes) > 0:
                    if is_novel_event:
                        novel_phases.append(phase_data_event[valid_spikes].data)
                    else:
                        rep_phases.append(phase_data_event[valid_spikes].data)

    # will be number of spikes x frequencies
    novel_phases = np.vstack(novel_phases)
    rep_phases = np.vstack(rep_phases)

    return novel_phases, rep_phases


def _copmute_novel_rep_spike_stats(novel_phases, rep_phases):

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












































