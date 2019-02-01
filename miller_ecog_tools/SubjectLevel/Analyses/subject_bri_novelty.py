"""

"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import signal
from scipy.stats import zscore, ttest_ind, ttest_rel
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData

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

        # frequencies at which to compute power
        self.power_freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # how much time (in s) to remove from each end of the data after wavelet convolution
        self.buffer = 1.5

        # settings for guassian kernel used to smooth spike trains
        # enter .kern_width in milliseconds
        self.kern_width = 150
        self.kern_sd = 10

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
                    f = compute_power_novelty_effect
                    memory_effect_channel = parallel((delayed(f)(eeg_channel, freq, self.buffer)
                                                      for freq in self.power_freqs))

                    self.res[channel_grp.name]['delta_z'] = pd.concat([x[0] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_t'] = pd.concat([x[1] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_z_lag'] = pd.concat([x[2] for x in memory_effect_channel])
                    self.res[channel_grp.name]['delta_t_lag'] = pd.concat([x[3] for x in memory_effect_channel])

                    # also store region and hemisphere for easy reference. and time
                    self.res[channel_grp.name]['region'] = eeg_channel.event.data['region'][0]
                    self.res[channel_grp.name]['hemi'] = eeg_channel.event.data['hemi'][0]

                    # for each cluster in the channel, compute smoothed firing rate
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        self.res[channel_grp.name]['firing_rates'][clust_str] = {}

                        # compute number of spikes at each timepoint
                        spike_counts = self._create_spiking_counts(cluster_grp, events, eeg_channel.shape[1])

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

    def _create_spiking_counts(self, cluster_grp, events, n):
        spike_counts = []

        # loop over each event
        for index, e in events.iterrows():
            # load the spike times for this cluster
            spike_times = np.array(cluster_grp[str(index)])

            # interpolate the timestamps for this event
            start = e.stTime + self.start_ms * 1000
            stop = e.endTime + self.stop_ms * 1000
            timestamps = np.linspace(start, stop, n, endpoint=True)

            # find the closest timestamp to each spike (technically, the closest timestamp following a spike, but I
            # think this level of accuracy is fine). This is the searchsorted command. Then count the number of spikes
            # that occurred at each timepoint with histogram
            bin_counts, _ = np.histogram(np.searchsorted(timestamps, spike_times), np.arange(len(timestamps) + 1))
            spike_counts.append(bin_counts)
        return np.stack(spike_counts, 0)

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


def compute_power_at_single_freq(eeg, freq, buffer_len):

    # compute phase
    power_data = MorletWaveletFilter(eeg,
                                     np.array([freq]),
                                     output='power',
                                     width=5,
                                     cpus=12,
                                     verbose=False).filter()

    # remove the buffer from each end
    power_data = power_data.remove_buffer(buffer_len)
    return power_data.squeeze()


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


def compute_power_novelty_effect(eeg, freq, buffer_len):

    # compute the power first
    power_data = compute_power_at_single_freq(eeg, freq, buffer_len)

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
    return df_zpower_diff, df_tstat_diff, df_lag_zpower_diff, df_lag_tstat_diff












































