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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData
from miller_ecog_tools.Utils import neurtex_bri_helpers as bri

# figure out the number of cores available for a parallel pool. Will use half
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()


class SubjectBRIRetrievalAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """

    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRIRetrievalAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # this needs to be an event-locked analyses
        self.do_event_locked = True

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
        self.res_str = 'retrieval.p'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        1. Computes ERP for the LFP channel as a function of lag
        2. Computes mean phase relative to an oscillation as a function of lag
        3. Computes firing rate as a function as the lag between the first and second presentions
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
                    self.res[channel_grp.name]['spiking'] = {}

                    # load behavioral events
                    events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')

                    # load eeg for this channel
                    eeg_channel = self._create_eeg_timeseries(channel_grp, events)

                    # 1. compute ERP as a funcion of lag
                    erp_by_lag_channel = self.compute_mean_by_lag(eeg_channel, [-.5, 0])
                    self.res[channel_grp.name]['erp_by_lag'] = erp_by_lag_channel

                    # length of buffer in samples. Used below for extracting smoothed spikes
                    samples = int(np.ceil(float(eeg_channel['samplerate']) * self.buffer))

                    # next, compute phase as a function of time for each event and each hilbert band
                    power_phase_data = [compute_hilbert_at_single_band(eeg_channel, freq, self.buffer)
                                        for freq in self.hilbert_bands]
                    phase_data = xarray.concat([x[1] for x in power_phase_data],
                                               dim='frequency').transpose('event', 'time', 'frequency')

                    # also store region and hemisphere for easy reference
                    self.res[channel_grp.name]['region'] = eeg_channel.event.data['region'][0]
                    self.res[channel_grp.name]['hemi'] = eeg_channel.event.data['hemi'][0]

                    # for each cluster in the channel, compute smoothed firing rate and spike phaes
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        self.res[channel_grp.name]['spiking'][clust_str] = {}

                        # compute number of spikes at each timepoint and the time in samples when each occurred
                        spike_counts, spike_rel_times = self._create_spiking_counts(cluster_grp, events,
                                                                                    eeg_channel.shape[1])

                        # 2. compute the phase of each spike at each frequency using the already computed phase data
                        # for this channel.
                        spike_phase_df = self._compute_spike_phase_by_freq(spike_rel_times,
                                                                           self.phase_bin_start,
                                                                           self.phase_bin_stop,
                                                                           phase_data,
                                                                           events)
                        self.res[channel_grp.name]['spiking'][clust_str]['spike_phase'] = spike_phase_df

                        # smooth the spike train
                        kern_width_samples = int(eeg_channel.samplerate.data / (1000/self.kern_width))
                        kern = signal.gaussian(kern_width_samples, self.kern_sd)
                        kern /= kern.sum()
                        smoothed_spike_counts = np.stack([signal.convolve(x, kern, mode='same')
                                                          for x in spike_counts], 0)

                        # make into timeseries and pass to .compute_mean_by_lag (same computation as ERP)
                        smoothed_spike_counts = self._create_spike_timeseries(smoothed_spike_counts,
                                                                              eeg_channel.time.data,
                                                                              channel_grp.attrs['samplerate'],
                                                                              events)
                        firing_rate_by_lag_cluster = self.compute_mean_by_lag(smoothed_spike_counts)
                        self.res[channel_grp.name]['spiking'][clust_str]['firing_rate'] = firing_rate_by_lag_cluster

    def compute_mean_by_lag(self, data_ts, baseline_time=None):

        # get the lags and also whether it was the first presentation
        lags = data_ts.event.data['lag']
        is_novel = data_ts.event.data['isFirst']
        pressed_old_key = data_ts.event.data['oldKey']

        # for the purpose of this presentation, label all novel items as lag of 0
        lags[is_novel] = 0

        # for only looking at items with correct responses
        correct = (pressed_old_key & ~is_novel) | (~pressed_old_key & is_novel)

        # if baseline_time_interval is given, subtract mean of that interval for each channel
        if baseline_time is not None:
            baseline_inds = (data_ts.time.data > baseline_time[0]) & (data_ts.time.data < baseline_time[1])
            baselines = np.mean(data_ts[:, baseline_inds], axis=1)
            data_ts = data_ts - baselines

        # now we can just loop over all unique values of lag
        lag_dfs = []
        lags_to_bin = np.unique(lags).tolist()
        lags_to_bin.append([4, 8])
        lags_to_bin.append([16, 32])
        for this_lag in lags_to_bin:

            # filter to just this lag (or range of lags)
            lag_inds = np.in1d(lags, [this_lag] if not isinstance(this_lag, list) else this_lag)
            lag_str = '{}-{}'.format(*this_lag) if isinstance(this_lag, list) else str(this_lag)
            eeg_lag = data_ts[lag_inds].squeeze()

            # remove the begining and end buffer
            eeg_lag = eeg_lag.remove_buffer(self.buffer)

            # store the lag as a coord in the xarray so that it will be there when we convert to DF
            eeg_lag.coords['lag'] = lag_str

            # take the mean and sem and convert to df
            eeg_lag_mean = eeg_lag.mean(dim='event').to_dataframe(name='y')
            eeg_lag_mean = eeg_lag_mean[['lag', 'y']]
            eeg_lag_sem = eeg_lag.std(dim='event') / np.sqrt(eeg_lag.shape[0] - 1)
            eeg_lag_mean['sem'] = eeg_lag_sem.data

            # also compute the mean and sem for just the correct items
            correct_this_lag = correct[lag_inds]
            mean_correct_lag = eeg_lag[correct_this_lag].mean(dim='event')
            sem_correct_lag = eeg_lag[correct_this_lag].std(dim='event') / np.sqrt(np.sum(correct_this_lag) - 1)
            eeg_lag_mean['y_correct'] = mean_correct_lag.data
            eeg_lag_mean['sem_correct'] = sem_correct_lag.data

            lag_dfs.append(eeg_lag_mean)

        return pd.concat(lag_dfs)

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

    def _compute_spike_phase_by_freq(self, spike_rel_times, phase_bin_start, phase_bin_stop, phase_data, events):

        # only will count samples the occurred within window defined by phase_bin_start and _stop
        valid_samps = np.where((phase_data.time > phase_bin_start) & (phase_data.time < phase_bin_stop))[0]

        # will grow as we iterate over spikes in each condition.
        phase_lag_correct = []
        for (index, e), spikes, phase_data_event in zip(events.iterrows(), spike_rel_times, phase_data):

            is_novel_event = e.isFirst
            event_lag = e.lag
            is_correct_event = (e.oldKey & ~is_novel_event) | (~e.oldKey & is_novel_event)

            if len(spikes) > 0:
                valid_spikes = spikes[np.in1d(spikes, valid_samps)]
                if len(valid_spikes) > 0:
                    event_df = pd.DataFrame([*phase_data_event[valid_spikes].data.T,
                                             [event_lag] * len(valid_spikes),
                                             [is_correct_event] * len(valid_spikes),
                                             [is_novel_event] * len(valid_spikes),
                                             [index] * len(valid_spikes)]).T
                    event_df.columns = [*np.array(['{}-{}'.format(*x) for x in self.hilbert_bands]),
                                        'lag', 'correct', 'is_novel', 'event']
                    phase_lag_correct.append(event_df)
        return pd.concat(phase_lag_correct).reset_index(drop=True)

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






































