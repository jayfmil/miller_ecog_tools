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


class SubjectBRINoveltySpikePhaseWithShuffleAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """

    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRINoveltySpikePhaseWithShuffleAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # this needs to be an event-locked analyses
        self.do_event_locked = True

        # also compute power/phase at frequencies in specific bands using hilbert transform, if desired
        self.hilbert_bands = np.array([[1, 4], [4, 9]])

        # how much time (in s) to remove from each end of the data after power calculation
        self.buffer = 1.5

        # window to use when computing spike phase
        self.phase_bin_start = 0.0
        self.phase_bin_stop = 1.0

        # set to True to only include hits and correct rejections
        # self.only_correct_items = False
        self.max_lag = 8

        # number of shuffles to do when created the permuted data
        self.num_perms = 500

        # string to use when saving results files
        self.res_str = 'novelty_phase_stats_with_shuff.hdf5'

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

        # open a parallel pool using joblib
        with Parallel(n_jobs=int(NUM_CORES/2) if NUM_CORES != 1 else 1,  verbose=5) as parallel:

            # loop over sessions
            for session_name, session_grp in self.subject_data.items():
                print('{} processing.'.format(session_grp.name))

                # and channels
                for channel_num, channel_grp in tqdm(session_grp.items()):
                    res_channel_grp = res_file.create_group(channel_grp.name)

                    # load behavioral events
                    events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')
                    events['item_name'] = events.name.apply(lambda x: x.split('_')[1])

                    # filter to just events of a certain lag, if desired
                    if self.max_lag is not None:
                        events_to_keep = events.lag.values <= self.max_lag
                    else:
                        events_to_keep = np.array([True] * events.shape[0])

                    # load eeg for this channel
                    eeg_channel = self._create_eeg_timeseries(channel_grp, events)[events_to_keep]

                    # and for hilbert bands
                    phase_data_hilbert = compute_phase(eeg_channel, self.hilbert_bands, self.buffer, parallel)

                    # also store region and hemisphere for easy reference
                    res_channel_grp.attrs['region'] = eeg_channel.event.data['region'][0]
                    res_channel_grp.attrs['hemi'] = eeg_channel.event.data['hemi'][0]

                    # and clusters
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        res_cluster_grp = res_channel_grp.create_group(clust_str)

                        # find number of spikes at each timepoint and the time in samples when each occurred
                        spike_counts, spike_rel_times = self._create_spiking_counts(cluster_grp, events[events_to_keep],
                                                                                    eeg_channel.shape[1])

                        # following function 1: computes the phase at which each spike occurred, based on the the
                        # already computed phase data, 2: runs stats
                        phase_stats, phase_stats_percentiles = run_phase_stats_with_shuffle(events[events_to_keep],
                                                                                            spike_rel_times,
                                                                                            phase_data_hilbert,
                                                                                            self.phase_bin_start,
                                                                                            self.phase_bin_stop,
                                                                                            parallel, self.num_perms)

                        res_cluster_grp.create_dataset('novel_rayleigh_stat', data=phase_stats[0][0])
                        res_cluster_grp.create_dataset('rep_rayleigh_stat', data=phase_stats[0][1])
                        res_cluster_grp.create_dataset('watson_williams_stat', data=phase_stats[0][2])
                        res_cluster_grp.create_dataset('kuiper_stat', data=phase_stats[0][3])

                        res_cluster_grp.create_dataset('novel_rayleigh_perc', data=phase_stats[1][0])
                        res_cluster_grp.create_dataset('rep_rayleigh_perc', data=phase_stats[1][1])
                        res_cluster_grp.create_dataset('watson_williams_perc', data=phase_stats[1][2])
                        res_cluster_grp.create_dataset('kuiper_perc', data=phase_stats[1][3])

        res_file.close()
        self.res = h5py.File(self.res_save_file, 'r')

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

        return np.stack(spike_counts, 0), np.array(spike_ts)

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

    return phase_data


def compute_wavelet_at_single_freq(eeg, freq, buffer_len):

    # compute phase
    data = MorletWaveletFilter(eeg,
                               np.array([freq]),
                               output=['phase'],
                               width=5,
                               cpus=1,
                               verbose=False).filter()

    # remove the buffer from each end
    data = data.remove_buffer(buffer_len)
    return data.squeeze()


def compute_phase(eeg, freqs, buffer_len, parallel=None):

    f = compute_wavelet_at_single_freq if isinstance(freqs[0], float) else compute_hilbert_at_single_band
    if parallel is None:
        phase_data = []
        for freq in freqs:
            phase_data.append(f(eeg, freq, buffer_len))
    else:
        phase_data = parallel((delayed(f)(eeg, freq, buffer_len) for freq in freqs))
    phase_data = xarray.concat(phase_data, dim='frequency').transpose('event', 'time', 'frequency')
    return phase_data


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

    return np.array(phases)


def _bin_phases_into_cond(spike_phases, events):

    # get the novel and repeated spike phases for this event condition. Some events have no
    # spikes, so filter those out
    novel_phases = spike_phases[events.isFirst]
    novel_phases = novel_phases[np.array([len(x) > 0 for x in novel_phases])]
    if novel_phases.shape[0] == 0:
        novel_phases = []
    else:
        novel_phases = np.vstack(novel_phases)

    rep_phases = spike_phases[~events.isFirst]
    rep_phases = rep_phases[np.array([len(x) > 0 for x in rep_phases])]
    if rep_phases.shape[0] == 0:
        rep_phases = []
    else:
        rep_phases = np.vstack(rep_phases)

    return novel_phases, rep_phases


def compute_phase_stats_with_shuffle(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                     phase_bin_stop, do_permute=False):

    spike_rel_times_tmp = spike_rel_times.copy()
    if do_permute:

        # permute the novel
        novel_events = np.where(events.isFirst.values)[0]
        perm_novel_events = np.random.permutation(novel_events)
        spike_rel_times_tmp[novel_events] = spike_rel_times_tmp[perm_novel_events]

        # and repeated separately
        rep_events = np.where(~events.isFirst.values)[0]
        perm_rep_events = np.random.permutation(rep_events)
        spike_rel_times_tmp[rep_events] = spike_rel_times_tmp[perm_rep_events]

    # get the phases at which the spikes occurred and bin into novel and repeated items for each hilbert band
    spike_phases_hilbert = _compute_spike_phase_by_freq(spike_rel_times_tmp,
                                                        phase_bin_start,
                                                        phase_bin_stop,
                                                        phase_data_hilbert,
                                                        events)

    # bin into repeated and novel phases
    novel_phases, rep_phases = _bin_phases_into_cond(spike_phases_hilbert, events)

    if (len(novel_phases) > 0) & (len(rep_phases) > 0):

        # rayleigh test for uniformity
        _, z_novel = pycircstat.rayleigh(novel_phases, axis=0)
        _, z_rep = pycircstat.rayleigh(rep_phases, axis=0)

        # watson williams test for equal means
        _, ww_tables = pycircstat.watson_williams(novel_phases, rep_phases, axis=0)
        ww_fstat = np.array([x.loc['Columns'].F for x in ww_tables])

        # kuiper test, to test for difference in dispersion (not mean, because I'm making them equal)
        _, stat_kuiper = pycircstat.kuiper(novel_phases - pycircstat.mean(novel_phases),
                                           rep_phases - pycircstat.mean(rep_phases), axis=0)

        return z_novel, z_rep, ww_fstat, stat_kuiper

    else:
        return np.nan, np.nan, np.nan, np.nan


def run_phase_stats_with_shuffle(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                 phase_bin_stop, parallel=None, num_perms=100):

    # first, get the stats on the non-permuted data
    stats_real = compute_phase_stats_with_shuffle(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                                  phase_bin_stop, do_permute=False)

    # then run the permutations
    f = compute_phase_stats_with_shuffle
    if ~np.isnan(np.any(stats_real)):

        if isinstance(parallel, Parallel):
            shuff_res = parallel((delayed(f)(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                             phase_bin_stop, True) for _ in range(num_perms)))
        else:
            shuff_res = []
            for _ in range(num_perms):
                shuff_res.append(f(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                   phase_bin_stop, do_permute=True))

        # compare the true stats to the distributions of permuted stats
        stats_percentiles = np.mean(np.array(stats_real) > np.array(shuff_res), axis=0)

        return np.array(stats_real), stats_percentiles

    else:
        return np.array([np.nan, np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan, np.nan])








































