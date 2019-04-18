"""

"""
import os
import pycircstat
import numpy as np
import pandas as pd
import xarray
import h5py

from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter
from scipy.signal import hilbert
from scipy.stats import sem, zscore
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter

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

        # buffer (s) on each side of spike to when computing spike triggered average
        self.sta_buffer = 0.75

        # set to True to only include hits and correct rejections
        self.only_correct_items = False
        self.max_lag = 8
        self.min_lag = 0

        # number of shuffles to do when created the permuted data
        self.num_perms = 500

        # whether to skip different parts of the analysis
        self.skip_phase_stats = False
        self.skip_sta_stats = False
        self.skip_power_fr_stats = False

        # string to use when saving results files
        self.res_str = 'novelty_phase_stats_with_shuff.hdf5'

        # run with all lags? Exc lag 1 as well?
        # what about only correct items
        # response locked?

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
        res_file = h5py.File(self.res_save_file, 'a')

        # open a parallel pool using joblib
        with Parallel(n_jobs=int(NUM_CORES/2) if NUM_CORES != 1 else 1,  verbose=5) as parallel:

            # loop over sessions
            for session_name, session_grp in self.subject_data.items():
                print('{} processing.'.format(session_grp.name))

                # and channels
                for channel_num, channel_grp in tqdm(session_grp.items()):
                    if channel_grp.name not in res_file:
                        res_channel_grp = res_file.create_group(channel_grp.name)
                    else:
                        res_channel_grp = res_file[channel_grp.name]

                    # load behavioral events
                    events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')
                    events['item_name'] = events.name.apply(lambda x: x.split('_')[1])

                    # filter to just events of a certain lag, if desired
                    events_to_keep = np.array([True] * events.shape[0])
                    if self.max_lag is not None:
                        events_to_keep = events_to_keep & (((events.lag.values <= self.max_lag) & ~events.isFirst.values) | events.isFirst.values)
                    if self.min_lag is not None:
                        events_to_keep = events_to_keep & (((events.lag.values >= self.min_lag) & ~events.isFirst.values) | events.isFirst.values)
                    if self.only_correct_items:
                        events_to_keep = events_to_keep & self._filter_to_correct_items_paired(events)

                    # load eeg for this channel
                    eeg_channel = self._create_eeg_timeseries(channel_grp, events)[events_to_keep]

                    # and for hilbert bands
                    phase_data_hilbert, power_data_hilbert, band_pass_eeg = compute_phase(eeg_channel,
                                                                                          self.hilbert_bands,
                                                                                          self.buffer,
                                                                                          parallel)

                    # also store region and hemisphere for easy reference
                    res_channel_grp.attrs['region'] = eeg_channel.event.data['region'][0]
                    res_channel_grp.attrs['hemi'] = eeg_channel.event.data['hemi'][0]

                    # and clusters
                    for cluster_num, cluster_grp in channel_grp['spike_times'].items():
                        clust_str = cluster_grp.name.split('/')[-1]
                        if clust_str not in res_channel_grp:
                            res_cluster_grp = res_channel_grp.create_group(clust_str)
                        else:
                            res_cluster_grp = res_channel_grp[clust_str]

                        # find number of spikes at each timepoint and the time in samples when each occurred
                        spike_counts, spike_rel_times = self._create_spiking_counts(cluster_grp, events[events_to_keep],
                                                                                    eeg_channel.shape[1])

                        # compute spike triggered average of eeg
                        if not self.skip_sta_stats:
                            _sta_by_event_cond(spike_rel_times, self.phase_bin_start, self.phase_bin_stop,
                                               self.sta_buffer, eeg_channel, band_pass_eeg, events[events_to_keep],
                                               res_cluster_grp)

                        if not self.skip_power_fr_stats:
                            # buffer length in samples for removing from spiking data when computing firing rate
                            samples = int(np.ceil(float(power_data_hilbert['samplerate']) * self.buffer))

                            # compute mean power and firing rate by condition
                            _power_fr_by_event_cond(spike_counts, power_data_hilbert,
                                                    self.phase_bin_start, self.phase_bin_stop,
                                                    events[events_to_keep], res_cluster_grp)

                        if not self.skip_phase_stats:

                            # following function 1: computes the phase at which each spike occurred, based on the the
                            # already computed phase data, 2: runs stats
                            phase_stats, phase_stats_percentiles, orig_pvals, novel_phases, rep_phases = \
                                run_phase_stats_with_shuffle(events[events_to_keep],
                                                             spike_rel_times,
                                                             phase_data_hilbert,
                                                             self.phase_bin_start,
                                                             self.phase_bin_stop,
                                                             parallel, self.num_perms)

                            res_cluster_grp.create_dataset('novel_rvl_stat', data=phase_stats[0])
                            res_cluster_grp.create_dataset('rep_rvl_stat', data=phase_stats[1])
                            res_cluster_grp.create_dataset('rvl_diff_stat', data=phase_stats[2])
                            res_cluster_grp.create_dataset('watson_williams_stat', data=phase_stats[3])
                            res_cluster_grp.create_dataset('kuiper_stat', data=phase_stats[4])
                            res_cluster_grp.create_dataset('novel_rayleigh_stat', data=phase_stats[5])
                            res_cluster_grp.create_dataset('rep_rayleigh_stat', data=phase_stats[6])
                            res_cluster_grp.create_dataset('rayleigh_diff_stat', data=phase_stats[7])

                            res_cluster_grp.create_dataset('novel_rvl_perc', data=phase_stats_percentiles[0])
                            res_cluster_grp.create_dataset('rep_rvl_perc', data=phase_stats_percentiles[1])
                            res_cluster_grp.create_dataset('rvl_diff_perc', data=phase_stats_percentiles[2])
                            res_cluster_grp.create_dataset('watson_williams_perc', data=phase_stats_percentiles[3])
                            res_cluster_grp.create_dataset('kuiper_perc', data=phase_stats_percentiles[4])
                            res_cluster_grp.create_dataset('novel_rayleigh_perc', data=phase_stats_percentiles[5])
                            res_cluster_grp.create_dataset('rep_rayleigh_perc', data=phase_stats_percentiles[6])
                            res_cluster_grp.create_dataset('rayleigh_diff_perc', data=phase_stats_percentiles[7])

                            res_cluster_grp.create_dataset('novel_rayleigh_orig_pval', data=orig_pvals[0])
                            res_cluster_grp.create_dataset('rep_rayleigh_orig_pval', data=orig_pvals[1])
                            res_cluster_grp.create_dataset('watson_williams_orig_pval', data=orig_pvals[2])
                            res_cluster_grp.create_dataset('kuiper_orig_pval', data=orig_pvals[3])

                            res_cluster_grp.create_dataset('novel_phases', data=novel_phases)
                            res_cluster_grp.create_dataset('rep_phases', data=rep_phases)

        res_file.close()
        self.res = h5py.File(self.res_save_file, 'r')

    def _filter_to_correct_items_paired(self, events):

        # get boolean of correct responses
        novel_items = events['isFirst'].values
        pressed_old_key = events['oldKey'].values
        hits = pressed_old_key & ~novel_items
        correct_rejections = ~pressed_old_key & novel_items
        correct = hits | correct_rejections

        # find instances where both novel and repeated responses for an item are correct
        c = Counter(events[correct].item_name.values)
        correct_items = [k for k, v in c.items() if v==2]

        # return boolean of correct
        return events.item_name.isin(correct_items).values

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
    band_eeg.coords['frequency'] = np.mean(freq_band)

    # run hilbert to get the complexed valued result
    complex_hilbert_res = hilbert(band_eeg.data, N=band_eeg.shape[-1], axis=-1)

    # get phase at each timepoint
    phase_data = band_eeg.copy()
    phase_data.data = np.angle(complex_hilbert_res)
    # phase_data = phase_data.remove_buffer(buffer_len)
    # phase_data.coords['frequency'] = np.mean(freq_band)

    # and power
    power_data = band_eeg.copy()
    power_data.data = np.log10(np.abs(complex_hilbert_res) ** 2)
    # power_data = power_data.remove_buffer(buffer_len)
    # power_data.coords['frequency'] = np.mean(freq_band)

    return phase_data, power_data, band_eeg


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


def _power_fr_by_event_cond(spike_counts, power_data_hilbert, phase_bin_start, phase_bin_stop, events, h_file):

    time_ind = (power_data_hilbert.time.data > phase_bin_start) & (power_data_hilbert.time.data < phase_bin_stop)
    is_novel = events.isFirst.values

    # compute (zscored) power for each condition
    zpower_data_hilbert = zscore(power_data_hilbert[:, time_ind], axis=0)
    pow_novel = zpower_data_hilbert[is_novel].mean(axis=1)
    pow_rep = zpower_data_hilbert[~is_novel].mean(axis=1)

    # compute firing rate for each condition
    frs = np.sum(spike_counts[:, time_ind], axis=1) / (phase_bin_stop - phase_bin_start)
    fr_novel = frs[is_novel]
    fr_rep = frs[~is_novel]

    if h_file is not None:
        add_to_hd5f_file(h_file, 'pow_novel', pow_novel)
        add_to_hd5f_file(h_file, 'pow_rep', pow_rep)
        add_to_hd5f_file(h_file, 'fr_novel', fr_novel)
        add_to_hd5f_file(h_file, 'fr_rep', fr_rep)

    else:
        return pow_novel, pow_rep, fr_novel, fr_rep


def _sta_by_event_cond(spike_rel_times, phase_bin_start, phase_bin_stop, sta_buffer, eeg, filtered_eeg, events,
                       h_file=None):
    valid_samps = np.where((eeg.time > phase_bin_start) & (eeg.time < phase_bin_stop))[0]
    nsamples = int(np.ceil(float(eeg['samplerate']) * sta_buffer))

    # throw out novel items that were never repeated
    good_events = events[~((events['isFirst']) & (events['lag'] == 0))].index.values

    # loop over each event
    stas = []
    stas_filt = []
    is_novel = []
    for (index, e), spikes, eeg_data_event, eeg_filt_data_event in zip(events.iterrows(), spike_rel_times, eeg, filtered_eeg):
        if index in good_events:
            if len(spikes) > 0:
                valid_spikes = spikes[np.in1d(spikes, valid_samps)]
                if len(valid_spikes) > 0:
                    for this_spike in valid_spikes:
                        stas.append(eeg_data_event[this_spike - nsamples:this_spike + nsamples].data)
                        stas_filt.append(eeg_filt_data_event[this_spike - nsamples:this_spike + nsamples].data)
                        is_novel.append(e.isFirst)
    is_novel = np.array(is_novel)

    # sta by condition for raw eeg
    if len(stas) > 0:
        stas = np.stack(stas)
        novel_sta_mean = stas[is_novel].mean(axis=0)
        novel_sta_sem = sem(stas[is_novel], axis=0)
        rep_sta_mean = stas[~is_novel].mean(axis=0)
        rep_sta_sem = sem(stas[~is_novel], axis=0)
        sta_time = np.linspace(-sta_buffer, sta_buffer, novel_sta_mean.shape[0])

        # sta by condition for filtered eeg
        stas_filt = np.stack(stas_filt)
        novel_sta_filt_mean = stas_filt[is_novel].mean(axis=0)
        novel_sta_filt_sem = sem(stas_filt[is_novel], axis=0)
        rep_sta_filt_mean = stas_filt[~is_novel].mean(axis=0)
        rep_sta_filt_sem = sem(stas_filt[~is_novel], axis=0)

        if h_file is not None:
            add_to_hd5f_file(h_file, 'novel_sta_mean', novel_sta_mean)
            add_to_hd5f_file(h_file, 'novel_sta_sem', novel_sta_sem)
            add_to_hd5f_file(h_file, 'rep_sta_mean', rep_sta_mean)
            add_to_hd5f_file(h_file, 'rep_sta_sem', rep_sta_sem)

            add_to_hd5f_file(h_file, 'novel_sta_filt_mean', novel_sta_filt_mean)
            add_to_hd5f_file(h_file, 'novel_sta_filt_sem', novel_sta_filt_sem)
            add_to_hd5f_file(h_file, 'rep_sta_filt_mean', rep_sta_filt_mean)
            add_to_hd5f_file(h_file, 'rep_sta_filt_sem', rep_sta_filt_sem)

            add_to_hd5f_file(h_file, 'sta_time', sta_time)
        else:
            return novel_sta_mean, novel_sta_sem, rep_sta_mean, rep_sta_sem, novel_sta_filt_mean, novel_sta_filt_sem, \
                   rep_sta_filt_mean, rep_sta_filt_sem, sta_time


def add_to_hd5f_file(h_file, data_name, data):
    if data_name not in h_file:
        h_file.create_dataset(data_name, data=data)


def compute_phase(eeg, freqs, buffer_len, parallel=None):

    f = compute_hilbert_at_single_band
    if parallel is None:
        phase_data = []
        power_data = []
        band_eeg_data = []
        for freq in freqs:
            phases, powers, band_eeg = f(eeg, freq, buffer_len)
            phase_data.append(phases)
            power_data.append(powers)
            band_eeg_data.append(band_eeg)
    else:
        phase_eeg_data = parallel((delayed(f)(eeg, freq, buffer_len) for freq in freqs))
        phase_data = [x[0] for x in phase_eeg_data]
        power_data = [x[1] for x in phase_eeg_data]
        band_eeg_data = [x[2] for x in phase_eeg_data]

    phase_data = xarray.concat(phase_data, dim='frequency').transpose('event', 'time', 'frequency')
    power_data = xarray.concat(power_data, dim='frequency').transpose('event', 'time', 'frequency')
    band_eeg_data = xarray.concat(band_eeg_data, dim='frequency').transpose('event', 'time', 'frequency')

    return phase_data, power_data, band_eeg_data


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
        rvl_novel = pycircstat.resultant_vector_length(novel_phases, axis=0)
        rvl_rep = pycircstat.resultant_vector_length(rep_phases, axis=0)
        rvl_diff = rvl_novel - rvl_rep

        # compute rayleigh test for each condition
        rayleigh_pval_novel, rayleigh_z_novel = pycircstat.rayleigh(novel_phases, axis=0)
        rayleigh_pval_rep, rayleigh_z_rep = pycircstat.rayleigh(rep_phases, axis=0)
        rayleigh_diff = rayleigh_z_novel - rayleigh_z_rep

        # watson williams test for equal means
        ww_pval, ww_tables = pycircstat.watson_williams(novel_phases, rep_phases, axis=0)
        ww_fstat = np.array([x.loc['Columns'].F for x in ww_tables])

        # kuiper test, to test for difference in dispersion (not mean, because I'm making them equal)
        kuiper_pval, stat_kuiper = pycircstat.kuiper(novel_phases - pycircstat.mean(novel_phases),
                                                     rep_phases - pycircstat.mean(rep_phases), axis=0)

        return (rvl_novel, rvl_rep, rvl_diff, ww_fstat, stat_kuiper, rayleigh_z_novel, rayleigh_z_rep, rayleigh_diff), \
               (rayleigh_pval_novel, rayleigh_pval_rep, ww_pval, kuiper_pval), novel_phases, rep_phases

    else:
        return (np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2])), \
               (np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2]),
                np.array([np.nan] * phase_data_hilbert.shape[2])), novel_phases, rep_phases


def run_phase_stats_with_shuffle(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                 phase_bin_stop, parallel=None, num_perms=100):

    # first, get the stats on the non-permuted data
    stats_real, pvals_real, novel_phases, rep_phases = compute_phase_stats_with_shuffle(events, spike_rel_times,
                                                                                        phase_data_hilbert,
                                                                                        phase_bin_start,
                                                                                        phase_bin_stop,
                                                                                        do_permute=False)

    # then run the permutations
    f = compute_phase_stats_with_shuffle
    if ~np.any(np.isnan(stats_real)):

        if isinstance(parallel, Parallel):
            shuff_res = parallel((delayed(f)(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                             phase_bin_stop, True) for _ in range(num_perms)))
        else:
            shuff_res = []
            for _ in range(num_perms):
                shuff_res.append(f(events, spike_rel_times, phase_data_hilbert, phase_bin_start,
                                   phase_bin_stop, do_permute=True))
        shuff_res = [x[0] for x in shuff_res]

        # compare the true stats to the distributions of permuted stats
        stats_percentiles = np.mean(np.array(stats_real) > np.array(shuff_res), axis=0)

        return np.array(stats_real), stats_percentiles, np.array(pvals_real), novel_phases, rep_phases

    else:
        return np.full((8, phase_data_hilbert.shape[2]), np.nan), np.full((8, phase_data_hilbert.shape[2]), np.nan), \
               np.full((4, phase_data_hilbert.shape[2]), np.nan), novel_phases, rep_phases








































