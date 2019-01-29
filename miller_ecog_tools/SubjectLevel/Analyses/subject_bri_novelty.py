"""

"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycircstat
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData


class SubjectNoveltyAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """

    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectNoveltyAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # frequencies at which to compute power
        self.power_freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # how much time (in s) to remove from each end of the data after wavelet convolution
        self.buffer = 1.5

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

        # loop over sessions
        for session_name, session_grp in self.subject_data.items():
            print('{} processing.'.format(session_grp.name))

            # and channels
            for channel_num, channel_grp in tqdm(session_grp.items()):

                # and clusters within channel
                for cluster_num, cluster_grp in channel_grp.items():

                    # load the data and metadata for this cluster and channel and session, and create an TimeSeries
                    # (in order to make use of their wavelet calculation)
                    eeg = self._create_eeg_timeseries(cluster_grp)

                    # remove the moment of spiking from the eeg and interpolate the missing data
                    if self.half_spike_length_ms is not None:
                        print('{}: interpolating spiking interval.'.format(cluster_grp.name))
                        eeg = self._interp_spike_interval(eeg, self.half_spike_length_ms)
                        print('Done.')

                    # next we want to compute the phase at all the frequencies in self.phase_freqs and at all the
                    # timepoints in eeg. Can easily take up a lot of memory. So we will process one frequency at a time
                    f = compute_phase_return_rayleigh_wrapper
                    clust_res = Parallel(n_jobs=12, verbose=5)(delayed(f)(eeg, freq, self.phase_buffer)
                                                               for freq in self.phase_freqs)

                    # store results in res seperately
                    self.res[cluster_grp.name] = {}
                    self.res[cluster_grp.name]['ps'] = np.stack([x[0] for x in clust_res])
                    self.res[cluster_grp.name]['zs'] = np.stack([x[1] for x in clust_res])
                    self.res[cluster_grp.name]['time'] = clust_res[0][2]

                    # also store the p value of the freq with the strongest rayleigh stat at zero. Store what the freq
                    # is as well
                    zero_point = np.argmin(np.abs(self.res[cluster_grp.name]['time']))
                    max_freq_ind = np.argmax(self.res[cluster_grp.name]['zs'][:, zero_point])
                    p_at_zero = self.res[cluster_grp.name]['ps'][max_freq_ind, zero_point]
                    self.res[cluster_grp.name]['zero_point'] = zero_point
                    self.res[cluster_grp.name]['max_freq_ind'] = max_freq_ind
                    self.res[cluster_grp.name]['max_freq_at_zero'] = self.phase_freqs[max_freq_ind]
                    self.res[cluster_grp.name]['p_at_zero'] = p_at_zero

                    # number of spikes in case we want threshold things
                    self.res[cluster_grp.name]['n_spikes'] = eeg.shape[0]

                    # store region in res for easy access
                    self.res[cluster_grp.name]['region'] = eeg.event.data['region'][0]
                    self.res[cluster_grp.name]['hemi'] = eeg.event.data['hemi'][0]

                    # finally, compute phase at the frequency with the strongest clustering
                    # nvm let's just compute on the fly if desired. makes files kind of big
                    # phase_at_max_freq = MorletWaveletFilter(eeg,
                    #                                         np.array([self.res[cluster_grp.name]['max_freq_at_zero']]),
                    #                                         output='phase',
                    #                                         width=5,
                    #                                         cpus=12,
                    #                                         verbose=False).filter()
                    # self.res[cluster_grp.name]['phase_at_max_freq'] = np.squeeze(phase_at_max_freq.data.astype('float32'))

    def compute_phase_hist_for_freq(self, cluster_grp_path, freq):

        do_unload = False
        if self.subject_data is None:
            do_unload = True
            self.load_data()

        eeg = self._create_eeg_timeseries(self.subject_data[cluster_grp_path])

        # remove the moment of spiking from the eeg and interpolate the missing data
        if self.half_spike_length_ms is not None:
            print('{}: interpolating spiking interval.'.format(cluster_grp_path))
            eeg = self._interp_spike_interval(eeg, self.half_spike_length_ms)
            print('Done.')

        phase_data = compute_phase_at_single_freq(eeg, freq, self.phase_buffer)

        if do_unload:
            self.unload_data()

    def _create_eeg_timeseries(self, grp):
        data = np.array(grp['ev_eeg'])
        events = pd.read_hdf(self.subject_data.filename, grp.name + '/event')
        time = grp.attrs['time']
        channel = grp.attrs['channel']
        sr = grp.attrs['samplerate']

        # create an TimeSeries object (in order to make use of their wavelet calculation)
        dims = ('event', 'time', 'channel')
        coords = {'event': events[events.columns[events.columns != 'index']].to_records(),
                  'time': time,
                  'channel': [channel]}

        return TimeSeries.create(data, samplerate=sr, dims=dims, coords=coords)

    @staticmethod
    def _interp_spike_interval(eeg, half_spike_length_ms):

        half_spike_length_s = half_spike_length_ms / 1000
        spike_ind = (eeg.time > -half_spike_length_s) & (eeg.time < half_spike_length_s)

        spike_time = eeg.time[spike_ind].data
        perc = (spike_time - spike_time.min()) / np.ptp(spike_time)

        for i, this_spike in enumerate(eeg):
            spike_int = np.squeeze(this_spike[spike_ind].data)
            volt_range = spike_int[-1] - spike_int[0]
            new_vals = perc * volt_range + spike_int[0]
            eeg[i, spike_ind] = np.expand_dims(new_vals, 1)

        return eeg

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))


def compute_phase_at_single_freq(eeg, freq, buffer_len):

    # compute phase
    phase_data = MorletWaveletFilter(eeg,
                                     np.array([freq]),
                                     output='phase',
                                     width=5,
                                     cpus=12,
                                     verbose=False).filter()

    # remove the buffer from each end
    phase_data = phase_data.remove_buffer(buffer_len)
    return phase_data


def compute_rayleigh_stat(phase_data):

    # run rayleight test for this frequency
    ps, zs = pycircstat.rayleigh(phase_data.data, axis=1)
    return ps, zs, phase_data.time.data


def compute_phase_return_rayleigh_wrapper(eeg, freq, buffer_len):
    phase_data = compute_phase_at_single_freq(eeg, freq, buffer_len)
    ps, zs, time = compute_rayleigh_stat(phase_data)
    return ps, zs, time
