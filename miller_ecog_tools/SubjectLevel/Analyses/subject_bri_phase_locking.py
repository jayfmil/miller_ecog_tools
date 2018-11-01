"""
Phase locking of spikes to LFP as a function of time and frequency
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


class SubjectPhaseLockingAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """

    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectPhaseLockingAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # frequencies at which to compute phase
        self.phase_freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # how much time (in s) to remove from each end of the data
        self.phase_buffer = 1.5

        # how much time before and after each spike to remove from the eeg before phase calculation
        self.half_spike_length_ms = 15

        # string to use when saving results files
        self.res_str = 'spike_lfp_phase_locking.p'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        For each session, channel, and cluster on that channel:

        1. Read in eeg for the cluster
        2. Compute phase at each timepoint and frequency defined in self.phase_freqs
        3. Perform rayleigh test at each timepoint and frequency across events
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
                    f = compute_phase_and_rayleigh_stat
                    clust_res = Parallel(n_jobs=12, verbose=5)(delayed(f)(eeg, freq, self.phase_buffer)
                                                               for freq in self.phase_freqs)

                    # store results in res seperately
                    self.res[cluster_grp.name] = {}
                    self.res[cluster_grp.name]['ps]'] = np.stack([x[0] for x in clust_res])
                    self.res[cluster_grp.name]['zs'] = np.stack([x[1] for x in clust_res])
                    self.res[cluster_grp.name]['time'] = clust_res[0][2]

                    # also store the p value of the freq with the strongest rayleigh stat at zero. Store what the freq
                    # is as well
                    zero_point = np.argmin(np.abs(self.res[cluster_grp.name]['time']))
                    max_freq_ind = np.argmax(self.res[cluster_grp.name]['zs'][:, zero_point])
                    p_at_zero = self.res[cluster_grp.name]['ps]'][max_freq_ind, zero_point]
                    self.res[cluster_grp.name]['zero_point'] = zero_point
                    self.res[cluster_grp.name]['max_freq_ind'] = max_freq_ind
                    self.res[cluster_grp.name]['max_freq_at_zero'] = self.phase_freqs[max_freq_ind]
                    self.res[cluster_grp.name]['p_at_zero'] = p_at_zero

                    # number of spikes in case we want threshold things
                    self.res[cluster_grp.name]['n_spikes'] =

                    # store region in res for easy access
                    self.res[cluster_grp.name]['region'] = eeg.event.data['region'][0]
                    self.res[cluster_grp.name]['hemi'] = eeg.event.data['hemi'][0]

                    # finally, compute phase at the frequency with the strongest clustering
                    phase_at_max_freq = MorletWaveletFilter(eeg,
                                                            np.array([self.res[cluster_grp.name]['max_freq_at_zero']]),
                                                            output='phase',
                                                            width=5,
                                                            cpus=12,
                                                            verbose=False).filter()
                    self.res[cluster_grp.name]['phase_at_max_freq'] = np.squeeze(phase_at_max_freq)

    def plot_rayleigh(self, res_key, sig_thresh=None, vmax=None):
        """
        Plots frequency by time heatmap of rayleigh z-values.
        """

        # pull out the info about this cluster
        zs = self.res[res_key]['zs']
        time = self.res[res_key]['time']
        zero_point = self.res[res_key]['zero_point']
        p = self.res[res_key]['p_at_zero']
        freq = self.res[res_key]['max_freq_at_zero']
        freq_ind = self.res[res_key]['max_freq_ind']

        # see if there is significant phase clustering
        if sig_thresh is None:
            sig_thresh = 0.05 / len(self.phase_freqs)
        is_sig = p < sig_thresh

        with plt.style.context('seaborn-white'):
            with mpl.rc_context({'ytick.labelsize': 22,
                                 'xtick.labelsize': 22}):

                # make the initial plot
                fig, ax = plt.subplots()
                im = ax.imshow(zs, aspect='auto', interpolation='quadric', cmap='jet', vmax=vmax)
                ax.invert_yaxis()

                # set the x values to be specific timepoints
                x_vals = [self.start_spike_ms + self.phase_buffer * 1000,
                          0,
                          self.stop_spike_ms - self.phase_buffer * 1000]
                new_xticks = np.round(np.interp(x_vals, time, np.arange(len(time))))
                ax.set_xticks(new_xticks)
                ax.set_xticklabels([int(x) for x in x_vals], fontsize=22)
                ax.set_xlabel('Time (ms)', fontsize=20)

                # now the y
                ax.set_yticks(np.arange(len(self.phase_freqs))[::5])
                ax.set_yticklabels(np.round(self.phase_freqs[::5], 2), fontsize=20)
                ax.set_ylabel('Frequency (Hz)', fontsize=22)

                # plot an x at the most significant point at t=0
                if is_sig:
                    plt.plot(zero_point, freq_ind, 'x', markersize=20, mew=5, c='w')
                    y = np.where(self.phase_freqs == freq)[0]
                    props = dict(boxstyle='round', facecolor='k', alpha=.3, pad=.1, ec='k')
                    ax.text(zero_point, y + 2, str(np.round(freq, 2)) + ' Hz',
                            fontdict={'fontsize': 24,
                                      'color': 'w',
                                      'verticalalignment': 'bottom',
                                      'horizontalalignment': 'center'},
                            bbox=props)

                # add colorbar and that's it
                cbar = plt.colorbar(im)
                ticklabels = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabels, fontsize=16)
                fig.set_size_inches(15, 12)

    def _create_eeg_timeseries(self, channel_grp):
        data = np.array(channel_grp['ST_eeg'])
        events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')
        time = channel_grp.attrs['time']
        channel = channel_grp.attrs['channel']
        sr = channel_grp.attrs['samplerate']

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


def compute_phase_and_rayleigh_stat(eeg, freq, buffer_len):

    # compute phase
    phase_data = MorletWaveletFilter(eeg,
                                     np.array([freq]),
                                     output='phase',
                                     width=5,
                                     cpus=12,
                                     verbose=False).filter()

    # remove the buffer from each end
    phase_data = phase_data.remove_buffer(buffer_len)

    # run rayleight test for this frequency
    ps, zs = pycircstat.rayleigh(phase_data.data, axis=1)
    return ps, zs, phase_data.time.data