"""
Phase locking of spikes to LFP as a function of time and frequency
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

        # string to use when saving results files
        self.res_str = 'spike_lfp_phase_locking.p'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        Just averages the data really.
        """
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # will hold info for each unit
        sta_list = []
        p_spect_list = []
        region_list = []
        hemi_list = []
        session_list = []
        channel_list = []
        unit_list = []

        # loop over sessions
        for session_name, session_grp in self.subject_data.items():

            # and channels
            channels = session_grp.keys()
            for channel_num, channel_grp in session_grp.items():

                # load the data and metadata for this channel and session, and create an TimeSeries
                # (in order to make use of their wavelet calculation)
                eeg = self._create_eeg_timeseries(channel_grp)

                # compute phase
                phase_data = MorletWaveletFilter(eeg,
                                                 self.phase_freqs,
                                                 output='phase',
                                                 width=5,
                                                 cpus=12,
                                                 verbose=False).filter()

                # remove the buffer from each end
                phase_data = phase_data.remove_buffer(self.phase_buffer)

                # and units within a channel
                chan_units = np.unique(data.event.data['cluster_num'])
                for this_unit in chan_units:
                    unit_data = data[data.event.data['cluster_num'] == this_unit]

                    # compute spike-triggered average and store, along with attributes
                    sta_list.append(unit_data.mean(axis=0))
                    session_list.append(session_name)
                    channel_list.append(channel)
                    unit_list.append(this_unit)
                    region_list.append(np.unique(unit_data.event.data['region'])[0])
                    hemi_list.append(np.unique(unit_data.event.data['hemi'])[0])

                    # also store power spectra
                    p_spect_list.append(session_dict[channel]['power_spectra'][this_unit])

        # create multi indexed dataframe
        names = ['session', 'channel', 'unit', 'region', 'hemi']
        index = pd.MultiIndex.from_arrays([session_list, channel_list, unit_list, region_list, hemi_list], names=names)
        sta_df = pd.DataFrame(data=np.squeeze(np.stack(sta_list, 0)), index=index, columns=unit_data.time)

        # and power spectra
        p_spect_df = pd.DataFrame(data=np.squeeze(np.stack(p_spect_list, 0)), index=index, columns=self.freqs)

        # store results.
        self.res['sta'] = sta_df.sort_index()
        self.res['p_spect'] = p_spect_df.sort_index()

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

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
