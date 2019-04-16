"""
Spike-triggered averages from the brain research institute data.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData


# CURRENTLY BUSTED NEED TO UPDATE TO BE COMPAT WITH HDF5!

class SubjectSTAAnalysis(SubjectAnalysisBase, SubjectBRIData):
    """
    Computes spike triggered averages.
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectSTAAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'sta.p'

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

        for session_name, session_grp in self.subject_data.items():
            print('{} processing.'.format(session_grp.name))

            # and channels
            for channel_num, channel_grp in session_grp.items():

                # load behavioral events
                events = pd.read_hdf(self.subject_data.filename, channel_grp.name + '/event')
                events['item_name'] = events.name.apply(lambda x: x.split('_')[1])

                # filter to just events of a certain lag, if desired
                events_to_keep = np.array([True] * events.shape[0])
                if self.max_lag is not None:
                    events_to_keep = events_to_keep & (
                                ((events.lag.values <= self.max_lag) & ~events.isFirst.values) | events.isFirst.values)
                if self.min_lag is not None:
                    events_to_keep = events_to_keep & (
                                ((events.lag.values >= self.min_lag) & ~events.isFirst.values) | events.isFirst.values)
                if self.only_correct_items:
                    events_to_keep = events_to_keep & self._filter_to_correct_items_paired(events)

                # load eeg for this channel
                eeg_channel = self._create_eeg_timeseries(channel_grp, events)[events_to_keep]

        # loop over sessions
        for session_name, session_dict in self.subject_data.items():

            # and channels
            channels = session_dict.keys()
            for channel in channels:
                data = session_dict[channel]['ST_eeg']

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

    def _create_eeg_timeseries(self, grp):
        data = np.array(grp['ST_eeg'])
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

    def plot_sta_single(self, session, channel, unit, save_dir=''):
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                fig, (ax1, ax2) = plt.subplots(1, 2)

                # axis 1: STA
                sta = self.res['sta'].loc[(session, channel, unit)]
                ax1.plot(sta.columns.values, np.squeeze(sta.values), lw=2)
                ax1.set_ylabel('Microvolts')
                ax1.set_xlabel('Time (s)')
                for tick in ax1.get_xticklabels():
                    tick.set_rotation(45)

                # axis 2: power spectra
                p_spect = self.res['p_spect'].loc[(session, channel, unit)]
                ax2.plot(np.log10(p_spect.columns), np.squeeze(p_spect.values))

                new_x = self.compute_pow_two_series()
                _ = ax2.xaxis.set_ticks(np.log10(new_x))
                _ = ax2.xaxis.set_ticklabels(new_x, rotation=0)
                ax2.set_ylabel('log(power)')
                ax2.set_xlabel('Frequency (Hz)')

                title_str = '{} {} - channel {}, unit {}: {} {}'.format(self.subject,
                                                                        session,
                                                                        channel,
                                                                        unit,
                                                                        sta.reset_index()['region'].values[0],
                                                                        sta.reset_index()['hemi'].values[0])
                plt.suptitle(title_str)
                fig.set_size_inches(15, 6)
                plt.tight_layout()

                if save_dir:
                    fname = '{}_{}_{}_{}.pdf'.format(self.subject, session, channel, unit)
                    fname = os.path.join(save_dir, fname)
                    plt.savefig(fname)

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
