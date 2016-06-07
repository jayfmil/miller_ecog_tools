import numpy as np
import re
import os
from glob import glob
import pdb
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter
from ptsa.data.TimeSeriesX import TimeSeriesX
import cPickle as pickle
import cluster_helper.cluster


# I don't know why parallel code can't be member of the class, but whatever. Define it here
def process_elec_eeg_and_pow(info):

    # pull out the info we need from the input list
    elecs = info[0]
    elec_ind = info[1]
    events = info[2]
    params = info[3]

    # if doing bipolar, load in pair of channels
    if params['bipolar']:

        # load eeg for channels in this bipolar_pair
        # pdb.set_trace()
        eeg_reader = EEGReader(events=events, channels=np.array(list(elecs[elec_ind])), start_time=params['start_time'],
                               end_time=params['end_time'], buffer_time=params['buffer_len'])
        eegs = eeg_reader.read()

        # convert to bipolar
        bipolar_recarray = np.recarray(shape=1, dtype=[('ch0', '|S3'), ('ch1', '|S3')])
        bipolar_recarray[0] = elecs[elec_ind]
        m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=bipolar_recarray)
        eegs = m2b.filter()
    else:

        # load eeg for for single channe;
        eeg_reader = EEGReader(events=events, channels=np.array(elecs[elec_ind]), start_time=params['start_time'],
                               end_time=params['end_time'], buffer_time=params['buffer_len'])
        eegs = eeg_reader.read()

    # filter 60 Hz line noise
    b_filter = ButterworthFilter(time_series=eegs, freq_range=[58., 62.], filt_type='stop', order=4)
    eegs_filtered = b_filter.filter()

    # compute power
    wf = MorletWaveletFilter(time_series=eegs_filtered, freqs=params['freqs'], output='power')
    pow_elec, phase_dummy = wf.filter()

    # remove buffer
    pow_elec = pow_elec.remove_buffer(duration=params['buffer_len'])

    # log transform
    np.log10(pow_elec.data, out=pow_elec.data)

    # downsample
    ds_pow_elec = downsample_power(pow_elec, params['window_size'], params['step_size'])
    return ds_pow_elec


def downsample_power(power_timeseries, window_size, step_size):
    # get time axis data and define start and end point
    y = power_timeseries['time'] * 1000.
    start_ms = PowerCalc.start_time * 1000.
    end_ms = PowerCalc.end_time * 1000.

    # get the start of each bin based on step size. Include endpoint if valid
    edges = np.arange(start_ms, end_ms, step_size)
    if np.ptp([start_ms, end_ms]) % step_size == 0:
        edges = np.append(edges, end_ms)

    # find center of each bin
    bin_centers = np.mean(np.array(zip(edges, edges[1:])), axis=1)

    # add plus/minus half width of each window to bin center, as long as it's within entire range
    power_bins = [[x - window_size / 2, x + window_size / 2] for x in bin_centers
                  if ((x + window_size / 2 <= end_ms) & (x - window_size / 2 >= start_ms))]

    # create a list where each entry is a boolean specifying which elements of the power data go in that bin
    bin_inds_lst = [(y >= bin[0]) & (y <= bin[1]) for bin in power_bins]

    # now bin power data
    ds_pow = [power_timeseries[:, :, :, bin].mean('time').data for bin in bin_inds_lst]

    # bin time axis data
    ds_time = [power_timeseries['time'][bin].mean().data for bin in bin_inds_lst]
    # pdb.set_trace()
    # stack all the time bins
    ds_time_series = np.stack(ds_pow, -1)

    return ds_time_series, ds_time


class PowerCalc:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    data_path = '/data/eeg'
    save_dir = '/scratch/jfm2/python_power/TH'

    # power calculation time
    start_time = -2.0
    end_time = 2.0
    buffer_len = 2.0

    def __init__(self, subjs=None, bipolar=True, freqs=None, window_size=1000., step_size=100., do_par=False):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = self.get_th_subjs()
        self.subjs = subjs

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute power
        self.freqs = freqs

        # and the window size and step size for downsampling
        self.window_size = window_size
        self.step_size = step_size

        # if doing parallel jobs
        self.do_par = do_par

        # create dict of parameters to send to parallel function
        keys = ['bipolar', 'freqs', 'window_size', 'step_size', 'do_par', 'start_time', 'end_time', 'buffer_len']
        vals = [bipolar, freqs, window_size, step_size, do_par, PowerCalc.start_time, PowerCalc.end_time,
                PowerCalc.buffer_len]
        self.params = {k: v for (k, v) in zip(keys, vals)}

    def create_pow_for_all_subjs(self):
        for subj in self.subjs:
            print 'Processing %s.' % subj
            subj_pow = self.create_pow_for_single_subj(subj)

            # define save directory
            subj_save_dir = os.path.join(PowerCalc.save_dir, '%d_freqs' % (len(self.freqs)),
                                         'window_size_%d_step_size_%d' % (self.window_size, self.step_size))
            if not os.path.exists(subj_save_dir):
                os.makedirs(subj_save_dir)

            # open file and save
            save_file = os.path.join(subj_save_dir, subj+'.p')
            with open(save_file, 'wb') as f:
                pickle.dump(subj_pow, f, protocol=-1)

    def create_pow_for_single_subj(self, subj):

        # get electrode numbers and events
        elecs_bipol, elecs_monopol = self.load_subj_elecs(subj)
        events = self.load_subj_events(subj)

        # construct input to main prcoessing function
        elecs = elecs_bipol if self.bipolar else elecs_monopol
        num_elecs = len(elecs)
        elec_range = range(num_elecs)

        # info is the big list that map() iterates over
        info = zip([elecs] * num_elecs, elec_range, [events] * num_elecs, [self.params] * num_elecs)

        if not self.do_par:
            downsampled_elec_list = map(process_elec_eeg_and_pow, info)
        else:
            with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=50,
                                                     cores_per_job=1, extra_params={'mem': 10}) as view:
                downsampled_elec_list = view.map(process_elec_eeg_and_pow, info)

        # create np array from list of new downsampled data
        ds_data = [x[0] for x in downsampled_elec_list]
        ds_data = np.concatenate(ds_data, axis=1)

        # pull out what will be the new time axis
        ds_time = np.stack(downsampled_elec_list[0][1])

        # new time series object
        elec_str = 'bipolar_pairs' if self.bipolar else 'channels'
        new_ts = TimeSeriesX(data=ds_data, dims=['frequency', elec_str, 'events', 'time'],
                             coords={'frequency': self.freqs,
                                     elec_str: elecs,
                                     'events': events,
                                     'time': ds_time,
                                     'window_size': self.window_size,
                                     'step_size': self.step_size})
        return new_ts

    @classmethod
    def load_subj_events(cls, subj):
        """Returns subject event structure."""
        subj_ev_path = os.path.join(cls.event_path, subj + '_events.mat')
        e_reader = BaseEventReader(filename=subj_ev_path, eliminate_events_with_no_eeg=True)
        events = e_reader.read()

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]
        ev_order = np.argsort(events, order=('session', 'trial', 'mstime'))
        events = events[ev_order]

        # filter to just item presentation events
        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        return events

    @classmethod
    def load_subj_elecs(cls, subj):
        """Returns array of electrode numbers  (monopolar and bipolar)."""
        tal_path = os.path.join(cls.data_path, subj, 'tal', subj + '_talLocs_database_bipol.mat')
        tal_reader = TalReader(filename=tal_path)
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_pairs = tal_reader.get_bipolar_pairs()
        return bipolar_pairs, monopolar_channels

    @classmethod
    def get_th_subjs(cls):
        """Returns list of subjects who performed TH1."""
        subjs = glob(os.path.join(cls.event_path, 'R*_events.mat'))
        subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
        subjs.sort()
        return subjs
