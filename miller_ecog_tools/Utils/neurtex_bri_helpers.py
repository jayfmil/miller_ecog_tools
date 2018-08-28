"""
Code for loading Neurtex Brain Research Institute data. Basically, loads custom neuralynx-like files. Only for use
with this project, as there a lot of custom file types and directory paths/filenames.

Built for use with data from the continuous recognition (CRM) paradigm.
"""

import numpy as np
import pandas as pd
import re

from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ResampleFilter
from ptsa.data.timeseries import TimeSeries
from scipy.signal import resample

# file constants
HEADER_SIZE = 16 * 1024
BLOCK_SIZE = 512

# behavioral data is all in one big table. Defined at module scope for caching purposes.
my_globals = {'master_table_path': '/scratch/josh/BniData/BniData/Analysis/CRM/masterPseudoZ_r5.txt',
              'master_table_data': None}


def set_master_table(filepath):
    """
    Set the master_table_path variable. Also unloads the cached master data if loaded.

    Parameters
    ----------
    filepath: str
        The path to the file.

    """
    my_globals['master_table_path'] = filepath
    my_globals['master_table_data'] = None


def load_master_table():
    """
    Loads master data table and stores it in module variable my_globals['master_table_data']. We are caching it
    because it'll be faster than having to load this over and over.
    """

    # read table
    my_globals['master_table_data'] = pd.read_table(my_globals['master_table_path'])

    # also make a new column with a unique subject identifier
    my_globals['master_table_data']['subject'] = my_globals['master_table_data'].expID.apply(lambda x: x.split('e')[0])


def get_subjs(task='crm'):
    """
    Parameters
    ----------
    task: str
        Task to use. CURRENTLY ONLY 'crm' IS SUPPORTED

    Returns
    -------
    numpy.ndarray
        array of subject names. Technically these are sessions, not subjects.

    """
    if task != 'crm':
        print('CURRENTLY ONLY crm TASK IS SUPPORTED.')
        return

    # load master data if not cached already
    if my_globals['master_table_data'] is None:
        load_master_table()

    # return unique subjects
    return my_globals['master_table_data'].subject.unique()


def load_subj_events(task='crm', subject=''):
    """
    Parameters
    ----------
    task: str
        Task to use. CURRENTLY ONLY 'crm' IS SUPPORTED
    subject: str
        Subject to use

    Returns
    -------
    pandas.DataFrame
        DataFrame of events for the given subject.
    """

    if task != 'crm':
        print('CURRENTLY ONLY crm TASK IS SUPPORTED.')
        return

    # load master data if not cached already
    if my_globals['master_table_data'] is None:
        load_master_table()

    # filter to just this subject
    df_subj = my_globals['master_table_data'][my_globals['master_table_data'].subject == subject]

    # reduce to only columns with relavent behavioral data and unique rows
    df_subj = df_subj[['expID', 'rep', 'name', 'stTime', 'endTime', 'firstResp', 'keyEarly', 'oldKey', 'otherKey',
                       'multiPress', 'delay', 'isPaired', 'pairedWithDup', 'isFirst', 'lag']].drop_duplicates()
    df_subj = df_subj.reset_index(drop=True)

    return df_subj


def stat_ncs(channel_file):
    """

    Parameters
    ----------
    channel_file: str
        Path to an ncs neuralynx file

    Returns
    -------
    dict
        A dictionary with the file's header parameters

    """
    header_keys = [('NLX_Base_Class_Type', None),
                   ('AmpHiCut', float),
                   ('ADChannel', None),
                   ('ADGain', float),
                   ('AmpGain', float),
                   ('SubSamplingInterleave', int),
                   ('ADMaxValue', int),
                   ('ADBitVolts', float),
                   ('SamplingFrequency', float),
                   ('AmpLowCut', float),
                   ('HardwareSubSystemName', None),
                   ('HardwareSubSystemType', None)]

    # load header from file
    with open(channel_file, 'rb') as f:
        txt_header = f.read(HEADER_SIZE)
    txt_header = txt_header.strip(b'\x00').decode('latin-1')

    # find values and make dict
    info = {}
    for k, type_ in header_keys:
        pattern = '-(?P<name>' + k + ')\t(?P<value>[\S ]*)'
        matches = re.findall(pattern, txt_header)
        for match in matches:
            name = match[0]
            val = match[1].rstrip(' ')
            if type_ is not None:
                val = type_(val)
            info[name] = val
    return info


def load_ncs(channel_file):
    """

    Parameters
    ----------
    channel_file: str
        Path to an ncs neuralynx file

    Returns
    -------
    signals: np.ndarry
        The eeg data for this channel, length is the number of samples
    timestamps: np.ndarry
        The timestamp (in microseconds) corresponding to each sample
    sr: float
        the sampleing rate of the data

    """

    # load header info
    info = stat_ncs(channel_file)

    # define datatype for memmap
    ncs_dtype = [('timestamp', 'uint64'), ('channel', 'uint32'), ('sample_rate', 'uint32'),
                 ('nb_valid', 'uint32'), ('samples', 'int16', (BLOCK_SIZE,))]

    # load it all at once... sorry.
    data = np.memmap(channel_file, dtype=ncs_dtype, mode='r', offset=HEADER_SIZE)

    # loop over each block and create timestamps for each sample
    signals = []
    timestamps = []
    for this_block in data:
        # extend our list of the data
        signals.extend(this_block[4])

        # create timestamps for each sample in this block
        timestamps_block = np.linspace(this_block[0], this_block[0] + (1e6 / this_block[2] * (BLOCK_SIZE - 1)),
                                       BLOCK_SIZE)
        timestamps.append(timestamps_block)

    # get our final arrays
    timestamps = np.concatenate(timestamps)
    signals = np.array(signals)
    signals = signals * info['ADBitVolts']

    return signals, timestamps, info['SamplingFrequency']


def load_nse(channel_file, return_waveforms=False):
    """

    Parameters
    ----------
    channel_file: str
        Path to an nse neuralynx file (NOTE: not ncs file)
    return_waveforms: bool
        Whether to return as a second output the spike waveforms

    Returns
    -------
    timestamps: numpy.ndarray
        An array of timestamps when spikes occured
    spike_waveforms: numpy.ndarray
        If return_waveforms, num spikes x 32 array
    """

    # nse dtype
    dtype = [('timestamp', 'uint64'), ('channel', 'uint32'), ('unit_id', 'uint32')]
    dtype += [('params', 'uint32', (8,))]
    dtype += [('samples', 'uint16', (32,))]

    # load spiking data
    data = np.memmap(channel_file, dtype=dtype, mode='r', offset=HEADER_SIZE)

    # get timestamps
    timestamps = np.array([x[0] for x in data])

    # get spike waveforms
    if return_waveforms:
        spike_waveforms = np.array([x[4] for x in data])
        return timestamps, spike_waveforms
    else:
        return timestamps


def load_eeg(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms=0, noise_freq=[58., 62.],
             resample_freq=None, pass_band=None, demean=False, do_average_ref=False):
    """
    Returns an EEG TimeSeries object.

    Parameters
    ----------
    events: pandas.DataFrame
        An events dataframe
    rel_start_ms: int
        Initial time (in ms), relative to the onset of each event
    rel_stop_ms: int
        End time (in ms), relative to the onset of each event
    channel_list: list
        list of paths to channels (ncs files)
    buf_ms:
        Amount of time (in ms) of buffer to add to both the begining and end of the time interval
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    resample_freq: float
        Sampling rate to resample to after loading eeg.
    pass_band: list
        If given, the eeg will be band pass filtered in the given range.
    demean: bool
        If True, will subject the mean voltage between rel_start_ms and rel_stop_ms from each channel
    do_average_ref: bool
        If True, will compute the average reference based on the mean voltage across channels

    Returns
    -------
    TimeSeries
        EEG timeseries object with dimensions event x time x channel

    """

    # load eeg and downsampled to 1000 Hz. This is hardcoded for now because I don't want 30 KHz data ever..
    # eeg is a PTSA timeseries
    eeg = _load_eeg_timeseries(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms, 1000)

    # compute average reference by subracting the mean across channels
    if do_average_ref:
        eeg = eeg - eeg.mean(dim='channel')

    # baseline correct subracting the mean within the baseline time range
    if demean:
        eeg = eeg.baseline_corrected([rel_start_ms, rel_stop_ms])

    # filter line noise
    if noise_freq is not None:
        if isinstance(noise_freq[0], float):
            noise_freq = [noise_freq]
        for this_noise_freq in noise_freq:
            b_filter = ButterworthFilter(eeg, this_noise_freq, filt_type='stop', order=4)
            eeg = b_filter.filter()

    # resample if desired. Note: can be a bit slow especially if have a lot of eeg data
    if resample_freq is not None:
        r_filter = ResampleFilter(eeg, resample_freq)
        eeg = r_filter.filter()

    # do band pass if desired.
    if pass_band is not None:
        eeg = band_pass_eeg(eeg, pass_band)

    return eeg


def band_pass_eeg(eeg, freq_range, order=4):
    """
    Runs a butterworth band pass filter on an eeg time seriesX object.

    Parameters
    ----------
    eeg: timeseries
        A ptsa.timeseries object
    freq_range: list
        List of two floats defining the range to filter in
    order: int
        Order of butterworth filter

    Returns
    -------
    timeseries
        Filtered EEG object
    """
    return ButterworthFilter(eeg, freq_range, filt_type='pass', order=order).filter()


def _load_eeg_timeseries(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms=0, resample_freq=1000):
    """

    Parameters
    ----------
    events
    rel_start_ms
    rel_stop_ms
    channel_list
    buf_ms
    resample_freq

    Returns
    -------

    """

    # will build a list of eeg data that we will concatenate across channels
    eeg_list = []

    # timeseries dims and coords
    dims = ('event', 'time', 'channel')
    coords = {'event': events.to_records(),
              'time': [],
              'channel': []}

    # epochs will be a list of tuples of start and stop sample offsets
    epochs = None
    for channel in channel_list:
        print(channel)

        # load channel data
        signals, timestamps, sr = load_ncs(channel)

        # get start and stop samples (only once)
        # assumes all channels have the same timestamps..
        if epochs is None:
            print('epochs')
            epochs = _compute_epochs(events, rel_start_ms - buf_ms, rel_stop_ms + buf_ms, timestamps, sr)

        # segment the continuous eeg into epochs. Also resample.
        eeg, new_time = _segment_eeg_single_channel(signals, epochs, sr, timestamps, resample_freq)
        eeg_list.append(eeg)
        coords['channel'].append(channel)

    # create timeseries
    coords['time'] = (new_time[0] - events.stTime[0])/1e6
    eeg_all_chans = TimeSeries.create(np.stack(eeg_list, -1), samplerate=resample_freq, dims=dims, coords=coords)

    return eeg_all_chans


def _compute_epochs(events, rel_start_ms, rel_stop_ms, timestamps, sr):
    """
    convert timestamps into start and start sample offsets
    """
    offsets = events.stTime.apply(lambda x: np.where(timestamps >= x)[0][0])
    rel_start_micro = int(rel_start_ms * sr / 1e3)
    rel_stop_micro = int(rel_stop_ms * sr / 1e3)
    epochs = [(offset + rel_start_micro, offset + rel_stop_micro) for offset in offsets]
    return epochs


def _segment_eeg_single_channel(signals, epochs, sr, timestamps, resample_freq):
    eeg = np.array([signals[x[0]:x[1]] for x in epochs])
    time_data = np.array([timestamps[x[0]:x[1]] for x in epochs])

    if resample_freq is not None:
        new_length = int(np.round(eeg.shape[1] * resample_freq / sr))
        resampled_eeg = []
        resampled_t = []

        # looping because otherwise I get a value error that doesn't make
        # and that I should probably try to figure out...
        for i in range(eeg.shape[0]):
            e, t = resample(eeg[i], new_length, t=time_data[i], axis=0)
            resampled_eeg.append(e)
            resampled_t.append(t)
        return np.array(resampled_eeg), np.array(resampled_t)
    else:
        return eeg, time_data






































