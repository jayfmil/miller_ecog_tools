"""
Code for loading Neurtex Brain Research Institute data. Basically, loads custom neuralynx-like files. Only for use
with this project, as there a lot of custom file types and directory paths/filenames.

Built for use with data from the continuous recognition (CRM) paradigm.
"""

import re
import os
import numexpr
import numpy as np
import pandas as pd

from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ResampleFilter
from ptsa.data.timeseries import TimeSeries
from scipy.signal import resample, filtfilt, butter
from glob import glob

# file constants
HEADER_SIZE = 16 * 1024
BLOCK_SIZE = 512

# behavioral data is all in one big table. Defined at module scope for caching purposes. Also define path subject dirs
my_globals = {'master_table_path': '/scratch/josh/BniData/BniData/Analysis/CRM/masterPseudoZ_r5.txt',
              'master_table_data': None,
              'subject_dir': '/scratch/josh/BniData/BniData/Subjects'}


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


def get_subj_files_by_sess(task='crm', subject=''):
    """
    Parameters
    ----------
    task: str
        Task to use. CURRENTLY ONLY 'crm' IS SUPPORTED
    subject: str
        Subject to use

    Returns
    -------
    dict
        Dictionary with a key for each session ID, with a list of all the .Ncs files for that session.
    """

    if task != 'crm':
        print('CURRENTLY ONLY crm TASK IS SUPPORTED.')
        return

    # load master data if not cached already
    if my_globals['master_table_data'] is None:
        load_master_table()

    # get session list
    sessions = my_globals['master_table_data'][my_globals['master_table_data'].subject == subject].expID.unique()

    # paths on disk have extra 0 in the subject names..
    subj_str = subject[0] + '0' + subject[-2:] if len(subject) == 3 else subject[0] + '00' + subject[-1:]

    # loop over each session
    file_dict = {}
    for session in sessions:
        session_dir = os.path.join(my_globals['subject_dir'], subj_str, 'analysis', session)

        # will be dictionary where keys are channel numbers
        session_dict = {}

        # get list of channel files
        ncs_files = glob(session_dir + '/*.Ncs')

        # store data for this channel, both the Ncs and Nse files. Also storing subject and session for convenience
        for ncs_file in ncs_files:
            chan_num = re.split(r'(\d+)', ncs_file)[-2]
            session_dict[int(chan_num)] = {'ncs': ncs_file,
                                           'nse': os.path.join(session_dir, 'KK', 'CSC' + chan_num + '.Nse'),
                                           'clusters': os.path.join(session_dir, 'KK', 'CSC' + chan_num + '.clu.1'),
                                           'subject': subject,
                                           'session': session}

        # store in dictionary
        file_dict[session] = session_dict

    return file_dict


def get_localization_by_sess(subject, session, channel_num, clusters):
    """
    Look up region and hemisphere information in the big master table for a given channel and cluster numbers.

    Parameters
    ----------
    subject: str
        subject string
    session: str
        session string
    channel_num: int
        channel number of the data file
    clusters: np.ndarray
        array of cluster id numbers

    Returns
    -------
    np.chararray, np.chararray
        Character arrays (same length as input clusters) with corresponding region and hemisphere labels

    """
    # reduce master data table to just this subject and session
    df = my_globals['master_table_data']
    cluster_qual_df = df[(df.subject == subject) & (df.expID == session)][['clustId', 'side', 'area']].drop_duplicates()

    # pull out the channel number and cluster number from the clustId string
    chan_clust = cluster_qual_df.clustId.apply(lambda x: np.array([int(y) for y in re.findall(r'\d+', x)]))
    channels, cluster_ids = np.stack(chan_clust.values).T

    # will hold region and hemisphere for each cluster entry
    region = np.chararray(clusters.shape, 2, unicode=True)
    hemisphere = np.chararray(clusters.shape, 2, unicode=True)

    # loop over each cluster
    for this_cluster in np.unique(clusters):
        ind_df = (channels == channel_num) & (cluster_ids == this_cluster)
        ind_clusters = clusters == this_cluster
        region[ind_clusters] = cluster_qual_df[ind_df].area
        hemisphere[ind_clusters] = cluster_qual_df[ind_df].side

    return region, hemisphere


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
                       'multiPress', 'delay', 'isPaired', 'pairedWithDup', 'isFirst', 'lag', 'subject']].drop_duplicates()
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

    # convert to microvolts
    signals = signals * info['ADBitVolts'] * 1e6

    actual_samplerate = 1e6 / (np.mean(np.diff([x[0] for x in data]))/BLOCK_SIZE)
    #info['SamplingFrequency']
    return signals, timestamps, actual_samplerate


def load_spikes_cluster_with_qual(session_file_dict, chan_num, quality=list(['SPIKE'])):
    """

    Parameters
    ----------
    session_file_dict: dict
        A subdictionary returned by .get_subj_files_by_sess()
    chan_num: int
        Channel number to use as key into the dictionary
    quality: list of strings
        Either ['SPIKE'], ['POTENTIAL'], or ['SPIKE', 'POTENTIAL'] specifying which spikes to load

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        arrays of spike times and cluster IDs

    """

    # get the channel info from the dict
    subject = session_file_dict[chan_num]['subject']
    session = session_file_dict[chan_num]['session']
    channel_file = session_file_dict[chan_num]['nse']
    cluster_file = session_file_dict[chan_num]['clusters']

    # reduce master data table to just this subject and session
    df = my_globals['master_table_data']
    cluster_qual_df = df[(df.subject == subject) & (df.expID == session)][['clustId', 'quality']].drop_duplicates()

    # pull out the channel number and cluster number from the clustId string
    chan_clust = cluster_qual_df.clustId.apply(lambda x: np.array([int(y) for y in re.findall(r'\d+', x)]))
    channels, cluster_ids = np.stack(chan_clust.values).T

    # get non-noise clusters ids of requested quality for this channel
    good_clusts = cluster_ids[(channels == chan_num) & cluster_qual_df.quality.isin(quality)]

    # if we have good clusters, load the data
    if good_clusts.size > 0:
        spikes = load_nse(channel_file, return_waveforms=False)
        clusters = load_cluster_ids(cluster_file)
        if len(spikes) != len(clusters):
            print('Something wrong, number of spikes and cluster ids not equal.')
            return

        # do the filtering to the good clusters
        good_spikes = np.in1d(clusters, good_clusts)
        return spikes[good_spikes], clusters[good_spikes]
    else:
        return np.array([]), np.array([])


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


def load_cluster_ids(cluster_file):
    """

    Parameters
    ----------
    cluster_file: str
        Path to a 'clusters' file in the .get_subj_files_by_sess() dict This is .clu file with the IDs of the spikes.

    Returns
    -------
    np.ndarray
        Array with an integers representing the cluser of ID of each spike

    """

    # return array of cluster IDs, skipping the first entry which is not a cluster ID
    return np.fromfile(cluster_file, dtype=int, sep='\n')[1:]


def load_eeg_from_times(df, channel_file, rel_start_ms, rel_stop_ms, buf_ms=0, noise_freq=[58., 62.],
                        downsample_freq=1000, resample_freq=None, pass_band=None):
    """

    Parameters
    ----------
    df: pandas.DataFrame
        An dataframe with a stTime column
    channel_file: str
        Path to Ncs file from which to load eeg.
    rel_start_ms: int
        Initial time (in ms), relative to the onset of each spike
    rel_stop_ms: int
        End time (in ms), relative to the onset of each spike
    buf_ms: int
        Amount of time (in ms) of buffer to add to both the beginning and end of the time interval
    noise_freq: list
        Stop filter will be applied to the given range. Default=[58. 62]
    downsample_freq
    resample_freq
    pass_band

    Returns
    -------

    """

    # # make a df with 'stTime' column to pass to _load_eeg_timeseries
    # events = pd.DataFrame(data=np.stack([s_times, clust_nums], -1), columns=['stTime', 'cluster_num'])

    # load eeg for this channel
    eeg = _load_eeg_timeseries(df, rel_start_ms, rel_stop_ms, [channel_file], buf_ms, downsample_freq, resample_freq)

    # filter line noise
    if noise_freq is not None:
        if isinstance(noise_freq[0], float):
            noise_freq = [noise_freq]
        for this_noise_freq in noise_freq:
            b_filter = ButterworthFilter(eeg, this_noise_freq, filt_type='stop', order=4)
            eeg = b_filter.filter()

    # mean center the data
    eeg = eeg.baseline_corrected([rel_start_ms, rel_stop_ms])

    # do band pass if desired.
    if pass_band is not None:
        eeg = band_pass_eeg(eeg, pass_band)

    return eeg


def power_spectra_from_spike_times(s_times, clust_nums, channel_file, rel_start_ms, rel_stop_ms, freqs,
                                           noise_freq=[58., 62.], downsample_freq=250, mean_over_spikes=True):
    """
    Function to compute power relative to spike times. This computes power at given frequencies for the ENTIRE session
    and then bins it relative to spike times. You WILL run out of memory if you don't let it downsample first. Default
    downsample is to 250 Hz.

    Parameters
    ----------
    s_times: np.ndarray
        Array (or list) of timestamps of when spikes occured. EEG will be loaded relative to these times.
    clust_nums:
        s_times: np.ndarray
        Array (or list) of cluster IDs, same size as s_times
    channel_file: str
        Path to Ncs file from which to load eeg.
    rel_start_ms: int
        Initial time (in ms), relative to the onset of each spike
    rel_stop_ms: int
        End time (in ms), relative to the onset of each spike
    freqs: np.ndarray
        array of frequencies at which to compute power
    noise_freq: list
        Stop filter will be applied to the given range. Default=[58. 62]
    downsample_freq: int or float
        Frequency to downsample the data. Use decimate, so we will likely not reach the exact frequency.
    mean_over_spikes: bool
        After computing the spike x frequency array, do we mean over spikes and return only the mean power spectra

    Returns
    -------
    dict
        dict of either spike x frequency array of power values or just frequencies, if mean_over_spikes. Keys are
        cluster numbers
    """

    # make a df with 'stTime' column for epoching
    events = pd.DataFrame(data=np.stack([s_times, clust_nums], -1), columns=['stTime', 'cluster_num'])

    # load channel data
    signals, timestamps, sr = load_ncs(channel_file)

    # downsample the session
    if downsample_freq is not None:
        signals, timestamps, sr = _my_downsample(signals, timestamps, sr, downsample_freq)
    else:
        print('I HIGHLY recommend you downsample the data before computing power across the whole session...')
        print('You will probably run out of memory.')

    # make into timeseries
    eeg = TimeSeries.create(signals, samplerate=sr, dims=['time'], coords={'time': timestamps / 1e6})

    # filter line noise
    if noise_freq is not None:
        if isinstance(noise_freq[0], float):
            noise_freq = [noise_freq]
        for this_noise_freq in noise_freq:
            b_filter = ButterworthFilter(eeg, this_noise_freq, filt_type='stop', order=4)
            eeg = b_filter.filter()

    # compute power
    wave_pow = MorletWaveletFilter(eeg, freqs, output='power', width=5, cpus=12, verbose=False).filter()

    # log the power
    data = wave_pow.data
    wave_pow.data = numexpr.evaluate('log10(data)')

    # get start and stop relative to the spikes
    epochs = _compute_epochs(events, rel_start_ms, rel_stop_ms, timestamps, sr)
    bad_epochs = (np.any(epochs < 0, 1)) | (np.any(epochs > len(signals), 1))
    epochs = epochs[~bad_epochs]
    events = events[~bad_epochs].reset_index(drop=True)

    # mean over time within epochs
    spikes_x_freqs = np.stack([np.mean(wave_pow.data[:, x[0]:x[1]], axis=1) for x in epochs])

    # make dict with keys being cluster numbers. Mean over spikes if desired.
    pow_spect_dict = {}
    for this_cluster in events.cluster_num.unique():
        if mean_over_spikes:
            pow_spect_dict[this_cluster] = spikes_x_freqs[events.cluster_num == this_cluster].mean(axis=0)
        else:
            pow_spect_dict[this_cluster] = spikes_x_freqs[events.cluster_num == this_cluster]

    return pow_spect_dict


def load_eeg_from_event_times(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms=0, noise_freq=[58., 62.],
                              downsample_freq=1000,
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
    buf_ms: int
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

    # eeg is a PTSA timeseries
    eeg = _load_eeg_timeseries(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms, downsample_freq)

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


def _load_eeg_timeseries(events, rel_start_ms, rel_stop_ms, channel_list, buf_ms=0, downsample_freq=1000,
                         resample_freq=None):
    """

    Parameters
    ----------
    events: pandas.DataFrame
        DataFrame with the column 'stTime', specifying the timestamp when the event occurred
    rel_start_ms: int
        Relative time (ms) to add to the stTime to define the start of the time interval
    rel_stop_ms: int
        Relative time (ms) to add to the stTime to define the end of the time interval
    channel_list: list
        List of channel Ncs files
    buf_ms:
        Buffer (ms) to add to the start and end of the time period
    downsample_freq: int
        sample rate to downsample sample initial data immediately after loading the full file
    resample_freq: int
        Resample eeg to this value. Done after epoching.

    Returns
    -------
        ptsa.TimeSeries with dims event x time x channel
    """

    # will build a list of eeg data that we will concatenate across channels
    eeg_list = []

    # epochs will be a list of tuples of start and stop sample offsets
    epochs = None
    for channel in channel_list:

        # load channel data
        signals, timestamps, sr = load_ncs(channel)

        if downsample_freq is not None:
            signals, timestamps, sr = _my_downsample(signals, timestamps, sr, downsample_freq)

        # get start and stop samples (only once)
        # assumes all channels have the same timestamps..
        if epochs is None:
            epochs = _compute_epochs(events, rel_start_ms - buf_ms, rel_stop_ms + buf_ms, timestamps, sr)

            # remove any epochs < 0
            bad_epochs = (np.any(epochs < 0, 1)) | (np.any(epochs > len(signals), 1))
            epochs = epochs[~bad_epochs]
            events = events[~bad_epochs].reset_index(drop=True)

        # segment the continuous eeg into epochs. Also resample.
        eeg, new_time = _segment_eeg_single_channel(signals, epochs, sr, timestamps, resample_freq)
        eeg_list.append(eeg)

    # create timeseries
    dims = ('event', 'time', 'channel')
    coords = {'event': events.to_records(),
              'time': (new_time[0] - events.stTime.values[0])/1e6,
              'channel': channel_list}
    sr_for_ptsa = resample_freq if resample_freq is not None else sr
    eeg_all_chans = TimeSeries.create(np.stack(eeg_list, -1), samplerate=sr_for_ptsa, dims=dims, coords=coords)
    return eeg_all_chans


def _compute_epochs(events, rel_start_ms, rel_stop_ms, timestamps, sr):
    """
    convert timestamps into start and start sample offsets
    """

    # THIS IS SO MUCH FASTER THAN NP.WHERE, CRAZY
    offsets = events.stTime.apply(lambda x: np.searchsorted(timestamps, x))
    # offsets = events.stTime.apply(lambda x: np.where(timestamps >= x)[0][0])
    rel_start_micro = int(rel_start_ms * sr / 1e3)
    rel_stop_micro = int(rel_stop_ms * sr / 1e3)
    epochs = np.array([(offset + rel_start_micro, offset + rel_stop_micro) for offset in offsets])
    return epochs


def _segment_eeg_single_channel(signals, epochs, sr, timestamps, resample_freq):
    """
    Chunk eeg signal and timestamps by epochs. Also resample if desired
    """
    eeg = np.stack([signals[x[0]:x[1]] for x in epochs])
    time_data = np.stack([timestamps[x[0]:x[1]] for x in epochs])

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


def _my_downsample(signal, timestamps, sr, desired_downsample_rate):
    """
    Downsample using a decimate style. Not using scipy because I also want to return
    the new timestamps. scipy.resample is super slow for large arrays.
    """

    # figure out our decimate factor. Must be int, so we'll try to get as close
    # as possible to the desired rate. Will not be exactly.
    ts_diff_sec = np.mean(np.diff(timestamps)) / 1e6
    dec_factor = np.floor(1. / (desired_downsample_rate * ts_diff_sec))

    # new sampling rate
    ts_diff_down = dec_factor * ts_diff_sec
    new_sr = 1. / ts_diff_down

    # apply a low pass filter before decimating
    low_pass_freq = new_sr / 2.
    [b, a] = butter(4, low_pass_freq / (sr / 2), 'lowpass')
    signals_low_pass = filtfilt(b, a, signal, axis=0)

    # now decimate
    inds = np.arange(0, len(signals_low_pass), dec_factor, dtype=int)
    new_sigals = signals_low_pass[inds]
    new_ts = timestamps[inds]
    return new_sigals, new_ts, new_sr


































