import numexpr
import json
import joblib
import os
# import re
import warnings
# import cluster_helper.cluster
import numpy as np
import xarray as xr
# from ptsa.data.readers import BaseEventReader
from ptsa.data.readers import EEGReader
# from ptsa.data.readers.tal import TalReader
# from ptsa.data.readers.index import JsonIndexReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ResampleFilter
from ptsa.data.timeseries import TimeSeries
from scipy.stats.mstats import zscore
from tqdm import tqdm
from glob import glob

import pandas as pd
from cmlreaders import CMLReader, get_data_index
from scipy.io import loadmat



# load json database of subject information. Doing this on import because it's not
# super fast, and I don't want to do it each call to get_subjs or whatever functions need it
try:
    # reader = JsonIndexReader('/protocols/r1.json')
    r1_data = get_data_index("r1")
except(IOError):
    print('JSON protocol file not found')


# use cmlreaders to get r1 tasks
# ram_tasks = r1_data.experiment.unique()


def get_subjs_and_montages(task):
    """Returns a DataFrame with columns 'subject' and 'montage' listing participants in a given experiment.

    Parameters
    ----------
    task: str
        The experiment name (ex: TH1, FR1, ...).

    Returns
    -------
    pandas.DataFrame
        A DataFrame of all subjects who performed the task.
    """

    # if this is RAM task, load the subject/montage directly from the r1 database
    task = task.replace('RAM_', '')
    if task in r1_data.experiment.unique():
        df = r1_data[r1_data['experiment'] == task][['subject', 'montage']].drop_duplicates().reset_index(drop=True)

    # otherwise, need to look for *events.mat files in '/data/events/task
    else:
        subj_list = []
        mont_list = []
        subjs = glob(os.path.join('/data/events/', task, '*_events.mat'))
        subjs = [os.path.split(f.replace('_events.mat', ''))[1] for f in subjs]
        subjs.sort()
        for subj in subjs:
            m = 0
            if '_' in subj:
                subj_split = subj.split('_')
                if len(subj_split[-1]) == 1:
                    m = int(subj_split[-1])
                    subj = subj[:-2]
            subj_list.append(subj)
            mont_list.append(m)
        df = pd.DataFrame({'subject': np.array(subj_list, dtype=object), 'montage': np.array(mont_list, dtype=int)})
    return df


def load_subj_events(task, subject, montage, as_df=True):
    """Returns a DataFrame of the events.

    Parameters
    ----------
    task: str
        The experiment name (ex: RAM_TH1, RAM_FR1, ...).
    subject: str
        The subject code
    montage: int
        The montage number for the subject
    as_df: bool
        If true, the events will returned as a pandas.DataFrame, otherwise a numpy.recarray

    Returns
    -------
    pandas.DataFrame
        A DataFrame of of the events
    """
    task = task.replace('RAM_', '')

    # if a RAM task, get info from r1 database and load as df using cmlreader
    if task in r1_data.experiment.unique():
        # get list of sessions for this subject, experiment, montage
        inds = (r1_data['subject'] == subject) & (r1_data['experiment'] == task) & (r1_data['montage'] == int(montage))
        sessions = r1_data[inds]['session'].unique()

        # load all and concat
        events = pd.concat([CMLReader(subject=subject,
                                      experiment=task,
                                      session=session).load('events')
                            for session in sessions])
        if not as_df:
            events = events.to_records(index=False)

    # otherwise load matlab files
    else:
        subj_file = subject + '_events.mat'
        if int(montage) != 0:
            subj_file = subject + '_' + str(montage) + '_events.mat'
        subj_ev_path = str(os.path.join('/data/events/', task, subj_file))
        # events = read_mat(subj_ev_path, 'events')
        events = loadmat(subj_ev_path, squeeze_me=True)['events']
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]

        if as_df:
            events = pd.DataFrame.from_records(events)

    return events


def load_elec_info(subject, montage=0, bipolar=True, as_df=True, return_raw=False):
    """

    Parameters
    ----------
    subject: str
        subject code
    montage: int
        montage number
    bipolar: bool
        whether to return electrode info for bipolar or monopolar electrode configuration
    return_raw: bool
        whether to return the data as it originally was loaded, or to return the standardized version. The
        standardized version doesn't contain all the possible fields and only returns the most useful (imo) stuff. It
        also makes sure the field/column names are the same between the old .mat version and the json data.


    Returns
    -------

    """

    # check if this subject/montage is in r1. If it is, use cmlreaders to load it. Easy.
    if np.any((r1_data['subject'] == subject) & (r1_data['montage'] == montage)):
        elec_df = CMLReader(subject=subject, montage=montage).load('pairs' if bipolar else 'contacts')

    # otherwise, load the mat file and do some reorganization to make it a nice dataframe
    else:
        # load appropriate .mat file
        subj_mont = subject
        if int(montage) != 0:
            subj_mont = subject + '_' + str(montage)
        file_str = '_bipol' if bipolar else ''
        tal_path = os.path.join('/data/eeg', subj_mont, 'tal', subj_mont + '_talLocs_database' + file_str + '.mat')
        elec_raw = loadmat(tal_path, squeeze_me=True)
        elec_raw = elec_raw[np.setdiff1d(list(elec_raw.keys()), ['__header__', '__version__', '__globals__'])[0]]

        # sume of the data is in subarrays, flatten it, and make dataframe. Eeessh
        # also rename some of the fields/columns
        # make average surface dataframe
        surf_data = []
        exclude = []
        if 'avgSurf' in elec_raw.dtype.names:
            avg_surf = pd.concat([pd.DataFrame(index=[i], data=e) for (i, e) in enumerate(elec_raw['avgSurf'])],
                                 sort=False)
            avg_surf = avg_surf.rename(columns={x: 'avg.{}'.format(x) for x in avg_surf.columns})
            surf_data.append(avg_surf)
            exclude.append('avgSurf')

        # make indiv surface dataframe
        if 'indivSurf' in elec_raw.dtype.names:
            ind_surf = pd.concat([pd.DataFrame(index=[i], data=e) for (i, e) in enumerate(elec_raw['indivSurf'])],
                                 sort=False)
            ind_surf = ind_surf.rename(columns={x: 'ind.{}'.format(x) for x in ind_surf.columns})
            surf_data.append(ind_surf)
            exclude.append('indivSurf')

            # concat them, excluding the original subarrays
        elec_df = pd.DataFrame.from_records(elec_raw, exclude=exclude)
        elec_df = pd.concat([elec_df] + surf_data, axis='columns')

    return elec_df


def load_eeg(events, rel_start_ms, rel_stop_ms, buf_ms=0, elec_scheme=None, noise_freq=[58., 62.],
             resample_freq=None, pass_band=None, use_mirror_buf=False, demean=False):
    """
    Returns an EEG TimeSeries object.

    Parameters
    ----------
    events: pandas.DataFrame
        An events dataframe that contains eegoffset and eegfile fields
    rel_start_ms: int
        Initial time (in ms), relative to the onset of each event
    rel_stop_ms: int
        End time (in ms), relative to the onset of each event
    buf_ms:
        Amount of time (in ms) of buffer to add to both the begining and end of the time interval
    elec_scheme: pandas.DataFrame
        DESRCIBE THIS
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    resample_freq: float
        Sampling rate to resample to after loading eeg.
    pass_band: list
        If given, the eeg will be band pass filtered in the given range.
    use_mirror_buf: bool
        If True, the buffer will be data taken from within the rel_start_ms to rel_stop_ms interval,
        mirrored and prepended and appended to the timeseries. If False, data outside the rel_start_ms and rel_stop_ms
        interval will be read.

    Returns
    -------
    TimeSeries
        EEG timeseries object with dimensions channels x events x time (or bipolar_pairs x events x time)

        NOTE: The EEG data is returned with time buffer included. If you included a buffer and want to remove it,
              you may use the .remove_buffer() method. EXTRA NOTE: INPUT SECONDS FOR REMOVING BUFFER, NOT MS!!

    """

    # add buffer is using
    if (buf_ms is not None) and not use_mirror_buf:
        actual_start = rel_start_ms - buf_ms
        actual_stop = rel_stop_ms + buf_ms
    else:
        actual_start = rel_start_ms
        actual_stop = rel_stop_ms

    # load eeg
    # Should auto convert to PTSA? Any reason not to?
    eeg = CMLReader(subject=events.iloc[0].subject).load_eeg(events, rel_start=actual_start, rel_stop=actual_stop,
                                                        scheme=elec_scheme).to_ptsa()
    if demean:
        eeg = eeg.baseline_corrected([rel_start_ms, rel_stop_ms])

    # add mirror buffer if using. PTSA is expecting this to be in seconds.
    if use_mirror_buf:
        eeg = eeg.add_mirror_buffer(buf_ms / 1000.)

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

    # reorder dims to make events first
    eeg = make_events_first_dim(eeg)
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


def compute_power(events, freqs, wave_num, rel_start_ms, rel_stop_ms, buf_ms=1000, elec_scheme=None,
                  noise_freq=[58., 62.], resample_freq=None, mean_over_time=True, log_power=True, loop_over_chans=True,
                  cluster_pool=None, use_mirror_buf=False):
    """
    Returns a TimeSeries object of power values with dimensions 'events' x 'frequency' x 'bipolar_pairs/channels' x
    'time', unless mean_over_time is True, then no 'time' dimenstion.

    Parameters
    ----------
    events: pandas.DataFrame
        An events structure that contains eegoffset and eegfile fields
    freqs: np.array or list
        A set of frequencies at which to compute power using morlet wavelet transform
    wave_num: int
        Width of the wavelet in cycles (I THINK)
    rel_start_ms: int
        Initial time (in ms), relative to the onset of each event
    rel_stop_ms: float
        End time (in ms), relative to the onset of each event
    buf_ms:
        Amount of time (in ms) of buffer to add to both the begining and end of the time interval
    elec_scheme: pandas.DataFrame:
        EHHHHHHHHHHHHHHH
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    resample_freq: float
        Sampling rate to resample to after loading eeg but BEFORE computing power. So be careful. Don't downsample below
        your nyquist.
    mean_over_time: bool
        Whether to mean power over time, and return the power data with no time dimension
    log_power: bool
        Whether to log the power values
    loop_over_chans: bool
        Whether to process each channel independently, or whether to try to do all channels at once. Default is to loop
    cluster_pool: None or ipython cluster helper pool
        If given, will parallelize over channels
    use_mirror_buf: bool
        If True, a mirror buffer will be (used see load_eeg) instead of a normal buffer

    Returns
    -------
    timeseries object of power values

    """

    # warn people if they set the resample_freq too low
    if (resample_freq is not None) and (resample_freq< (np.max(freqs)*2.)):
        print('Resampling EEG below nyquist frequency.')
        warnings.warn('Resampling EEG below nyquist frequency.')

    # make freqs a numpy array if it isn't already because PTSA is kind of stupid and can't handle a list of numbers
    if isinstance(freqs, list):
        freqs = np.array(freqs)

    # We will loop over channels if desired or if we are are using a pool to parallelize
    if cluster_pool or loop_over_chans:

        # LET'S ASSUME THAT CMLREADERS WILL BE UPDATED TO ALLOW FOR BETTER MONOPOLAR SUPPORT.
        if (elec_scheme is not None) and ('contact' in elec_scheme.columns):
            raise NotImplementedError

        # put all the inputs into one list. This is so because it is easier to parallize this way. Parallel functions
        # accept one input. The pool iterates over this list.

        arg_list = [(events, freqs, wave_num, elec_scheme.iloc[r:r + 1], rel_start_ms, rel_stop_ms,
                     buf_ms, noise_freq, resample_freq, mean_over_time, log_power, use_mirror_buf)
                    for r in range(elec_scheme.shape[0])]

        # if no pool, just use regular map
        if cluster_pool is not None:
            pow_list = cluster_pool.map(_parallel_compute_power, arg_list)
        else:
            pow_list = list(map(_parallel_compute_power, tqdm(arg_list, disable=True if len(arg_list) == 1 else False)))

        # This is the stupidest thing in the world. I should just be able to do concat(pow_list, dim='channels') or
        # concat(pow_list, dim='bipolar_pairs'), but for some reason it breaks. I don't know. So I'm creating a new
        # TimeSeries object

        # concatenate data
        chan_dim = pow_list[0].get_axis_num('channel')
        elecs = np.concatenate([x[x.dims[chan_dim]].data for x in pow_list])
        pow_cat = np.concatenate([x.data for x in pow_list], axis=chan_dim)

        # create new coordinates and Timeseries with concatenated data and electrode info
        new_coords = {x: (pow_list[0].coords[x] if x != 'channel' else elecs) for x in pow_list[0].coords.keys()}
        wave_pow = TimeSeries(data=pow_cat, coords=new_coords, dims=pow_list[0].dims)

    # if not looping, sending all the channels at once
    else:
        arg_list = [events, freqs, wave_num, elec_scheme, rel_start_ms, rel_stop_ms, buf_ms, noise_freq,
                    resample_freq, mean_over_time, log_power, use_mirror_buf]
        wave_pow = _parallel_compute_power(arg_list)

    # reorder dims to make events first
    wave_pow = make_events_first_dim(wave_pow)

    return wave_pow


def _parallel_compute_power(arg_list):
    """
    Returns a timeseries object of power values. Accepts the inputs of compute_power() as a single list. Probably
    don't really need to call this directly.
    """

    events, freqs, wave_num, elec_scheme, rel_start_ms, rel_stop_ms, buf_ms, noise_freq, resample_freq, mean_over_time,\
    log_power, use_mirror_buf = arg_list

    # first load eeg
    eeg = load_eeg(events, rel_start_ms, rel_stop_ms, buf_ms=buf_ms, elec_scheme=elec_scheme,
                   noise_freq=noise_freq, resample_freq=resample_freq, use_mirror_buf=use_mirror_buf)

    # then compute power
    wave_pow = MorletWaveletFilter(eeg, freqs, output='power', width=wave_num, cpus=12,
                                   verbose=False).filter()

    # remove the buffer
    wave_pow = wave_pow.remove_buffer(buf_ms / 1000.)

    # are we taking the log?
    if log_power:
        data = wave_pow.data
        wave_pow.data = numexpr.evaluate('log10(data)')

    # mean over time if desired
    if mean_over_time:
        wave_pow = wave_pow.mean(dim='time')
    return wave_pow


def make_events_first_dim(ts, event_dim_str='event'):
    """
    Transposes a TimeSeriesX object to have the events dimension first. Returns transposed object.

    Parameters
    ----------
    ts: TimeSeries
        A PTSA TimeSeries object
    event_dim_str: str
        the name of the event dimension

    Returns
    -------
    TimeSeries
        A transposed version of the orginal timeseries
    """

    # if events is already the first dim, do nothing
    if ts.dims[0] == event_dim_str:
        return ts

    # make sure events is the first dim because I think it is better that way
    ev_dim = np.where(np.array(ts.dims) == event_dim_str)[0]
    new_dim_order = np.hstack([ev_dim, np.setdiff1d(range(ts.ndim), ev_dim)])
    ts = ts.transpose(*np.array(ts.dims)[new_dim_order])
    return ts


def zscore_by_session(ts):
    """
    Returns a numpy array the same shape as the original timeseries, where all the elements have been zscored by
    session

    Returns
    -------
    numpy array
    """
    sessions = ts.events.data['session']
    z_pow = np.empty(ts.shape)
    uniq_sessions = np.unique(sessions)
    for sess in uniq_sessions:
        sess_inds = sessions == sess
        z_pow[sess_inds] = zscore(ts[sess_inds], axis=0)
    return z_pow










#
#
#
# def load_tal(subj, montage=0, bipol=True, use_json=True):
#     """
#     Loads subject electrode information from either bipolar ('pairs') or monopolar ('contacts') database.
#     Returns a numpy recarray with the following fields:
#
#          channel - list of electrode numbers, stores as zero padded strings (because that's how the EEGReader wants it.)
#                    If bipolar, each list entry will have two elements, otherwise just one.
#      anat_region - Freesurfer cortical parcellation
#          loc_tag - Joel Stein's localization tag (mainly for MTL electrodes)
#         tag_name - The clinical electrode tag
#          xyz_avg - x,y,z electrode coordinates, registered to our average brain
#        xyz_indiv - x,y,z electrode coordinates, registered to subject specific brain
#           e_type - G or D or S for grid or depth or strip
#
#     Parameters
#     ----------
#     subj: str
#         Subject code
#     montage: int
#         Montage number of electrode configuration
#     bipol: bool
#         Whether to load the monopolar or bipolar electrode localizations
#     use_json: bool
#         Whether to load the electrode info from the .mat tal structures or the json database
#
#     Returns
#     -------
#     numpy.recarray
#         Recarray containing electrode location information.
#     """
#
#     if use_json:
#
#         # load appropriate json file
#         montage = int(montage)
#         elec_key = 'pairs' if bipol else 'contacts'
#         f_path = reader.aggregate_values(elec_key, subject=subj, montage=montage)
#         elec_json = open(list(f_path)[0], 'r')
#         if montage == 0:
#             elec_data = json.load(elec_json)[subj][elec_key]
#         else:
#             elec_data = json.load(elec_json)[subj + '_' + str(montage)][elec_key]
#         elec_json.close()
#
#         # create empty recarray, then fill it in
#         elec_array = np.recarray(len(elec_data, ), dtype=[('channel', list),
#                                                           ('anat_region', 'U30'),
#                                                           ('loc_tag', 'U30'),
#                                                           ('tag_name', 'U30'),
#                                                           ('xyz_avg', list),
#                                                           ('xyz_indiv', list),
#                                                           ('e_type', 'U1')
#                                                           ])
#
#         # loop over each electrode
#         for i, elec in enumerate(np.sort(list(elec_data.keys()))):
#             elec_array[i]['tag_name'] = elec
#
#             # store channel numbers
#             if bipol:
#                 elec_array[i]['channel'] = [str(elec_data[elec]['channel_1']).zfill(3),
#                                             str(elec_data[elec]['channel_2']).zfill(3)]
#                 elec_array[i]['e_type'] = elec_data[elec]['type_1']
#             else:
#                 elec_array[i]['channel'] = [str(elec_data[elec]['channel']).zfill(3)]
#                 elec_array[i]['e_type'] = elec_data[elec]['type']
#
#             # 'ind' information, subject specific brain
#             if 'ind' in elec_data[elec]['atlases']:
#                 ind = elec_data[elec]['atlases']['ind']
#                 elec_array[i]['anat_region'] = ind['region']
#                 elec_array[i]['xyz_indiv'] = np.array([ind['x'], ind['y'], ind['z']])
#             else:
#                 elec_array[i]['anat_region'] = ''
#                 elec_array[i]['xyz_indiv'] = np.array([np.nan, np.nan, np.nan])
#
#             # 'average' information, average brain
#             if 'avg' in elec_data[elec]['atlases']:
#                 avg = elec_data[elec]['atlases']['avg']
#                 elec_array[i]['xyz_avg'] = np.array([avg['x'], avg['y'], avg['z']])
#             else:
#                 elec_array[i]['xyz_avg'] = np.array([np.nan, np.nan, np.nan])
#
#             # add joel stein loc tags if they exist
#             if 'stein' in elec_data[elec]['atlases']:
#                 loc_tag = elec_data[elec]['atlases']['stein']['region']
#                 if (loc_tag is not None) and (loc_tag != '') and (loc_tag != 'None'):
#                     elec_array[i]['loc_tag'] = loc_tag
#                 else:
#                     elec_array[i]['loc_tag'] = ''
#             else:
#                 elec_array[i]['loc_tag'] = ''
#     else:
#
#         # load appropriate .mat file
#         subj_mont = subj
#         if int(montage) != 0:
#             subj_mont = subj + '_' + str(montage)
#
#         file_str = 'bipol' if bipol else 'monopol'
#         struct_name = 'bpTalStruct' if bipol else 'talStruct'
#         tal_path = os.path.join('/data/eeg', subj_mont, 'tal', subj_mont + '_talLocs_database_' + file_str + '.mat')
#         tal_reader = TalReader(filename=tal_path, struct_name=struct_name)
#         tal_struct = tal_reader.read()
#
#         # get electrode cooridinates
#         xyz_avg = np.array(zip(tal_struct.avgSurf.x_snap, tal_struct.avgSurf.y_snap, tal_struct.avgSurf.z_snap))
#         xyz_indiv = np.array(
#             zip(tal_struct.indivSurf.x_snap, tal_struct.indivSurf.y_snap, tal_struct.indivSurf.z_snap))
#
#         # region based on individual freesurfer parecellation
#         anat_region = tal_struct.indivSurf.anatRegion_snap
#
#         # region based on locTag, if available
#         if 'locTag' in tal_struct.dtype.names:
#             loc_tag = tal_struct.locTag
#         else:
#             loc_tag = np.array(['[]'] * len(tal_struct), dtype='|U256')
#
#         # get bipolar or monopolar channels
#         if bipol:
#             channels = tal_reader.get_bipolar_pairs()
#         else:
#             channels = np.array([str(x).zfill(3) for x in tal_struct.channel])
#
#         elec_array = np.recarray(len(tal_struct.tagName, ), dtype=[('channel', list),
#                                                         ('anat_region', 'U30'),
#                                                         ('loc_tag', 'U30'),
#                                                         ('tag_name', 'U30'),
#                                                         ('xyz_avg', list),
#                                                         ('xyz_indiv', list),
#                                                         ('e_type', 'U1')
#                                                         ])
#
#         # fill in the recarray
#         for i, elec in enumerate(zip(loc_tag, anat_region, tal_struct.tagName,
#                                      xyz_avg, xyz_indiv, tal_struct.eType, channels)):
#             elec_array[i]['loc_tag'] = elec[0]
#             elec_array[i]['anat_region'] = elec[1]
#             elec_array[i]['tag_name'] = elec[2]
#             elec_array[i]['xyz_avg'] = elec[3]
#             elec_array[i]['xyz_indiv'] = elec[4]
#             elec_array[i]['e_type'] = elec[5]
#             elec_array[i]['channel'] = list(elec[6]) if bipol else [elec[6]]
#
#     return elec_array
#
#
# def get_channel_numbers(subj, montage=0, bipol=True, use_json=True):
#     """
#     Wrapper around load_tal() that just returns arrays of channel numbers that you can pass to eeg or power calculations
#     functions.
#
#     Parameters
#     ----------
#     subj: str
#         Subject code
#     montage: int
#         Montage number of electrode configuration
#     bipol: bool
#         Whether to load the monopolar or bipolar electrode localizations
#     use_json: bool
#         Whether to load the electrode info from the .mat tal structures or the json database
#
#     Returns
#     -------
#     numpy.array
#         Array of zero-padded electrode strings. If bipolar, then each element is a list of entries and the type is a
#         recarray because that's how PTSA wants it... If not, each element is the channel string.
#     """
#
#     tal = load_tal(subj, montage=montage, bipol=bipol, use_json=use_json)
#     if bipol:
#         e1 = [chan[0] for chan in tal['channel']]
#         e2 = [chan[1] for chan in tal['channel']]
#         channels = np.array(list(zip(e1, e2)), dtype=[('ch0', '|U3'), ('ch1', '|U3')]).view(np.recarray)
#     else:
#         channels = np.array([chan[0] for chan in tal['channel']])
#     return channels


def load_eeg_full_timeseries(events, monopolar_channels, noise_freq=[58., 62.], bipol_channels=None,
                             resample_freq=None, pass_band=None):
    """
    Function for loading continuous EEG data from a full session, not based on event times.
    Returns a list of EEG TimeSeriesX objects. Each entry is a session's worth of data.

    events: np.recarray
        An events structure that contains eegoffset and eegfile fields
    monopolar_channels: np.array
        Array of zero padded strings indicating the channel numbers of electrodes
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    bipol_channels: np.recarray
        A recarray indicating pairs of channels. If given, monopolar channels will be converted to bipolar.
    resample_freq: float
        Sampling rate to resample to after loading eeg
    pass_band: list
        If given, the eeg will be band pass filtered in the given range

    Returns
    -------
    list
        A list of TimeSeriesX objects.
    """

    # figure out number of eeg files. Usually, one per session, but not always
    eeg_files = np.unique(events.eegfile)
    eeg_list = []
    for f_num, this_eeg_file in enumerate(eeg_files):
        print('%s: loading EEG for eegfile %d of %d.' % (events[0].subject, f_num+1, len(eeg_files)))

        # for each channel loop, either load one channel or two and compute the bipolar pair
        if bipol_channels is None:
            chans = list(zip([np.array(list([x])) for x in monopolar_channels], [None] * len(monopolar_channels)))
        else:
            chans = list(zip([np.array(x.tolist()) for x in bipol_channels],
                             [bipol_channels[x:x + 1] for x in range(len(bipol_channels))]))

        this_eeg = []
        for chan in chans:
            this_eeg_chan = EEGReader(channels=chan[0], session_dataroot=this_eeg_file)
            this_eeg_chan = this_eeg_chan.read()

            # convert to bipolar if desired
            if chan[0].shape[0] == 2:
                this_eeg_chan = MonopolarToBipolarMapper(this_eeg_chan, bipolar_pairs=chan[1]).filter()

            # filter line noise
            if noise_freq is not None:
                b_filter = ButterworthFilter(this_eeg_chan, freq_range=noise_freq, filt_type='stop', order=4)
                this_eeg_chan = b_filter.filter()

            # resample if desired. Note: can be a bit slow especially if have a lot of eeg data
            if resample_freq is not None:
                this_eeg_chan = this_eeg_chan.resampled(resample_freq)

            # do band pass if desired.
            if pass_band is not None:
                this_eeg_chan = band_pass_eeg(this_eeg_chan, pass_band)

            # store this channel
            this_eeg.append(this_eeg_chan)

        # concatenate all the channels
        this_eeg = xr.concat(this_eeg, dim='channels' if bipol_channels is None else 'bipolar_pairs')

        # squeeze away the useless start offset dimension and add this timeseries to our list
        eeg_list.append(this_eeg.squeeze())
    return eeg_list


# def load_eeg(events, monopolar_channels, start_s, stop_s, buf=0.0, noise_freq=[58., 62.],
#              bipol_channels=None, resample_freq=None, pass_band=None, use_mirror_buf=False, demean=False):
#     """
#     Returns an EEG TimeSeriesX object.
#
#     Parameters
#     ----------
#     events: np.recarray
#         An events structure that contains eegoffset and eegfile fields
#     monopolar_channels: np.array
#         Array of zero padded strings indicating the channel numbers of electrodes
#     start_s: float
#         Initial time (in seconds), relative to the onset of each event
#     stop_s: float
#         End time (in seconds), relative to the onset of each event
#     buf:
#         Amount of time (in seconds) of buffer to add to both the begining and end of the time interval
#     noise_freq: list
#         Stop filter will be applied to the given range. Default=(58. 62)
#     bipol_channels: np.recarray
#         A recarray indicating pairs of channels. If given, monopolar channels will be converted to bipolar.
#     resample_freq: float
#         Sampling rate to resample to after loading eeg.
#     pass_band: list
#         If given, the eeg will be band pass filtered in the given range.
#     use_mirror_buf: bool
#         If True, the buffer will be data taken from within the start_s to stop_s interval, mirrored and prepended and
#         appended to the timeseries. If False, data outside the start_s and stop_s interval will be read.
#
#     Returns
#     -------
#     TimeSeriesX
#         EEG TimeSeriesX object with dimensions channels x events x time (or bipolar_pairs x events x time)
#
#         NOTE: The EEG data is returned with time buffer included. If you included a buffer and want to remove it,
#               you may use the .remove_buffer() method of EEGReader.
#
#     """
#
#     # load eeg for given events, channels, and timing parameters
#     eeg_reader = EEGReader(events=events, channels=monopolar_channels, start_time=start_s, end_time=stop_s,
#                            buffer_time=buf if not use_mirror_buf else 0.0)
#     eeg = eeg_reader.read()
#
#     if demean:
#         eeg = eeg.baseline_corrected([start_s, stop_s])
#
#     # add mirror buffer if using
#     if use_mirror_buf:
#         eeg = eeg.add_mirror_buffer(buf)
#
#     # if bipolar channels are given as well, convert the eeg to bipolar
#     if bipol_channels is not None:
#         if len(bipol_channels) > 0:
#             eeg = MonopolarToBipolarMapper(eeg, bipolar_pairs=bipol_channels).filter()
#
#     # filter line noise
#     if noise_freq is not None:
#         if isinstance(noise_freq[0], float):
#             noise_freq = [noise_freq]
#         for this_noise_freq in noise_freq:
#             b_filter = ButterworthFilter(eeg, this_noise_freq, filt_type='stop', order=4)
#             eeg = b_filter.filter()
#
#     # resample if desired. Note: can be a bit slow especially if have a lot of eeg data
#     if resample_freq is not None:
#         eeg = eeg.resampled(resample_freq)
#
#     # do band pass if desired.
#     if pass_band is not None:
#         eeg = band_pass_eeg(eeg, pass_band)
#
#     # reorder dims to make events first
#     eeg = make_events_first_dim(eeg)
#     return eeg


# def compute_power(events, rel_start_ms, rel_stop_ms, buf_ms=0, elec_scheme=None)



