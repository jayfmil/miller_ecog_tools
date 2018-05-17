import numexpr
import json
import joblib
import os
import re
import warnings
# import cluster_helper.cluster
import numpy as np
import xarray as xr
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers import EEGReader
from ptsa.data.readers.tal import TalReader
from ptsa.data.readers.index import JsonIndexReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp
from ptsa.data.TimeSeriesX import TimeSeriesX
from scipy.stats.mstats import zscore
from tqdm import tqdm
from glob import glob

# load json database of subject information. Doing this on import because it's not
# super fast, and I don't want to do it each call to get_subjs or whatever functions need it
try:
    reader = JsonIndexReader('/protocols/r1.json')
except(IOError):
    print('JSON protocol file not found')



def get_subjs_and_montages(task, use_json=True):
    """Returns list of subjects who performed a given task, along with the montage numbers.

    Parameters
    ----------
    task: str
        The experiment name (ex: RAM_TH1, RAM_FR1, ...).
    use_json: bool
        Whether to look for subjects in the json database or matlab database

    Returns
    -------
    numpy.array
        Array of arrays. Each subarray has two elements, the subject code and the montage number.
    """

    out = []
    if use_json:
        subjs = reader.subjects(experiment=task.replace('RAM_', ''))
        for subj in subjs:
            m = reader.aggregate_values('montage', subject=subj, experiment=task.replace('RAM_', ''))
            out.extend(zip([subj] * len(m), m))
    else:
        subjs = glob(os.path.join('/data/events/', task, 'R*_events.mat'))
        subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
        subjs.sort()
        for subj in subjs:
            m = subj[-1] if '_' in subj else '0'
            out.extend(zip([subj], m))
    return np.array(out)


def load_subj_events(task, subj, montage=0, use_json=True, use_reref_eeg=False):
    """Returns subject event structure.

    Parameters
    ----------
    task: str
        The experiment name (ex: RAM_TH1, RAM_FR1, ...).
    subj: str
        Subject code
    montage: int
        Montage number of electrode configuration (0 is the most common)
    use_json: bool
        Whether to load the matlab events or the json events
    use_reref_eeg: bool
        Whether the eeg.eegfile field of the events structure should point to the refef directory.
        You generally want this to be true if you are NOT using bipolar referencing.
        NOTE: This has no effect for the json events. json events have no reref data.

    Returns
    -------
    numpy.recarray
        Subject event structure

    """

    if not use_json:
        subj_file = subj + '_events.mat'
        if int(montage) != 0:
            subj_file = subj + '_' + str(montage) + '_events.mat'
        subj_ev_path = str(os.path.join('/data/events/', task, subj_file))
        e_reader = BaseEventReader(filename=subj_ev_path, eliminate_events_with_no_eeg=True,
                                   use_reref_eeg=use_reref_eeg)
        events = e_reader.read()
    else:
        event_paths = reader.aggregate_values('task_events', subject=subj, montage=montage,
                                              experiment=task.replace('RAM_', ''))
        events = [BaseEventReader(filename=path).read() for path in sorted(event_paths)]
        events = np.concatenate(events)
        events = events.view(np.recarray)
    events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]
    return events


def load_tal(subj, montage=0, bipol=True, use_json=True):
    """
    Loads subject electrode information from either bipolar ('pairs') or monopolar ('contacts') database.
    Returns a numpy recarray with the following fields:

         channel - list of electrode numbers, stores as zero padded strings (because that's how the EEGReader wants it.)
                   If bipolar, each list entry will have two elements, otherwise just one.
     anat_region - Freesurfer cortical parcellation
         loc_tag - Joel Stein's localization tag (mainly for MTL electrodes)
        tag_name - The clinical electrode tag
         xyz_avg - x,y,z electrode coordinates, registered to our average brain
       xyz_indiv - x,y,z electrode coordinates, registered to subject specific brain
          e_type - G or D or S for grid or depth or strip

    Parameters
    ----------
    subj: str
        Subject code
    montage: int
        Montage number of electrode configuration
    bipol: bool
        Whether to load the monopolar or bipolar electrode localizations
    use_json: bool
        Whether to load the electrode info from the .mat tal structures or the json database

    Returns
    -------
    numpy.recarray
        Recarray containing electrode location information.
    """

    if use_json:

        # load appropriate json file
        montage = int(montage)
        elec_key = 'pairs' if bipol else 'contacts'
        f_path = reader.aggregate_values(elec_key, subject=subj, montage=montage)
        elec_json = open(list(f_path)[0], 'r')
        if montage == 0:
            elec_data = json.load(elec_json)[subj][elec_key]
        else:
            elec_data = json.load(elec_json)[subj + '_' + str(montage)][elec_key]
        elec_json.close()

        # create empty recarray, then fill it in
        elec_array = np.recarray(len(elec_data, ), dtype=[('channel', list),
                                                          ('anat_region', 'U30'),
                                                          ('loc_tag', 'U30'),
                                                          ('tag_name', 'U30'),
                                                          ('xyz_avg', list),
                                                          ('xyz_indiv', list),
                                                          ('e_type', 'U1')
                                                          ])

        # loop over each electrode
        for i, elec in enumerate(np.sort(list(elec_data.keys()))):
            elec_array[i]['tag_name'] = elec

            # store channel numbers
            if bipol:
                elec_array[i]['channel'] = [str(elec_data[elec]['channel_1']).zfill(3),
                                            str(elec_data[elec]['channel_2']).zfill(3)]
                elec_array[i]['e_type'] = elec_data[elec]['type_1']
            else:
                elec_array[i]['channel'] = [str(elec_data[elec]['channel']).zfill(3)]
                elec_array[i]['e_type'] = elec_data[elec]['type']

            # 'ind' information, subject specific brain
            if 'ind' in elec_data[elec]['atlases']:
                ind = elec_data[elec]['atlases']['ind']
                elec_array[i]['anat_region'] = ind['region']
                elec_array[i]['xyz_indiv'] = np.array([ind['x'], ind['y'], ind['z']])
            else:
                elec_array[i]['anat_region'] = ''
                elec_array[i]['xyz_indiv'] = np.array([np.nan, np.nan, np.nan])

            # 'average' information, average brain
            if 'avg' in elec_data[elec]['atlases']:
                avg = elec_data[elec]['atlases']['avg']
                elec_array[i]['xyz_avg'] = np.array([avg['x'], avg['y'], avg['z']])
            else:
                elec_array[i]['xyz_avg'] = np.array([np.nan, np.nan, np.nan])

            # add joel stein loc tags if they exist
            if 'stein' in elec_data[elec]['atlases']:
                loc_tag = elec_data[elec]['atlases']['stein']['region']
                if (loc_tag is not None) and (loc_tag != '') and (loc_tag != 'None'):
                    elec_array[i]['loc_tag'] = loc_tag
                else:
                    elec_array[i]['loc_tag'] = ''
            else:
                elec_array[i]['loc_tag'] = ''
    else:

        # load appropriate .mat file
        subj_mont = subj
        if int(montage) != 0:
            subj_mont = subj + '_' + str(montage)

        file_str = 'bipol' if bipol else 'monopol'
        struct_name = 'bpTalStruct' if bipol else 'talStruct'
        tal_path = os.path.join('/data/eeg', subj_mont, 'tal', subj_mont + '_talLocs_database_' + file_str + '.mat')
        tal_reader = TalReader(filename=tal_path, struct_name=struct_name)
        tal_struct = tal_reader.read()

        # get electrode cooridinates
        xyz_avg = np.array(zip(tal_struct.avgSurf.x_snap, tal_struct.avgSurf.y_snap, tal_struct.avgSurf.z_snap))
        xyz_indiv = np.array(
            zip(tal_struct.indivSurf.x_snap, tal_struct.indivSurf.y_snap, tal_struct.indivSurf.z_snap))

        # region based on individual freesurfer parecellation
        anat_region = tal_struct.indivSurf.anatRegion_snap

        # region based on locTag, if available
        if 'locTag' in tal_struct.dtype.names:
            loc_tag = tal_struct.locTag
        else:
            loc_tag = np.array(['[]'] * len(tal_struct), dtype='|U256')

        # get bipolar or monopolar channels
        if bipol:
            channels = tal_reader.get_bipolar_pairs()
        else:
            channels = np.array([str(x).zfill(3) for x in tal_struct.channel])

        elec_array = np.recarray(len(tal_struct.tagName, ), dtype=[('channel', list),
                                                        ('anat_region', 'U30'),
                                                        ('loc_tag', 'U30'),
                                                        ('tag_name', 'U30'),
                                                        ('xyz_avg', list),
                                                        ('xyz_indiv', list),
                                                        ('e_type', 'U1')
                                                        ])

        # fill in the recarray
        for i, elec in enumerate(zip(loc_tag, anat_region, tal_struct.tagName,
                                     xyz_avg, xyz_indiv, tal_struct.eType, channels)):
            elec_array[i]['loc_tag'] = elec[0]
            elec_array[i]['anat_region'] = elec[1]
            elec_array[i]['tag_name'] = elec[2]
            elec_array[i]['xyz_avg'] = elec[3]
            elec_array[i]['xyz_indiv'] = elec[4]
            elec_array[i]['e_type'] = elec[5]
            elec_array[i]['channel'] = list(elec[6]) if bipol else [elec[6]]

    return elec_array


def get_channel_numbers(subj, montage=0, bipol=True, use_json=True):
    """
    Wrapper around load_tal() that just returns arrays of channel numbers that you can pass to eeg or power calculations
    functions.

    Parameters
    ----------
    subj: str
        Subject code
    montage: int
        Montage number of electrode configuration
    bipol: bool
        Whether to load the monopolar or bipolar electrode localizations
    use_json: bool
        Whether to load the electrode info from the .mat tal structures or the json database

    Returns
    -------
    numpy.array
        Array of zero-padded electrode strings. If bipolar, then each element is a list of entries and the type is a
        recarray because that's how PTSA wants it... If not, each element is the channel string.
    """

    tal = load_tal(subj, montage=montage, bipol=bipol, use_json=use_json)
    if bipol:
        e1 = [chan[0] for chan in tal['channel']]
        e2 = [chan[1] for chan in tal['channel']]
        channels = np.array(list(zip(e1, e2)), dtype=[('ch0', '|U3'), ('ch1', '|U3')]).view(np.recarray)
    else:
        channels = np.array([chan[0] for chan in tal['channel']])
    return channels


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
                this_eeg_chan = MonopolarToBipolarMapper(time_series=this_eeg_chan, bipolar_pairs=chan[1]).filter()

            # filter line noise
            if noise_freq is not None:
                b_filter = ButterworthFilter(time_series=this_eeg_chan, freq_range=noise_freq, filt_type='stop', order=4)
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


def load_eeg(events, monopolar_channels, start_s, stop_s, buf=0.0, noise_freq=[58., 62.],
             bipol_channels=None, resample_freq=None, pass_band=None, use_mirror_buf=False, demean=False):
    """
    Returns an EEG TimeSeriesX object.

    Parameters
    ----------
    events: np.recarray
        An events structure that contains eegoffset and eegfile fields
    monopolar_channels: np.array
        Array of zero padded strings indicating the channel numbers of electrodes
    start_s: float
        Initial time (in seconds), relative to the onset of each event
    stop_s: float
        End time (in seconds), relative to the onset of each event
    buf:
        Amount of time (in seconds) of buffer to add to both the begining and end of the time interval
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    bipol_channels: np.recarray
        A recarray indicating pairs of channels. If given, monopolar channels will be converted to bipolar.
    resample_freq: float
        Sampling rate to resample to after loading eeg.
    pass_band: list
        If given, the eeg will be band pass filtered in the given range.
    use_mirror_buf: bool
        If True, the buffer will be data taken from within the start_s to stop_s interval, mirrored and prepended and
        appended to the timeseries. If False, data outside the start_s and stop_s interval will be read.

    Returns
    -------
    TimeSeriesX
        EEG TimeSeriesX object with dimensions channels x events x time (or bipolar_pairs x events x time)

        NOTE: The EEG data is returned with time buffer included. If you included a buffer and want to remove it,
              you may use the .remove_buffer() method of EEGReader.

    """

    # load eeg for given events, channels, and timing parameters
    eeg_reader = EEGReader(events=events, channels=monopolar_channels, start_time=start_s, end_time=stop_s,
                           buffer_time=buf if not use_mirror_buf else 0.0)
    eeg = eeg_reader.read()

    if demean:
        eeg = eeg.baseline_corrected([start_s, stop_s])

    # add mirror buffer if using
    if use_mirror_buf:
        eeg = eeg.add_mirror_buffer(buf)

    # if bipolar channels are given as well, convert the eeg to bipolar
    if bipol_channels is not None:
        if len(bipol_channels) > 0:
            eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=bipol_channels).filter()

    # filter line noise
    if noise_freq is not None:
        b_filter = ButterworthFilter(time_series=eeg, freq_range=noise_freq, filt_type='stop', order=4)
        eeg = b_filter.filter()

    # resample if desired. Note: can be a bit slow especially if have a lot of eeg data
    if resample_freq is not None:
        eeg = eeg.resampled(resample_freq)

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
    eeg: TimeSeriesX
        A TimeSeriesX object created with the EEGReader
    freq_range: list
        List of two floats defining the range to filter in
    order: int
        Order of butterworth filter

    Returns
    -------
    TimeSeriesX
        Filtered EEG object
    """
    return ButterworthFilter(time_series=eeg, freq_range=freq_range, filt_type='pass', order=order).filter()


def compute_power(events, freqs, wave_num, monopolar_channels, start_s, stop_s, buf=1.0, noise_freq=[58., 62.],
                  bipol_channels=None, resample_freq=None, mean_over_time=True, log_power=True, loop_over_chans=True,
                  cluster_pool=None, use_mirror_buf=False):
    """
    Returns a TimeSeriesX object of power values with dimensions 'events' x 'frequency' x 'bipolar_pairs/channels' x
    'time', unless mean_over_time is True, then no 'time' dimenstion.

    Parameters
    ----------
    events: np.recarray
        An events structure that contains eegoffset and eegfile fields
    freqs: np.array or list
        A set of frequencies at which to compute power using morlet wavelet transform
    wave_num: int
        Width of the wavelet in cycles (I THINK)
    monopolar_channels: np.array
        Array of zero padded strings indicating the channel numbers of electrodes
    start_s: float
        Initial time (in seconds), relative to the onset of each event
    stop_s: float
        End time (in seconds), relative to the onset of each event
    buf:
        Amount of time (in seconds) of buffer to add to both the begining and end of the time interval
    noise_freq: list
        Stop filter will be applied to the given range. Default=(58. 62)
    bipol_channels: np.recarray
        A recarray indicating pairs of channels. If given, monopolar channels will be converted to bipolar.
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
    TimeSeriesX object of power values

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

        # create the channel inputs
        if bipol_channels is None:
            if len(monopolar_channels) == 0:
                chans = [[np.array([]), np.array([])]]
            else:
                chans = zip([np.array(list([x])) for x in monopolar_channels], [None] * len(monopolar_channels))
        else:
            chans = zip([np.array(x.tolist()) for x in bipol_channels], [bipol_channels[x:x+1] for x in range(len(bipol_channels))])

        # put all the inputs into one list. This is so because it is easier to parallize this way. Parallel functions
        # accept one input. The pool iterates over this list.
        arg_list = [(events, freqs, wave_num, chan[0], start_s, stop_s, buf, noise_freq,
                    chan[1], resample_freq, mean_over_time, log_power, use_mirror_buf) for chan in chans]

        # if no pool, just use regular map
        if cluster_pool is not None:
            pow_list = cluster_pool.map(_parallel_compute_power, arg_list)
        else:
            pow_list = list(map(_parallel_compute_power, tqdm(arg_list, disable=True if len(arg_list) == 1 else False)))

        # This is the stupidest thing in the world. I should just be able to do concat(pow_list, dim='channels') or
        # concat(pow_list, dim='bipolar_pairs'), but for some reason it breaks. I don't know. So I'm creating a new
        # TimeSeriesX object
        chan_str = 'bipolar_pairs' if 'bipolar_pairs' in pow_list[0].dims else 'channels'
        chan_dim = pow_list[0].get_axis_num(chan_str)
        elecs = np.concatenate([x[x.dims[chan_dim]].data for x in pow_list])
        pow_cat = np.concatenate([x.data for x in pow_list], axis=chan_dim)
        coords = pow_list[0].coords

        coords[chan_str] = elecs
        wave_pow = TimeSeriesX(data=pow_cat, coords=coords, dims=pow_list[0].dims)

    # if not looping, sending all the channels at once
    else:
        arg_list = [events, freqs, wave_num, monopolar_channels, start_s, stop_s, buf, noise_freq,
                    bipol_channels, resample_freq, mean_over_time, log_power]
        wave_pow = _parallel_compute_power(arg_list)

    # reorder dims to make events first
    wave_pow = make_events_first_dim(wave_pow)

    return wave_pow


def _parallel_compute_power(arg_list):
    """
    Returns a TimeSeriesX object of power values. Accepts the inputs of compute_power() as a single list. Probably
    don't really need to call this directly.
    """

    events, freqs, wave_num, monopolar_channels, start_s, stop_s, buf, noise_freq, bipol_channels, resample_freq, \
    mean_over_time, log_power, use_mirror_buf = arg_list

    # first load eeg
    eeg = load_eeg(events, monopolar_channels, start_s, stop_s, buf, noise_freq, bipol_channels, resample_freq,
                   use_mirror_buf=use_mirror_buf)

    # then compute power
    wave_pow, _ = MorletWaveletFilterCpp(time_series=eeg, freqs=freqs, output='power', width=wave_num, cpus=20,
                                         verbose=False).filter()

    # remove the buffer
    wave_pow = wave_pow.remove_buffer(buf)

    # are we taking the log?
    if log_power:
        data = wave_pow.data
        wave_pow.data = numexpr.evaluate('log10(data)')

    # mean over time if desired
    if mean_over_time:
        wave_pow = wave_pow.mean(dim='time')
    return wave_pow


def make_events_first_dim(ts):
    """
    Transposes a TimeSeriesX object to have the events dimension first. Returns transposed object.

    Parameters
    ----------
    ts: TimeSeriesX
        A PTSA TimeSeriesX object

    Returns
    -------
    TimeSeriesX
        A transposed version of the orginal timeseries
    """

    # if events is already the first dim, do nothing
    if ts.dims[0] == 'events':
        return ts

    # make sure events is the first dim because I think it is better that way
    ev_dim = np.where(np.array(ts.dims) == 'events')[0]
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
