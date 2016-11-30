"""
This module contains a number of functions for loading raw voltage data from electrodes and processing the signals.
"""

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
from scipy.signal import hilbert
from scipy.stats.mstats import zscore
import ram_data_helpers
import behavioral.add_conf_time_to_events
import cPickle as pickle
import cluster_helper.cluster
import pycircstat
import ram_data_helpers
import h5py
from scipy.io import loadmat
from sklearn.preprocessing import Imputer


# parallelizable function to compute power for a single electrode (or electrode pair)
def load_elec_func_watrous_freq(params, events):
    subj = params['subj']

    data_dir = '/home2/andrew.watrous/Results/TH1_FrequencySliding'
    subj_data_dir = os.path.join(data_dir, subj, 'Oct4_2016')
    sessions = np.array(sorted([int(x[8:]) for x in os.listdir(subj_data_dir) if 'Session' in x]))
    ev_sessions = np.unique(events.session)

    if not np.array_equal(sessions, ev_sessions):
        print('%s: sessions not equal' % subj)
        return
    if subj == 'R1201P_1':
        sessions = sessions[[0, 3, 1, 2]]

    fs_data = None
    fs_data_list = []
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    for sess_count, session in enumerate(sessions):
        sess_file = os.path.join(subj_data_dir, 'Session_'+str(session), 'FS_Results.mat')
        f = h5py.File(sess_file)
        fs = f['FS']

        band_file = os.path.join(subj_data_dir, 'Stable_Bands.mat')
        f2 = loadmat(band_file)
        elec_bands = f2['stable_bands'][0]

        # get the indices of all the encoding events
        enc_corr = np.array(f['params']['trialIndices']['enc_corr'])
        enc_incorr = np.array(f['params']['trialIndices']['enc_incorr'])
        enc = np.sort(np.concatenate([enc_corr.flatten(), enc_incorr.flatten()]))

        # get the indices of the time points we are interested in
        eeg_times = np.array(f['params']['eeg']['times'])/1e3
        eeg_inds = (eeg_times > params['start_time']) & (eeg_times < params['end_time'])

        # fs is hdf5 dataset, each entry is a reference to the data for an electrode
        num_elecs = fs.shape[0]
        fs_data_sess = np.empty(shape=(len(enc), num_elecs))
        fs_data_sess = None

        count = 0
        bad_cols = []
        for i, elec in enumerate(fs):

            # load data for this electrode
            fs_data_elec = np.array(f[elec[0]]['FS'])

            # sessions, bands, stable for elec
            sess_elec = elec_bands[i][0][0][2][0]
            bands_elec = elec_bands[i][0][0][1]
            stable_elec = elec_bands[i][0][0][3][0]
            sess_inds = sess_elec == sess_count+1
            good_inds = stable_elec[sess_inds] > 0
            if ~np.any(good_inds):
                continue
            count += np.sum(good_inds)


            # limit to just time points of interest
            elec_data_in_time = fs_data_elec[eeg_inds.flatten()]

            # limit to just events of interest and mean over time. Also only the first band
            # fs_data_sess[:, i] = np.nanmean(elec_data_in_time[:, enc.astype(int), 0], axis=0)
            # if i == 9:
            #     pdb.set_trace()
            # fs_data_mean = np.nanmean(~np.isnan(elec_data_in_time[:, enc.astype(int), :]), axis=0)
            fs_data_mean = np.nanmean(elec_data_in_time[:, enc.astype(int), :], axis=0)
            if len(good_inds) < fs_data_mean.shape[1]:
                pdb.set_trace()
            fs_data_mean = fs_data_mean[:, good_inds]

            # normalize within the range 0 - 1 based on the band
            # elec_band = np.array(f[elec[0]]['freq_bands'])[:, 0]
            # fs_data_sess[:, i] = (fs_data_sess[:, i] - elec_band[0]) / elec_band.ptp()

            fs_data_sess = np.concatenate([fs_data_sess, fs_data_mean], axis=1) if fs_data_sess is not None else fs_data_mean
            # print i, count, fs_data_sess.shape[1]
        # pdb.set_trace()
        # fs_data_sess = imp.fit_transform(fs_data_sess)

        # concatentate the sessions
        # fs_data = np.concatenate([fs_data, fs_data_sess], axis=0) if fs_data is not None else fs_data_sess
        fs_data_list.append(fs_data_sess)
        # make nans zeros? Does that make sense?
        # fs_data[np.isnan(fs_data)] = 0
    bad = [np.where(np.all(np.isnan(x), axis=0))[0] for x in fs_data_list]
    bad = np.unique(np.concatenate(bad))
    for fs_data_sess in fs_data_list:
        if np.any(bad):
            fs_data_sess = np.delete(fs_data_sess, bad, axis=1)
        # pdb.set_trace()
        fs_data_sess = imp.fit_transform(fs_data_sess)
        # pdb.set_trace()
        fs_data = np.concatenate([fs_data, fs_data_sess], axis=0) if fs_data is not None else fs_data_sess

    return fs_data

def load_elec_func_pac(info):
    # pull out the info we need from the input list
    elecs = info[0]
    elec_ind = info[1]
    events = info[2]
    params = info[3]

    if 'subj_save_dir' not in params.keys():
        params['subj_save_dir'] = compute_subj_dir_path(params, events[0].subject)
    if params['bipolar']:
        save_file = os.path.join(params['subj_save_dir'], '%s_%s.p' % (elecs[elec_ind][0], elecs[elec_ind][1]))
    else:
        save_file = os.path.join(params['subj_save_dir'], '%s.p' % (elecs[elec_ind]))

    if os.path.exists(save_file):
        if params['save_chan']:
            return save_file
        else:
            with open(save_file, 'rb') as f:
                elec_data = pickle.load(f)
            pow_elec = elec_data['pow_elec']
            phase_elec = elec_data['phase_elec']
    else:

        # if doing bipolar, load in pair of channels
        if params['bipolar']:

            # load eeg for channels in this bipolar_pair
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
            eeg_reader = EEGReader(events=events, channels=np.array([elecs[elec_ind]]), start_time=params['start_time'],
                                   end_time=params['end_time'], buffer_time=params['buffer_len'])
            eegs = eeg_reader.read()

        # filter 60 Hz line noise
        b_filter = ButterworthFilter(time_series=eegs, freq_range=[58., 62.], filt_type='stop', order=4)
        eegs_filtered = b_filter.filter()


        # filter within band and run hilbert transform
        hb_vals = hilbert(eegs_filtered.filtered(params['hilbert_phase_band'], filt_type='pass', order=2),
                          axis=eegs_filtered.get_axis_num('time'))

        # copy eeg timeseries (to keep metadata) and replace data with hilbert phase
        hb_ts = eegs_filtered.copy(deep=True)
        # hb_ts.data = (np.angle(hb_vals) + 5*np.pi/4) % 2*np.pi
        hb_ts.data = np.angle(hb_vals)
        hb_ts = hb_ts.remove_buffer(duration=params['buffer_len'])

        # now compute power at freqs
        wf = MorletWaveletFilter(time_series=eegs_filtered, freqs=params['freqs'])
        pow_elec, phase_elec = wf.filter()
        pow_elec = pow_elec.remove_buffer(duration=params['buffer_len'])

        # log transform
        np.log10(pow_elec.data, out=pow_elec.data)

        # subtract 1/f
        x = np.log10(params['freqs'])
        A = np.vstack([x, np.ones(len(x))]).T
        y = np.mean(np.mean(pow_elec.data, axis=3), axis=2)
        m, c = np.linalg.lstsq(A, y)[0]
        fit = m * x + c
        pow_elec.data = np.transpose(pow_elec.data.T - fit)

        uniq_sessions = np.unique(events.session)
        # for sess in uniq_sessions:
        #     sess_event_mask = (sessions == sess)
        #     feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

        # bin pow based on phase of hilbert transformed signal
        num_bins = params['num_phase_bins']
        # bins = np.linspace(0, 2 * np.pi, num_bins+1)
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        # inds = np.digitize(hb_ts, bins)
        inds = np.digitize(-hb_ts, bins)


        # only include time points when mean power is above treshhold
        # m = np.mean(pow_elec.data, axis=2).flatten().mean()
        # s = np.mean(pow_elec.data, axis=2).flatten().std()
        # thresh = m + 2*s
        # pow_elec.data[pow_elec.data < thresh] = np.nan

        # hold the binned values. There is surely a better way to do this
        pow_by_phase_bin = np.empty(shape=(pow_elec.shape[0], pow_elec.shape[1], pow_elec.shape[2], num_bins))
        for bin in range(1, num_bins+1):
            bin_inds = inds == bin
            for ev in range(len(pow_elec.events)):
                pow_by_phase_bin[:, :, ev, bin-1] = np.nanmean(pow_elec[:, :, ev, bin_inds[:, ev, :].flatten()], axis=2)
            # for sess in uniq_sessions:
            #     sess_event_mask = (events.session == sess)
            #     pow_by_phase_bin[:, :, sess_event_mask, bin - 1] = zscore(pow_by_phase_bin[:, :, sess_event_mask, bin - 1], axis=2)
        # pdb.set_trace()
        # lastly, bin power by freq_bands
        num_bands = np.shape(params['freq_bands'])[0]
        pow_double_binned = np.empty(shape=(num_bands, pow_by_phase_bin.shape[1], pow_by_phase_bin.shape[2],
                                            pow_by_phase_bin.shape[3]))
        bands = params['freq_bands'] if params['freq_bands'].ndim > 1 else [params['freq_bands']]
        band_array = np.recarray(shape=num_bands, dtype=[('bin_start', 'float'), ('bin_end', 'float')])
        for i, band in enumerate(bands):
            band_array[i] = band
            band_inds = (params['freqs'] >= band[0]) & (params['freqs'] <= band[1])
            pow_double_binned[i, :, :, :] = np.nanmean(pow_by_phase_bin[band_inds], axis=0)


        elec_str = 'bipolar_pairs' if params['bipolar'] else 'channels'
        new_ts = TimeSeriesX(data=pow_double_binned, dims=['bands', elec_str, 'events', 'theta_phase'],
                             coords={'bands': band_array,
                                     elec_str: pow_elec.bipolar_pairs.data if params['bipolar'] else pow_elec.channels.data,
                                     'events': pow_elec.events,
                                     'theta_phase': np.mean(np.vstack([bins[0:-1], bins[1:]]), axis=0) - np.pi/4})

        # either return data or save to file. If save to file, return path to file
        if params['save_chan']:
            with open(save_file, 'wb') as f:
                pickle.dump({'pow_elec': new_ts}, f, protocol=-1)
            return save_file

        return new_ts

# parallelizable function to compute power for a single electrode (or electrode pair)
def load_elec_func(info):
    # pull out the info we need from the input list
    elecs = info[0]
    elec_ind = info[1]
    events = info[2]
    params = info[3]

    if 'subj_save_dir' not in params.keys():
        params['subj_save_dir'] = compute_subj_dir_path(params, events[0].subject)
    if params['bipolar']:
        save_file = os.path.join(params['subj_save_dir'], '%s_%s.p' % (elecs[elec_ind][0], elecs[elec_ind][1]))
    else:
        save_file = os.path.join(params['subj_save_dir'], '%s.p' % (elecs[elec_ind]))

    if os.path.exists(save_file):
        if params['save_chan']:
            return save_file
        else:
            with open(save_file, 'rb') as f:
                elec_data = pickle.load(f)
            pow_elec = elec_data['pow_elec']
            phase_elec = elec_data['phase_elec']
    else:

        # if doing bipolar, load in pair of channels
        if params['bipolar']:

            # load eeg for channels in this bipolar_pair
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
            eeg_reader = EEGReader(events=events, channels=np.array([elecs[elec_ind]]), start_time=params['start_time'],
                                   end_time=params['end_time'], buffer_time=params['buffer_len'])
            eegs = eeg_reader.read()

        # filter 60 Hz line noise
        b_filter = ButterworthFilter(time_series=eegs, freq_range=[58., 62.], filt_type='stop', order=4)
        eegs_filtered = b_filter.filter()

        # resample (downsample) to 500 Hz to speed things up a bit.
        eegs_filtered = eegs_filtered.resampled(500)
        eegs_filtered['samplerate'] = 500.
        # print eegs_filtered['samplerate']

        # compute power and phase
        wf = MorletWaveletFilter(time_series=eegs_filtered, freqs=params['freqs'])
        pow_elec, phase_elec = wf.filter()

        # remove buffer
        pow_elec = pow_elec.remove_buffer(duration=params['buffer_len'])
        phase_elec = phase_elec.remove_buffer(duration=params['buffer_len'])

        # log transform
        np.log10(pow_elec.data, out=pow_elec.data)

        # mean power over time
        if params['time_bins'] is None:
            pow_elec = pow_elec.mean('time')
        else:
            pow_list = []
            for tbin in params['time_bins']:
                inds = (pow_elec.time.data >= tbin[0]) & (pow_elec.time.data < tbin[1])
                pow_list.append(np.mean(pow_elec.data[:, :, :, inds], axis=pow_elec.get_axis_num('time')))
            pow_elec_tbins = np.stack(pow_list, axis=3)
            pow_elec = TimeSeriesX(data=pow_elec_tbins, dims=['frequency', pow_elec.dims[1], 'events', 'time'],
                                 coords={'frequency': pow_elec.frequency,
                                         pow_elec.dims[1]: pow_elec.coords[pow_elec.dims[1]],
                                         'events': pow_elec.events,
                                         'time': params['time_bins'].mean(axis=1)})

        # either return data or save to file. If save to file, return path to file
        if params['save_chan']:
            with open(save_file, 'wb') as f:
                pickle.dump({'pow_elec': pow_elec, 'phase_elec': phase_elec}, f, protocol=-1)
            return save_file

    if params['feat_type'] == 'tilt':
        sessions = pow_elec.events.data['session']
        uniq_sessions = np.unique(sessions)
        tilt = np.empty(shape=pow_elec.data.T.shape)
        for sess in uniq_sessions:
            sess_event_mask = (sessions == sess)
            tilt[sess_event_mask] = zscore(pow_elec.data.T[sess_event_mask], axis=0)
        low_f = params['freqs'] < 16
        tilt = np.mean(tilt[:, :, low_f], axis=2) - np.mean(tilt[:, :, ~low_f], axis=2)
        tilt_ts = TimeSeriesX(data=np.squeeze(tilt), dims=['events'],
                                 coords={'events': pow_elec.events})
        return tilt_ts

    if params['feat_type'] == 'power':
        return pow_elec
    elif params['feat_type'] in ['rbar', 'phase_diff']:
        return phase_elec
    elif params['feat_type'] == 'power_and_rbar':
        return pow_elec, phase_elec


def load_features(subj, task, task_phase, start_time, end_time, time_bins, freqs, freq_bands, hilbert_phase_band,
                  num_phase_bins, bipolar, feat_type, mean_pow, save_chan, subj_save_dir, ROIs, pool, session=None):

    # get electrode numbers and events
    elecs_bipol, elecs_monopol = ram_data_helpers.load_subj_elecs(subj)
    events = ram_data_helpers.load_subj_events(task, subj, task_phase, session, False if bipolar else True)

    # construct input to main prcoessing function
    elecs = elecs_bipol if bipolar else elecs_monopol

    # filter ROIs
    loc_tag, anat_region, chan_tags = ram_data_helpers.load_subj_elec_locs(subj, bipolar)
    if ROIs is not None:
        loc_dict = ram_data_helpers.bin_elec_locs(loc_tag, anat_region)
        roi_elecs = np.array([False]*len(loc_tag))
        if isinstance(ROIs, str):
            ROIs = [ROIs]
        for roi in ROIs:
            roi_elecs = roi_elecs | loc_dict[roi]
        elecs = elecs[roi_elecs]
        loc_tag = loc_tag[roi_elecs]
        anat_region = anat_region[roi_elecs]
        chan_tags = chan_tags[roi_elecs]
    num_elecs = len(elecs)
    elec_range = range(num_elecs)

    # info is the big list that map() iterates over
    keys = ['freqs', 'task_phase', 'start_time', 'end_time', 'bipolar', 'feat_type', 'buffer_len', 'save_chan',
            'subj_save_dir', 'mean_pow', 'freq_bands', 'hilbert_phase_band', 'num_phase_bins', 'subj', 'time_bins']
    vals = [freqs, task_phase, start_time, end_time, bipolar, feat_type, 3.0, save_chan, subj_save_dir, mean_pow,
            freq_bands, hilbert_phase_band, num_phase_bins, subj, time_bins]
    params = {k: v for (k, v) in zip(keys, vals)}
    info = zip([elecs] * num_elecs, elec_range, [events] * num_elecs, [params] * num_elecs)

    # if we have an open pool, send the iterable to the workers
    if feat_type == 'pow_by_phase':
        if pool is None:
            feature_list = map(load_elec_func_pac, info)
        else:
            feature_list = pool.map(load_elec_func_pac, info)

        # cat the list into ndarray of freq x elecs x events
        pow_features = np.concatenate(feature_list, axis=1)

        # new time series object
        elec_str = 'bipolar_pairs' if bipolar else 'channels'
        new_ts = TimeSeriesX(data=pow_features, dims=feature_list[0].dims,
                             coords={'bands': feature_list[0].bands,
                                     elec_str: elecs,
                                     'events': feature_list[0].events,
                                     'theta_phase': feature_list[0].theta_phase},
                             attrs={'loc_tag': loc_tag,
                                    'anat_region': anat_region,
                                    'chan_tags': chan_tags,
                                    'start_time': start_time,
                                    'end_time': end_time})
        return new_ts
    elif feat_type == 'freq_sliding':
        fs_data = load_elec_func_watrous_freq(params, events)
        features = TimeSeriesX(data=fs_data.T, dims=['bipolar_pairs', 'events'],
                               coords={'bipolar_pairs': range(fs_data.shape[1]),
                                       'events': events},
                               attrs={'start_time': start_time,
                                      'end_time': end_time})
    else:
        if pool is None:
            feature_list = map(load_elec_func, info)
        else:
            feature_list = pool.map(load_elec_func, info)

        if feat_type == 'power':
            if time_bins is None:
                features = mean_power_features(subj, feature_list, start_time, end_time, freqs, bipolar, elecs)
            else:
                features = mean_power_features_tbins(subj, feature_list, time_bins, freqs, bipolar, elecs)
        elif feat_type == 'tilt':
            features = tilt_features(subj, feature_list, start_time, end_time, freqs, bipolar, elecs)
        elif (feat_type == 'rbar') or (feat_type == 'phase_diff'):
            features = pairwise_features(feature_list, start_time, end_time, freqs, bipolar, elecs, events, pool, feat_type)
        elif feat_type == 'power_and_rbar':
            power_feature_list = [x[0] for x in feature_list]
            features_power = mean_power_features(subj, power_feature_list, start_time, end_time, freqs, bipolar, elecs, events, pool)

            phase_feature_list = [x[1] for x in feature_list]
            features_rbar = pairwise_features(phase_feature_list, start_time, end_time, freqs, bipolar, elecs, events, pool, feat_type)

            features = combine_pow_and_rbar(features_power, features_rbar, start_time, end_time, freqs, elecs, events)

    return features


def combine_pow_and_rbar(features_power, features_rbar, start_time, end_time, freqs, elecs, events):
    combined_features = np.concatenate((features_power, features_rbar), axis=1)
    # new time series object
    new_ts = TimeSeriesX(data=combined_features, dims=['frequency', 'elec_ind', 'events'],
                         coords={'frequency': freqs,
                                 'elec_ind': range(combined_features.shape[1]),
                                 'events': events,
                                 'start_time': start_time,
                                 'end_time': end_time})
    return new_ts


def tilt_features(subj, feature_list, start_time, end_time, freqs, bipolar, elecs):

    # cat the list into ndarray events x elecs
    tilt_features = np.stack(feature_list, axis=1)

    # load electrode location info and add as a timeseries attribute
    loc_tag, anat_region, chan_tags = ram_data_helpers.load_subj_elec_locs(subj, bipolar)

    # new time series object
    elec_str = 'bipolar_pairs' if bipolar else 'channels'
    new_ts = TimeSeriesX(data=tilt_features.T, dims=[elec_str, 'events'],
                         coords={elec_str: elecs,
                                 'events': feature_list[0].events,
                                 'start_time': start_time,
                                 'end_time': end_time},
                         attrs={'loc_tag': loc_tag,
                                'anat_region': anat_region,
                                'chan_tags': chan_tags})
    return new_ts


def mean_power_features(subj, feature_list, start_time, end_time, freqs, bipolar, elecs):
    # cat the list into ndarray of freq x elecs x events
    pow_features = np.concatenate(feature_list, axis=1)

    # load electrode location info and add as a timeseries attribute
    loc_tag, anat_region, chan_tags = ram_data_helpers.load_subj_elec_locs(subj, bipolar)

    # new time series object
    elec_str = 'bipolar_pairs' if bipolar else 'channels'
    new_ts = TimeSeriesX(data=pow_features, dims=['frequency', elec_str, 'events'],
                         coords={'frequency': freqs,
                                 elec_str: elecs,
                                 'events': feature_list[0].events,
                                 'start_time': start_time,
                                 'end_time': end_time},
                         attrs={'loc_tag': loc_tag,
                                'anat_region': anat_region,
                                'chan_tags': chan_tags})
    return new_ts


def mean_power_features_tbins(subj, feature_list, time_bins, freqs, bipolar, elecs):
    # cat the list into ndarray of freq x elecs x events
    pow_features = np.concatenate(feature_list, axis=1)

    # load electrode location info and add as a timeseries attribute
    loc_tag, anat_region, chan_tags = ram_data_helpers.load_subj_elec_locs(subj, bipolar)

    # new time series object
    elec_str = 'bipolar_pairs' if bipolar else 'channels'
    new_ts = TimeSeriesX(data=pow_features, dims=['frequency', elec_str, 'events', 'time'],
                         coords={'frequency': freqs,
                                 elec_str: elecs,
                                 'events': feature_list[0].events,
                                 'time': time_bins.mean(axis=1)},
                         attrs={'loc_tag': loc_tag,
                                'anat_region': anat_region,
                                'chan_tags': chan_tags,
                                'window_size': time_bins.ptp(axis=1)})
    return new_ts


def pairwise_features(feature_list, start_time, end_time, freqs, bipolar, elecs, events, pool, feat_type):
    from itertools import combinations

    if feat_type in ['rbar', 'power_and_rbar']:
        feat_func = rbar_par
    elif feat_type == 'phase_diff':
        feat_func = phase_diff_par

    # compute all combinations of electrodes
    combs = combinations(range(len(elecs)), 2)
    combs = [list(x) for x in combs]

    # cat feature list into one array freqs x elecs x events x time
    # phase_over_time = np.concatenate(feature_list, axis=1)
    # return phase_over_time
    # create empty array to hold rbar values. also keep track of pair labels
    num_pairs = len(elecs) * (len(elecs) - 1) / 2
    features = np.empty((feature_list[0].shape[0], feature_list[0].shape[2], num_pairs), dtype=np.float32)

    # there must be a more efficient way to do this.
    # loop over each pair of electrodes in combs. I want to parallelize the pairwise calculations, but I can't just
    # stick all the freqs x 2 x events x time arrays into one list. Do it in chucks of 50, whic still takes up a lot
    # of memory
    chunks = [combs[i:i + 25] for i in xrange(0, len(combs), 25)]
    start = 0
    for i, chunk in enumerate(chunks):
        print 'Processing chunk %d of %d.' % (i + 1, len(chunks))

        stop = start + len(chunk)
        feature_chunk = []
        for pair in chunk:
            # this feels kind of stupid, having to put all of this into one object in memory
            feature_chunk.append(np.concatenate([feature_list[pair[0]], feature_list[pair[1]]], axis=1))

        # process chunk
        if pool is None:
            rbar_chunk = map(feat_func, feature_chunk)
        else:
            rbar_chunk = pool.map(feat_func, feature_chunk)

        features[:, :, start:stop] = np.stack(rbar_chunk, axis=2)
        start = stop

    # transpose to be consistent with output of other features functions
    features = np.transpose(features, [0, 2, 1])

    # create new list of electrodes pairs corresponding to new dim
    # this doesn't work sometimes wiht less than 10 elecs?
    #elec_pairs = elecs[combs]
    elec_pairs = [list(elecs[comb]) for comb in combs]
    if bipolar:
            elec_pairs_list = [list(i[0]) + list(i[1]) for i in elec_pairs]
            elec_pairs_array = np.recarray(shape=(len(combs)),
                                           dtype=[('ch0', '|S3'), ('ch1', '|S3'), ('ch2', '|S3'), ('ch3', '|S3')])
    else:
        elec_pairs_list = elec_pairs
        elec_pairs_array = np.recarray(shape=(len(combs)),
                                       dtype=[('ch0', '|S3'), ('ch1', '|S3')])
    for i, all_elecs in enumerate(elec_pairs_list):
        elec_pairs_array[i] = tuple(all_elecs)

    # put combs instead of elec nums in coords?

    # new time series object
    elec_str = 'bipolar_pairs' if bipolar else 'channels'
    new_ts = TimeSeriesX(data=features, dims=['frequency', 'elec_pairs', 'events'],
                         coords={'frequency': freqs,
                                 'elec_pairs': elec_pairs_array,
                                 'events': events,
                                 'start_time': start_time,
                                 'end_time': end_time})#,
                                 #elec_str: elecs})
    return new_ts


def rbar_par(pair_data):
    phase_diff = pair_data[:, 0, :, :] - pair_data[:, 1, :, :]
    rbar = pycircstat.resultant_vector_length(phase_diff, axis=2)
    return rbar

def phase_diff_par(pair_data):
    from time import time

    phase_diff = pair_data[:, 0, :, :] - pair_data[:, 1, :, :]
    phase_diff_mean = pycircstat.mean(phase_diff, axis=2)
    return phase_diff_mean


def compute_subj_dir_path(params, subj_str):
    # directory to save data
    f1 = params['freqs'][0]
    f2 = params['freqs'][-1]
    subj_save_dir = os.path.join(params['save_dir'], '%d_freqs_%.1f_%.1f' % (len(params['freqs']), f1, f2),
                                 '%s_start_%.1f_stop_%.1f' % (params['task_phase'], params['start_time'],
                                                              params['end_time']), subj_str)
    return subj_save_dir


