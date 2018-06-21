import RAM_helpers
import numpy as np
# import seaborn as sns
# import numexpr
# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd
import joblib
import pdb
# import os
import h5py

from tqdm import tqdm
# from xarray import concat
from sklearn.metrics import roc_auc_score, roc_curve
# from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp
from ptsa.data.TimeSeriesX import TimeSeriesX
# from SubjectLevel.Analyses import subject_SME
from SubjectLevel.Analyses.subject_classifier_by_region import SubjectClassifier as sc
from scipy.stats import ttest_rel, levene, fisher_exact, zscore, zmap, ttest_1samp


def fr6_stim_info(events, elec_info):
    """
    Figure out a few things about the catFR6/FR6 events.

    - What electrode pairs were stimulated
    - What type of stim list is it? A, B, or A+B?
    """

    # make sure alll elec_info strings are upper
    for e_ind in range(len(elec_info.tag_name)):
        elec_info[e_ind].tag_name = elec_info[e_ind].tag_name.upper()

    stim_on = events[(events.type == 'STIM_ON') & (events.stim_list == 1)]
    non_stim = events[(events.type == 'WORD') & (events.phase == 'NON-STIM')]
    sessions = np.unique(events.session)
    stim_info = {}
    for sess in sessions:
        sess_ev = events[(events.type == 'WORD') & (events.session == sess)]
        stim_on_sess = stim_on[stim_on.session == sess]

        # get unique stim params
        anode_label = stim_on_sess.stim_params['anode_label']
        cathode_label = stim_on_sess.stim_params['cathode_label']
        stim_sites, indices = np.unique(np.concatenate([anode_label, cathode_label], 1), axis=0, return_inverse=True)

        # if there aren't 3 unqiue stim params, then somthint is weird
        if stim_sites.shape[0] != 3:
            print('%s: something wrong?' % events[0].subject)
            continue
        # return

        # now we will label lists as being stim locaiton A, B, or both
        stim_info[sess] = {}
        ab = ['B', 'A']
        stim_site_trial_labels = np.empty(indices.shape, dtype='U2')
        stim_site_trial_nums = np.empty(indices.shape, dtype=int)
        for i, this_stim_site in enumerate(stim_sites):
            if len(list(filter(None, this_stim_site))) == 2:
                this_label = ab.pop()
            elif len(list(filter(None, this_stim_site))) == 4:
                this_label = 'AB'
            stim_site_trial_labels[indices == i] = this_label
            stim_site_trial_nums[indices == i] = stim_on_sess[indices == i].list

        # get electrode tag and region A
        A_tag = anode_label[stim_site_trial_labels == 'A'][0, 0] + '-' + cathode_label[stim_site_trial_labels == 'A'][
            0, 0]
        A_tag = A_tag.upper()
        A_elec_info = elec_info[elec_info.tag_name == A_tag]
        A_elec_str = A_elec_info.loc_tag[0] if A_elec_info.loc_tag else A_elec_info.anat_region[0]
        A_region = get_elec_region(A_elec_str, A_tag, events[0].subject)
        A_hemi = 'right' if A_elec_info.xyz_indiv[0][0] > 0 else 'left'

        # get electrode tag and region B
        B_tag = anode_label[stim_site_trial_labels == 'B'][0, 0] + '-' + cathode_label[stim_site_trial_labels == 'B'][
            0, 0]
        B_tag = B_tag.upper()
        B_elec_info = elec_info[elec_info.tag_name == B_tag]
        B_elec_str = B_elec_info.loc_tag[0] if B_elec_info.loc_tag else B_elec_info.anat_region[0]
        B_region = get_elec_region(B_elec_str, B_tag, events[0].subject)
        B_hemi = 'right' if B_elec_info.xyz_indiv[0][0] > 0 else 'left'

        # store the info for this session
        stim_info[sess]['stim_site_trial_labels'] = stim_site_trial_labels
        stim_info[sess]['stim_site_trial_nums'] = stim_site_trial_nums
        stim_info[sess]['stim_events'] = stim_on_sess
        stim_info[sess]['A_tag'] = A_tag
        stim_info[sess]['A_region'] = A_region
        stim_info[sess]['A_hemi'] = A_hemi
        stim_info[sess]['B_tag'] = B_tag
        stim_info[sess]['B_region'] = B_region
        stim_info[sess]['B_hemi'] = B_hemi

        # also store non-stim events
        stim_info[sess]['sham_events'] = non_stim[non_stim.session == sess]
        stim_info[sess]['session_events'] = sess_ev

        # finally add the recalled info to the stim events
        for stim_ev in range(stim_info[sess]['stim_events'].shape[0]):
            word = stim_info[sess]['stim_events'][stim_ev].item_name
            recalled = sess_ev[sess_ev.item_name == word].recalled
            sess_ev[sess_ev.item_name == word][0].is_stim = 1
            stim_info[sess]['stim_events'][stim_ev].recalled = recalled
            stim_info[sess]['session_events'].is_stim[sess_ev.item_name == word] = 1

    return stim_info


# i think i should rework this whole thing
def process_subj(task, subj):
    # load events, electrode_info
    events = RAM_helpers.load_subj_events(task=task, subj=subj[0], montage=subj[1])
    elec_info = RAM_helpers.load_tal(subj=subj[0], montage=subj[1], bipol=True, use_json=True)
    stim_info = fr6_stim_info(events, elec_info)

    # sham_timing
    sham_pre_start = 0.85
    sham_pre_end = 1.45
    sham_post_start = 2.05
    sham_post_end = 2.65

    # loop over each session in stim_info
    res = {}
    for sess in stim_info:
        res[sess] = {}
        stim_info_sess = stim_info[sess]
        print('%s: stim A: %s (%s), stim B: %s (%s)' % (subj[0], stim_info_sess['A_tag'],
                                                        stim_info_sess['A_region'],
                                                        stim_info_sess['B_tag'],
                                                        stim_info_sess['B_region']))

        # These are all ENS subjects. get the actual channel data from the h5 file
        eegfile = np.unique(events.eegfile)[0]
        with h5py.File(eegfile, 'r') as f:

            bipolar_channels = np.array([])
            monopolar_channels = np.array([])
            orig_chan_tags = np.array(f['bipolar_info']['contact_name'])
            elec_info_sess = modify_tal_for_ens(elec_info, orig_chan_tags)
            mono_chans = np.array(['{:03d}'.format(x).encode() for x in f['/ports'][:]])

            for e_ind in range(len(elec_info_sess.tag_name)):
                elec_info_sess[e_ind].tag_name = elec_info_sess[e_ind].tag_name.upper()
        elec_regions, elec_hemis = get_region_and_hemi_for_all_elecs(elec_info_sess)

        # load eeg to determine artifact channels
        print('%s: loading eeg for stim artifact detection for session %d.' % (subj[0], sess))
        ev_fields = ['mstime', 'eegoffset', 'eegfile', 'session']
        evs_for_eeg = stim_info_sess['stim_events'][ev_fields]
        eeg = RAM_helpers.load_eeg(evs_for_eeg,
                                   np.array([]), -0.6, 1.1, use_mirror_buf=False,
                                   bipol_channels=np.array([]), buf=0.0, noise_freq=None, demean=True)

        # find channels with saturation
        # print('dumping')
        # return eeg
        # joblib.dump(eeg, '/home1/jfm2/whyyyyyyyyyyyyyy.p')
        print('sat events')
        sat_events = find_sat_events(eeg)
        print('{0}: {1:d} channel/events bad.'.format(subj[0], sat_events.sum()))

        # find channels with unequal means or variances pre to post
        pre_range = [-.5, -.1]
        post_range = [.6, 1.0]
        ps, lev_ps = get_bad_chans_pre_post(eeg, pre_range, post_range, sat_events)

        # combine tests to get array of bad channels
        #         bad_chans = (ps<0.01) | (lev_ps<0.01) | (sat_events.mean(axis=0) > .95)
        bad_chans_with_voltage_test = (ps < 0.01) | (sat_events.mean(axis=0) > .95)
        # bad_chans = (ps < 0.001) | (sat_events.mean(axis=0) > .95)
        bad_chans = sat_events.mean(axis=0) > .95
        print('{0}: {1:.2f}% of channels bad.'.format(subj[0], np.sum(bad_chans) / len(bad_chans) * 100))

        # build two classifiers.
        # 1: with bad channels excluded
        # 2: with all channels (excluding sitm channels only)
        stim_elecs = (elec_info_sess.tag_name == stim_info_sess['A_tag']) | (
        elec_info_sess.tag_name == stim_info_sess['B_tag'])
        subj_classifier_bad_chans = load_subject_classifier(subj, bad_chans | stim_elecs)
        subj_classifier_all_chans = load_subject_classifier(subj, stim_elecs)
        freqs = subj_classifier_all_chans.freqs

        ########
        ###### THINK ABOUT THE NORMALIZATION HERE... Pre and post normalized independently may not be right...
        ######## but maybe it is
        ####### maybe pre and post should be normalized together. concat and normalize
        #########

        # compute pre stim power (could just load a longer period of eeg earler)
        # and pass the timeseries into MorletWaveletFilterCpp, but this is slightly easier / less efficient
        print('{0}: computing pre-stim power.'.format(subj[0]))
        evs_for_pow = stim_info_sess['stim_events'][ev_fields]
        powers_pre = RAM_helpers.compute_power(evs_for_pow, freqs, 5, np.array([]), -.65, -.05,
                                               bipol_channels=None,
                                               mean_over_time=False, use_mirror_buf=True, buf=.59)
        powers_pre = powers_pre[:, :, :, (powers_pre.time >= -.6) & (powers_pre.time <= -.1)].mean(dim='time')

        print('{0}: computing post-stim power.'.format(subj[0]))
        powers_post = RAM_helpers.compute_power(evs_for_pow, freqs, 5, np.array([]), .550, 1.15,
                                                bipol_channels=None,
                                                mean_over_time=False, use_mirror_buf=True, buf=.59)
        powers_post = powers_post[:, :, :, (powers_post.time >= 0.6) & (powers_post.time <= 1.1)].mean(dim='time')
        zmap_data = np.concatenate([powers_pre, powers_post], axis=0)
        powers_pre = zmap(powers_pre, zmap_data, axis=0)
        powers_post = zmap(powers_post, zmap_data, axis=0)

        # load power during encoding for all items
        enc_events = stim_info_sess['session_events'][stim_info_sess['session_events']['phase'] != 'BASELINE']
        enc_events = stim_info_sess['session_events'][stim_info_sess['session_events']['list'] != -1]
        enc_events_for_pow = enc_events[ev_fields]
        arg_list = [(enc_events_for_pow,
                     freqs,
                     5, np.array([chan]),
                     0.0,
                     1.366,
                     1.355,
                     [[58., 62.], [118., 122.], [178., 182.]],
                     np.array([]),
                     None,
                     True,
                     True,
                     True) for chan in mono_chans]
        print('Computing encoding power')
        pow_list = list(map(RAM_helpers._parallel_compute_power, tqdm(arg_list, disable=True if len(arg_list) == 1 else False)))
        # pow_list = list(map(RAM_helpers._parallel_compute_power, arg_list))
        chan_str = 'bipolar_pairs' if 'bipolar_pairs' in pow_list[0].dims else 'channels'
        chan_dim = pow_list[0].get_axis_num(chan_str)
        print('Concatenating power')
        elecs = np.concatenate([x[x.dims[chan_dim]].data for x in pow_list])
        pow_cat = np.concatenate([x.data for x in pow_list], axis=chan_dim)
        coords = pow_list[0].coords
        print('Done')

        print('getting elecs')
        coords[chan_str] = elecs
        print('Making new TS of power')
        enc_power = TimeSeriesX(data=pow_cat, coords=coords, dims=pow_list[0].dims)
        # print('dumping')
        # joblib.dump(enc_power, '/home1/jfm2/whyyyyyyyyyyyyyy.p')
        print('make events first')
        enc_power = RAM_helpers.make_events_first_dim(enc_power)
        print('zscoring')
        enc_power = RAM_helpers.zscore_by_session(enc_power)
        print('Done')

        # apply classifier
        # region_elecs_all_chans = subj_classifier_all_chans.res['all']['region_elecs']
        # region_elecs_bad_chans = subj_classifier_bad_chans.res['all']['region_elecs']

        # with all channels
        powers_pre_all_chans = powers_pre[:, :, ~stim_elecs]  # [:, :, region_elecs_all_chans]
        powers_post_all_chans = powers_post[:, :, ~stim_elecs]  # [:, :, region_elecs_all_chans]
        pre_probs_all_chans = subj_classifier_all_chans.res['all']['model'].predict_proba(
            powers_pre_all_chans.reshape(powers_pre_all_chans.shape[0], -1))[:, 1]
        post_probs_all_chans = subj_classifier_all_chans.res['all']['model'].predict_proba(
            powers_post_all_chans.reshape(powers_post_all_chans.shape[0], -1))[:, 1]

        # and with bad channels excluded
        powers_pre_bad_chans = powers_pre[:, :, ~(bad_chans | stim_elecs)]  # [:, :, region_elecs_bad_chans]
        powers_post_bad_chans = powers_post[:, :, ~(bad_chans | stim_elecs)]  # [:, :, region_elecs_bad_chans]
        pre_probs_bad_chans = subj_classifier_bad_chans.res['all']['model'].predict_proba(
            powers_pre_bad_chans.reshape(powers_pre_bad_chans.shape[0], -1))[:, 1]
        post_probs_bad_chans = subj_classifier_bad_chans.res['all']['model'].predict_proba(
            powers_post_bad_chans.reshape(powers_post_bad_chans.shape[0], -1))[:, 1]

        # adn with bad channels with the voltage test included
        powers_pre_bad_volt_chans = powers_pre[:, :, ~(bad_chans_with_voltage_test | stim_elecs)]  # [:, :, region_elecs_bad_chans]
        powers_post_bad_volt_chans = powers_post[:, :, ~(bad_chans_with_voltage_test | stim_elecs)]  # [:, :, region_elecs_bad_chans]

        # apply classifier to all encoding events
        enc_power_all_chans = enc_power[:, :, ~stim_elecs]
        enc_probs_all_chans = subj_classifier_all_chans.res['all']['model'].predict_proba(
            enc_power_all_chans.reshape(enc_power_all_chans.shape[0], -1))[:, 1]
        non_stim_auc = roc_auc_score(enc_events[enc_events.stim_list == 0].recalled,
                                     enc_probs_all_chans[enc_events.stim_list == 0])
        print(non_stim_auc)

        # apply classifier to all encoding events excluding saturated channels
        enc_power_bad_chans = enc_power[:, :, ~(bad_chans | stim_elecs)]
        enc_probs_bad_chans = subj_classifier_bad_chans.res['all']['model'].predict_proba(
            enc_power_bad_chans.reshape(enc_power_bad_chans.shape[0], -1))[:, 1]
        non_stim_auc_bad_chans = roc_auc_score(enc_events[enc_events.stim_list == 0].recalled,
                                               enc_probs_bad_chans[enc_events.stim_list == 0])
        print(non_stim_auc_bad_chans)


        # power change from pre to post
        power_change_all_chans = powers_post_all_chans - powers_pre_all_chans
        all_chan_regions = np.array(list(map('-'.join, zip(elec_hemis, elec_regions))))[~stim_elecs]
        power_change_bad_chans = powers_post_bad_chans - powers_pre_bad_chans
        bad_chan_regions = np.array(list(map('-'.join, zip(elec_hemis, elec_regions))))[~(bad_chans | stim_elecs)]
        power_change_bad_volt_chans = powers_post_bad_volt_chans - powers_pre_bad_volt_chans
        bad_volt_chan_regions = np.array(list(map('-'.join, zip(elec_hemis, elec_regions))))[~(bad_chans_with_voltage_test | stim_elecs)]


        res[sess]['A_tag'] = stim_info_sess['A_tag']
        res[sess]['A_region'] = stim_info_sess['A_region']
        res[sess]['A_hemi'] = stim_info_sess['A_hemi']
        res[sess]['B_tag'] = stim_info_sess['B_tag']
        res[sess]['B_region'] = stim_info_sess['B_region']
        res[sess]['B_hemi'] = stim_info_sess['B_hemi']
        res[sess]['pre_probs_all_chans'] = pre_probs_all_chans
        res[sess]['post_probs_all_chans'] = post_probs_all_chans
        res[sess]['pre_probs_bad_chans'] = pre_probs_bad_chans
        res[sess]['post_probs_bad_chans'] = post_probs_bad_chans
        res[sess]['power_change_all_chans'] = power_change_all_chans
        res[sess]['power_change_bad_chans'] = power_change_bad_chans
        res[sess]['power_change_bad_volt_chans'] = power_change_bad_volt_chans
        res[sess]['all_chan_regions'] = all_chan_regions
        res[sess]['bad_chan_regions'] = bad_chan_regions
        res[sess]['bad_volt_chan_regions'] = bad_volt_chan_regions
        res[sess]['recalled'] = stim_info_sess['stim_events'].recalled
        res[sess]['recalled_all_items'] = stim_info_sess['session_events'].recalled
        res[sess]['is_stim'] = enc_events.is_stim
        res[sess]['lists_all_items'] = stim_info_sess['session_events'].list
        res[sess]['stim_site_trial_labels'] = stim_info_sess['stim_site_trial_labels']
        res[sess]['stim_site_trial_nums'] = stim_info_sess['stim_site_trial_nums']
        res[sess]['enc_probs_all_chans'] = enc_probs_all_chans
        res[sess]['enc_probs_bad_chans'] = enc_probs_bad_chans
        res[sess]['stim_list_field'] = enc_events.stim_list
        res[sess]['subj'] = subj[0]
        res[sess]['non_stim_auc'] = non_stim_auc
        res[sess]['non_stim_auc_bad_chans'] = non_stim_auc_bad_chans

    return res




def modify_tal_for_ens(tal, chan_tags):

    orig_chan_tags = chan_tags.astype(str)
    elec_array = np.recarray(len(orig_chan_tags, ), dtype=[('channel', list),
                                                          ('anat_region', 'U30'),
                                                          ('loc_tag', 'U30'),
                                                          ('tag_name', 'U30'),
                                                          ('xyz_avg', list),
                                                          ('xyz_indiv', list),
                                                          ('e_type', 'U1')
                                                          ])

    for i, this_tag in enumerate(orig_chan_tags):
        elec_tal = tal[tal['tag_name'] == this_tag]
        if len(elec_tal) > 0:
            for field in elec_array.dtype.names:
                elec_array[i][field] = elec_tal[field][0]
        else:
            elec_array[i]['xyz_indiv'] = np.array([np.nan, np.nan, np.nan])
            elec_array[i]['xyz_avg'] = np.array([np.nan, np.nan, np.nan])
            elec_array[i]['anat_region'] = ''
            elec_array[i]['loc_tag'] = ''
            elec_array[i]['e_type'] = ''
            elec_array[i]['tag_name'] = this_tag
            elec_array[i]['channel'] = ['', '']

    return elec_array


def load_subject_classifier(subj, bad_chans=None):
    # load classifier for a given subject. Classifier will be based on whichever combination of
    # FR1 and catFR1 data is present

    tasks = ['RAM_FR1', 'RAM_catFR1']
    if subj[0] == 'R1247P':
        tasks = ['RAM_catFR1']

    subj_joint = []
    for task in tasks:
        subj_classifier = SubjectClassifier(subject=subj[0], montage=subj[1], task=task)
        if subj_classifier.subj is not None:
            subj_classifier.freqs = np.logspace(np.log10(6), np.log10(180), 8)
            subj_classifier.do_not_compute = False
            subj_classifier.start_time = [0.0]
            subj_classifier.end_time = [1.366]
            subj_classifier.freqs = np.logspace(np.log10(6), np.log10(180), 8)
            subj_joint.append(subj_classifier)
    if len(subj_joint) == 2:
        subj_classifier = create_joint_subj(subj_joint[0], subj_joint[1])
    else:
        subj_classifier = subj_joint[0]
        subj_classifier.load_data()
        subj_classifier.save_data()

    subj_classifier.regions_to_classify = ['all', 'TC', 'FC', 'IPC', 'SPC', 'OC', ['MTL', 'Hipp'], ['TC', 'FC']]
    subj_classifier.verbose = False
    subj_classifier.load_res_if_file_exists = False
    subj_classifier.save_res = False
    subj_classifier.classify_by_hemi = True
    if bad_chans is not None:
        subj_classifier.remove_bad_chans_from_data(bad_chans)
    subj_classifier.run()
    return subj_classifier


def create_joint_subj(subj_obj1, subj_obj2):
    # load data from each exp
    subj_obj1.load_data()
    subj_obj1.save_data()
    subj_obj2.load_data()
    subj_obj2.save_data()

    # modify session numbers of events in one of the experiments
    subj2_events = subj_obj2.subject_data.events.data
    subj2_events['session'] += 100

    # concat events with just desired (common) fields
    fields = ['protocol', 'session', 'subject', 'item_name', 'mstime', 'type', 'eegoffset', 'recalled', 'list',
              'eegfile', 'msoffset']
    new_events = np.concatenate([subj_obj1.subject_data.events.data[fields], subj2_events[fields]])

    # concat data
    new_data = np.concatenate([subj_obj1.subject_data.data, subj_obj2.subject_data.data])

    # create new timeseries object
    if 'time' in subj_obj1.subject_data.dims:
        new_ts = TimeSeriesX(data=new_data, dims=['events', 'frequency', subj_obj1.subject_data.dims[2], 'time'],
                             coords={'events': new_events,
                                     'frequency': subj_obj1.freqs,
                                     subj_obj1.subject_data.dims[2]: subj_obj1.subject_data[
                                         subj_obj1.subject_data.dims[2]].data,
                                     'time': subj_obj1.subject_data['time'].data,
                                     'samplerate': subj_obj1.subject_data.samplerate})
    else:
        new_ts = TimeSeriesX(data=new_data, dims=['events', 'frequency', subj_obj1.subject_data.dims[2]],
                             coords={'events': new_events,
                                     'frequency': subj_obj1.freqs,
                                     subj_obj1.subject_data.dims[2]: subj_obj1.subject_data[
                                         subj_obj1.subject_data.dims[2]].data,
                                     'samplerate': subj_obj1.subject_data.samplerate})

    if 'orig_chan_tags' in subj_obj1.subject_data.attrs:
        new_ts.attrs['orig_chan_tags'] = subj_obj1.subject_data.attrs['orig_chan_tags']

    # new task phase array
    new_task_phase = np.concatenate([subj_obj1.task_phase, subj_obj2.task_phase])

    # make new subject object
    new_subj = subj_obj1
    new_subj.subject_data = new_ts
    new_subj.task_phase = new_task_phase
    #     new_subj.task = '+'.join([subj_obj1.task, subj_obj2.task])
    return new_subj


def find_sat_events(eegs):
    sat_events = np.zeros([eegs.shape[0], eegs.shape[1]])

    for i in range(eegs.shape[0]):
        for j in range(eegs.shape[1]):

            ts = eegs[i, j].data
            changes = np.array(ts[1:] != ts[:-1])
            changes_inds = np.append(np.where(changes), len(ts) - 1)
            run_lengths = np.diff(np.append(-1, changes_inds))

            if (run_lengths > 10).any():
                sat_events[i, j] = 1

    return sat_events.astype(bool)


def get_bad_chans_pre_post(eeg, pre_range, post_range, bad_events):
    # mean eeg over pre and post intervals
    pre_eeg = eeg[:, :, (eeg.time >= pre_range[0]) & (eeg.time <= pre_range[1])].mean(dim='time').data
    post_eeg = eeg[:, :, (eeg.time >= post_range[0]) & (eeg.time <= post_range[1])].mean(dim='time').data
    if np.any(bad_events):
        pre_eeg[bad_events] = np.nan
        post_eeg[bad_events] = np.nan

    # ttest comparing pre and post. one value per channel
    ts, ps = ttest_rel(post_eeg, pre_eeg, nan_policy='omit')

    # levene for equal variance
    lev_ts = []
    lev_ps = []
    for chan in zip(post_eeg.T, pre_eeg.T):
        chan_t, chan_p = levene(chan[0][~np.isnan(chan[0])], chan[1][~np.isnan(chan[1])], center='mean')
        lev_ts.append(chan_t)
        lev_ps.append(chan_p)

    return np.array(ps), np.array(lev_ps)


class SubjectClassifier(sc):
    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectClassifier, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

        # adding parameter for bad channels
        self.bad_chans = []

    def analysis(self, permute=False):
        """
        Does the actual classification. I wish I could simplify this a bit, but, it's got a lot of steps to do. Maybe
        break some of this out into seperate functions
        """
        if not self.cross_val_dict:
            print('Cross validation labels must be computed before running classifier. Use .make_cross_val_labels()')
            return

        # The bias correct only works for subjects with multiple sessions ofdata, see comment below.
        if (len(np.unique(self.subject_data.events.data['session'])) == 1) & (len(self.C) > 1):
            print('Multiple C values cannot be tested for a subject with only one session of data.')
            return

        # Get class labels
        Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)
        if permute:
            Y = np.random.permutation(Y)

        # normalize data by session if the features are oscillatory power
        if self.feat_type == 'power':
            X = self.normalize_power(self.subject_data.data)

        # revert C value to default C value not multi session subejct
        Cs = self.C
        loso = True
        if len(np.unique(self.subject_data.events.data['session'])) == 1:
            Cs = SubjectClassifier.default_C
            loso = False

        if self.classify_by_hemi:
            for k in list(self.elec_locs.keys()):
                if k != 'is_right':
                    self.elec_locs['right-{}'.format(k)] = self.elec_locs[k] & self.elec_locs['is_right']
                    self.elec_locs['left-{}'.format(k)] = self.elec_locs[k] & ~self.elec_locs['is_right']
            regions_to_classify = []
            for r in self.regions_to_classify:
                if isinstance(r, str) and r != 'all':
                    regions_to_classify.append('{}-{}'.format('right', r))
                    regions_to_classify.append('{}-{}'.format('left', r))
                elif isinstance(r, list):
                    regions_to_classify.append(['{}-{}'.format('right', x) for x in r])
                    regions_to_classify.append(['{}-{}'.format('left', x) for x in r])
                else:
                    regions_to_classify.append(r)
        else:
            regions_to_classify = self.regions_to_classify

        self.res = {}
        for region in regions_to_classify:
            region_str = region if isinstance(region, str) else '-'.join(region)
            region_elecs = []
            if isinstance(region, list) and np.any([np.any(self.elec_locs[r]) for r in region]):
                region_elecs = np.any(np.stack([self.elec_locs[r] for r in region]), axis=0)
            elif isinstance(region, str):
                region_elecs = self.elec_locs[region] if region != 'all' else np.ones(X.shape[-1], dtype=bool)
            if np.any(region_elecs):
                if self.verbose:
                    print('%s: classifying %s.' % (self.subj, region_str))
                region_x = X[:, :, region_elecs].reshape(X.shape[0], -1)
                res = self.run_classifier_for_region(region_x, Y, Cs, loso)
                res['region'] = region
                res['n'] = np.sum(region_elecs)
                res['region_elecs'] = region_elecs
                self.res[region_str] = res
            else:
                if self.verbose:
                    print('%s: no %s elecs.' % (self.subj, region_str))

    def remove_bad_chans_from_data(self, bad_chan_bool):
        self.subject_data = self.subject_data[:, :, ~bad_chan_bool]
        for key in self.elec_locs.keys():
            self.elec_locs[key] = np.delete(self.elec_locs[key], np.where(bad_chan_bool), axis=0)


def get_elec_region(elec_str, stim_tag, subj):
    stim_info = {}
    stim_info['R1170J'] = {}
    stim_info['R1170J']['L7ST7-L7ST8'] = 'IPC'
    stim_info['R1223E'] = {}
    stim_info['R1223E']['7LD7-7LD9'] = 'TC'
    if subj in stim_info:
        key = stim_info[subj][stim_tag]
        return key

    # hipp_tags = ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
    #                  'Right CA3', 'Right DG', 'Right Sub']
    mtl_tags = ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC', 'Left CA1', 'Left CA2',
                'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2', 'Right CA3', 'Right DG', 'Right Sub',
                'Left MTL WM', 'Right MTL WM', 'Left PRC', 'Right PRC']
    #     mtl_tags = ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC']
    ifg_tags = ['parsopercularis', 'parsorbitalis', 'parstriangularis']
    mfg_tags = ['caudalmiddlefrontal', 'rostralmiddlefrontal']
    sfg_tags = ['superiorfrontal']
    fc_tags = ['parsopercularis', 'parsorbitalis', 'parstriangularis', 'caudalmiddlefrontal', 'rostralmiddlefrontal',
               'superiorfrontal', 'Left DLPFC', 'Right DLPFC', 'Left Caudal Middle Frontal Cor',
               'Right Caudal Middle Frontal Cor',
               'Right Superior Frontal Gyrus', 'Left Superior Frontal Gyrus']
    tc_tags = ['superiortemporal', 'middletemporal', 'inferiortemporal', 'Left TC', 'Right TC',
               'Left Middle Temporal Gyrus', 'Right Middle Temporal Gyrus', 'Left STG', 'Right STG']
    ipc_tags = ['inferiorparietal', 'supramarginal', 'Right Supramarginal Gyrus', 'Left Supramarginal Gyrus']
    spc_tags = ['superiorparietal', 'precuneus']
    oc_tags = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']

    loc_dict = {'MTL-Hipp': mtl_tags,
                'FC': fc_tags,
                'TC': tc_tags,
                'IPC': ipc_tags,
                'SPC': spc_tags,
                'OC': oc_tags}

    for key in loc_dict.keys():
        if elec_str in loc_dict[key]:
            return key
    return ''


def get_region_and_hemi_for_all_elecs(elec_info_sess):
    regions = []
    hemi = []
    for this_elec_info in elec_info_sess:
        elec_str = this_elec_info.loc_tag if this_elec_info.loc_tag else this_elec_info.anat_region
        regions.append(get_elec_region(elec_str,None,None))
        hemi.append('right' if this_elec_info.xyz_indiv[0] > 0 else 'left')
    return np.array(regions), np.array(hemi)