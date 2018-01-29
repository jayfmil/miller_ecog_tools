"""
A variety of helper functions for working with RAM data.
"""

import re
import os
import numpy as np
from glob import glob
from copy import deepcopy
from numpy.lib.recfunctions import stack_arrays
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from ptsa.data.readers.ParamsReader import ParamsReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from numpy.lib.recfunctions import append_fields, merge_arrays
import behavioral.add_conf_time_to_events
import behavioral.make_move_events
import pdb
import json
from scipy.stats import ttest_1samp, ttest_ind

import platform
basedir = ''
if platform.system() == 'Darwin':
    basedir = '/Users/jmiller/python'
try:
    reader = JsonIndexReader(basedir + '/protocols/r1.json')
except(IOError):
    print('JSON protocol file not found')


# This file contains a bunch of helper functions for
def load_subj_events(task, subj, montage=0, task_phase=['enc'], session=None, use_reref_eeg=False, use_json=True):
    """Returns subject event structure."""

    if not use_json:
        subj_file = subj + '_events.mat'
        if int(montage) != 0:
            subj_file = subj + '_' + str(montage) + '_events.mat'
        subj_ev_path = str(os.path.join(basedir+'/data/events/', task, subj_file))
        e_reader = BaseEventReader(filename=subj_ev_path, eliminate_events_with_no_eeg=True, use_reref_eeg=use_reref_eeg)
        events = e_reader.read()
    else:
        # reader = JsonIndexReader(basedir+'/protocols/r1.json')
        event_paths = reader.aggregate_values('task_events', subject=subj, montage=montage, experiment=task.replace('RAM_', ''))
        events = [BaseEventReader(filename=path).read() for path in sorted(event_paths)]
        events = np.concatenate(events)
        events = events.view(np.recarray)

    if task == 'RAM_TH1':

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]

        # add some new fields to events
        error_percentiles = calc_norm_dist_error(events.locationX, events.locationY, events.distErr)
        if not use_json:
            events = append_fields(events, 'norm_err', error_percentiles, dtypes=float, usemask=False, asrecarray=True)
        else:
            events = merge_arrays([events, np.array(error_percentiles, dtype=[('norm_err', float)])], flatten=True,
                                  asrecarray=True)

        events = calc_min_dist_to_any_chest(events)
        # events = behavioral.add_conf_time_to_events.process_event_file(events)

        # filter to our task phase(s) of interest
        phase_list = task_phase if isinstance(task_phase, list) else [task_phase]
        ev_list = []
        for phase in phase_list:

            if phase == 'chest':
                # filter to just item presentation events
                ev_list.append(events[(events.type == 'CHEST')])

            if phase == 'enc':
                # filter to just item presentation events
                ev_list.append(events[(events.type == 'CHEST') & (events.confidence >= 0)])

            elif phase == 'rec':
                # filter to just recall probe events
                ev_list.append(events[(events.type == 'REC') & (events.confidence >= 0)])
            elif phase == 'move':
                ev = events[(events.type == 'CHEST')]
                move_ev = behavioral.make_move_events.process_event_file(ev, use_json)
                ev_list.append(move_ev)

            elif phase == 'rec_circle':
                circle_events = events[(events.type == 'REC') & (events.confidence >= 0)]

                sessions = circle_events.session
                uniq_sessions = np.unique(sessions)
                for sess in uniq_sessions:
                    sess_inds = sessions == sess
                    ev_sess = circle_events[sess_inds]
                    dataroot = ev_sess[0].eegfile
                    p_reader = ParamsReader(dataroot=dataroot)
                    params = p_reader.read()
                    samplerate = params['samplerate']
                    circle_events.eegoffset[sess_inds] += (ev_sess.reactionTime / 1e3 * samplerate).astype(int)
                    circle_events.eegoffset[sess_inds] += (ev_sess.reactionTime / 1e3 * samplerate).astype(int)
                circle_events.mstime = circle_events.mstime + circle_events.reactionTime
                ev_list.append(circle_events)

            elif phase == 'rec_choice':
                choice_events = events[(events.type == 'REC') & (events.confidence >= 0)]

                sessions = choice_events.session
                uniq_sessions = np.unique(sessions)
                for sess in uniq_sessions:
                    sess_inds = sessions == sess
                    ev_sess = choice_events[sess_inds]
                    dataroot = ev_sess[0].eegfile
                    p_reader = ParamsReader(dataroot=dataroot)
                    params = p_reader.read()
                    samplerate = params['samplerate']
                    choice_events.eegoffset[sess_inds] += (ev_sess.choice_rt / 1e3 * samplerate).astype(int)
                choice_events.mstime = choice_events.mstime + choice_events.choice_rt
                ev_list.append(choice_events)

            elif phase == 'both':
                events = events[((events.type == 'REC') | (events.type == 'CHEST')) & (events.confidence >= 0)]
                # events[events.type == 'REC'].mstime = events[events.type == 'REC'].mstime + events[events.type == 'REC'].reactionTime

        # concatenate the different types of events if needed
        if len(ev_list) == 1:
            events = ev_list[0]
        else:
            events = stack_arrays(ev_list, asrecarray=True, usemask=False)

        # make sure events are in time order. this doesn't really matter
        # ev_order = np.argsort(events, order=('session', 'trial', 'mstime'))
        ev_order = np.argsort(events, order=('session', 'mstime'))
        events = events[ev_order]

    elif task in ['RAM_THR', 'RAM_THR1']:
        # filter to our task phase(s) of interest
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]
        phase_list = task_phase if isinstance(task_phase, list) else [task_phase]
        ev_list = []
        for phase in phase_list:

            if phase == 'enc':
                # filter to just item presentation events
                ev_list.append(events[events.type == 'CHEST'])

            elif phase == 'rec_probe':
                # filter to just recall probe events
                ev_list.append(events[events.type == 'PROBE'])

            elif phase == 'rec_word':
                # filter to just recall spoken events
                rec_events = events[events.type == 'REC_EVENT']

                probe_events = events[events.type == 'PROBE']

                # this finds all all times there was more than one response for an item. Or I could just loop over
                # each list, that might be better
                good_evs = []
                uniq_sessions = np.unique(rec_events.session)
                for sess in uniq_sessions:
                    sess_inds = rec_events.session == sess
                    repeats = np.array([True if x in set(rec_events[sess_inds].item_name[:i]) else False for i, x in
                                        enumerate(rec_events[sess_inds].item_name)])

                    # also find all '<>', which are vocalizations. Get rid of those too
                    vocs = rec_events[sess_inds].resp_word == '<>'

                    # make sure the first TRICK is excluded too. The repeats line above should catch the rest
                    trick = rec_events[sess_inds].item_name == 'TRICK'

                    # also exclude based on IRT. This is essentially just making just there was no vocalization just
                    # prior to the recall event
                    irts = np.concatenate([[10000], np.diff(rec_events[sess_inds].mstime)]) < 2000

                    bad = repeats | vocs | trick | irts

                    # item probes without responses
                    sess_probes = probe_events[probe_events.session == sess]
                    no_voc_items = ~np.in1d(sess_probes.item_name, np.unique(rec_events[sess_inds].item_name))

                    # for incorrect recalls, get mstimes and eegoffsets
                    incorr = rec_events[sess_inds].recalled == 0

                    # for probes without responses, create surragate events using the same time offsets as the incorr
                    surrogate_evs = []
                    if np.any(no_voc_items):
                        for ind in np.where(no_voc_items)[0]:

                            tmp_ev = deepcopy(sess_probes[ind:ind+1])
                            tmp_ev.type = 'REC_EVENT'
                            tmp_ev.resp_word = 'DUMMY'
                            tmp_ev.recalled = 0

                            # pick random incorrect data
                            # x = np.random.randint(0, incorr_mstimes.shape[0])
                            x = np.random.choice(np.where(incorr)[0])

                            # get probe mstime and eegoffset for this response
                            probe_ind = (sess_probes.item_name == rec_events[sess_inds][x].item_name) & \
                                        (sess_probes.trial == rec_events[sess_inds][x].trial)
                            this_probe_mstime = sess_probes[probe_ind].mstime
                            this_probe_eegoffset = sess_probes[probe_ind].eegoffset

                            # compute the offsets from the probe
                            delta_mstime = rec_events[sess_inds][x].mstime - this_probe_mstime
                            delta_eegoffset = rec_events[sess_inds][x].eegoffset - this_probe_eegoffset

                            # add the offsets this event
                            tmp_ev.mstime += delta_mstime
                            tmp_ev.eegoffset += delta_eegoffset

                            # store
                            surrogate_evs.append(tmp_ev)

                        if len(surrogate_evs) == 1:
                            surrogate_evs = surrogate_evs[0]
                        else:
                            surrogate_evs = np.concatenate(surrogate_evs)
                            surrogate_evs = surrogate_evs.view(np.recarray)

                        # concaat the new events to the original rectrieval events
                        sess_evs = np.concatenate([rec_events[sess_inds][~bad], surrogate_evs])
                        sess_evs = sess_evs.view(np.recarray)
                        ev_order = np.argsort(sess_evs, order=('trial', 'mstime'))
                        sess_evs = sess_evs[ev_order]

                    else:
                        sess_evs = rec_events[sess_inds][~bad]
                    good_evs.append(sess_evs)

                if len(good_evs) == 1:
                    rec_events = good_evs[0]
                else:
                    rec_events = np.concatenate(good_evs)
                    rec_events = rec_events.view(np.recarray)

                # good = np.concatenate([[10000], np.diff(rec_events.mstime)]) > 2000
                # rec_events = rec_events[good]
                ev_list.append(rec_events)

        # concatenate the different types of events if needed
        if len(ev_list) == 1:
            events = ev_list[0]
        else:
            events = stack_arrays(ev_list, asrecarray=True, usemask=False)

        # make sure events are in time order. this doesn't really matter
        ev_order = np.argsort(events, order=('session', 'trial', 'mstime'))
        events = events[ev_order]

    elif 'RAM_FR' in task:
        is_clustered = add_temp_clust_field(events)
        if not use_json:
            events = append_fields(events, 'is_clustered', is_clustered, dtypes=float, usemask=False, asrecarray=True)
        else:
            events = merge_arrays([events, np.array(is_clustered, dtype=[('is_clustered', float)])], flatten=True,
                                  asrecarray=True)

        phase_list = task_phase if isinstance(task_phase, list) else [task_phase]
        ev_list = []
        for phase in phase_list:

            if phase == 'enc':
                # filter to just item presentation events
                ev_list.append(events[(events.type == 'WORD')])
            elif phase == 'rec':
                tmp_ev = events[(events.type == 'REC_WORD')]
                # rec_time_diffs = np.diff([0] + tmp_ev[0].mstime.tolist())
                ev_list.append(tmp_ev)
        # pdb.set_trace()
        # concatenate the different types of events if needed
        if len(ev_list) == 1:
            events = ev_list[0]
        else:
            events = stack_arrays(ev_list, asrecarray=True, usemask=False)

        ev_order = np.argsort(events, order=('session', 'list', 'mstime'))
        events = events[ev_order]

    elif 'RAM_PAL' in task:
        ev_order = np.argsort(events, order=('session', 'list', 'mstime'))
        events = events[ev_order]

        if task_phase == 'enc':
            # filter to just item presentation events
            events = events[(events.type == 'STUDY_PAIR')]
        elif task_phase == 'rec':
            events = events[(events.type == 'TEST_PROBE')]

    elif 'RAM_YC' in task:

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]

        # add some new fields to events
        test_errs = add_err_to_test_YC(events)
        # events = append_fields(events, 'norm_err', test_errs, dtypes=float, usemask=False, asrecarray=True)
        events = merge_arrays([events, np.array(test_errs, dtype=[('norm_err', float)])], flatten=True,
                              asrecarray=True)

        ev_order = np.argsort(events, order=('session', 'itemno', 'mstime'))
        events = events[ev_order]

        if task_phase == 'enc':
            # filter to just item presentation events
            events = events[(events.type == 'NAV_LEARN')]
        elif task_phase == 'rec':
            events = events[(events.type == 'NAV_TEST')]

    if session is not None:
        events = events[np.array([True if x in session else False for x in events.session])]
        # events = events[events.session == session]

    return events


def get_event_mtime(task, subj, montage, use_json=True):
    """
    Returns the modification time of the event file
    # """
    if not use_json:
        subj_file = subj + '_events.mat'
        if int(montage) != 0:
            subj_file = subj + '_' + str(montage) + '_events.mat'
        subj_ev_path = os.path.join(basedir+'/data/events/', task, subj_file)
        return os.path.getmtime(subj_ev_path)
    else:
        # reader = JsonIndexReader(basedir+'/protocols/r1.json')
        event_paths = list(
        reader.aggregate_values('task_events', subject=subj, montage=montage, experiment=task.replace('RAM_', '')))
        return np.max([os.path.getmtime(x) for x in event_paths])


def load_subj_elecs(subj, montage=0, use_json=True):
    """Returns array of electrode numbers  (monopolar and bipolar)."""

    subj = str(subj)
    if not use_json:
        if int(montage) != 0:
            subj = subj + '_' + str(montage)
        bipol_tal_path = os.path.join(basedir + '/data/eeg', subj, 'tal', subj + '_talLocs_database_bipol.mat')
        bipol_tal_reader = TalReader(filename=bipol_tal_path)
        bipolar_pairs = bipol_tal_reader.get_bipolar_pairs()

        mono_tal_path = os.path.join(basedir + '/data/eeg', subj, 'tal', subj + '_talLocs_database_monopol.mat')
        mono_tal_reader = TalReader(filename=mono_tal_path, struct_name='talStruct')
        mono_tal_struct = mono_tal_reader.read()
        monopolar_channels = np.array([str(x).zfill(3) for x in mono_tal_struct.channel])
    else:

        mp_struct = load_tal(subj, montage, False)
        monopolar_channels = np.array([chan[0] for chan in mp_struct['channel']])

        bp_struct = load_tal(subj, montage, True)
        e1 = [chan[0] for chan in bp_struct['channel']]
        e2 = [chan[1] for chan in bp_struct['channel']]
        bipolar_pairs = np.array(list(zip(e1, e2)), dtype=[('ch0', '|S3'), ('ch1', '|S3')])

    return bipolar_pairs, monopolar_channels


def load_subj_elec_locs(subj, bipol=True):
    """Returns arrays of (localization tag, freesurfer region, clinical tag)."""

    subj = str(subj)
    file_str = 'bipol' if bipol else 'monopol'
    struct_name = 'bpTalStruct' if bipol else 'talStruct'
    tal_path = os.path.join(basedir+'/data/eeg', subj, 'tal', subj + '_talLocs_database_' + file_str + '.mat')
    tal_reader = TalReader(filename=tal_path, struct_name=struct_name)
    tal_struct = tal_reader.read()

    xyz_avg = np.array(zip(tal_struct.avgSurf.x_snap, tal_struct.avgSurf.y_snap, tal_struct.avgSurf.z_snap))
    xyz_indiv = np.array(zip(tal_struct.indivSurf.x_snap, tal_struct.indivSurf.y_snap, tal_struct.indivSurf.z_snap))

    # region based on individual freesurfer parecellation
    anat_region = tal_struct.indivSurf.anatRegion_snap

    # region based on locTag, if available
    if 'locTag' in tal_struct.dtype.names:
        loc_tag = tal_struct.locTag
    else:
        loc_tag = np.array(['[]']*len(tal_struct),dtype='|S256')
    return loc_tag, anat_region, tal_struct.tagName, xyz_avg, xyz_indiv, tal_struct.eType


def load_tal(subj, montage=0, bipol=True, use_json=True):

    if use_json:
        montage = int(montage)
        elec_key = 'pairs' if bipol else 'contacts'

        # reader = JsonIndexReader(basedir + '/protocols/r1.json')
        f_path = reader.aggregate_values(elec_key, subject=subj, montage=montage)
        elec_json = open(list(f_path)[0], 'r')

        if montage == 0:
            elec_data = json.load(elec_json)[subj][elec_key]
        else:
            elec_data = json.load(elec_json)[subj+'_'+str(montage)][elec_key]
        elec_json.close()

        elec_array = np.recarray(len(elec_data, ), dtype=[('channel', list),
                                                          ('anat_region', 'S30'),
                                                          ('loc_tag', 'S30'),
                                                          ('tag_name', 'S30'),
                                                          ('xyz_avg', list),
                                                          ('xyz_indiv', list),
                                                          ('e_type', 'S1')
                                                          ])

        for i, elec in enumerate(np.sort(list(elec_data.keys()))):
            elec_array[i]['tag_name'] = elec
            if bipol:
                elec_array[i]['channel'] = [str(elec_data[elec]['channel_1']).zfill(3),
                                            str(elec_data[elec]['channel_2']).zfill(3)]
                elec_array[i]['e_type'] = elec_data[elec]['type_1']
            else:
                elec_array[i]['channel'] = [str(elec_data[elec]['channel']).zfill(3)]
                elec_array[i]['e_type'] = elec_data[elec]['type']

            if 'ind' in elec_data[elec]['atlases']:
                ind = elec_data[elec]['atlases']['ind']
                elec_array[i]['anat_region'] = ind['region']
                elec_array[i]['xyz_indiv'] = np.array([ind['x'], ind['y'], ind['z']])
            else:
                elec_array[i]['anat_region'] = ''
                elec_array[i]['xyz_indiv'] = np.array([np.nan, np.nan, np.nan])

            if 'avg' in elec_data[elec]['atlases']:
                avg = elec_data[elec]['atlases']['avg']
                elec_array[i]['xyz_avg'] = np.array([avg['x'], avg['y'], avg['z']])
            else:
                elec_array[i]['xyz_avg'] = np.array([np.nan, np.nan, np.nan])

            if 'stein' in elec_data[elec]['atlases']:
                loc_tag = elec_data[elec]['atlases']['stein']['region']
                if (loc_tag is not None) and (loc_tag != '') and (loc_tag != 'None'):
                    elec_array[i]['loc_tag'] = loc_tag
                else:
                    elec_array[i]['loc_tag'] = ''
            else:
                elec_array[i]['loc_tag'] = ''

    else:

        subj_mont = subj
        if int(montage) != 0:
            subj_mont = subj + '_' + str(montage)

        loc_tag, anat_region, tagName, xyz_avg, xyz_indiv, eType = load_subj_elec_locs(subj_mont, bipol)
        bipolar_pairs, monopolar_channels = load_subj_elecs(subj, montage, use_json=False)
        elec_array = np.recarray(len(tagName, ), dtype=[('channel', list),
                                                          ('anat_region', 'S30'),
                                                          ('loc_tag', 'S30'),
                                                          ('tag_name', 'S30'),
                                                          ('xyz_avg', list),
                                                          ('xyz_indiv', list),
                                                          ('e_type', 'S1')
                                                          ])
        for i, elec in enumerate(zip(loc_tag, anat_region, tagName, xyz_avg, xyz_indiv, eType,
                                     bipolar_pairs if bipol else monopolar_channels)):
            elec_array[i]['loc_tag'] = elec[0]
            elec_array[i]['anat_region'] = elec[1]
            elec_array[i]['tag_name'] = elec[2]
            elec_array[i]['xyz_avg'] = elec[3]
            elec_array[i]['xyz_indiv'] = elec[4]
            elec_array[i]['e_type'] = elec[5]
            elec_array[i]['channel'] = elec[6]

    return elec_array


def bin_elec_locs(loc_tags, anat_regions, coords):
    """Returns dictionary of boolean arrays indicated whether electrodes are in a given region"""

    hipp_tags = ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                 'Right CA3', 'Right DG', 'Right Sub']
    mtl_tags = ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC', 'Left CA1', 'Left CA2',
                'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2', 'Right CA3', 'Right DG', 'Right Sub']
    mtl_tags = ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC']
    ifg_tags = ['parsopercularis', 'parsorbitalis', 'parstriangularis']
    mfg_tags = ['caudalmiddlefrontal', 'rostralmiddlefrontal']
    sfg_tags = ['superiorfrontal']
    fc_tags = ['parsopercularis', 'parsorbitalis', 'parstriangularis', 'caudalmiddlefrontal', 'rostralmiddlefrontal',
               'superiorfrontal']
    tc_tags = ['superiortemporal', 'middletemporal', 'inferiortemporal']
    ipc_tags = ['inferiorparietal', 'supramarginal']
    spc_tags = ['superiorparietal', 'precuneus']
    oc_tags = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']

    loc_dict = {}
    loc_dict['Hipp'] = np.array([x in hipp_tags for x in loc_tags])
    loc_dict['MTL'] = np.array([x in mtl_tags for x in loc_tags])
    loc_dict['IFG'] = np.array([x in ifg_tags for x in anat_regions])
    loc_dict['MFG'] = np.array([x in mfg_tags for x in anat_regions])
    loc_dict['SFG'] = np.array([x in sfg_tags for x in anat_regions])
    loc_dict['FC'] = np.array([x in fc_tags for x in anat_regions])
    loc_dict['TC'] = np.array([x in tc_tags for x in anat_regions])
    loc_dict['IPC'] = np.array([x in ipc_tags for x in anat_regions])
    loc_dict['SPC'] = np.array([x in spc_tags for x in anat_regions])
    loc_dict['OC'] = np.array([x in oc_tags for x in anat_regions])
    loc_dict['is_right'] = coords[:, 0] > 0
    return loc_dict


def get_subjs(task, use_json=True):
    """Returns list of subjects who performed a given task."""

    if not use_json:
        subjs = glob(os.path.join(basedir+'/data/events/', task, 'R*_events.mat'))
        subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
        subjs.sort()
        subjs = np.array(subjs)
    else:
        # reader = JsonIndexReader(basedir + '/protocols/r1.json')
        subjs = np.array(reader.subjects(experiment=task.replace('RAM_', '')))

    return subjs


def get_subjs_and_montages(task):
    """Returns list of subjects who performed a given task, along with the montage numbers."""

    # reader = JsonIndexReader(basedir + '/protocols/r1.json')
    subjs = reader.subjects(experiment=task.replace('RAM_', ''))

    out = []
    for subj in subjs:
        m = reader.aggregate_values('montage', subject=subj, experiment=task.replace('RAM_', ''))
        out.extend(zip([subj] * len(m), m))
    return np.array(out)


##########################################################
############### Event filtering functions ################
##########################################################
def filter_events_to_chest(task, events):
    """True if item was presented, really only makes sense if you've load CHEST events"""

    recalled = events['confidence'] >= 0
    return recalled


def filter_events_to_not_low_conf(task, events):
    """True if not low confidence response"""

    recalled = events['confidence'] > 0
    return recalled


def filter_events_to_high_conf(task, events):
    """True only if highest confidence response"""

    recalled = events['confidence'] == 2
    return recalled


def filter_events_to_recalled(task, events, thresh=None):
    """True if not low confidence and better than median or radius size (or threshold if given)"""

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        if thresh is None:
            thresh = np.max([np.median(events['distErr']), events['radius_size'][0]])
        not_far_dist = events['distErr'] < thresh
        recalled = not_low_conf & not_far_dist

    elif task == 'RAM_YC1':
        recalled = events['norm_err'] < np.median(events['norm_err'])
    elif task == 'RAM_PAL1':
        recalled = events['correct'] == 1
    else:
        recalled = events['recalled'] == 1
    return recalled

def filter_events_to_clustered(task, events, thresh=None):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        if thresh is None:
            thresh = np.max([np.median(events['distErr']), events['radius_size'][0]])
        not_far_dist = events['distErr'] < thresh
        recalled = not_low_conf & not_far_dist

    elif task == 'RAM_YC1':
        recalled = events['norm_err'] < np.median(events['norm_err'])
    elif task == 'RAM_PAL1':
        recalled = events['correct'] == 1
    else:
        recalled = events['is_clustered']
        recalled[events['recalled'] == 0] = np.nan
    return recalled


def filter_events_to_recalled_min_err(task, events, thresh=None):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        if thresh is None:
            thresh = np.max([np.median(events[not_low_conf]['min_err']), events['radius_size'][0]])
            thresh = np.median(events[not_low_conf]['min_err'])
            print('%s: thresh %.2f' % (events[0]['subject'], thresh))
        not_far_dist = events['min_err'] < thresh
        recalled = not_low_conf & not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_min_err_just_median(task, events, thresh=None):

    if task == 'RAM_TH1':
        if thresh is None:
            thresh = np.max([np.median(events['min_err']), events['radius_size'][0]])
        recalled = events['min_err'] < thresh
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_reaction_time(task, events, thresh=None):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        if thresh is None:
            thresh = np.median(events['reactionTime'])
        not_far_dist = events['reactionTime'] < thresh
        recalled = not_low_conf & not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_just_median(task, events, thresh=None):
    "True if better than median distance error or given threshold"

    if task == 'RAM_TH1':
        if thresh is None:
            thresh = np.max([np.median(events['distErr']), events['radius_size'][0]])
        recalled = events['distErr'] < thresh
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_sess_level(task, events, thresh=None):

    if task == 'RAM_TH1':

        sessions = events['session']
        recalled = np.zeros(shape=sessions.shape, dtype=bool)
        not_low_conf = events['confidence'] > 0
        if len(np.unique(sessions)) == 1:
            if thresh is None:
                thresh = np.max([np.median(events['distErr']), events['radius_size'][0]])
            not_far_dist = events['distErr'] < thresh
            recalled = not_low_conf & not_far_dist

        else:

            uniq_sess = np.unique(sessions)
            for i, sess in enumerate(uniq_sess):
                sess_inds = sessions == sess
                if i == 0:
                    thresh = np.max([np.median(events[sess_inds]['distErr']), events['radius_size'][0]])
                    # not_far_dist = events[sess_inds]['distErr'] < thresh
                    # recalled[sess_inds] = not_low_conf[sess_inds] & not_far_dist
                else:
                    prev_sess = uniq_sess[:i]
                    prev_inds = np.array([True if x in prev_sess else False for x in sessions])
                    prev_errs = events[prev_inds]['distErr']
                    curr_errs = events[sess_inds]['distErr']
                    st, pval = ttest_ind(prev_errs, curr_errs)
                    if pval < .05:
                        print('session %d subject %s differs' %(sess, events['subject'][0]))
                        errs = curr_errs
                    else:
                        errs = np.concatenate([prev_errs, curr_errs])
                    thresh = np.max([np.median(errs), events['radius_size'][0]])
                not_far_dist = events[sess_inds]['distErr'] < thresh
                recalled[sess_inds] = not_low_conf[sess_inds] & not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_smart_low(task, events, thresh=None):

    if task == 'RAM_TH1':

        not_low_conf = events['confidence'] > 0
        t, pval = ttest_1samp(events[events['confidence'] == 0]['norm_err'], .5)
        if (t < 0) & (pval < .05):
            not_low_conf = events['confidence'] >= 0
            print('Confidence not reliable for %s' % events['subject'][0])
        if thresh is None:
            thresh = np.max([np.median(events[not_low_conf]['distErr']), events['radius_size'][0]])
        not_far_dist = events['distErr'] < thresh
        recalled = not_low_conf & not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_norm_thresh(task, events, thresh):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        # thresh = np.median(events['norm_err'])
        # radius = events['radius_size'][0]
        # correct = events['distErr'] < radius
        not_far_dist = events['norm_err'] < thresh
        # recalled = not_low_conf & (not_far_dist | correct)
        recalled = not_low_conf & not_far_dist
        # recalled = not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_norm_thresh_no_conf(task, events, thresh):

    if task == 'RAM_TH1':
        # not_low_conf = events['confidence'] > 0
        # thresh = np.median(events['norm_err'])
        # radius = events['radius_size'][0]
        # correct = events['distErr'] < radius
        not_far_dist = events['norm_err'] < thresh
        # recalled = not_low_conf & (not_far_dist | correct)
        recalled = not_far_dist
        # recalled = not_far_dist
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_norm(task, events, thresh):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        if thresh is None:
            thresh = np.median(events['norm_err'])
        radius = events['radius_size'][0]
        correct = events['distErr'] < radius
        not_far_dist = events['norm_err'] < thresh
        recalled = not_low_conf & (not_far_dist | correct)
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_norm_thresh_exc_low(task, events):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        thresh = np.median(events[not_low_conf]['norm_err'])
        radius = events['radius_size'][0]
        correct = events['distErr'] < radius
        not_far_dist = events['norm_err'] < thresh
        recalled = not_low_conf & (not_far_dist | correct)
    else:
        recalled = events['recalled'] == 1
    return recalled


def filter_events_to_recalled_multi_thresh(task, events):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        flipped = events['isRecFromStartSide'] == 0
        far = events['isRecFromNearSide'] == 0
        recalled = np.zeros(shape=flipped.shape, dtype=bool)

        flipped_thresh = np.max([np.median(events[flipped]['distErr']), events[flipped]['radius_size'][0]])
        recalled[flipped] = (events[flipped]['distErr'] < flipped_thresh) & not_low_conf[flipped]

        non_flipped_thresh = np.max([np.median(events[~flipped]['distErr']), events[~flipped]['radius_size'][0]])
        recalled[~flipped] = (events[~flipped]['distErr'] < non_flipped_thresh) & not_low_conf[~flipped]

        # flipped_far_thresh = np.max([np.median(events[flipped & far]['distErr']), events[flipped & far]['radius_size'][0]])
        # recalled[flipped & far] = (events[flipped & far]['distErr'] < flipped_far_thresh) & not_low_conf[flipped & far]
        #
        # flipped_near_thresh = np.max([np.median(events[flipped & ~far]['distErr']), events[flipped & ~far]['radius_size'][0]])
        # recalled[flipped & ~far] = (events[flipped & ~far]['distErr'] < flipped_near_thresh) & not_low_conf[flipped & ~far]
        #
        # non_flipped_far_thresh = np.max([np.median(events[~flipped & far]['distErr']), events[~flipped & far]['radius_size'][0]])
        # recalled[~flipped & far] = (events[~flipped & far]['distErr'] < non_flipped_far_thresh) & not_low_conf[~flipped & far]
        #
        # non_flipped_near_thresh = np.max([np.median(events[~flipped & ~far]['distErr']), events[~flipped & ~far]['radius_size'][0]])
        # recalled[~flipped & ~far] = (events[~flipped & ~far]['distErr'] < non_flipped_near_thresh) & not_low_conf[~flipped & ~far]

    else:
        recalled = events['recalled'] == 1
    return recalled

def filter_events_to_recalled_multi_thresh_near(task, events):

    if task == 'RAM_TH1':
        not_low_conf = events['confidence'] > 0
        flipped = events['isRecFromStartSide'] == 0
        far = events['isRecFromNearSide'] == 0
        recalled = np.zeros(shape=flipped.shape, dtype=bool)

        far_thresh = np.max([np.median(events[far]['distErr']), events[far]['radius_size'][0]])
        recalled[far] = (events[far]['distErr'] < far_thresh) & not_low_conf[far]

        near_thresh = np.max([np.median(events[~far]['distErr']), events[~far]['radius_size'][0]])
        recalled[~far] = (events[~far]['distErr'] < near_thresh) & not_low_conf[~far]

        # flipped_far_thresh = np.max([np.median(events[flipped & far]['distErr']), events[flipped & far]['radius_size'][0]])
        # recalled[flipped & far] = (events[flipped & far]['distErr'] < flipped_far_thresh) & not_low_conf[flipped & far]
        #
        # flipped_near_thresh = np.max([np.median(events[flipped & ~far]['distErr']), events[flipped & ~far]['radius_size'][0]])
        # recalled[flipped & ~far] = (events[flipped & ~far]['distErr'] < flipped_near_thresh) & not_low_conf[flipped & ~far]
        #
        # non_flipped_far_thresh = np.max([np.median(events[~flipped & far]['distErr']), events[~flipped & far]['radius_size'][0]])
        # recalled[~flipped & far] = (events[~flipped & far]['distErr'] < non_flipped_far_thresh) & not_low_conf[~flipped & far]
        #
        # non_flipped_near_thresh = np.max([np.median(events[~flipped & ~far]['distErr']), events[~flipped & ~far]['radius_size'][0]])
        # recalled[~flipped & ~far] = (events[~flipped & ~far]['distErr'] < non_flipped_near_thresh) & not_low_conf[~flipped & ~far]

    else:
        recalled = events['recalled'] == 1
    return recalled


def calc_norm_dist_error(x_pos, y_pos, act_errs):
    rand_x = np.random.uniform(362.9, 406.9, 100000)
    rand_y = np.random.uniform(321.0, 396.3, 100000)

    error_percentiles = np.zeros(np.shape(act_errs), dtype=float)
    for i, this_item in enumerate(zip(x_pos, y_pos, act_errs)):
        if np.isnan(this_item[2]):
            error_percentiles[i] = np.nan
        else:
            possible_errors = np.sqrt((rand_x - this_item[0]) ** 2 + (rand_y - this_item[1]) ** 2)
            error_percentiles[i] = np.mean(possible_errors < this_item[2])
    return error_percentiles


def calc_min_dist_to_any_chest(events):

    # need the center locations to make the coordinates relative to zero
    x_center = 384.8549
    y_center = 358.834

    sessions = events.session
    uniq_sess = np.unique(sessions)
    trials = events.trial

    # adjust response and actual locations
    x_resp = events.chosenLocationX - x_center
    y_resp = events.chosenLocationY - y_center
    x_act = events.locationX - x_center
    y_act = events.locationY - y_center
    min_err = np.zeros(np.shape(x_act), dtype=float)

    for sess in uniq_sess:
        uniq_trial = np.unique(trials[sessions == sess])
        for trial in uniq_trial:
            trial_inds = np.where((sessions == sess) & (trials == trial))[0]
            xy = np.stack([x_act[trial_inds], y_act[trial_inds]], axis=1)
            xy = np.concatenate([xy, -xy], axis=0)
            for ev in trial_inds:
                if np.isnan(x_resp[ev]):
                    min_err[ev] = np.nan
                else:
                    xy_resp = np.array([x_resp[ev], y_resp[ev]])
                    min_err[ev] = np.sqrt(np.sum(np.square(xy - xy_resp), axis=1)).min()
    try:
        events = append_fields(events, 'min_err', min_err, dtypes=float, usemask=False, asrecarray=True)
    except:
        events = merge_arrays([events, np.array(min_err, dtype=[('min_err', float)])], flatten=True, asrecarray=True)

    return events


def add_err_to_test_YC(events):

    rec_events = events.type == 'NAV_TEST'

    test_error = np.zeros((len(events)))
    test_error[:] = np.nan

    for this_rec in np.where(rec_events)[0]:
        session = events[this_rec].session
        trial = events[this_rec].blocknum;
        this_err = events[this_rec].respPerformanceFactor

        this_enc = (events.blocknum == trial) & (events.session == session)
        test_error[this_enc] = this_err
    return test_error


def add_temp_clust_field(events):
    uniq_sessions = np.unique(events.session)
    is_clustered = np.zeros(len(events))
    rec_inds = events.type == 'REC_WORD'
    enc_inds = events.type == 'WORD'
    for sess in uniq_sessions:
        sess_inds = events.session == sess
        sess_rec_inds = sess_inds & rec_inds
        sess_enc_inds = sess_inds & enc_inds

        sess_trials = np.unique(events[sess_rec_inds].list)
        for trial in sess_trials:
            trial_rec_events = events[sess_rec_inds & (events.list == trial)]
            trial_enc_inds = sess_enc_inds & (events.list == trial)
            rec_serial_pos = np.zeros(len(trial_rec_events))
            rec_serial_pos[:] = np.nan
            for i, trial_rec_event in enumerate(trial_rec_events):
                spos_ind = events[trial_enc_inds].item_name == trial_rec_event.item_name
                if np.any(spos_ind):
                    rec_serial_pos[i] = events[trial_enc_inds][spos_ind].serialpos
            clustered = (np.abs(np.diff(np.concatenate([[100], rec_serial_pos]))) == 1) | (
            np.abs(np.diff(np.concatenate([rec_serial_pos, [100]])[::-1])[::-1]) == 1)
            for this_clusterd_item in np.where(clustered)[0]:
                this_enc_event = trial_enc_inds & (events.serialpos == rec_serial_pos[this_clusterd_item])
                is_clustered[this_enc_event] = 1
    return is_clustered
