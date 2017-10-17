import numpy as np
import pandas as pd
from SubjectLevel import subject_exclusions
from SubjectLevel.Analyses import subject_SME
from GroupLevel.Analyses import group_SME, group_move_vs_still
import ram_data_helpers

def filter_out_bad_elecs(subj, bad_elec_table, onset_only=True, only_bad=False):

    bad_ictal = bad_elec_table[bad_elec_table.index == [int(subj.subj[2:5])]]['IctalOnset']
    if not isinstance(bad_ictal.values[0], str):
        bad_ictal = ''
    else:
        bad_ictal = bad_ictal.values[0]

    bad_spiking = bad_elec_table[bad_elec_table.index == [int(subj.subj[2:5])]]['IctalSpiking']
    if not isinstance(bad_spiking.values[0], str):
        bad_spiking = ''
    else:
        bad_spiking = bad_spiking.values[0]

    if onset_only:
        bad_elecs = bad_ictal
    else:
        bad_elecs = bad_spiking + ' ' + bad_ictal
    bad_elecs = np.array(bad_elecs.split())

    if len(bad_elecs) == 0:
        print('%s: no bad elecs.' % subj.subj)
        return subj

    bad_bools = np.zeros(len(subj.subject_data.attrs['chan_tags'])).astype(bool)
    for bad_elec in bad_elecs:
        is_bad = np.any(np.array([pair.split('-') for pair in subj.subject_data.attrs['chan_tags']]) == bad_elec, axis=1)
        bad_bools[is_bad] = True
    if only_bad:
        bad_bools = ~bad_bools

    subj.res['ts'] = subj.res['ts'][:, ~bad_bools]
    subj.res['ps'] = subj.res['ps'][:, ~bad_bools]
    subj.res['zs'] = subj.res['zs'][:, ~bad_bools]
    subj.elec_xyz_avg = subj.elec_xyz_avg[~bad_bools]
    subj.elec_xyz_indiv = subj.elec_xyz_indiv[~bad_bools]
    for key in subj.elec_locs.keys():
        subj.elec_locs[key] = subj.elec_locs[key][~bad_bools]
    subj.subject_data = subj.subject_data[:, :, ~bad_bools]
    return subj


def plot_good_and_bad_hipp_elecs(subj,
                                 table_path='/Users/jmiller/Documents/papers/jacobsPapers/TH_SME/bad_elecs.csv'):

    bad_elec_table = pd.read_csv(table_path, index_col=0)

    # load sme data
    sme = group_SME.GroupSME(load_data_if_file_exists=True, subject_settings='default_50_freqs',
                                load_res_if_file_exists=False, bipolar=True, use_json=True,
                                subjs=[subj],
                                base_dir='/Users/jmiller/data/python',
                                do_not_compute=True,start_time=[0.0],end_time=[1.5],
                                recall_filter_func=ram_data_helpers.filter_events_to_recalled_norm)
    sme.process()
    subj_sme = sme.subject_objs[0]
    subj_sme.load_data()
    subj_sme = subject_exclusions.remove_abridged_sessions(subj_sme)

    # load move data
    move = group_move_vs_still.GroupMoveStill(subjs=[subj],
                                              load_data_if_file_exists=True,
                                              load_res_if_file_exists=True, use_json=True,
                                              base_dir='/Users/jmiller/data/python',
                                              do_not_compute=True)
    move.process()
    subj_move = move.subject_objs[0]
    subj_move.load_data()
    subj_move = subject_exclusions.remove_abridged_sessions(subj_move)

    # filter to just good electrodes
    subj_sme_good_elecs = filter_out_bad_elecs(subj_sme, bad_elec_table, onset_only=True, only_bad=False)
    subj_move_good_elecs = filter_out_bad_elecs(subj_move, bad_elec_table, onset_only=True, only_bad=False)

    freqs_inds = (sme.subject_objs[0].freqs >= 1) & (sme.subject_objs[0].freqs <= 3)
    left_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (~subj_sme_good_elecs.elec_locs['is_right'])
    right_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (subj_sme_good_elecs.elec_locs['is_right'])

    # left_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_elecs])
    # right_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_elecs])
    left_move_good_elecs = np.nanmean(subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_good_elecs])
    right_move_good_elecs = np.nanmean(subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_good_elecs])

    # now just bad electrodes
    subj_sme_bad_elecs = filter_out_bad_elecs(subj_sme, bad_elec_table, onset_only=True, only_bad=True)
    subj_move_bad_elecs = filter_out_bad_elecs(subj_move, bad_elec_table, onset_only=True, only_bad=True)

    left_hipp_bad_elecs = (subj_sme_bad_elecs.elec_locs['Hipp']) & (~subj_sme_bad_elecs.elec_locs['is_right'])
    right_hipp_bad_elecs = (subj_sme_bad_elecs.elec_locs['Hipp']) & (subj_sme_bad_elecs.elec_locs['is_right'])

    # left_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_elecs])
    # right_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_elecs])
    left_move_bad_elecs = np.nanmean(subj_move_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_bad_elecs])
    right_move_bad_elecs = np.nanmean(subj_move_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_bad_elecs])
