import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_ind
from SubjectLevel import subject_exclusions
from SubjectLevel.Analyses import subject_SME
from GroupLevel.Analyses import group_SME, group_move_vs_still
import ram_data_helpers
import matplotlib.pyplot as plt
import platform

table_path='/home1/jfm2/python/RAM_classify/Projects/TH_SME_Paper/bad_elecs.csv'
basedir = '/scratch/jfm2/python'
if platform.system() == 'Darwin':
    basedir = '/Users/jmiller/data/python'
    table_path = '/Users/jmiller/Documents/papers/jacobsPapers/TH_SME/bad_elecs.csv'


def filter_to_move(task, events, thresh):
    return events['type'] == 'move'


# normalize nav and sme together
def mean_within_region_wrapper(subjs=None, region='Hipp', do_move=False, freq_range=[1., 3.], remove_elecs=True,
                               remove_hipp=False, do_mean_before_stats=True):
    bad_elec_table = pd.read_csv(table_path, index_col=0)
    if subjs is None:
        subjs = ram_data_helpers.get_subjs_and_montages('RAM_TH1')
        subjs = subjs[np.array([False if s[0] == 'R1132C' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1201P' else True for s in subjs])]
        # subjs = subjs[np.array([False if s[0] == 'R1212P' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1219C' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1231M' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1243T' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1244J' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1258T' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1230J' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1269E' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1259E' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1226D' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1214M' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1263C' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1160C' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1282C' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1227T' else True for s in subjs])]
        subjs = subjs[np.array([False if s[0] == 'R1182C' else True for s in subjs])]

    # load sme data
    # if not do_move:
    sme = group_SME.GroupSME(load_data_if_file_exists=True, subject_settings='default_50_freqs',
                             load_res_if_file_exists=True, bipolar=True, use_json=True,
                             subjs=subjs,
                             base_dir=basedir,
                             do_not_compute=True, start_time=[0.0], end_time=[1.5],
                             recall_filter_func=ram_data_helpers.filter_events_to_recalled_norm)
    sme.process()
    # else:
    nav = group_move_vs_still.GroupMoveStill(subjs=subjs,
                                             load_data_if_file_exists=True,
                                             load_res_if_file_exists=True, use_json=True,
                                             base_dir=basedir,
                                             do_not_compute=True,
                                             recall_filter_func=filter_to_move)
    nav.process()

    res = []
    subj_list = []
    # import pdb
    # pdb.set_trace()
    for subj in zip(sme.subject_objs, nav.subject_objs):

        if subj[0].subj != subj[1].subj:
            print('Subject mismatch %s %s' % (subj[0].subj, subj[1].subj))
            continue

        for i, ana in enumerate(subj):
            ana.load_data()
            ana = subject_exclusions.remove_abridged_sessions(ana)

            if ana.subject_data is not None:
                subj_list.append(ana.subj)
                if remove_elecs:
                    if remove_hipp:
                        ana = filter_out_bad_mtl(ana, bad_elec_table, onset_only=True)
                    else:
                        ana = filter_out_bad_elecs(ana, bad_elec_table, onset_only=True, only_bad=False)

        rec_mean_l, nrec_mean_l, nav_mean_l, still_mean_l = sme_within_region(subj, region=region, hemi='l',
                                                                              freq_range=freq_range,
                                                                              do_mean_before_stats=do_mean_before_stats)
        rec_mean_r, nrec_mean_r, nav_mean_r, still_mean_r = sme_within_region(subj, region=region, hemi='r',
                                                                              freq_range=freq_range,
                                                                              do_mean_before_stats=do_mean_before_stats)
        res.append(np.array([rec_mean_l, nrec_mean_l, rec_mean_r, nrec_mean_r, nav_mean_l, still_mean_l, nav_mean_r, still_mean_r]))
    return res, subj_list



def sme_within_region(subj, region='Hipp', hemi=None, freq_range=None, do_mean_before_stats=True):

    data = np.concatenate([subj[0].subject_data, subj[1].subject_data], axis=0)
    sessions = np.concatenate([subj[0].subject_data.events.data['session'], subj[1].subject_data.events.data['session']])
    ana = np.concatenate([np.zeros(len(subj[0].subject_data.events.data['session'])), np.ones(len(subj[1].subject_data.events.data['session']))])

    # mean power within ROI first
    elecs_to_mean = subj[0].elec_locs[region]
    if hemi == 'l':
        elecs_to_mean = elecs_to_mean & ~subj[0].elec_locs['is_right']
    elif hemi == 'r':
        elecs_to_mean = elecs_to_mean & subj[0].elec_locs['is_right']
    if do_mean_before_stats:
        region_mean = np.nanmean(data[:, :, elecs_to_mean], axis=2)
    else:
        region_mean = data[:, :, elecs_to_mean]

    # now zscore by session
    uniq_sessions = np.unique(sessions)
    for sess in uniq_sessions:
        sess_inds = sessions == sess
        region_mean[sess_inds] = zscore(region_mean[sess_inds], axis=0)

    # mean over frequencies if desired
    if freq_range is not None:
        freq_inds = (subj[0].freqs >= freq_range[0]) & (subj[0].freqs <= freq_range[1])
        region_mean = np.mean(region_mean[:, freq_inds], axis=1)

    recalled = subj[0].recall_filter_func(subj[0].task, subj[0].subject_data.events.data, None)
    # import pdb
    # pdb.set_trace()
    rec_mean_sme = np.nanmean(region_mean[ana == 0][recalled], axis=0)
    nrec_mean_sme = np.nanmean(region_mean[ana == 0][~recalled], axis=0)

    recalled = subj[1].recall_filter_func(subj[1].task, subj[1].subject_data.events.data, None)
    rec_mean_nav = np.nanmean(region_mean[ana == 1][recalled], axis=0)
    nrec_mean_nav = np.nanmean(region_mean[ana == 1][~recalled], axis=0)

    if not do_mean_before_stats:
        rec_mean_sme = np.nanmean(rec_mean_sme, 1)
        nrec_mean_sme = np.nanmean(nrec_mean_sme, 1)
        rec_mean_nav = np.nanmean(rec_mean_nav, 1)
        nrec_mean_nav = np.nanmean(nrec_mean_nav, 1)

    # print(ttest_ind(region_mean[recalled], region_mean[~recalled], axis=0, nan_policy='omit'))
    return rec_mean_sme, nrec_mean_sme, rec_mean_nav, nrec_mean_nav


def get_bad_elec_str_array(subj_code, bad_elec_table, onset_only=True):

    bad_ictal = bad_elec_table[bad_elec_table.index == [subj_code]]['IctalOnset']
    if not isinstance(bad_ictal.values[0], str):
        bad_ictal = ''
    else:
        bad_ictal = bad_ictal.values[0]

    bad_spiking = bad_elec_table[bad_elec_table.index == [subj_code]]['IctalSpiking']
    if not isinstance(bad_spiking.values[0], str):
        bad_spiking = ''
    else:
        bad_spiking = bad_spiking.values[0]

    if onset_only:
        bad_elecs = bad_ictal
    else:
        bad_elecs = bad_spiking + ' ' + bad_ictal
    bad_elecs = np.array(bad_elecs.split())
    return bad_elecs


def filter_out_bad_mtl(subj, bad_elec_table, onset_only=True):
    bad_elecs = get_bad_elec_str_array(int(subj.subj[2:5]), bad_elec_table, onset_only)

    bad_right_mtl = False
    bad_left_mtl = False
    # fix this so it doesn't exclude whole subject, just hemisphere
    for bad_elec in bad_elecs:
        is_bad = np.any(np.array([pair.split('-') for pair in subj.subject_data.attrs['chan_tags']]) == bad_elec,
                        axis=1)

        if (np.any(subj.elec_locs['Hipp'][is_bad])) | (np.any(subj.elec_locs['MTL'][is_bad])):

            # import pdb
            # pdb.set_trace()
            if np.any(subj.elec_locs['is_right'][is_bad]):
                bad_right_mtl = True
            else:
                bad_left_mtl = True

    if bad_right_mtl & bad_left_mtl:
        # import pdb
        # pdb.set_trace()

        mtl = (subj.elec_locs['Hipp']) | (subj.elec_locs['MTL'])
        bad = np.zeros(len(subj.subject_data.attrs['chan_tags'])).astype(bool)
        bad[mtl] = True
        print('%s: bad right MTL found, removing.' % subj.subj)
        subj.res['ts'] = subj.res['ts'][:, ~bad]
        subj.res['ps'] = subj.res['ps'][:, ~bad]
        subj.res['zs'] = subj.res['zs'][:, ~bad]
        subj.elec_xyz_avg = subj.elec_xyz_avg[~bad]
        subj.elec_xyz_indiv = subj.elec_xyz_indiv[~bad]
        for key in subj.elec_locs.keys():
            subj.elec_locs[key] = subj.elec_locs[key][~bad]
        subj.subject_data = subj.subject_data[:, :, ~bad]

    elif bad_right_mtl:
        right_mtl = ((subj.elec_locs['Hipp']) | (subj.elec_locs['MTL'])) & (subj.elec_locs['is_right'])
        bad = np.zeros(len(subj.subject_data.attrs['chan_tags'])).astype(bool)
        bad[right_mtl] = True
        print('%s: bad right MTL found, removing.' % subj.subj)
        subj.res['ts'] = subj.res['ts'][:, ~bad]
        subj.res['ps'] = subj.res['ps'][:, ~bad]
        subj.res['zs'] = subj.res['zs'][:, ~bad]
        subj.elec_xyz_avg = subj.elec_xyz_avg[~bad]
        subj.elec_xyz_indiv = subj.elec_xyz_indiv[~bad]
        for key in subj.elec_locs.keys():
            subj.elec_locs[key] = subj.elec_locs[key][~bad]
        subj.subject_data = subj.subject_data[:, :, ~bad]
    elif bad_left_mtl:
        left_mtl = ((subj.elec_locs['Hipp']) | (subj.elec_locs['MTL'])) & (~subj.elec_locs['is_right'])
        bad = np.zeros(len(subj.subject_data.attrs['chan_tags'])).astype(bool)
        bad[left_mtl] = True
        print('%s: bad left MTL found, removing.' % subj.subj)
        subj.res['ts'] = subj.res['ts'][:, ~bad]
        subj.res['ps'] = subj.res['ps'][:, ~bad]
        subj.res['zs'] = subj.res['zs'][:, ~bad]
        subj.elec_xyz_avg = subj.elec_xyz_avg[~bad]
        subj.elec_xyz_indiv = subj.elec_xyz_indiv[~bad]
        for key in subj.elec_locs.keys():
            subj.elec_locs[key] = subj.elec_locs[key][~bad]
        subj.subject_data = subj.subject_data[:, :, ~bad]
    return subj




def filter_out_bad_elecs(subj, bad_elec_table, onset_only=True, only_bad=False):

    bad_elecs = get_bad_elec_str_array(int(subj.subj[2:5]), bad_elec_table, onset_only)

    if len(bad_elecs) == 0:
        print('%s: no bad elecs.' % subj.subj)
        # return subj

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


def plot_good_mtl_subjs(subj, onset_only=True,
                        table_path='/Users/jmiller/Documents/papers/jacobsPapers/TH_SME/bad_elecs.csv',
                        do_plot=False,
                        save_dir='/Users/jmiller/Desktop/hipp_move_good_bad_mtl/'):

    bad_elec_table = pd.read_csv(table_path, index_col=0)

    # load sme data
    sme = group_SME.GroupSME(load_data_if_file_exists=True, subject_settings='default_50_freqs',
                             load_res_if_file_exists=False, bipolar=True, use_json=True,
                             subjs=[subj],
                             # base_dir='/Users/jmiller/data/python',
                             do_not_compute=True, start_time=[0.0], end_time=[1.5],
                             recall_filter_func=ram_data_helpers.filter_events_to_recalled_norm)
    sme.process()
    if len(sme.subject_objs) == 0:
        return

    subj_sme = sme.subject_objs[0]
    subj_sme.load_data()
    subj_sme = subject_exclusions.remove_abridged_sessions(subj_sme)

    # load move data
    move = group_move_vs_still.GroupMoveStill(subjs=[subj],
                                              load_data_if_file_exists=True,
                                              load_res_if_file_exists=True, use_json=True,
                                              # base_dir='/Users/jmiller/data/python',
                                              do_not_compute=True)
    move.process()
    if len(move.subject_objs) == 0:
        return
    subj_move = move.subject_objs[0]
    subj_move.load_data()
    subj_move = subject_exclusions.remove_abridged_sessions(subj_move)

    # filter to just good mtl
    subj_sme_good_elecs = filter_out_bad_elecs(subj_sme, bad_elec_table, onset_only=onset_only)
    subj_move_good_elecs = filter_out_bad_elecs(subj_move, bad_elec_table, onset_only=onset_only)

    freqs_inds = (sme.subject_objs[0].freqs >= 1) & (sme.subject_objs[0].freqs <= 3)
    left_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (~subj_sme_good_elecs.elec_locs['is_right'])
    left_good_n = np.sum(left_hipp_good_elecs)
    right_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (subj_sme_good_elecs.elec_locs['is_right'])
    right_good_n = np.sum(right_hipp_good_elecs)

    left_sme_good_elecs = subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_good_elecs]
    left_sme_good_elecs_mean = np.nanmean(left_sme_good_elecs)
    right_sme_good_elecs = subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_good_elecs]
    right_sme_good_elecs_mean = np.nanmean(right_sme_good_elecs)

    left_move_good_elecs = subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_good_elecs]
    left_move_good_elecs_mean = np.nanmean(left_move_good_elecs)
    right_move_good_elecs = subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_good_elecs]
    right_move_good_elecs_mean = np.nanmean(right_move_good_elecs)

    if do_plot:
        y = [left_move_good_elecs_mean, right_move_good_elecs_mean]
        plt.bar(np.arange(0, 2)-.125, y, align='center', width=.25, zorder=5, color=[.5, .5, .5])
        y = [left_sme_good_elecs_mean, right_sme_good_elecs_mean]
        plt.bar(np.arange(0, 2)+.125, y, align='center', width=.25, zorder=5, color=[.5, .5, .5])
        plt.plot([-2,3], [0,0], '-k', lw=2)
        plt.xlim(-.5,1.5)
        plt.xticks(range(4), ['L. Good (%d)' % left_good_n,
                              'R. Good (%d)' % right_good_n])
        plt.ylabel('Navigation t-stat', fontsize=20)

        for i, elec_data in enumerate([left_move_good_elecs, right_move_good_elecs]):
            plt.plot([i-.125]*len(elec_data), elec_data, '.', c='k', markersize=24, zorder=10)
        for i, elec_data in enumerate([left_sme_good_elecs, right_sme_good_elecs]):
            plt.plot([i+.125]*len(elec_data), elec_data, '.', c='k', markersize=24, zorder=10)
        ylim = plt.ylim()
        plt.ylim(-np.max(np.abs(ylim)), np.max(np.abs(ylim)))
        plt.tight_layout()
        plt.savefig(save_dir + '%s_%s.pdf' % (subj[0], subj[1]))
        plt.show()


    res = {'left_move_good_elecs': left_move_good_elecs_mean,
           'right_move_good_elecs': right_move_good_elecs_mean,
           'left_good_n': left_good_n,
           'left_sme_good_elecs': left_sme_good_elecs_mean,
           'right_sme_good_elecs': right_sme_good_elecs_mean,
           'right_good_n': right_good_n}
    return res



def plot_good_and_bad_hipp_elecs(subj, onset_only=True,
                                 table_path='/Users/jmiller/Documents/papers/jacobsPapers/TH_SME/bad_elecs.csv',
                                 do_plot=False,
                                 save_dir='/Users/jmiller/Desktop/hipp_move_good_bad/'):

    bad_elec_table = pd.read_csv(table_path, index_col=0)

    # load sme data
    sme = group_SME.GroupSME(load_data_if_file_exists=True, subject_settings='default_50_freqs',
                                load_res_if_file_exists=False, bipolar=True, use_json=True,
                                subjs=[subj],
                                base_dir='/Users/jmiller/data/python',
                                do_not_compute=True,start_time=[0.0],end_time=[1.5],
                                recall_filter_func=ram_data_helpers.filter_events_to_recalled_norm)
    sme.process()
    if len(sme.subject_objs) == 0:
        return

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
    subj_sme_good_elecs = filter_out_bad_elecs(subj_sme, bad_elec_table, onset_only=onset_only, only_bad=False)
    subj_move_good_elecs = filter_out_bad_elecs(subj_move, bad_elec_table, onset_only=onset_only, only_bad=False)

    freqs_inds = (sme.subject_objs[0].freqs >= 1) & (sme.subject_objs[0].freqs <= 3)
    left_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (~subj_sme_good_elecs.elec_locs['is_right'])
    left_good_n = np.sum(left_hipp_good_elecs)
    right_hipp_good_elecs = (subj_sme_good_elecs.elec_locs['Hipp']) & (subj_sme_good_elecs.elec_locs['is_right'])
    right_good_n = np.sum(right_hipp_good_elecs)

    left_sme_good_elecs = subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_good_elecs]
    left_sme_good_elecs_mean = np.nanmean(left_sme_good_elecs)
    right_sme_good_elecs = subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_good_elecs]
    right_sme_good_elecs_mean = np.nanmean(right_sme_good_elecs)

    left_move_good_elecs = subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_good_elecs]
    left_move_good_elecs_mean = np.nanmean(left_move_good_elecs)
    right_move_good_elecs = subj_move_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_good_elecs]
    right_move_good_elecs_mean = np.nanmean(right_move_good_elecs)


    # now just bad electrodes
    subj_sme.load_data()
    subj_sme = subject_exclusions.remove_abridged_sessions(subj_sme)
    subj_sme.load_res_data()
    subj_move.load_data()
    subj_move = subject_exclusions.remove_abridged_sessions(subj_move)
    subj_move.load_res_data()

    subj_sme_bad_elecs = filter_out_bad_elecs(subj_sme, bad_elec_table, onset_only=onset_only, only_bad=True)
    subj_move_bad_elecs = filter_out_bad_elecs(subj_move, bad_elec_table, onset_only=onset_only, only_bad=True)

    left_hipp_bad_elecs = (subj_sme_bad_elecs.elec_locs['Hipp']) & (~subj_sme_bad_elecs.elec_locs['is_right'])
    left_bad_n = np.sum(left_hipp_bad_elecs)
    right_hipp_bad_elecs = (subj_sme_bad_elecs.elec_locs['Hipp']) & (subj_sme_bad_elecs.elec_locs['is_right'])
    right_bad_n = np.sum(right_hipp_bad_elecs)

    # left_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_elecs])
    # right_sme = np.nanmean(subj_sme_good_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_elecs])

    left_sme_bad_elecs = subj_sme_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_bad_elecs]
    left_sme_bad_elecs_mean = np.nanmean(left_sme_bad_elecs)
    right_sme_bad_elecs = subj_sme_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_bad_elecs]
    right_sme_bad_elecs_mean = np.nanmean(right_sme_bad_elecs)

    left_move_bad_elecs = subj_move_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[left_hipp_bad_elecs]
    left_move_bad_elecs_mean = np.nanmean(left_move_bad_elecs)
    right_move_bad_elecs = subj_move_bad_elecs.res['ts'][freqs_inds].mean(axis=0)[right_hipp_bad_elecs]
    right_move_bad_elecs_mean = np.nanmean(right_move_bad_elecs)

    if do_plot:
        y = [left_move_good_elecs_mean, left_move_bad_elecs_mean, right_move_good_elecs_mean, right_move_bad_elecs_mean]
        plt.bar(np.arange(0, 4)-.125, y, align='center', width=.25, zorder=5, color=[.5, .5, .5])
        y = [left_sme_good_elecs_mean, left_sme_bad_elecs_mean, right_sme_good_elecs_mean, right_sme_bad_elecs_mean]
        plt.bar(np.arange(0, 4)+.125, y, align='center', width=.25, zorder=5, color=[.5, .5, .5])
        plt.plot([-2,5], [0,0], '-k', lw=2)
        plt.xlim(-.5,3.5)
        plt.xticks(range(4), ['L. Good (%d)' % left_good_n,
                              'L. Bad (%d)' % left_bad_n,
                              'R. Good (%d)' % right_good_n,
                              'R. Bad (%d)' % right_bad_n])
        plt.ylabel('Navigation t-stat', fontsize=20)

        for i, elec_data in enumerate([left_move_good_elecs, left_move_bad_elecs, right_move_good_elecs, right_move_bad_elecs]):
            plt.plot([i-.125]*len(elec_data), elec_data, '.', c='k', markersize=24, zorder=10)
        for i, elec_data in enumerate([left_sme_good_elecs, left_sme_bad_elecs, right_sme_good_elecs, right_sme_bad_elecs]):
            plt.plot([i+.125]*len(elec_data), elec_data, '.', c='k', markersize=24, zorder=10)
        ylim = plt.ylim()
        plt.ylim(-np.max(np.abs(ylim)), np.max(np.abs(ylim)))
        plt.tight_layout()
        plt.savefig(save_dir + '%s_%s.pdf' % (subj[0], subj[1]))
        plt.show()


    res = {'left_move_good_elecs': left_move_good_elecs_mean,
           'right_move_good_elecs': right_move_good_elecs_mean,
           'left_move_bad_elecs': left_move_bad_elecs_mean,
           'right_move_bad_elecs': right_move_bad_elecs_mean,
           'left_good_n': left_good_n,
           'left_sme_good_elecs': left_sme_good_elecs_mean,
           'right_sme_good_elecs': right_sme_good_elecs_mean,
           'left_sme_bad_elecs': left_sme_bad_elecs_mean,
           'right_sme_bad_elecs': right_sme_bad_elecs_mean,
           'right_good_n': right_good_n,
           'left_bad_n': left_bad_n,
           'right_bad_n': right_bad_n}
    return res
