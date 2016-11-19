import numpy as np
import re
import os
import pdb
from ptsa.data.TimeSeriesX import TimeSeriesX
import cPickle as pickle
import cluster_helper.cluster
import pycircstat
import ram_data_helpers
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ttest_ind
from scipy.spatial.distance import squareform
from TH_load_features import *
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time

class stPPC:
    # default paths
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', task_phase='enc', bipolar=True, freqs=None,
                 start_time=-1.2, end_time=0.5, save_cv_features=True, pool=None):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task

        # classifying encoding or retrieval ('enc' | 'rec')
        self.task_phase = task_phase

        # this is stupid but don't let it precess R1132C - this subject didn't use the confident judgements so we can't
        #  work with the data
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute powe
        self.freqs = freqs

        # time bin to use
        self.start_time = start_time
        self.end_time = end_time

        self.save_cv_features = save_cv_features
        if save_cv_features:
            self.func = compute_stPPC_cross_val
        else:
            self.func = compute_stPPC

        # if doing parallel jobs, pool with be a cluster_helper object, otherwise None. Running with a pool is highly
        # recommended
        self.pool = pool

        # where to save data
        self.save_dir = os.path.join(stPPC.base_dir, task)

    def stPPC_run_for_all_subjs(self):
        for subj in self.subjs:
            print 'Processing %s.' % subj

            # create stPPC network for subject
            ppc_stats_subj, subj_save_dir = self.stPPC_single_subj(subj)

            # plot summary of network
            if not self.save_cv_features:
                self.plot_stPPC_network_subj(ppc_stats_subj, subj_save_dir, subj)

    def stPPC_single_subj(self, subj):
        # get electrode numbers and events
        elecs_bipol, elecs_monopol = ram_data_helpers.load_subj_elecs(subj)
        events = ram_data_helpers.load_subj_events(self.task, subj)

        # construct input to main prcoessing function
        elecs = elecs_bipol if self.bipolar else elecs_monopol
        num_elecs = len(elecs)
        elec_range = range(num_elecs)

        # info is the big list that map() iterates over
        keys = ['freqs', 'task_phase', 'start_time', 'end_time', 'bipolar', 'feat_type', 'buffer_len', 'save_chan', 'save_dir']
        vals = [self.freqs, self.task_phase, self.start_time, self.end_time, self.bipolar, 'phase_diff', 2.0, True,
                self.save_dir]
        params = {k: v for (k, v) in zip(keys, vals)}

        subj_save_dir = compute_subj_dir_path(params, events[0].subject)
        subj_save_dir_elecs = os.path.join(subj_save_dir, 'elec_data')
        subj_save_dir_features = os.path.join(subj_save_dir, 'ppc')
        params['subj_save_dir'] = subj_save_dir_elecs
        if not os.path.exists(subj_save_dir_elecs):
            os.makedirs(subj_save_dir_elecs)
        if not os.path.exists(subj_save_dir_features):
            os.makedirs(subj_save_dir_features)

        if self.save_cv_features:
            save_file = os.path.join(subj_save_dir, 'ppc', events[0].subject + '_features.p')
            if len(np.unique(events['session'])) == 1:
                print '%s: Cannot compute PPC features with only one session' % events[0].subject
                return None, None
        else:
            save_file = os.path.join(subj_save_dir, events[0].subject + '_stPPC.p')
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                ppc_stats = pickle.load(f)
        else:

            # if we have an open pool, send the iterable to the workers, otherwise normal map. This will load the phase
            #  data
            info = zip([elecs] * num_elecs, elec_range, [events] * num_elecs, [params] * num_elecs)
            if self.pool is None:
                feature_list = map(load_elec_func, info)
            else:
                feature_list = self.pool.map(load_elec_func, info)

            ppc_stats = self.stPPC_wrapper(feature_list, elecs, events)
            with open(save_file, 'wb') as f:
                pickle.dump(ppc_stats, f, protocol=-1)
        return ppc_stats, subj_save_dir

    def stPPC_wrapper(self, feature_list, elecs, events):
        """
        Computes single trial PPC for all electrode pairs.

        Outer loop - all combinations of electrode pairs (in parallel if pool exists)
        Inner loop - all combinations of trials
        """

        # compute all combinations of electrodes
        combs = np.array(list(combinations(range(len(elecs)), 2)))
        recalls = ram_data_helpers.filter_events_to_recalled(self.task, events)

        # list to iteratate over: [[elec1_file, elec2_file], recall boolean, session number]
        feature_pairs = [[feature_list[x[0]], feature_list[x[1]]] for x in combs]
        info = zip(feature_pairs, [recalls] * len(feature_pairs), [events['session']] * len(feature_pairs))
        if self.pool is None:
            ppc_stats = map(self.func, info)
        else:
            ppc_stats = self.pool.map(self.func, info)

        if self.save_cv_features:
            feat_array = np.stack(ppc_stats, axis=1)
            elec_pairs = [list(elecs[comb]) for comb in combs]
            elec_pairs_list = [list(i[0]) + list(i[1]) for i in elec_pairs]
            elec_pairs_array = np.recarray(shape=(len(combs)),
                                           dtype=[('ch0', '|S3'), ('ch1', '|S3'), ('ch2', '|S3'), ('ch3', '|S3')])
            for i, all_elecs in enumerate(elec_pairs_list):
                elec_pairs_array[i] = tuple(all_elecs)

            # new time series object
            new_ts = TimeSeriesX(data=feat_array, dims=['frequency', 'elec_pairs', 'events', 'sess_cv_fold'],
                                 coords={'frequency': self.freqs,
                                         'elec_pairs': elec_pairs_array,
                                         'events': events,
                                         'sess_cv_fold': range(feat_array.shape[3]),
                                         'start_time': self.start_time,
                                         'end_time': self.end_time})
            return new_ts
        else:
            return ppc_stats

    def plot_stPPC_network_subj(self, ppc_stats_subj, subj_save_dir, subj):

        plot_file = os.path.join(subj_save_dir,subj + '_connections.pdf')
        with PdfPages(plot_file) as pdf:
            for sess in range(np.shape(ppc_stats_subj[0][0])[0]):

                ts = [x[0][sess] for x in ppc_stats_subj]
                ts_mean = np.array([np.mean(x, axis=1) for x in ts]).T
                d = int(np.ceil(np.sqrt(ts_mean.shape[1] * 2)))
                squares_by_freqs = np.empty((d, d, len(self.freqs)))
                for i, data_freq in enumerate(ts_mean):
                    squares_by_freqs[:, :, i] = squareform(data_freq)

                fig, axs = plt.subplots(2, 4)
                fig.tight_layout()
                fig.set_size_inches(19, 8)

                # clim = 1
                for i, ax in enumerate(fig.axes):
                    clim = np.max(np.abs([squares_by_freqs[:, :, i].min(), squares_by_freqs[:, :, i].max()]))
                    heatmap = ax.imshow(squares_by_freqs[:, :, i],
                                        extent=[1, squares_by_freqs.shape[0], squares_by_freqs.shape[0], 1], cmap='RdBu_r',
                                        interpolation='none', vmin=-clim, vmax=clim)
                    divider1 = make_axes_locatable(ax)
                    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                    #         cbar1 = plt.colorbar(heatmap, cax=cax1, ticks=MultipleLocator(1), format="%.1f")
                    cbar1 = plt.colorbar(heatmap, cax=cax1, format="%.2f")
                    # fig.colorbar(heatmap)
                    ax.xaxis.set_tick_params(labeltop='on', labelbottom='off')
                    sessions = 'sessions: ' + ' '.join([str(x) for x in ppc_stats_subj[0][2][sess]])
                    ax.set_title(sessions + ', Freq: %.2f' % self.freqs[i], y=1.08)
                    ax.set_ylabel('Electrode y')
                    ax.yaxis.labelpad = 0
                    ax.set_xlabel('Electrode x')
                pdf.savefig()
                plt.close()

def compute_stPPC_cross_val(pair_data):
    """
    Does the single trial ppc calculation.
    """

    # load electrode 1 data
    elec1_f = pair_data[0][0]
    with open(elec1_f, 'rb') as f:
        elec_data1 = pickle.load(f)
        phase_data1 = elec_data1['phase_elec']

    # load electrode 2 data
    elec2_f = pair_data[0][1]
    with open(elec2_f, 'rb') as f:
        elec_data2 = pickle.load(f)
        phase_data2 = elec_data2['phase_elec']
    # print(elec1_f, elec2_f)
    # pull out recalled/not-recalled info and session info
    recalls = pair_data[1]
    sessions = pair_data[2]

    # we want to caluclate stPPC on all sessions aggregated together, as well as well as on all combinations of n-1
    # sessions
    rec_out_sess_bool = []
    non_rec_out_sess_bool = []
    in_sess_bool = []


    #rec_by_sess_bool.append(recalls)
    #non_rec_by_sess_bool.append(~recalls)
    uniq_sessions = list(np.unique(sessions))
    #sessions_used = [uniq_sessions]


    if len(uniq_sessions) > 1:
        # compute phase differences between electrode
        phase_diff = np.squeeze(phase_data1.data - phase_data2.data)

        # downsample (take every 10 samples)
        phase_diff = phase_diff[:, :, 0::10]

        in_sess_ppc = []
        out_sess_ppc = []

        # pdb.set_trace()
        ppc = np.empty((phase_diff.shape[0], phase_diff.shape[1], len(uniq_sessions)), dtype=np.float32)
        for sess_count, sess in enumerate(uniq_sessions):

            # out of sessions recalls and non recalls indices
            rec_out_sess_bool = recalls & (sessions != sess)
            non_rec_out_sess_bool = ~recalls & (sessions != sess)

            # in session, all events
            in_sess_bool = sessions == sess

            phase_diff_recalled = phase_diff[:, rec_out_sess_bool, :]
            phase_diff_non_recalled = phase_diff[:, non_rec_out_sess_bool, :]
            phase_diff_in_sess = phase_diff[:, in_sess_bool, :]

            phase_diff_out_sess = phase_diff[:, ~in_sess_bool, :]
            recalls_out_sess = recalls[~in_sess_bool]

            #### COMPUTATION OF OUT OF SAMPLE PPC
            out_sess_trial_ppc_rec = np.empty(phase_diff_out_sess.shape, dtype=np.float32)
            out_sess_trial_ppc_non_rec = np.empty(phase_diff_out_sess.shape, dtype=np.float32)
            pd_range = range(phase_diff_out_sess.shape[1])
            for i in pd_range:
                pd1 = phase_diff_out_sess[:, i, :]
                pd_rec = phase_diff_out_sess[:, (pd_range != i) and recalls_out_sess, :]
                out_sess_cos_trial_diffs_recalled = np.cos(pd1[:, :, np.newaxis] -
                                                           np.transpose(pd_rec, (0, 2, 1)))
                out_sess_trial_ppc_rec[:, i, :] = np.mean(out_sess_cos_trial_diffs_recalled, axis=2)

                pd_non_rec = phase_diff_out_sess[:, (pd_range != i) and ~recalls_out_sess, :]
                out_sess_cos_trial_diffs_non_recalled = np.cos(pd1[:, :, np.newaxis] -
                                                               np.transpose(pd_non_rec, (0, 2, 1)))
                out_sess_trial_ppc_non_rec[:, i, :] = np.mean(out_sess_cos_trial_diffs_non_recalled, axis=2)

                # out_sess_cos_trial_diffs_recalled = np.cos(pd1[:, :, np.newaxis] -
                #                                           np.transpose(phase_diff_recalled, (0, 2, 1)))
                # out_sess_trial_ppc_rec[:, i, :] = np.mean(out_sess_cos_trial_diffs_recalled, axis=2)

                # out_sess_cos_trial_diffs_non_recalled = np.cos(pd1[:, :, np.newaxis] -
                #                                               np.transpose(phase_diff_non_recalled, (0, 2, 1)))
                # out_sess_trial_ppc_non_rec[:, i, :] = np.mean(out_sess_cos_trial_diffs_non_recalled, axis=2)

            out_sess_ppc_diff = out_sess_trial_ppc_rec - out_sess_trial_ppc_non_rec
            ppc[:, ~in_sess_bool, sess_count] = np.mean(out_sess_ppc_diff, axis=2)
            # out_sess_ppc.append(np.mean(out_sess_ppc_diff, axis=2))


            # compute ppc for recalled
            # combos = np.array(list(combinations(range(phase_diff_recalled.shape[1]), 2)))
            # cos_trial_diffs_recalled = np.empty(
            #     [phase_diff_recalled.shape[0], phase_diff_recalled.shape[2], combos.shape[0]],
            #     dtype=np.float32)
            # for i in range(combos.shape[0]):
            #     pds1 = phase_diff_recalled[:, combos[i, 0], :]
            #     pds2 = phase_diff_recalled[:, combos[i, 1], :]
            #     cos_trial_diffs_recalled[:, :, i] = np.cos(pds1 - pds2)
            #
            # trial_ppc_rec = np.empty(phase_diff_recalled.shape)
            # for i in range(phase_diff_recalled.shape[1]):
            #     keep_combos_index = np.sum(combos == i, 1).astype(bool)
            #     trial_ppc_rec[:, i, :] = np.mean(cos_trial_diffs_recalled[:, :, keep_combos_index], axis=2)

            # # compute ppc for non recalled
            # combos = np.array(list(combinations(range(phase_diff_non_recalled.shape[1]), 2)))
            # cos_trial_diffs_non_recalled = np.empty([phase_diff_non_recalled.shape[0], phase_diff_non_recalled.shape[2],
            #                                          combos.shape[0]], dtype=np.float32)
            # for i in range(combos.shape[0]):
            #     pds1 = phase_diff_non_recalled[:, combos[i, 0], :]
            #     pds2 = phase_diff_non_recalled[:, combos[i, 1], :]
            #     cos_trial_diffs_non_recalled[:, :, i] = np.cos(pds1 - pds2)
            #
            # trial_ppc_non_rec = np.empty(phase_diff_non_recalled.shape)
            # for i in range(phase_diff_non_recalled.shape[1]):
            #     keep_combos_index = np.sum(combos == i, 1).astype(bool)
            #     trial_ppc_non_rec[:, i, :] = np.mean(cos_trial_diffs_non_recalled[:, :, keep_combos_index], axis=2)

            ### Compare in sample trials to out of sample recalled and non recalled distributions
            in_sess_trial_ppc_rec = np.empty(phase_diff_in_sess.shape, dtype=np.float32)
            in_sess_trial_ppc_non_rec = np.empty(phase_diff_in_sess.shape, dtype=np.float32)
            for i in range(phase_diff_in_sess.shape[1]):
                pd1 = phase_diff_in_sess[:, i, :]

                in_sess_cos_trial_diffs_recalled = np.cos(pd1[:, :, np.newaxis] -
                                                          np.transpose(phase_diff_recalled, (0, 2, 1)))
                in_sess_trial_ppc_rec[:, i, :] = np.mean(in_sess_cos_trial_diffs_recalled, axis=2)

                in_sess_cos_trial_diffs_non_recalled = np.cos(pd1[:, :, np.newaxis] -
                                                              np.transpose(phase_diff_non_recalled, (0, 2, 1)))
                in_sess_trial_ppc_non_rec[:, i, :] = np.mean(in_sess_cos_trial_diffs_non_recalled, axis=2)

            in_sess_ppc_diff = in_sess_trial_ppc_rec - in_sess_trial_ppc_non_rec
            ppc[:, in_sess_bool, sess_count] = np.mean(in_sess_ppc_diff, axis=2)
            # in_sess_ppc.append(np.mean(in_sess_ppc_diff, axis=2))
            # pdb.set_trace()
        # in_sess_ppc = np.concatenate(in_sess_ppc, axis=1)
        # pdb.set_trace()
        return ppc
    else:
        return None

def compute_stPPC(pair_data):
    """
       Does the single trial ppc calculation.
       """

    # load electrode 1 data
    elec1_f = pair_data[0][0]
    with open(elec1_f, 'rb') as f:
        elec_data1 = pickle.load(f)
        phase_data1 = elec_data1['phase_elec']

    # load electrode 2 data
    elec2_f = pair_data[0][1]
    with open(elec2_f, 'rb') as f:
        elec_data2 = pickle.load(f)
        phase_data2 = elec_data2['phase_elec']

    # pull out recalled/not-recalled info and session info
    recalls = pair_data[1]
    sessions = pair_data[2]

    # we want to caluclate stPPC on all sessions aggregated together, as well as well as on all combinations of n-1
    # sessions
    rec_by_sess_bool = []
    non_rec_by_sess_bool = []
    rec_by_sess_bool.append(recalls)
    non_rec_by_sess_bool.append(~recalls)
    uniq_sessions = list(np.unique(sessions))
    sessions_used = [uniq_sessions]

    if len(uniq_sessions) > 1:
        for sess in uniq_sessions:
            rec_by_sess_bool.append(recalls & (sessions != sess))
            non_rec_by_sess_bool.append(~recalls & (sessions != sess))
            sessions_used.append(list(set(uniq_sessions) - set([sess])))
    # t0 = time()
    # compute phase differences between electrode
    phase_diff = np.squeeze(phase_data1.data - phase_data2.data)

    # downsample (take every 10 samples)
    phase_diff = phase_diff[:, :, 0::10]

    # slice into all recalled trials and all non-recalled trials, looping over aggregate and hold out sessions
    ts = []
    pval = []
    for rec_info in zip(rec_by_sess_bool, non_rec_by_sess_bool):
        rec_sess = rec_info[0]
        non_rec_session = rec_info[1]

        phase_diff_recalled = phase_diff[:, rec_sess, :]
        phase_diff_non_recalled = phase_diff[:, non_rec_session, :]

        # compute ppc for recalled
        combos = np.array(list(combinations(range(phase_diff_recalled.shape[1]), 2)))
        cos_trial_diffs_recalled = np.empty(
            [phase_diff_recalled.shape[0], phase_diff_recalled.shape[2], combos.shape[0]],
            dtype=np.float32)
        for i in range(combos.shape[0]):
            pds1 = phase_diff_recalled[:, combos[i, 0], :]
            pds2 = phase_diff_recalled[:, combos[i, 1], :]
            cos_trial_diffs_recalled[:, :, i] = np.cos(pds1 - pds2)

        trial_ppc_rec = np.empty(phase_diff_recalled.shape)
        for i in range(phase_diff_recalled.shape[1]):
            keep_combos_index = np.sum(combos == i, 1).astype(bool)
            trial_ppc_rec[:, i, :] = np.mean(cos_trial_diffs_recalled[:, :, keep_combos_index], axis=2)

        # compute ppc for non recalled
        combos = np.array(list(combinations(range(phase_diff_non_recalled.shape[1]), 2)))
        cos_trial_diffs_non_recalled = np.empty([phase_diff_non_recalled.shape[0], phase_diff_non_recalled.shape[2],
                                                 combos.shape[0]], dtype=np.float32)
        for i in range(combos.shape[0]):
            pds1 = phase_diff_non_recalled[:, combos[i, 0], :]
            pds2 = phase_diff_non_recalled[:, combos[i, 1], :]
            cos_trial_diffs_non_recalled[:, :, i] = np.cos(pds1 - pds2)

        trial_ppc_non_rec = np.empty(phase_diff_non_recalled.shape)
        for i in range(phase_diff_non_recalled.shape[1]):
            keep_combos_index = np.sum(combos == i, 1).astype(bool)
            trial_ppc_non_rec[:, i, :] = np.mean(cos_trial_diffs_non_recalled[:, :, keep_combos_index], axis=2)

        # ttest recalled vs not recalled
        ts_sess, pval_sess = ttest_ind(trial_ppc_rec, trial_ppc_non_rec, axis=1)
        ts.append(ts_sess)
        pval.append(pval_sess)

    # print("done in %0.3fs" % (time() - t0))

    return ts, pval, sessions_used
