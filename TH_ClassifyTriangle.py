import numpy as np
import re
import os
from glob import glob
import pdb
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ranksums, ttest_1samp, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import cPickle as pickle
import ram_data_helpers
from TH_load_features import load_features, compute_subj_dir_path
import behavioral.add_conf_time_to_events
import create_stPPC_network
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.distance import squareform
from xray import concat


def compute_terciles(subj_res):
    binned_data = binned_statistic(subj_res['probs'], subj_res['classes'], statistic='mean',
                                   bins=np.percentile(subj_res['probs'], [0, 33, 67, 100]))
    tercile_delta_rec = (binned_data[0] - np.mean(subj_res['classes'])) / np.mean(subj_res['classes']) * 100
    return tercile_delta_rec


def classify_par(info):
    #self, subj, recalls, sessions, cv_sel, feat_mat, task_phase, good events
    params = info[0]
    recalls = info[1][info[6]]
    sessions = info[2][info[6]]
    cv_sel = info[3][info[6]]
    task_phase = info[4][info[6]]
    feat_mat = info[5][info[6]]
    # good_ev = info[6]

    # normalize data by session (this only makes sense for power?)
    if params['feat_type'] == 'power':
        uniq_sessions = np.unique(sessions)
        for sess in uniq_sessions:
            sess_event_mask = (sessions == sess)
            feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

    # initialize classifier
    lr_classifier = LogisticRegression(C=params['C'], penalty=params['norm'], class_weight='auto', solver='liblinear')

    # hold class probability
    probs = np.empty_like(recalls, dtype=np.float)

    # loop over each train set and classify test
    uniq_cv = np.unique(cv_sel)
    for cv_num, cv in enumerate(uniq_cv):
        # create mask of just training data
        mask_train = (cv_sel != cv)
        feats_to_use_bool = np.ones((feat_mat.shape[1]), dtype=bool)
        feats_train = feat_mat[mask_train]
        feats_test = feat_mat[(~mask_train) & (task_phase == params['test_phase'])]
        rec_train = recalls[mask_train]

        x_train = zscore(feats_train[:, feats_to_use_bool], axis=0)
        x_test = zmap(feats_test[:, feats_to_use_bool], feats_train[:, feats_to_use_bool], axis=0)

        # feat_mat_reduced = feat_mat[:, feats_to_use]
        # pow_train = zscore(feat_mat_reduced[mask_train], axis=0)
        # lr_classifier.fit(pow_train, rec_train)
        lr_classifier.fit(x_train, rec_train)

        # now estimate classes of train data
        # pow_test = zmap(feat_mat_reduced[~mask_train], feat_mat_reduced[mask_train], axis=0)
        # rec_train = recalls[~mask_train]
        # test_probs = lr_classifier.predict_proba(pow_test)[:, 1]
        test_probs = lr_classifier.predict_proba(x_test)[:, 1]
        probs[(~mask_train) & (task_phase == params['test_phase'])] = test_probs

    # compute AUC
    # probs = probs[task_phase==self.test_phase]
    # recalls = recalls[task_phase==self.test_phase]
    auc = roc_auc_score(recalls[task_phase == params['test_phase']], probs[task_phase == params['test_phase']])

    # output results, including AUC, lr object (fit to all data), prob estimates, and class labels
    subj_res = {}
    subj_res['auc'] = auc
    print auc

    subj_res['lr_classifier'] = lr_classifier.fit(feat_mat, recalls)
    subj_res['probs'] = probs[task_phase == params['test_phase']]
    subj_res['classes'] = recalls[task_phase == params['test_phase']]
    subj_res['tercile'] = compute_terciles(subj_res)
    subj_res['sessions'] = sessions
    subj_res['cv_sel'] = cv_sel
    subj_res['task_phase'] = task_phase
    subj_res['num_events_all'] = len(info[6])
    subj_res['num_events_used'] = np.sum(info[6])
    subj_res['cv_type'] = 'loso' if len(np.unique(sessions)) > 1 else 'lolo'

    # also compute forward model feature importance for each electrode if we are using power
    if params['feat_type'] in ['power', 'pow_by_phase']:
        probs_log = np.log(probs / (1 - probs))
        covx = np.cov(feat_mat.T)
        covs = np.cov(probs_log)
        W = subj_res['lr_classifier'].coef_
        A = np.dot(covx, W.T) / covs
        ts, ps = ttest_ind(feat_mat[recalls], feat_mat[~recalls])
        if params['feat_type'] == 'power':
            subj_res['forward_model'] = np.reshape(A, (-1, len(params['freqs'])))
            subj_res['univar_ts'] = np.reshape(ts, (-1, len(params['freqs'])))
        else:
            subj_res['forward_model'] = np.reshape(A, (-1, params['freq_bands'].shape[0] * params['num_phase_bins']))
            subj_res['univar_ts'] = np.reshape(ts, (-1, params['freq_bands'].shape[0] * params['num_phase_bins']))
    return subj_res


class ClassifyTH:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    data_path = '/data/eeg'
    save_dir = '/scratch/jfm2/python/TH'
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', train_phase='enc', test_phase='enc', bipolar=True,
                 freqs=None, freq_bands=None, hilbert_phase_band=None, num_phase_bins=None, start_time=-2.0,
                 end_time=2.0, step_size=0.1, window_size=np.arange(0.1, 4.1, .1), norm='l2', feat_type='power',
                 do_rand=False, recall_filter_func=None, rec_thresh=None,
                 force_reclass=False, C=7.2e-4, ROIs=None, pool=None):

        # if subjects not given, get list from /data/events/ directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task

        # task phase to train on ('enc', 'rec', or 'both')
        self.train_phase = train_phase

        # task phase to test fit model on ('enc' or 'rec', 'both' not currently supported)
        self.test_phase = test_phase

        # this is stupid but don't let it precess some subjects. R1132C didn't use the confident judgements so we can't
        # work with the data.
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1219C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1227T']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute power
        self.freqs = freqs

        # these following settings apply to the 'pow_by_phase' feature type
        self.freq_bands = freq_bands
        self.hilbert_phase_band = hilbert_phase_band
        self.num_phase_bins = num_phase_bins

        # start and end time to use, relative to events. If train_phase is 'both', enter a list of two times for start
        # and end (encoding time first)
        self.start_time = start_time
        self.end_time = end_time

        # compute all time bins based on start_time, end_time, step_size, and window_size
        edges = np.arange(start_time*1000, end_time*1000, step_size*1000)
        if int(np.ptp([start_time, end_time])*1000) % int(step_size * 1000) == 0:
            edges = np.append(edges, end_time*1000)
        bin_centers = np.mean(np.array(zip(edges, edges[1:])), axis=1)
        self.time_bins = np.array([[x - window / 2, x + window / 2] for x in bin_centers for window in
                                   (window_size*1000).astype(int) if ((x + window / 2 <= end_time*1000) &
                                                                      (x - window / 2 >= start_time*1000))])/1000

        # type of regularization, penalty parameter
        self.norm = norm
        self.C = C

        if recall_filter_func is None:
            self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        else:
            self.recall_filter_func = recall_filter_func
        self.rec_thresh = rec_thresh
        self.force_reclass = force_reclass

        # to restrict to just specific ROIs, enter a list of ROIs
        self.ROIs = ROIs

        # features to use for regression
        self.feat_type = feat_type
        self.mean_pow = False if self.feat_type == 'pow_by_phase' else True
        self.do_rand = do_rand

        # if doing parallel jobs, pool with be a cluster_helper object, otherwise None
        self.pool = pool

        # where to save data
        self.base_dir = os.path.join(ClassifyTH.base_dir, task)

        # holds the output from all subjects
        self.res = None
        self.df = None

    def run_classify_for_all_subjs(self):
        class_data_all_subjs = []
        for subj in self.subjs:
            print 'Processing %s.' % subj

            # define base directory for subject. Name the directory based on a variety of parameters so things stay
            # reasonably organized on disk
            f1 = self.freqs[0]
            f2 = self.freqs[-1]
            bipol_str = 'bipol' if self.bipolar else 'mono'
            tbin_str = '1_bin' if self.time_bins is None else str(self.time_bins.shape[0])+'_bins'
            if self.train_phase != 'both':
                subj_base_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f_%s' % (len(self.freqs), f1, f2, bipol_str),
                                             'triangle_%s_start_%.1f_stop_%.1f' % (self.train_phase, self.start_time,
                                                                                   self.end_time), tbin_str, subj)
            else:
                subj_base_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f_%s' % (len(self.freqs), f1, f2, bipol_str),
                                             'triangle_enc_start_%.1f_stop_%.1f_rec_%.1f_stop_%.1f' %
                                             (self.start_time[0], self.end_time[0], self.start_time[1], self.end_time[1]),
                                             tbin_str, subj)

            # sub directories hold electrode data, feature data, and classifier output
            subj_elec_dir = os.path.join(subj_base_dir, 'elec_data')
            subj_feature_dir = os.path.join(subj_base_dir, '%s' % self.feat_type)
            subj_class_dir = os.path.join(subj_base_dir, 'C_%.8f_norm_%s_%s' % (self.C, self.norm, self.recall_filter_func.__name__))

            # this holds the classifier results
            save_file = os.path.join(subj_class_dir, subj + '_' + self.feat_type + '.p')

            # features used for classification
            feat_file = os.path.join(subj_feature_dir, subj + '_features.p')

            # make sure we can even load the events
            try:
                events = ram_data_helpers.load_subj_events(self.task, subj, self.train_phase)
                # if self.train_phase == 'rec':
                #     n1 = len(events)
                #     events = behavioral.add_conf_time_to_events.process_event_file(events)
                #     events = events[events.move_rt > self.end_time * 1000]
                #     print('Removed %d of %d events (%.2f%%)' % (n1 - len(events), n1, (len(events))/float(n1)))
            except (ValueError, AttributeError, IOError):
                print 'Error processing %s. Could not load events.' % subj

            # If features exist, we will not overwrite them unless the number of events has changes (ie., a new session
            # was run)
            overwrite_features = False

            # If the classifier file already exists, load it. Again, only rerun if number of events has changed.
            if os.path.exists(save_file):
                with open(save_file, 'rb') as f:
                    subj_data = pickle.load(f)
                if len(events) == len(subj_data['events']):
                    if not self.force_reclass:
                        print 'Classifier exists for %s. Skipping.' % subj
                        class_data_all_subjs.append(subj_data)
                        self.res = class_data_all_subjs
                        continue
                else:
                    overwrite_features = True

            # Check the features vs the current events file
            if os.path.exists(feat_file):
                with open(feat_file, 'rb') as f:
                    feat_data = pickle.load(f)
                if len(events) != len(feat_data['events']):
                    overwrite_features = True

                # this is stupid, but this subject's events on disk don't align with features because some of the events
                # are too close the edge of the eeg file
                if subj == 'R1208C':
                    overwrite_features = False

            # make directories if missing
            if not os.path.exists(subj_class_dir):
                try:
                    os.makedirs(subj_class_dir)
                except OSError:
                    pass
                try:
                    os.makedirs(subj_feature_dir)
                except OSError:
                    pass
                try:
                    os.makedirs(subj_elec_dir)
                except OSError:
                    pass

            # run classifier for subject
            try:
                subj_classify = self.run_classify_for_single_subj(subj, feat_file,
                                                                  subj_elec_dir, overwrite_features)
                class_data_all_subjs.append(subj_classify)
                with open(save_file, 'wb') as f:
                    pickle.dump(subj_classify, f, protocol=-1)
            # except (ValueError, AttributeError, IOError, OSError):
            except:
              print 'Error processing %s.' % subj
            self.res = class_data_all_subjs

        subjs = []
        t_ax = []
        w_ax = []
        aucs = []
        cv_type = []
        for subj_res in self.res:
            subjs.extend([subj_res['subj']] * len(subj_res['aucs']))
            cv_type.extend([subj_res['cv_type']] * len(subj_res['aucs']))
            t_ax.extend(subj_res['time_bin'].astype(str))
            w_ax.extend(subj_res['window_size'].astype(str))
            aucs.extend(subj_res['aucs'])
        tuples = list(zip(cv_type, subjs, np.array(t_ax).astype(float), np.array(w_ax).astype(float)))
        index = pd.MultiIndex.from_tuples(tuples, names=['cv_type', 'subject', 'time', 'window'])
        self.df = pd.DataFrame(aucs, index=index, columns=['AUC']).T

    def run_classify_for_single_subj(self, subj, feat_file, subj_elec_dir, overwrite_features):
        """
        Runs logistic regression classifier to predict recalled/not-recalled items. Logic is as follows:
        1) Create classifier features, if they do not exist.
        2) Classify using leave one session out cross validation. If only one session, use leave one list out
        """

        # check if features exist. If not, or we are overwriting, create.
        if not os.path.exists(feat_file) or overwrite_features or self.feat_type[-2:] == '_r':
            print '%s features do not exist or overwriting for %s. Creating.' % (self.feat_type, subj)

            # if not training on both
            if self.train_phase != 'both':
                subj_features = load_features(subj, self.task, self.train_phase, self.start_time, self.end_time,
                                              self.time_bins, self.freqs, self.freq_bands, self.hilbert_phase_band,
                                              self.num_phase_bins, self.bipolar, self.feat_type, self.mean_pow,
                                              False, subj_elec_dir, self.ROIs, self.pool)

            # if training on both encoding and recall, load both and concat
            else:
                start_time = self.start_time[0]
                end_time = self.end_time[0]
                subj_features_enc = load_features(subj, self.task, 'enc', start_time, end_time,
                                                  self.freqs, self.freq_bands, self.hilbert_phase_band,
                                                  self.num_phase_bins, self.bipolar, self.feat_type, self.mean_pow,
                                                  False, subj_elec_dir, self.ROIs, self.pool)
                start_time = self.start_time[1]
                end_time = self.end_time[1]
                subj_features_rec = load_features(subj, self.task, 'rec', start_time, end_time,
                                                  self.freqs, self.freq_bands, self.hilbert_phase_band,
                                                  self.num_phase_bins, self.bipolar, self.feat_type, self.mean_pow,
                                                  False, subj_elec_dir, self.ROIs, self.pool)
                subj_features = concat([subj_features_enc, subj_features_rec], dim='events')

            # save features to disk
            with open(feat_file, 'wb') as f:
                pickle.dump(subj_features, f, protocol=-1)

        # if exist, load from disk
        else:
            with open(feat_file, 'rb') as f:
                subj_features = pickle.load(f)

        # if self.train_phase == 'rec':
        #     subj_features = subj_features[:, :, subj_features.events.data['move_rt'] > self.end_time * 1000.0, :]

        # determine classes
        recalls = self.recall_filter_func(self.task, subj_features.events.data, self.rec_thresh)

        # give encoding and retrieval events a common name regardless of task
        task_phase = subj_features.events.data['type']
        if self.task == 'RAM_TH1':
            task_phase[task_phase == 'CHEST'] = 'enc'
            task_phase[task_phase == 'REC'] = 'rec'
        elif self.task == 'RAM_FR1':
            task_phase[task_phase == 'WORD'] = 'enc'
            task_phase[task_phase == 'REC_WORD'] = 'rec'

        # create cross validation labels
        sessions = subj_features.events.data['session']
        if len(np.unique(subj_features.events.data['session'])) > 1:
            cv_sel = sessions
        else:
            trial_str = 'trial' if self.task == 'RAM_TH1' else 'list'
            cv_sel = subj_features.events.data[trial_str]

        # classify each time bin separately
        dims = subj_features.dims
        subj_features = subj_features.transpose(dims[3], dims[2], dims[1], dims[0])
        aucs = []
        terciles = []

        params = {'feat_type': self.feat_type,
                  'C': self.C,
                  'norm': self.norm,
                  'test_phase': self.test_phase,
                  'freqs': self.freqs,
                  'freq_bands': self.freq_bands,
                  'num_phase_bins': self.num_phase_bins}
        num_bins = subj_features.shape[0]
        features_list = []
        good_ev = []

        for t, features_tbin in enumerate(subj_features):
            features_list.append(features_tbin.data.reshape(features_tbin.data.shape[0], -1))
            if self.train_phase == 'rec':
                good_ev.append(subj_features.events.data['move_rt'] >
                               (features_tbin.time.data + features_tbin.window_size[t]/2.) * 1000.)
            else:
                good_ev.append(np.ones(subj_features.events.data.shape, dtype=bool))

        info = zip([params] * num_bins, [recalls] * num_bins, [sessions] * num_bins, [cv_sel] * num_bins,
                   [task_phase] * num_bins, features_list, good_ev)

        if self.pool is None:
            res = map(classify_par, info)
        else:
            res = self.pool.map(classify_par, info)


        subj_res = {'aucs': np.array([x['auc'] for x in res]),
                    'num_events_all': np.array([x['num_events_all'] for x in res]),
                    'num_events_used': np.array([x['num_events_used'] for x in res]),
                    'time_bin': subj_features.time.data,
                    'window_size': subj_features.window_size,
                    'subj': subj,
                    'terciles': np.stack([x['tercile'] for x in res]),
                    'cv_type': res[0]['cv_type'],
                    'events': subj_features['events']}
        subj_res['perc_events_used'] = subj_res['num_events_all']/subj_res['num_events_all']

        return subj_res

    def plot_terciles(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get terciles from specified subjects with specifed cv_type
        terciles = np.stack([x['tercile'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])

        yerr = sem(terciles, axis=0) * 1.96
        plt.bar(range(3), np.nanmean(terciles, axis=0), align='center', color=[.5, .5, .5], linewidth=2, yerr=yerr,
                error_kw={'ecolor': 'k', 'linewidth': 2, 'capsize': 5, 'capthick': 2})

    def plot_auc_hist(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get aucs from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])

        hist = np.histogram(aucs, np.linspace(0.025, .975, 20))
        plt.bar(np.linspace(0.05, .95, 19), hist[0].astype(np.float) / np.sum(hist[0]), .05, align='center',
                color=[.5, .5, .5], linewidth=2)

    def aucs(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get terciles from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])
        return aucs

    def mean_auc(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get terciles from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])
        return aucs.mean()

    def print_res_table(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        table = np.stack([[x['subj'], x['auc'], x['cv_type'], self.feat_type, np.mean(x['classes'])]
                          for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])

        print 'Subject\tAUC\tCV\tfeatures\t%% Rec'
        for row in table:
            print '%s\t%.3f\t%s\t%s' % (row[0][1:], float(row[1]), row[2], row[3])

    def compute_feature_heatmap(self, subjs=None, cv_type=('loso', 'lolo'),
                                regions=('IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC')):
        if subjs is None:
            subjs = self.subjs

        # concatenate all foward model features into one 3d array (freqs x regions x subjs)
        freq_x_regions_x_subjs = []
        for subj_res in self.res:
            if (subj_res['subj'] in subjs) and (subj_res['cv_type'] in cv_type):
                loc_dict = ram_data_helpers.bin_elec_locs(subj_res['loc_tag'], subj_res['anat_region'])
                A = subj_res['forward_model']
                freq_x_regions = np.empty(A.shape[1] * len(regions)).reshape(A.shape[1], len(regions))
                freq_x_regions[:] = np.nan
                for i, r in enumerate(regions):
                    freq_x_regions[:, i] = A[loc_dict[r], :].mean(axis=0)
                freq_x_regions_x_subjs.append(freq_x_regions)
        feature_heatmap = np.stack(freq_x_regions_x_subjs, -1)

        # plot heatmap. If just one subject, plot actual forward model weights. Otherwise, plot t-stat against zero
        if feature_heatmap.shape[2] > 1:
            plot_data, p = ttest_1samp(feature_heatmap, 0, axis=2, nan_policy='omit')
            colorbar_str = 't-stat'
        else:
            plot_data = np.nanmean(feature_heatmap, axis=2)
            colorbar_str = 'Feature Importance'

        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        fig, ax = plt.subplots(1, 1)
        plt.imshow(plot_data[::-1], interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
        cb = plt.colorbar()
        cb.set_label(label=colorbar_str, size=16)  # ,rotation=90)
        cb.ax.tick_params(labelsize=12)
        plt.xticks(range(len(regions)), regions, fontsize=16, rotation=-45)
        plt.yticks(range(0, len(self.freqs), 7)[::-1], np.round(self.freqs[range(0, len(self.freqs), 7)] * 10) / 10,
                   fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tight_layout()
        return feature_heatmap

    def plot_triangle(self, subjs=None, cv_type=('loso', 'lolo'),
                      regions=('IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC')):

        if subjs is None:
            subjs = self.subjs

        aucs_all = []
        for subj_res in self.res:
            if (subj_res['subj'] in subjs) and (subj_res['cv_type'] in cv_type):
                full_time_axis = np.round(subj_res['time_bin'] * 1000) / 1000
                uniq_full_time_axis = np.unique(full_time_axis)

                windows = np.round(subj_res['window_size'] * 1000) / 1000
                uniq_windows = np.unique(windows)

                aucs = np.empty((uniq_windows.shape[0], uniq_full_time_axis.shape[0]))
                aucs[:] = np.nan

                for row, window in enumerate(uniq_windows):
                    for col, t_bin in enumerate(uniq_full_time_axis):
                        bin = (windows == window) & (full_time_axis == t_bin)
                        if np.any(bin):
                            aucs[row, col] = subj_res['aucs'][bin]
                aucs_all.append(aucs)
        aucs_all = np.stack(aucs_all, -1)

        if aucs_all.shape[2] > 1:
            plot_data = np.nanmean(aucs_all, axis=2)
        else:
            plot_data = np.squeeze(aucs_all, axis=2)

        fig, ax = plt.subplots(1, 1)
        plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', aspect='auto', zorder=2)
        plt.xticks(range(uniq_full_time_axis.shape[0])[::3], uniq_full_time_axis[::3], rotation=45, fontsize=16)
        plt.xlabel('Bin Center (s)', fontsize=16)

        plt.yticks(range(uniq_windows.shape[0])[::3], uniq_windows[::3], fontsize=16)
        plt.ylabel('Window Size (s)', fontsize=16)
        plt.gca().invert_yaxis()

        clim = np.abs(np.array([np.nanmin(plot_data), np.nanmax(plot_data)]) - .5).max()
        plt.clim(.5 - clim, .5 + clim)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)

        ax.set_axisbelow(True)
        ax.yaxis.grid()
        ax.xaxis.grid()

        row, col = np.unravel_index(np.nanargmax(plot_data), plot_data.shape)
        best_time = uniq_full_time_axis[col]
        best_window = uniq_windows[row]
        return plot_data, best_time, best_window


    @classmethod
    def compute_terciles(cls, subj_res):
        binned_data = binned_statistic(subj_res['probs'], subj_res['classes'], statistic='mean',
                                       bins=np.percentile(subj_res['probs'], [0, 33, 67, 100]))
        tercile_delta_rec = (binned_data[0] - np.mean(subj_res['classes'])) / np.mean(subj_res['classes']) * 100
        return tercile_delta_rec

