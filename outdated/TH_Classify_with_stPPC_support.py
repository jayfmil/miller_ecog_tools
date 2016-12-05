import numpy as np
import re
import os
from glob import glob
import pdb
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ranksums, ttest_1samp
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import roc_auc_score, roc_curve
import cPickle as pickle
import cluster_helper.cluster
import ram_data_helpers
from TH_load_features import load_features, compute_subj_dir_path
import create_stPPC_network
import matplotlib
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.distance import squareform


class ClassifyTH:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    data_path = '/data/eeg'
    save_dir = '/scratch/jfm2/python/TH'
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', task_phase='enc', bipolar=True, freqs=None, freq_bands=None,
                 hilbert_phase_band=None, num_phase_bins=None, start_time=-1.2, end_time=0.5, norm='l2',
                 feat_type='power', stPPC_filtering=False, reduce_features=False, do_rand=False, C=7.2e-4, pool=None):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs
        print self.subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task

        # classifying encoding or retrieval ('enc' | 'rec')
        self.task_phase = task_phase

        # this is stupid but don't let it precess R1132C - this subject didn't use the confident judgements so we can't
        #  work with the data
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1219C']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute power
        self.freqs = freqs
        self.freq_bands = freq_bands
        self.hilbert_phase_band = hilbert_phase_band
        self.num_phase_bins = num_phase_bins
        # if self.freqs is not None and self.bands is not None:
        #     print 'Define either freqs or bands'

        # time bin to use
        self.start_time = start_time
        self.end_time = end_time

        # type of regularization, penalty parameter
        self.norm = norm
        self.C = C

        # features to use for regression
        self.feat_type = feat_type
        self.mean_pow = False if self.feat_type == 'pow_by_phase' else True
        self.do_rand = do_rand

        # if reduce_features, compute the most cignificant features within each training set and use # elecs * # freqs
        self.reduce_features = reduce_features
        self.stPPC_filtering = stPPC_filtering

        # if doing parallel jobs, pool with be a cluster_helper object, otherwise None
        self.pool = pool

        # where to save data
        self.base_dir = os.path.join(ClassifyTH.base_dir, task)

        ####
        self.res = None

    def run_classify_for_all_subjs(self):
        class_data_all_subjs = []
        for subj in self.subjs:
            print 'Processing %s.' % subj

            # define base directory for subject
            f1 = self.freqs[0]
            f2 = self.freqs[-1]
            subj_base_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f' % (len(self.freqs), f1, f2),
                                         '%s_start_%.1f_stop_%.1f' % (self.task_phase, self.start_time, self.end_time),
                                         subj)

            # sub directories hold electrode data, feature data, and classifier output
            subj_elec_dir = os.path.join(subj_base_dir, 'elec_data')
            subj_feature_dir = os.path.join(subj_base_dir, '%s' % self.feat_type)
            subj_class_dir = os.path.join(subj_base_dir, 'C_%.8f_norm_%s' % (self.C, self.norm))

            # this holds the classifier results
            save_file = os.path.join(subj_class_dir, subj + '_' + self.feat_type + '.p')
            stPPC_file = None
            if self.feat_type != 'power' and self.stPPC_filtering:
                save_file = os.path.join(subj_class_dir, subj + '_' + self.feat_type + '_stPPC.p')
                stPPC_file = os.path.join(subj_base_dir, subj + '_stPPC.p')

            # features used for classification
            feat_file = os.path.join(subj_feature_dir, subj + '_features.p')

            events = ram_data_helpers.load_subj_events(self.task, subj)

            # pdb.set_trace()
            # only run if file doesn't exist or number of events in saved data differs from current events structure
            # this isn't quite right. what if features exist but not results

            ############################################
            overwrite_features = False #################

            if os.path.exists(save_file):
                with open(save_file, 'rb') as f:
                    subj_data = pickle.load(f)
                events = ram_data_helpers.load_subj_events(self.task, subj)
                if len(events) == len(subj_data['events']):
                    print 'Classifier exists for %s. Skipping.' % subj
                    class_data_all_subjs.append(subj_data)
                    self.res = class_data_all_subjs
                    continue
                else:
                    # class_data_all_subjs.append(subj_data)
                    # self.res = class_data_all_subjs
                    # continue
                    overwrite_features = True
            # else:
            #     continue
            if os.path.exists(feat_file):
                with open(feat_file, 'rb') as f:
                    feat_data = pickle.load(f)
                events = ram_data_helpers.load_subj_events(self.task, subj)
                if len(events) != len(feat_data['events']):
                    overwrite_features = True
                if subj == 'R1208C':
                    overwrite_features = False

                # make directory if missing
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
                subj_classify = self.run_classify_for_single_subj(subj, feat_file, stPPC_file,
                                                                  subj_elec_dir, overwrite_features)
                class_data_all_subjs.append(subj_classify)
                with open(save_file, 'wb') as f:
                    pickle.dump(subj_classify, f, protocol=-1)
            except ValueError:
               print 'Error processing %s.' % subj
            self.res = class_data_all_subjs

        #return class_data_all_subjs

    def run_classify_for_single_subj(self, subj, feat_file, stPPC_file, subj_elec_dir, overwrite_features):
        """
        Runs logistic regression classifier to predict recalled/not-recalled items. Logic is as follows:
        1) Create classifier features, if they do not exist.
        2) Classify using leave one session out cross validation. If only one session, use leave one list out
        """

        # check if features exist. If not, create
        if not os.path.exists(feat_file) or overwrite_features or self.feat_type[-2:] == '_r':
            print '%s features do not exist or overwriting for %s. Creating.' % (self.feat_type, subj)

            # feat_func = self.get_feature_func(self.feat_type)
            # subj_features = feat_func(subj, self.start_time, self.end_time, self.freqs, self.bipolar, self.pool)
            if self.feat_type != 'ppc':
                subj_features = load_features(subj, self.task, self.task_phase, self.start_time, self.end_time,
                                              self.freqs, self.freq_bands, self.hilbert_phase_band, self.num_phase_bins,
                                              self.bipolar, self.feat_type, self.mean_pow, False, subj_elec_dir,
                                              self.pool)
            else:
                st = create_stPPC_network.stPPC(self.subjs, self.task, self.task_phase, self.bipolar, self.freqs,
                                                self.start_time, self.end_time, True, self.pool)
                subj_features, _ = st.stPPC_single_subj(subj)

            with open(feat_file, 'wb') as f:
                pickle.dump(subj_features, f, protocol=-1)

        # if exist, load from disk
        else:
            with open(feat_file, 'rb') as f:
                subj_features = pickle.load(f)

        # load single trial ppc data used for feature reduction
        stPPC_data = None
        if self.stPPC_filtering \
                and self.feat_type != 'power' \
                and len(np.unique(subj_features.events.data['session'])) > 1:
            if os.path.exists(stPPC_file):
                with open(stPPC_file, 'rb') as f:
                    stPPC_data = pickle.load(f)
            else:
                print 'stPPC filtering requested, but data not found for %s.' % subj
                return

        # determine classes
        if subj_features is None:
            print 'PPC NO %s.' % subj
            return
        # recalls = ram_data_helpers.filter_events_to_recalled(self.task, subj_features.events.data)
        recalls = ram_data_helpers.filter_events_to_recalled_sess_level(self.task, subj_features.events.data)
        # recalls = ram_data_helpers.filter_events_to_recalled_smart_low(self.task, subj_features.events.data)
        # recalls = ram_data_helpers.filter_events_to_recalled_multi_thresh(self.task, subj_features.events.data)
        # recalls = ram_data_helpers.filter_events_to_recalled_norm_thresh_exc_low(self.task, subj_features.events.data)

        # create cross validation labels
        sessions = subj_features.events.data['session']
        if len(np.unique(subj_features.events.data['session'])) > 1:
            cv_sel = sessions
        else:
            cv_sel = subj_features.events.data['trial']
        
        # transpose to have events as first dimensions
        # reshape to events by obs
        if self.feat_type != 'ppc':
            if self.feat_type == 'pow_by_phase':
                dims = subj_features.dims
                subj_features = subj_features.transpose(dims[2], dims[1], dims[0], dims[3])
            else:
                subj_features = subj_features.transpose()
            subj_data = subj_features.data.reshape(subj_features.data.shape[0], -1)
        else:
            subj_data = np.transpose(subj_features.data, (2, 1, 0, 3))
            subj_data = subj_data.reshape((subj_data.shape[0], subj_data.shape[1]*subj_data.shape[2], -1))

        # run classify
        subj_res = self.compute_classifier(subj, recalls, sessions, cv_sel, subj_data, stPPC_data)
        subj_res['subj'] = subj
        subj_res['events'] = subj_features['events']
        if self.feat_type == 'power':
            subj_res['loc_tag'] = subj_features.attrs['loc_tag']
            subj_res['anat_region'] = subj_features.attrs['anat_region']

        return subj_res

    def compute_classifier(self, subj, recalls, sessions, cv_sel, feat_mat, stPPC_data):
        if self.do_rand:
            feat_mat = np.random.rand(*feat_mat.shape)

        # normalize data by session (this only makes sense for power?)
        if self.feat_type == 'power':
            uniq_sessions = np.unique(sessions)
            for sess in uniq_sessions:
                sess_event_mask = (sessions == sess)
                feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

        # initialize classifier
        lr_classifier = LogisticRegression(C=self.C, penalty=self.norm, class_weight='auto', solver='liblinear')

        # hold class probability
        probs = np.empty_like(recalls, dtype=np.float)

        if stPPC_data is not None:
            stPPC_sessions = stPPC_data[0][2]

        # loop over each train set and classify test
        uniq_cv = np.unique(cv_sel)
        for cv_num, cv in enumerate(uniq_cv):

            # create mask of just training data
            mask_train = (cv_sel != cv)

            # if using stPPC data to filter features, pull out the stPPC data computing based on the training sessions
            # find the electrodes with the highest absolute connection strengths
            cvs_used = set(cv_sel[mask_train])
            if stPPC_data is not None:
                stPPC_to_use = [set(x) == cvs_used for x in stPPC_sessions]
                ts = [x[0][np.where(stPPC_to_use)[0]] for x in stPPC_data]
                ts_mean = np.array([np.mean(x, axis=1) for x in ts]).T

                feats_to_use_bool = np.zeros((np.shape(ts_mean)[1], np.shape(ts_mean)[0]), dtype=bool)
                d = int(np.ceil(np.sqrt(ts_mean.shape[1] * 2)))
                # feats_to_use_bool = np.zeros((d, d, np.shape(ts_mean)[0]), dtype=bool)

                squares_by_freqs = np.empty((d, d, len(self.freqs)))
                for i, data_freq in enumerate(ts_mean):
                    squares_by_freqs[:, :, i] = squareform(data_freq)


                    square_mat = squareform(data_freq)
                    abs_strengths = np.mean(np.abs(square_mat), axis=0)
                    sorted_strengths = np.argsort(-abs_strengths, axis=0)


                    square_mat_bool = np.zeros(np.shape(square_mat))
                    square_mat_bool[sorted_strengths[0:11], :] = 1
                    square_mat_bool[:, sorted_strengths[0:11]] = 1
                    square_mat_bool[sorted_strengths[0:11], sorted_strengths[0:11]] = 0
                    square_mat_bool = squareform(square_mat_bool).astype(bool)
                    feats_to_use_bool[:, i] = square_mat_bool


                # abs_strengths = np.mean(np.abs(squares_by_freqs), axis=0)
                # strongest_feats = abs_strengths > np.percentile(abs_strengths, 90)
                # inds = np.where(strongest_feats)
                # feats_to_use_bool[inds[0], :, inds[1]] = True
                # feats_to_use_bool[:, inds[0], inds[1]] = True
                # feats_to_use_bool[inds[0], inds[0], inds[1]] = False
                # feats_to_use_bool = np.stack([squareform(x) for x in feats_to_use_bool.T], axis=1)
                feats_to_use_bool = feats_to_use_bool.reshape(-1).astype(bool)
            else:
                feats_to_use_bool = np.ones((feat_mat.shape[1]), dtype=bool)

            if self.feat_type != 'ppc':
                feats_train = feat_mat[mask_train]
                feats_test = feat_mat[~mask_train]
            else:
                feats_train = feat_mat[mask_train, :, cv_num]
                feats_test = feat_mat[~mask_train, :, cv_num]
            rec_train = recalls[mask_train]

            # if self.reduce_features:
            #     test_stat = np.empty(feat_mat.shape[1], dtype=np.float)
            #     for i, col in enumerate(pow_train.T):
            #         t, p = ranksums(col[rec_train], col[~rec_train])
            #         test_stat[i] = t
            #
            #     # pick feature with strongest differences between conditions
            #     feat_abs_order = np.argsort(np.abs(test_stat))[::-1]
            #     num_elecs = self.get_num_elecs(subj)
            #     num_feats_to_use = num_elecs * len(self.freqs)
            #     feats_to_use = feat_abs_order[0:num_feats_to_use]
            # else:
            #     feats_to_use = np.ones((feat_mat.shape[1]), dtype=bool)


            if self.reduce_features:
                t0 = time()
                pca = RandomizedPCA(n_components=150, whiten=True).fit(pow_train)
                print("done in %0.3fs" % (time() - t0))
                x_train = pca.transform(pow_train)
                x_test = pca.transform(feat_mat[~mask_train])
            else:
                # feats_to_use = np.ones((feat_mat.shape[1]), dtype=bool)
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
            probs[~mask_train] = test_probs

        # compute AUC
        auc = roc_auc_score(recalls, probs)

        # output results, including AUC, lr object (fit to all data), prob estimates, and class labels
        subj_res = {}
        subj_res['auc'] = auc
        subj_res['subj'] = subj
        print auc

        # NEED TO ALSO MAKE PPC WITH ALL SESS
        subj_res['lr_classifier'] = lr_classifier.fit(feat_mat, recalls)
        subj_res['probs'] = probs
        subj_res['classes'] = recalls
        subj_res['tercile'] = self.compute_terciles(subj_res)
        subj_res['sessions'] = sessions
        subj_res['cv_sel'] = cv_sel
        subj_res['cv_type'] = 'loso' if len(np.unique(sessions)) > 1 else 'lolo'

        # also compute forward model feature importance for each electrode if we are using power
        if self.feat_type == 'power':
            probs_log = np.log(probs / (1 - probs))
            covx = np.cov(feat_mat.T)
            covs = np.cov(probs_log)
            W = subj_res['lr_classifier'].coef_
            A = np.dot(covx, W.T) / covs
            subj_res['forward_model'] = np.reshape(A, (-1, len(self.freqs)))

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

        table = np.stack([[x['subj'], x['auc'], x['cv_type'], self.feat_type] for x in self.res if
                          x['cv_type'] in cv_type and x['subj'] in self.subjs])

        print 'Subject\tAUC\tCV\tfeatures'
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


    @classmethod
    def compute_terciles(cls, subj_res):
        binned_data = binned_statistic(subj_res['probs'], subj_res['classes'], statistic='mean',
                                       bins=np.percentile(subj_res['probs'], [0, 33, 67, 100]))
        tercile_delta_rec = (binned_data[0] - np.mean(subj_res['classes'])) / np.mean(subj_res['classes']) * 100
        return tercile_delta_rec


    def get_num_elecs(self, subj):
        tal_path = os.path.join('/data/eeg', subj, 'tal', subj + '_talLocs_database_bipol.mat')
        tal_reader = TalReader(filename=tal_path)
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_pairs = tal_reader.get_bipolar_pairs()
        return len(bipolar_pairs) if self.bipolar else len(monopolar_channels)