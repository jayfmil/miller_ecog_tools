import os
import pdb
import ram_data_helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from SubjectLevel.subject_analysis import SubjectAnalysis
from rankpruning import RankPruning, other_pnlearning_methods


class SubjectClassifier(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to handle classification. Some options are specific to the Treasure Hunt
    (TH) task.
    """

    # class attribute: default regularization value. Event if self.C is modified, this will be used for subjects with
    # only one session of data
    default_C = [7.2e-4]

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectClassifier, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)
        self.train_phase = ['enc']  # ['enc'] or ['rec'] or ['enc', 'rec']
        self.test_phase = ['enc']   # ['enc'] or ['rec'] or ['enc', 'rec'] # PUT A CHECK ON THIS and others, properties?
        self.norm = 'l2'            # type of regularization (l1 or l2)
        self.C = SubjectClassifier.default_C
        self.scale_enc = 1.0
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        self.exclude_by_rec_time = False
        self.rec_thresh = None
        self.compute_new_y_labels = False
        self.compute_perc = 50
        self.do_rank_pruning = False
        self.rank_do_pu = True
        self.do_GBC = False

        # do we compute the foward model?
        self.do_compute_forward_model = True

        # will hold cross validation fold info after call to make_cross_val_labels(), task_phase will be an array with
        # either 'enc' or 'rec' for each entry in our data
        self.cross_val_dict = {}

        # option to divide all the data into two cv folds based on a randomly selecting half the data, ignoring trial
        # and session labels
        self.do_random_half_cv = False
        self.task_phase = None

        # if given, will mean power within a certain range and only use that frequency band
        self.freq_band_for_classification = None

        # string to use when saving results files
        self.res_str = 'classify.p'

    def run(self):
        """
        Basically a convenience function to do all the classification steps sequentially.
        """

        # Step 1: load data
        if self.subject_data is None:
            self.load_data()

        # Step 2: create (if needed) directory to save/load
        self.make_res_dir()

        # Step 3: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 4: if not loaded ...
        if not self.res:

            # Step 4A: make cross val labels before doing the actual classification
            self.make_cross_val_labels()

            # Step 4B: classify.
            print('%s: Running classify.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def make_cross_val_labels(self):
        """
        Creates the training and test folds. If a subject has multiple sessions of data, this will do leave-one-session-
        out cross validation. If only one session, this will do leave-one-list-out CV. Training data will only include
        experiment phases included in self.train_phase and test data will only include phases in self.test_phase.
        """
        if self.subject_data is None:
            print('Data must be loaded before computing cross validation labels. Use .load_data()')
            return

        # The classifier can train and test on different phases of our experiments, namely encoding and retrieval
        # (or both). These are coded different depending on the experiment.
        valid_train_inds = np.array([True if x in self.train_phase else False for x in self.task_phase])
        valid_test_inds = np.array([True if x in self.test_phase else False for x in self.task_phase])

        # For each task phase in the data, split half of it randomly into the first fold and half in the second
        if self.do_random_half_cv:
            folds = np.zeros(shape=len(self.task_phase))
            for phase in np.unique(self.task_phase):
                inds = np.where(self.task_phase == phase)[0]
                n = int(np.round(np.sum(self.task_phase == phase)/2.))
                folds[np.random.choice(inds, n, replace=False)] = 1

        # if not doing the random split, then base the folds on either the sessions or lists
        else:
            # create folds based on either sessions or lists within a session
            sessions = self.subject_data.events.data['session']
            if len(np.unique(self.subject_data.events.data['session'])) > 1:
                folds = sessions
            else:
                if self.task == 'RAM_YC1':
                    trial_str = 'blocknum'
                else:
                    trial_str = 'trial' if 'RAM_TH' in self.task else 'list'
                folds = self.subject_data.events.data[trial_str]

        # make dictionary to hold booleans for training and test indices for each fold, as well as the task phase for
        #  each fold
        cv_dict = {}
        uniq_folds = np.unique(folds)
        for fold in uniq_folds:
            cv_dict[fold] = {}
            cv_dict[fold]['train_bool'] = (folds != fold) & valid_train_inds
            cv_dict[fold]['test_bool'] = (folds == fold) & valid_test_inds
            cv_dict[fold]['train_phase'] = self.task_phase[(folds != fold) & valid_train_inds]
            cv_dict[fold]['test_phase'] = self.task_phase[(folds == fold) & valid_test_inds]
        self.cross_val_dict = cv_dict

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

        # reshape data to events x number of features
        # new_feats = self.add_prev_event_features()
        # X = np.concatenate([self.subject_data.data, new_feats], axis=1)
        # X = X.reshape(X.shape[0], -1)
        # pdb.set_trace()
        if self.freq_band_for_classification:
            freq_inds = (self.freqs >= self.freq_band_for_classification[0]) & (self.freqs <= self.freq_band_for_classification[1])
            X = np.nanmean(self.subject_data[:, freq_inds, :], axis=1)
            self.do_compute_forward_model = False
        else:
            X = self.subject_data.data.reshape(self.subject_data.shape[0], -1)
        # pdb.set_trace()

        # normalize data by session if the features are oscillatory power
        if self.feat_type == 'power':
            X = self.normalize_power(X)

        # revert C value to default C value not multi session subejct
        Cs = self.C
        loso = True
        if len(np.unique(self.subject_data.events.data['session'])) == 1:
            Cs = SubjectClassifier.default_C
            loso = False

        # if leave-one-session-out (loso) cross validation, this will hold area under the curve for each hold out
        fold_aucs = np.empty(shape=(len(self.cross_val_dict.keys()), len(self.C)), dtype=np.float)

        # will hold the predicted class probability for all the test data
        probs = np.empty(shape=(Y.shape[0], len(Cs)), dtype=np.float)

        # This outer loop is for all the different penalty (C) values given. If more than one, the optimal value will
        # be chosen using bias correction based on Tibshirani and Tibshirani 2009, Annals of Applied Statistics. This
        # only works for multi-session (loso) data, otherwise we don't have enough data to compute help out AUC.
        for c_num, c in enumerate(Cs):

            # create classifier with current C
            if not self.do_rank_pruning:
                if self.do_GBC:
                    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5,
                                                            subsample=0.8,
                                                            max_features='sqrt')
                else:
                    classifier = LogisticRegression(C=c, penalty=self.norm, solver='liblinear')
            else:
                rp_lr_classifier = RankPruning(clf=LogisticRegression(C=c, penalty=self.norm, solver='liblinear'))

            # now loop over all the cross validation folds
            for cv_num, cv in enumerate(self.cross_val_dict.keys()):
                # print(cv_num)

                # Training data for fold
                x_train = X[self.cross_val_dict[cv]['train_bool']]
                task_train = self.cross_val_dict[cv]['train_phase']
                y_train = Y[self.cross_val_dict[cv]['train_bool']]
                if self.compute_new_y_labels:
                    train_bool = self.compute_nrec_labels4(x_train, y_train, Y, self.cross_val_dict[cv]['train_bool'])
                    # pdb.set_trace()
                    # self.cross_val_dict[cv]['train_bool'] = train_bool
                    x_train = X[self.cross_val_dict[cv]['train_bool']]
                    y_train = Y[self.cross_val_dict[cv]['train_bool']]
                    task_train = self.task_phase[train_bool]
                    # task_train[~train_bool] = 'dummy'

                # Test data for fold
                x_test = X[self.cross_val_dict[cv]['test_bool']]
                task_test = self.cross_val_dict[cv]['test_phase']
                y_test = Y[self.cross_val_dict[cv]['test_bool']]

                # normalize train test and scale test data. This is ugly because it has to account for a few
                # different contingencies. If the test data phase is also a train data phase, then scale test data for
                # that phase based on the training data for that phase. Otherwise, just zscore the test data.
                for phase in self.train_phase:
                    x_train[task_train == phase] = zscore(x_train[task_train == phase], axis=0)

                for phase in self.test_phase:
                    if phase in self.train_phase:
                        x_test[task_test == phase] = zmap(x_test[task_test == phase],
                                                          x_train[task_train == phase], axis=0)
                    else:
                        x_test[task_test == phase] = zscore(x_test[task_test == phase], axis=0)

                # weight observations by number of positive and negative class
                y_ind = y_train.astype(int)

                # if we are training on both encoding and retrieval and we are scaling the encoding weights,
                # seperate the ecnoding and retrieval positive and negative classes so we can scale them later
                if len(self.train_phase) > 1:
                    y_ind[task_train == 'rec'] += 2

                # compute the weight vector as the reciprocal of the number of items in each class, divided by the mean
                # class frequency
                recip_freq = 1. / np.bincount(y_ind)
                recip_freq /= np.mean(recip_freq)

                # scale the encoding classes. Sorry for the identical if statements
                if len(self.train_phase) > 1:
                    recip_freq[:2] *= self.scale_enc
                    recip_freq /= np.mean(recip_freq)
                weights = recip_freq[y_ind]

                # Fit the model
                # pdb.set_trace()
                if not self.do_rank_pruning:
                    if self.do_GBC:
                        classifier.fit(x_train, y_train)
                    else:
                        classifier.fit(x_train, y_train, sample_weight=weights)
                else:
                    rp_lr_classifier.fit(x_train, y_train, pulearning=self.rank_do_pu, cv_n_folds=5)

                # now predict class probability of test data
                if not self.do_rank_pruning:
                    test_probs = classifier.predict_proba(x_test)[:, 1]
                else:
                    test_probs = rp_lr_classifier.predict_proba(x_test)
                probs[self.cross_val_dict[cv]['test_bool'], c_num] = test_probs
                if loso:
                    fold_aucs[cv_num, c_num] = roc_auc_score(y_test, test_probs)

            # If multi sessions, compute bias corrected AUC (if only one C values given, this has no effect) because
            # the bias will be 0. AUC is the average AUC across folds for the the bees value of C minus the bias term.
            train_bool = np.any(np.stack([self.cross_val_dict[x]['train_bool'] for x in self.cross_val_dict]), axis=0)
            test_bool = np.any(np.stack([self.cross_val_dict[x]['test_bool'] for x in self.cross_val_dict]), axis=0)

            if loso and len(self.cross_val_dict.keys()) > 1:
                aucs = fold_aucs.mean(axis=0)
                bias = np.mean(fold_aucs.max(axis=1) - fold_aucs[:, np.argmax(aucs)])
                auc = aucs.max() - bias
                C = Cs[np.argmax(aucs)]
                # auc = fold_aucs[-1][0]
            else:
                # is not multi session, AUC is just computed by aggregating all the hold out probabilities
                if len(np.unique(Y[test_bool])) == 1:
                    print('%s: only one class in Y. Cannot compute AUC. Setting to NaN.' % self.subj)
                    auc = np.nan
                else:
                    auc = roc_auc_score(Y[test_bool], probs[test_bool])
                C = Cs[0]

            # store classifier results
            res = {}

            # classifier performance as measure by area under the ROC curve
            res['auc'] = auc

            # estimated probabilities and true class labels
            res['probs'] = probs[test_bool, 0 if ~loso else np.argmax(aucs)]
            res['Y'] = Y[test_bool]

            # model fit on all the training data
            # pdb.set_trace()
            if not self.do_rank_pruning:
                if not self.do_GBC:
                    classifier.C = C
                res['model'] = classifier.fit(X[train_bool], Y[train_bool])
            else:
                rp_lr_classifier.fit(X[train_bool], Y[train_bool])
                res['model'] = rp_lr_classifier.clf
            # pdb.set_trace()
            # boolean array of all entries in subject data that were used to train classifier over all folds. Depending
            # on self.train_phase, this isn't necessarily all the data
            res['train_bool'] = train_bool
            self.res = res

            # store tercile measure
            self.res['tercile'] = self.compute_terciles()

            # store forward model
            if self.do_compute_forward_model:
                self.res['forward_model'] = self.compute_forward_model()
                self.res['forward_model_by_region'], self.res['regions'] = self.forward_model_by_region()

            # easy to check flag for multisession data
            self.res['loso'] = loso
            if self.verbose:
                print('%s: %.3f AUC.' % (self.subj, self.res['auc']))

    def compute_nrec_labels(self, x_train, y_train):

        # for the training data, treat correct recalls as correct
        new_y_train = np.zeros(y_train.shape).astype(bool)
        new_y_train[y_train] = True

        # cluster the nrec items into two groups
        kmeans = KMeans(n_clusters=2)
        pred_labels = kmeans.fit_predict(x_train[~y_train])

        # label the group closer to the recalled items as recalled
        dists = [np.linalg.norm(np.mean(x_train[y_train], axis=0) - x) for x in kmeans.cluster_centers_]
        rec_label = kmeans.labels_[np.argmin(dists)]
        new_y_train[~y_train] = pred_labels == rec_label

        return new_y_train

    def add_prev_event_features(self):

        # will hold new features
        new_feats = np.empty(self.subject_data.shape)

        # loop over each session
        sessions = self.subject_data.events.data['session']
        uniq_sessions = np.unique(sessions)
        for uniq_session in uniq_sessions:
            sess_inds = sessions == uniq_session

            # loop over each trial
            trial_str = 'trial' if 'RAM_TH' in self.task else 'list'
            trials = self.subject_data.events.data[trial_str]
            uniq_trials = np.unique(trials[sess_inds])
            for trial in uniq_trials:
                trial_inds = (trial == trials) & sess_inds

                # for each event in a trial, create a new feature that is the mean of the current
                # event and all previous in the trial
                trial_inds_where = np.where(trial_inds)[0]
                for i, ev_num in enumerate(trial_inds_where):
                    if i == 0:
                        new_feats[ev_num] = self.subject_data.data[ev_num]
                    # elif i == 1:
                    #     new_feats[ev_num] = self.subject_data.data[trial_inds_where[i - 1]]
                    else:
                        new_feats[ev_num] = self.subject_data.data[trial_inds_where[i-1]]
                        # new_feats[ev_num] = np.mean(self.subject_data.data[trial_inds_where[i-1]:trial_inds_where[i]], axis=0)
        return new_feats

    def compute_nrec_labels2(self, x_train, y_train):

        # for the training data, treat correct recalls as correct
        new_y_train = np.zeros(y_train.shape).astype(bool)

        # cluster the nrec items into two groups
        kmeans = KMeans(n_clusters=2)
        pred_labels = kmeans.fit_predict(x_train)

        # label the group closer to the recalled items as recalled
        dists = [np.linalg.norm(np.mean(x_train[y_train], axis=0) - x) for x in kmeans.cluster_centers_]
        rec_label = kmeans.labels_[np.argmin(dists)]
        new_y_train[pred_labels == rec_label] = True
        new_y_train[y_train] = True

        return new_y_train

    def compute_nrec_labels3(self, x_train, y_train):

        new_y_train = np.zeros(y_train.shape).astype(bool)
        new_y_train[y_train] = True

        # mean recalls features
        mean_feats = np.mean(x_train[y_train], axis=0)

        # compute distance from recalled mean to each nrec obs
        dists = np.array([np.linalg.norm(mean_feats - x) for x in x_train[~y_train]])

        # relabel the top to be recalled
        to_relabel = np.argsort(dists)[:np.int(len(dists) * .05)]
        new_y_train[to_relabel] = True

        return new_y_train

    def compute_nrec_labels4(self, x_train, y_train, Y, train_bool):

        rec_obs = x_train[y_train]
        rec_norms = np.array(
            [np.linalg.norm(np.mean(rec_obs[np.setdiff1d(range(rec_obs.shape[0]), i)], axis=0) - x) for i, x in
             enumerate(rec_obs)])
        nrec_norms = np.array([np.linalg.norm(np.mean(rec_obs, axis=0) - x) for x in x_train[~y_train]])
        train_bool[train_bool & ~Y] = nrec_norms > np.mean(rec_norms)
        train_bool[train_bool & Y] = rec_norms < np.mean(rec_norms)
        return train_bool

        # relabel the top to be recalled
        # to_relabel = np.argsort(dists)[:np.int(len(dists) * .05)]
        # new_y_train[to_relabel] = True
        #
        # return new_y_train

    def compute_auc_pval(self, n_iters=100):
        """
        Computes a distribution of AUC values based on permuted recall labels and computes a pvalue.
        """
        if not self.cross_val_dict:
            self.make_cross_val_labels()

        if not self.res:
            print('Classifier data must be loaded or computed.')
            return
        else:
            res_orig = deepcopy(self.res)
            orig_verbose = deepcopy(self.verbose)

        auc_null = np.zeros(n_iters)
        self.verbose = False
        print('%s: computing pval with %d iterations' % (self.subj, n_iters))
        for i in range(n_iters):
            self.analysis(permute=True)
            auc_null[i] = self.res['auc']

        self.verbose = orig_verbose
        self.res = res_orig
        self.res['perm_aucs'] = auc_null
        self.res['pval'] = np.mean(self.res['auc'] < auc_null)
        print('%s: p-value = %.3f' % (self.subj, self.res['pval']))

    def compute_terciles(self):
        """
        Compute change in subject recall rate as a function of three bins of classifier probability outputs.
        """
        if not self.res:
            print('Classifier data must be loaded or computed.')
            return

        binned_data = binned_statistic(self.res['probs'], self.res['Y'], statistic='mean',
                                       bins=np.percentile(self.res['probs'], [0, 33, 67, 100]))
        tercile_delta_rec = (binned_data[0] - np.mean(self.res['Y'])) / np.mean(self.res['Y']) * 100
        return tercile_delta_rec

    def plot_classifier_terciles(self):
        """
        Plot change in subject recall rate as a function of three bins of classifier probaility outputs.
        """
        if not self.res:
            print('Classifier data must be loaded or computed.')
            return

        with plt.style.context('myplotstyle.mplstyle'):
            plt.bar(range(3), self.res['tercile'], align='center', color=[.5, .5, .5], linewidth=2)

    def plot_roc(self):
        """
        Plot receiver operating charactistic curve for this subject's classifier.
        """
        if not self.res:
            print('Classifier data must be loaded or computed.')
            return

        fpr, tpr, _ = roc_curve(self.res['Y'], self.res['probs'])
        with plt.style.context('myplotstyle.mplstyle'):
            plt.plot(fpr, tpr, lw=4, label='ROC curve (AUC = %0.2f)' % self.res['auc'])
            plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='_nolegend_')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=24)
            plt.ylabel('True Positive Rate', fontsize=24)
            plt.legend(loc="lower right")
            plt.title('%s ROC' % self.subj)

    def compute_forward_model(self):
        """
        Compute "forward model" to make the classifier model weights interpretable. Based on Haufe et al, 2014 - On the
        interpretation of weight vectors of linear models in multivariate neuroimaging, Neuroimage
        """
        if not self.res or self.subject_data is None:
            print('Both classifier results and subject data must be loaded to compute forward model.')
            return

        # reshape data to events x number of features
        X = deepcopy(self.subject_data.data)
        X = X.reshape(X.shape[0], -1)

        # normalize data by session if the features are oscillatory power
        if self.feat_type == 'power':
            X = self.normalize_power(X)

        # compute forward model, using just the training data
        X = X[self.res['train_bool']]
        probs_log = np.log(self.res['probs'] / (1 - self.res['probs']))
        covx = np.cov(X.T)
        covs = np.cov(probs_log)
        W = self.res['model'].coef_
        A = np.dot(covx, W.T) / covs

        # reshape into elecs by freq
        A = A.reshape(self.subject_data.shape[1], -1)
        return A

    def forward_model_by_region(self):
        """
        Average the forward model weights within the subject's brain regions.
        """

        if 'forward_model' not in self.res:
            print('Must compute forward model before averaging by region. Use .compute_foward_model.')
            return

        # average all the elecs within each region.
        regions = np.array(sorted(self.elec_locs.keys()))
        regions = regions[regions != 'is_right']
        mean_array = np.stack([np.nanmean(self.res['forward_model'][:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
        return mean_array, regions

    def normalize_power(self, X):
        """
        Normalizes (zscores) each column in X. If rows of comprised of different task phases, each task phase is
        normalized to itself

        returns normalized X
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in set(self.train_phase + self.test_phase):
                task_mask = self.task_phase == phase
                X[sess_event_mask & task_mask] = zscore(X[sess_event_mask & task_mask], axis=0)
        return X

    def _generate_res_save_path(self):
        """
        Build path to where classification results should be saved (or loaded from). Return string.
        """

        if len(self.C) > 1:
            dir_str = 'C_bias_norm_%s_%s_train_%s_test_%s_enc_scale_%.3f'
            dir_str_vals = (
                self.norm,
                self.recall_filter_func.__name__,
                '_'.join(self.train_phase),
                '_'.join(self.test_phase),
                self.scale_enc)
        else:
            dir_str = 'C_%.8f_norm_%s_%s_train_%s_test_%s_enc_scale_%.3f'
            dir_str_vals = (
                self.C[0],
                self.norm,
                self.recall_filter_func.__name__,
                '_'.join(self.train_phase),
                '_'.join(self.test_phase),
                self.scale_enc)

        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str % dir_str_vals)

