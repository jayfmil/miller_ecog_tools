import os
import pdb
import ram_data_helpers
import cPickle as pickle
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from SubjectLevel.subject_analysis import SubjectAnalysis


class SubjectClassifier(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to handle classification. Some options are specific to the Treasure Hunt
    (TH) task.
    """

    # class attribute: default regularization value. Event if self.C is modified, this will be used for subjects with
    # only one session of data
    default_C = [7.2e-4]

    def __init__(self, task=None, subject=None):
        super(SubjectClassifier, self).__init__(task=task, subject=subject)
        self.train_phase = ['enc']  # ['enc'] or ['rec'] or ['enc', 'rec']
        self.test_phase = ['enc']   # ['enc'] or ['rec'] or ['enc', 'rec'] # PUT A CHECK ON THIS and others, properties?
        self.norm = 'l2'            # type of regularization (l1 or l2)
        self.C = SubjectClassifier.default_C
        self.scale_enc = 1.0
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        self.exclude_by_rec_time = False
        self.rec_thresh = None

        # will hold cross validation fold info after call to make_cross_val_labels(), task_phase will be an array with
        # either 'enc' or 'rec' for each entry in our data
        self.cross_val_dict = {}
        self.task_phase = None

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

        # create folds based on either sessions or lists within a session
        sessions = self.subject_data.events.data['session']
        if len(np.unique(self.subject_data.events.data['session'])) > 1:
            folds = sessions
        else:
            trial_str = 'trial' if self.task == 'RAM_TH1' else 'list'
            folds = self.subject_data.events.data[trial_str]

        # The classifier can train and test on different phases of our experiments, namely encoding and retrieval
        # (or both). These are coded different depending on the experiment.
        task_phase = self.subject_data.events.data['type']
        enc_str = 'CHEST' if 'RAM_TH' in self.task else 'WORD'
        rec_str = 'REC' if 'RAM_TH' in self.task else 'REC_WORD'
        task_phase[task_phase == enc_str] = 'enc'
        task_phase[task_phase == rec_str] = 'rec'
        valid_train_inds = np.array([True if x in self.train_phase else False for x in task_phase])
        valid_test_inds = np.array([True if x in self.test_phase else False for x in task_phase])

        # make dictionary to hold booleans for training and test indices for each fold, as well as the task phase for
        #  each fold
        cv_dict = {}
        uniq_folds = np.unique(folds)
        for fold in uniq_folds:
            cv_dict[fold] = {}
            cv_dict[fold]['train_bool'] = (folds != fold) & valid_train_inds
            cv_dict[fold]['test_bool'] = (folds == fold) & valid_test_inds
            cv_dict[fold]['train_phase'] = task_phase[(folds != fold) & valid_train_inds]
            cv_dict[fold]['test_phase'] = task_phase[(folds == fold) & valid_test_inds]
        self.cross_val_dict = cv_dict
        self.task_phase = task_phase

    def analysis(self):
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

        # reshape data to events x number of features
        X = self.subject_data.data.reshape(self.subject_data.shape[0], -1)

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
            lr_classifier = LogisticRegression(C=c, penalty=self.norm, solver='liblinear')

            # now loop over all the cross validation folds
            for cv_num, cv in enumerate(self.cross_val_dict.keys()):

                # Training data for fold
                x_train = X[self.cross_val_dict[cv]['train_bool']]
                task_train = self.cross_val_dict[cv]['train_phase']
                y_train = Y[self.cross_val_dict[cv]['train_bool']]

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
                lr_classifier.fit(x_train, y_train, sample_weight=weights)

                # now predict class probability of test data
                test_probs = lr_classifier.predict_proba(x_test)[:, 1]
                probs[self.cross_val_dict[cv]['test_bool'], c_num] = test_probs
                if loso:
                    fold_aucs[cv_num, c_num] = roc_auc_score(y_test, test_probs)

            # If multi sessions, compute bias corrected AUC (if only one C values given, this has no effect) because
            # the bias will be 0. AUC is the average AUC across folds for the the bees value of C minus the bias term.
            train_bool = np.any(np.stack([self.cross_val_dict[x]['train_bool'] for x in self.cross_val_dict]), axis=0)
            test_bool = np.any(np.stack([self.cross_val_dict[x]['test_bool'] for x in self.cross_val_dict]), axis=0)
            if loso:
                aucs = fold_aucs.mean(axis=0)
                bias = np.mean(fold_aucs.max(axis=1) - fold_aucs[:, np.argmax(aucs)])
                auc = aucs.max() - bias
                C = Cs[np.argmax(aucs)]
                # auc = fold_aucs[-1][0]
            else:
                # is not multi session, AUC is just computed by aggregating all the hold out probabilities
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
            lr_classifier.C = C
            res['model'] = lr_classifier.fit(X[train_bool], Y[train_bool])

            # boolean array of all entries in subject data that were used to train classifier over all folds. Depending
            # on self.train_phase, this isn't necessarily all the data
            res['train_bool'] = train_bool

            # easy to check flag for multisession data
            res['loso'] = loso
            print('%s: %.3f AUC.' % (self.subj, res['auc']))
            self.res = res

    # def plot_classifier_terciles(self):
    #     """
    #     Plot change in subject recall rate as a function of three bins of classifier probaility outputs.
    #     """
    #     if not self.res:
    #         print('Classifier data must be loaded or computed.')
    #         return
    #
    #     tercile_delta_rec = self.compute_terciles()
    #     plt.bar(range(3), tercile_delta_rec, align='center', color=[.5, .5, .5], linewidth=2)
    #
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

    #
    #
    #
    # def compute_forward_model(self):
    #     """
    #
    #     """
    #     if not self.res and not self.subject_data:
    #         print('Both classifier data and subject data must be loaded to compute forward model.')
    #         return
    #
    #     # reshape data to events x number of features
    #     X = self.subject_data.data.reshape(self.subject_data.shape[0], -1)
    #
    #     # normalize data by session if the features are oscillatory power
    #     if self.feat_type == 'power':
    #         X = self.normalize_power(X)
    #
    #     probs_log = np.log(self.res['probs'] / (1 - self.res['probs']))
    #     covx = np.cov(X.T)
    #     covs = np.cov(probs_log)
    #     W = self.res['model'].coef_
    #     A = np.dot(covx, W.T) / covs
    #     return A
    #         # ts, ps = ttest_ind(feat_mat[recalls], feat_mat[~recalls])
    #


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

