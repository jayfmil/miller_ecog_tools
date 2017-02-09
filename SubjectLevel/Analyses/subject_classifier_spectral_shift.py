import pdb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import ttest_ind
from scipy.stats.mstats import zscore, zmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from SubjectLevel.Analyses.subject_classifier import SubjectClassifier as SC
from SubjectLevel.par_funcs import par_robust_reg


class SubjectClassifier(SC):
    """
    Version of SubjectClassifier that classifies
    """

    valid_model_feats = ('resids', 'slopes', 'bband_power')
    res_str_tmp = 'classify_robust_%s.p'

    def __init__(self, task=None, subject=None, model_feats=('resids', 'slopes', 'bband_power')):
        super(SubjectClassifier, self).__init__(task=task, subject=subject)

        # string to use when saving results files
        self.res_str = SubjectClassifier.res_str_tmp

        #
        self.model_feats = model_feats

    # I'm using this property and setter to change the res_str whenever model_feats is set
    @property
    def model_feats(self):
        return self._model_feats

    # also make sure the model_feats are valid
    @model_feats.setter
    def model_feats(self, t):
        if isinstance(t, str):
            t = [t]
        if not np.all(np.array([x in SubjectClassifier.valid_model_feats for x in t])):
            print('model_feats must be any combination of ' + ', '.join(SubjectClassifier.valid_model_feats)+'.')
            self._model_feats = None
            return

        self._model_feats = t
        self.res_str = SubjectClassifier.res_str_tmp % '_'.join(t)

    def analysis(self):
        """

        """
        if self.model_feats is None:
            print('%s: must set .model_feats.' % self.subj)
            return

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
        X = deepcopy(self.subject_data.data)
        X = self.normalize_spectra(X)

        # x var is frequency of the power spectrum
        x = np.expand_dims(np.log10(self.subject_data.frequency.data), axis=1)
        x_rep = np.tile(x, X.shape[0]).T

        # run robust regression for each event and elec in order to get the residuals, slope, and offset
        print('%s: Running robust regression for %d elecs and %d events.' % (self.subj, X.shape[2], X.shape[0]))
        if self.pool is None:
            elec_res = map(par_robust_reg, zip(X, x_rep))
        else:
            elec_res = self.pool.map(par_robust_reg, zip(X, x_rep))

        tmp_res_dict = {}
        tmp_res_dict['intercepts'] = np.expand_dims(np.stack([foo[0] for foo in elec_res]), axis=1)
        tmp_res_dict['slopes'] = np.expand_dims(np.stack([foo[1] for foo in elec_res]), axis=1)
        tmp_res_dict['resids'] = np.stack([foo[2] for foo in elec_res])
        tmp_res_dict['bband_power'] = np.expand_dims(np.stack([foo[3] for foo in elec_res]), axis=1)


        X = np.concatenate([tmp_res_dict[feat] for feat in self.model_feats], axis=1)
        X = X.reshape(X.shape[0], -1)

        # I'm not sure we still need/should normalized by session?
        # X = self.normalize_power(X)

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

            if loso and len(self.cross_val_dict.keys()) > 1:
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
            self.res = res

            # store tercile measure
            self.res['tercile'] = self.compute_terciles()

            # store forward model
            self.res['forward_model'] = self.compute_forward_model_resids(X)
            self.res['forward_model_by_region'], self.res['regions'] = self.forward_model_by_region()

            # easy to check flag for multisession data
            self.res['loso'] = loso
            if self.verbose:
                print('%s: %.3f AUC.' % (self.subj, self.res['auc']))

    def compute_forward_model_resids(self, X):
        """
        Compute "forward model" to make the classifier model weights interpretable. Based on Haufe et al, 2014 - On the
        interpretation of weight vectors of linear models in multivariate neuroimaging, Neuroimage
        """
        if not self.res or self.subject_data is None:
            print('Both classifier results and subject data must be loaded to compute forward model.')
            return

        # compute forward model, using just the training data
        X = X[self.res['train_bool']]
        probs_log = np.log(self.res['probs'] / (1 - self.res['probs']))
        covx = np.cov(X.T)
        covs = np.cov(probs_log)
        W = self.res['model'].coef_
        A = np.dot(covx, W.T) / covs

        # reshape
        A = A.reshape(-1, self.subject_data.shape[2])
        return A

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in set(self.train_phase + self.test_phase):
                task_mask = self.task_phase == phase

                m = np.mean(X[sess_event_mask & task_mask], axis=1)
                m = np.mean(m, axis=0)
                s = np.std(X[sess_event_mask & task_mask], axis=1)
                s = np.mean(s, axis=0)
                X[sess_event_mask & task_mask] = (X[sess_event_mask & task_mask] - m) / s
        return X
