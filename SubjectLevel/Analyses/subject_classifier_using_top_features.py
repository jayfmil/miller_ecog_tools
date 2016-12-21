import pdb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import ttest_ind
from SubjectLevel.Analyses.subject_classifier import SubjectClassifier as SC


class SubjectClassifier(SC):
    """
    Version of SubjectClassifier that classifies the using only the best univariate electrode of frequency, then the
    best two, then the best three, then four, ....
    """

    res_str_tmp = 'classify_top_feats_%s.p'

    def __init__(self, task=None, subject=None, do_top_elecs=True):
        super(SubjectClassifier, self).__init__(task=task, subject=subject)

        # string to use when saving results files
        self.res_str = SubjectClassifier.res_str_tmp

        # If True, will classify based on best electrodes. If False, will classify based on top frequencies.
        self.do_top_elecs = do_top_elecs

        # Setting to divide the data into random halves. .analysis() will use one half to pick the features to use and
        # to train the classifier, and will test on the other half. This is repeated n_perms times. This analsysis
        # doesn't really make sense if do_random_half_cv is set to False, so don't do that.
        self.do_random_half_cv = True
        self.n_perms = 100

    # I'm using this property and setter to change the res_str whenever do_top_elecs is set
    @property
    def do_top_elecs(self):
        return self._do_top_elecs

    @do_top_elecs.setter
    def do_top_elecs(self, t):
        self._do_top_elecs = t
        self.res_str = SubjectClassifier.res_str_tmp % 'elecs' if t else SubjectClassifier.res_str_tmp % 'freqs'

    def analysis(self):
        """
        For each iteration, compute the best univariate features using a ttest comparing the remembered and not
        remembered items. Train a classifier using the best electrode (or best frequency), and test on help out data.
        Repeat n_perm times.

        res will have keys:
        """

        # make a copy of the data so we can restore to original after we are done
        subject_data = deepcopy(self.subject_data)

        N = self.subject_data.shape[2] if self.do_top_elecs else self.subject_data.shape[1]
        aucs_by_n_feats = np.empty(shape=(self.n_perms, N))
        ts_by_n_feats = np.empty(shape=(self.n_perms, N))

        # loop over n_perms times
        for perm in range(self.n_perms):
            print('%s: permutation %d of %d' % (self.subj, perm+1, 100))

            # call to .make_cross_val_labels generates a new random split of the data
            self.make_cross_val_labels()

            # these are the events to pick the features and to train on
            train_bool = self.cross_val_dict[1.0]['train_bool']

            # kind of a hack, but remove the other cross val info for the other half of data, as we don't ever want to
            # test on the same data we are training on, and the classifier code automatically would loop over all folds
            del(self.cross_val_dict[0])

            # Get class labels and filter to training events
            Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)
            Y = Y[train_bool]

            # make a copy of subject data that we will modify
            X = deepcopy(self.subject_data.data)
            X = X.reshape(X.shape[0], -1)

            # normalize data by session if the features are oscillatory power
            if self.feat_type == 'power':
                X = self.normalize_power(X)

            # filter data to just training events
            X = X[train_bool]

            # run ttest at each frequency and electrode comparing remembered and not remembered events
            ts, ps, = ttest_ind(X[Y], X[~Y])

            # reshape to make it easier to take the max across axis type of interest
            ts = ts.reshape(len(self.freqs), -1)
            ps = ps.reshape(len(self.freqs), -1)

            # most significant electrode (or freq), regardless of frequency (or elec) and sign, sort from most to least
            axis = 0 if self.do_top_elecs else 1
            t_inds_sorted = np.argsort(np.abs(ts).max(axis=axis))[::-1]

            # Use just the best univariate feature to classify. Then the best 2, then best 3, ...
            for i in range(N):

                # make .subj_data be data from just the elecrode(s) or freq(s) of interest
                if self.do_top_elecs:
                    self.subject_data = subject_data[:, :, t_inds_sorted[0:i+1]]
                else:
                    self.subject_data = subject_data[:, t_inds_sorted[0:i+1], :]

                # classify using the base classifier code
                super(SubjectClassifier, self).analysis()

                # store auc for this permation in an easier to access array
                aucs_by_n_feats[perm, i] = self.res['auc']

            # store the actual t values used when ordering the features
            row_idx = np.argmax(np.abs(ts), axis=axis)
            col_idx = np.arange(ts.shape[1 if self.do_top_elecs else 0])
            ts_by_n_feats[perm, :] = ts[row_idx, col_idx]

        # make subject data be the full dataset again
        self.subject_data = subject_data

        # finally, store results
        self.res['ts'] = ts_by_n_feats
        self.res['aucs_by_n_feats'] = aucs_by_n_feats

    def plot_auc_by_num_features(self):
        """
        Plots AUC as a function of the number of top features included in the model.
        """

        ts = np.nanmean(self.res['ts'], axis=0)
        ts = ts[np.argsort(np.abs(ts))][::-1]
        clim = np.max([ts.min(), ts.max()])
        with plt.style.context('myplotstyle.mplstyle'):
            y = np.mean(self.res['aucs_by_n_feats'], axis=0)
            plt.scatter(np.arange(len(y)) + 1, y, c=ts, cmap='RdBu_r', vmin=-clim, vmax=clim, s=100)
            cb = plt.colorbar()
            plt.ylabel('AUC', fontsize=20)
            plt.xlabel('# Electrodes' if self.do_top_elecs else '# Frequencies', fontsize=20)
            plt.title(self.subj, fontsize=20)

