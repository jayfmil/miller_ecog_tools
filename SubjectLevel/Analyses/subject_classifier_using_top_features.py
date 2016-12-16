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

        """
        if not self.cross_val_dict:
            print('Cross validation labels must be computed before running classifier. Use .make_cross_val_labels()')
            return

        # The bias correct only works for subjects with multiple sessions ofdata, see comment below.
        if (len(np.unique(self.subject_data.events.data['session'])) == 1) & (len(self.C) > 1):
            print('Multiple C values cannot be tested for a subject with only one session of data.')
            return

        # bool to filter to all training events
        train_bool = np.any(np.stack([self.cross_val_dict[x]['train_bool'] for x in self.cross_val_dict]), axis=0)

        # Get class labels and filter to training events
        Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)
        Y = Y[train_bool]

        # make a copy of subject data
        X = deepcopy(self.subject_data.data)
        X = X.reshape(X.shape[0], -1)

        # normalize data by session if the features are oscillatory power
        if self.feat_type == 'power':
            X = self.normalize_power(X)

        # filter to just training events
        X = X[train_bool]

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[Y], X[~Y])

        # reshape to make it easier to take the max across axis type of interest
        ts = ts.reshape(len(self.freqs), -1)
        ps = ps.reshape(len(self.freqs), -1)

        # most significant electrode (or freq), regardless of frequency (or elec) and sign, sorted from most to least
        axis = 0 if self.do_top_elecs else 1
        t_inds_sorted = np.argsort(np.abs(ts).max(axis=axis))[::-1]

        # create another copy of the data
        subject_data = deepcopy(self.subject_data)

        # make a new res dict to keep track of the results from all iterations
        res = {}

        # Use just the best univariate feature to classify. Then the best 2, then best 3, ...
        N = subject_data.shape[2] if self.do_top_elecs else subject_data.shape[1]
        for i in range(N):

            # make .subj_data be data from just the elecrode(s) or freq(s) of interest
            if self.do_top_elecs:
                self.subject_data = subject_data[:, :, t_inds_sorted[0:i+1]]
            else:
                self.subject_data = subject_data[:, t_inds_sorted[0:i+1], :]

            # classify using the base classifier code
            super(SubjectClassifier, self).analysis()

            # the above call modifies self.res, so store it as an entry in the new results dict
            res[i+1] = self.res

        # make subject data be the full dataset again
        self.subject_data = subject_data

        # overwrite the current self.res from the most recent loop with the full results from all loops
        self.res = res

    def plot_auc_by_num_features(self):
        """
        Plots AUC as a function of the number of top features included in the model.
        """

        # keys are the number of features included, so just sort them so we can plot in order
        keys = np.sort(self.res.keys())
        y = [self.res[x]['auc'] for x in keys]

        with plt.style.context('myplotstyle.mplstyle'):
            plt.plot(np.arange(len(y)) + 1, y, linewidth=4)
            plt.ylabel('AUC', fontsize=20)
            plt.xlabel('# Electrodes' if self.do_top_elecs else '# Frequencies', fontsize=20)
            plt.title(self.subj, fontsize=20)

