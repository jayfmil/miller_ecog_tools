"""
Uses a penalized logistic regression classifier to predict single-trial memory success or failure based on spectral
power features, here as a function of number of electrodes included.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, sem
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from tqdm import tqdm

from miller_ecog_tools.SubjectLevel.Analyses import subject_classifier


class SubjectClassifierNFeaturesAnalysis(subject_classifier.SubjectClassifierAnalysis):
    """
    Modified SubjectClassifierAnalysis that iteratively computes classification performance as a function of number of
    electrodes, from 1 .. total number.

    Electrodes will be included in order of decreasing strength univariate t-test vs good and bad memory. In other
    words, for the iteration with only 1 electrode, it will be the "best" electrode. This is Figure 7 of Miller et al.,
    2018. Big picture: this can test if the signals at different electrodes are redundant.
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectClassifierNFeaturesAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'classifier_n_elecs.p'

        # number of random splits to do when calculating mean auc as a function of number of electrodes
        self.num_rand_splits = 100

        # whether to use joblib to parallelize the random splits. If not, do in serial.
        self.use_joblib = True

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__ + '_res')

    def _make_cross_val_labels(self):
        """
        Unlike SubjectClassifierAnalysis, we here use a split-half CV. One half will be used to identify the features to
        use using a univariate t-test. The other half will be the test set.
        """

        # split the data into a random half
        folds = np.zeros(shape=self.subject_data.shape[0])
        folds[np.random.rand(self.subject_data.shape[0]) < .5] = 1

        # make dictionary to hold booleans for training and test indices for each fold
        cv_dict = {}
        uniq_folds = np.unique(folds)
        for fold in uniq_folds:
            cv_dict[fold] = {}
            cv_dict[fold]['train_bool'] = folds != fold
            cv_dict[fold]['test_bool'] = folds == fold
        return cv_dict

    def analysis(self, permute=False):
        """
        Classify based an iteratively increasing the number of features (electrodes) included in the model. Starts with
        the single best electrode (N=1) and increase until N = the number of electrodes.

        Note: permute is not used in this analysis, but kept to match the same signature as super.
        """
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s classifier: please provide a .recall_filter_func function.' % self.subject)
        y = self.recall_filter_func(self.subject_data)

        # zscore the data by session
        x = self.zscore_data()

        # create the classifier
        classifier = LogisticRegression(C=self.C, penalty=self.norm, solver='liblinear')

        # create .num_rand_splits of cv_dicts
        cv_dicts = [self._make_cross_val_labels() for _ in range(self.num_rand_splits)]

        # run permutations with joblib
        f = _par_compute_and_run_split
        if self.use_joblib:
            aucs = Parallel(n_jobs=12, verbose=5)(delayed(f)(cv, classifier, x, y) for cv in cv_dicts)
        else:
            aucs = []
            for cv in tqdm(cv_dicts):
                aucs.append(f(cv, classifier, x, y))

        # store results
        self.res['auc_x_n'] = np.stack(aucs)

    ######################
    # PLOTTING FUNCTIONS #
    ######################
    def plot_auc_x_num_features(self):
        """
        Plots AUC as a function of number of electrodes included, with SEM shadded error regions

        """
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                e = sem(self.res['auc_x_n'], axis=0)
                m = self.res['auc_x_n'].mean(axis=0)
                x = np.arange(1, len(m) + 1)
                plt.fill_between(x, m - e, m + e, alpha=.5)
                plt.plot(x, m, '-k', lw=3)
                plt.ylabel('AUC')
                plt.xlabel('# Electrodes')
                plt.gcf().set_size_inches(12, 9)

    def plot_roc(self):
        """
        Not applicable to this subject_classifier.SubjectClassifierAnalysis subclass
         """
        print('plot_roc not applicable to SubjectClassifierNFeaturesAnalysis.')

    def plot_elec_heat_map(self, sortby_column1='', sortby_column2=''):
        """
        Not applicable to this subject_classifier.SubjectClassifierAnalysis subclass
        """
        print('plot_elec_heat_map not applicable to SubjectClassifierNFeaturesAnalysis.')


def _par_compute_and_run_split(cv_dict, classifier, x, y):
    """
    Parallelizable function for computing AUC as a function of number of electrodes

    Returns list of length equal to number of electrodes, with AUC for each N.
    """
    # t-test on one half
    ts, ps = ttest_ind(x[y & cv_dict[0]['train_bool']], x[~y & cv_dict[0]['train_bool']], axis=0)

    # sort in order of descending max t-stat. First element will be the "best" electrode, and so on.
    t_inds_sorted = np.argsort(np.abs(ts).max(axis=0))[::-1]

    # now loop over all electrodes, including the first n elecs
    aucs_x_n = []
    for n in range(len(t_inds_sorted)):
        x_iter = x[:, :, t_inds_sorted[0:n + 1]].reshape(x.shape[0], -1)

        # do the actual classification for this set of electrodes
        auc, probs = subject_classifier.do_cv({'0': cv_dict[0]}, False, classifier, x_iter, y, permute=False)
        aucs_x_n.append(auc)
    return aucs_x_n
