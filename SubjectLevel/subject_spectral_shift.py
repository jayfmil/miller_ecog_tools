import os
import pdb
import ram_data_helpers
import cPickle as pickle
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn import linear_model
from subject_data import SubjectData


class SubjectSpectralShift(SubjectData):
    """
    Subclass of SubjectData with methods to analyze power spectrum of each electrode. More details..
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSpectralShift, self).__init__(task=task, subject=subject)

        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        self.exclude_by_rec_time = False
        self.rec_thresh = None

        self.task_phase = ['enc']  # ['enc'] or ['rec']

        # put a check on this, has to be power
        self.feat_type = 'power'

        self.load_res_if_file_exists = False
        self.save_class = False

        # will hold classifier results after loaded or computed
        self.res = {}

        # location to save or load classification results will be defined after call to make_res_dir()
        self.res_dir = None
        self.res_save_file = None

    def run(self):
        """
        Basically a convenience function to do all the .
        """
        if self.subject_data is None:
            print('%s: Data must be loaded before running. Use .load_data()' % self.subj)
            return

        # Step 1: create (if needed) directory to save/load
        self.make_res_dir()

        # Step 2: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 3: if not loaded ...
        if not self.res:

            # Step 3B: classify. how to handle normalization in particular the different phases. Do the different
            # phases need to be normalized relative to themselves, or can they be lumped together. together is easier..
            print('%s: Running classify.' % self.subj)
            self.classify()

            # save to disk
            if self.save_class:
                self.save_res_data()

    def make_res_dir(self):
        """
        Create directory where classifier data will be saved/loaded if it needs to be created. This also will define
        self.res_dir and self.res_save_file
        """

        self.res_dir = self._generate_res_save_path()
        self.res_save_file = os.path.join(self.res_dir, self.subj + '_' + self.feat_type + '.p')
        if not os.path.exists(self.res_dir):
            try:
                os.makedirs(self.res_dir)
            except OSError:
                pass

    def load_res_data(self):
        """
        Load classifier results if they exist and modify self.class_res to hold them.
        """
        if self.res_save_file is None:
            print('self.res_save_file must be defined before loading, .make_res_dir() will do this and create the '
                  'save directory for you.')
            return

        if os.path.exists(self.res_save_file):
            with open(self.res_save_file, 'rb') as f:
                res = pickle.load(f)
            self.res = res
        else:
            print('%s: No classifier data to load.' % self.subj)

    def save_res_data(self):
        """

        """
        if not self.res:
            print('Classifier data must be loaded or computed before saving. Use .load_data() or .classify()')
            return

        # write pickle file
        with open(self.res_save_file, 'wb') as f:
            pickle.dump(self.res, f, protocol=-1)


    def model(self):
        """
        Fits a robust regression model to the power spectrum of each electrode in order to get the slope and intercept.
        This fits every event individually in addition to each electrode, so it's a couple big loops. Sorry. It seems
        like you should be able to it all with one call by having multiple columns in y, but the results are different
        than looping, so..
        """

        # Get recalled or not labels
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # x var is frequency of the power spectrum
        x = np.expand_dims(np.log10(self.subject_data.frequency.data), axis=1)

        # create arrays to hold intercepts and slopes
        elec_str = 'bipolar_pairs' if self.bipolar else 'channels'
        intercepts = np.empty((len(self.subject_data['events']), len(self.subject_data[elec_str])))
        intercepts[:] = np.nan
        slopes = np.empty((len(self.subject_data['events']), len(self.subject_data[elec_str])))
        slopes[:] = np.nan

        # initialize regression model
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

        # loop over electrodes
        for elec in range(len(self.subject_data[elec_str])):

            # loop over events
            for ev in range(len(self.subject_data['events'])):

                # power values for this elec/event
                y = self.subject_data[ev, :, elec].data

                # fit line
                model_ransac.fit(x, y)

                # store result
                intercepts[ev, elec] = model_ransac.estimator_.intercept_
                slopes[ev, elec] = model_ransac.estimator_.coef_

        # store results
        res = {}

        # store the slopes and intercepts
        res['slopes'] = slopes
        res['intercepts'] = intercepts



        # estimated probabilities and true class labels
        res['probs'] = probs[test_bool, 0 if ~loso else np.argmax(aucs)]
        res['Y'] = Y[test_bool]

        # model fit on all the training data
        lr_classifier.C = C
        res['model'] = lr_classifier.fit(X[train_bool], Y[train_bool])

        # boolean array of all entries in subject data that were used to train classifier over all folds. Depending
        # on self.task_phase, this isn't necessarily all the data
        res['train_bool'] = train_bool

        # easy to check flag for multisession data
        res['loso'] = loso
        self.class_res = res

    def normalize_power(self, X):
        """
        Normalizes (zscores) each column in X. If rows of comprised of different task phases, each task phase is
        normalized to itself

        returns normalized X
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in set(self.task_phase + self.test_phase):
                task_mask = self.task_phase == phase
                X[sess_event_mask & task_mask] = zscore(X[sess_event_mask & task_mask], axis=0)
        return X

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = '%s_%s' %(self.recall_filter_func.__name__, self.task_phase)
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

