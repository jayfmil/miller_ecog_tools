"""
"""
# from __future__ import print_function
import os
import pdb
import ram_data_helpers
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import xarray as xray
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn import linear_model
from SubjectLevel.subject_analysis import SubjectAnalysis
plt.style.use('/home1/jfm2/python/RAM_classify/myplotstyle.mplstyle')


class SubjectSpectralShift(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode. More details..
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSpectralShift, self).__init__(task=task, subject=subject)

        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None
        self.task_phase = ['enc']  # ['enc'] or ['rec']

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'robust_reg.p'

    def run(self):
        """
        Basically a convenience function to do all the .
        """

        # Step 1: load data
        if self.subject_data is None:
            self.load_data()

        # Step 1: create (if needed) directory to save/load
        self.make_res_dir()

        # Step 2: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 3: if not loaded ...
        if not self.res:

            # Step 3A: fit model
            print('%s: Running robust regression.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
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
        n_elecs = len(self.subject_data[elec_str])
        n_events = len(self.subject_data['events'])
        intercepts = np.empty((n_events, n_elecs))
        intercepts[:] = np.nan
        slopes = np.empty((n_events, n_elecs))
        slopes[:] = np.nan

        # initialize regression model
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

        # loop over electrodes
        for elec in range(n_elecs):
            print('%s: elec %d of %d.' % (self.subj, elec+1, n_elecs))
            # loop over events
            for ev in range(n_events):

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
        self.res = res

    # Lab meeting topic: how does the SME manifest in terms of changes in the power spectrum?
    # Show example trials with fit line. Find examples of different kinds: tilt, shift, oscillation

    def plot_spectra_average(self):
        plt.plot(np.log10(s.subject_data.frequency), s.subject_data[recalled, :, 20].mean('events'), c='#8c564b')
        plt.plot(np.log10(s.subject_data.frequency), s.subject_data[~recalled, :, 20].mean('events'), c='#1f77b4')

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

        dir_str = 'robust_regress_%s_%s' %(self.recall_filter_func.__name__, self.task_phase[0])
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

