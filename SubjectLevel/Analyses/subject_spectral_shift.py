"""
"""
# from __future__ import print_function
import os
import pdb
import ram_data_helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import ttest_ind
# from SubjectLevel.subject_analysis import SubjectAnalysis
# from SubjectLevel.Analyses import subject_SME
from SubjectLevel.Analyses.subject_SME import SubjectSME as SME


class SubjectSME(SME):
    """
    Version of SubjectSME that, instead of performing the stats on normalized power, first fits a robust regression line
    to the power spectra, and then does stats on the residuals, the slope, and the offset.
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSME, self).__init__(task=task, subject=subject)

        # string to use when saving results files
        self.res_str = 'robust_reg.p'

    def analysis(self):
        """
        Fits a robust regression model to the power spectrum of each electrode in order to get the slope and intercept.
        This fits every event individually in addition to each electrode, so it's a couple big loops. Sorry. It seems
        like you should be able to it all with one call by having multiple columns in y, but the results are different
        than looping, so..
        """

        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # x var is frequency of the power spectrum
        x = np.expand_dims(np.log10(self.subject_data.frequency.data), axis=1)

        # create arrays to hold intercepts and slopes
        elec_str = 'bipolar_pairs' if self.bipolar else 'channels'
        n_elecs = len(self.subject_data[elec_str])
        n_events = len(self.subject_data['events'])

        # holds intercepts (offsets) of fit line
        intercepts = np.empty((n_events, n_elecs))
        intercepts[:] = np.nan

        # holds slope of fit line
        slopes = np.empty((n_events, n_elecs))
        slopes[:] = np.nan

        # holds residuals
        resids = np.empty((n_events, n_elecs, len(self.freqs)))
        resids[:] = np.nan

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

                # compute residuals
                resids[ev, elec, :] = y - model_ransac.predict(x)

        # make a new array that is the concatenation of residuals, slopes, intercepts.
        # shape is num events x num elecs x (num freqs + 2). Reshape to be num events x whatever so we can do a ttest
        # comparing recalled and now recalled by columns
        X = np.concatenate([resids, np.expand_dims(slopes, axis=2), np.expand_dims(intercepts, axis=2)], axis=2)
        X = X.reshape(X.shape[0], -1)

        # store results
        self.res = {}

        # run ttest comparing good and bad memory at each feature
        ts, ps, = ttest_ind(X[recalled], X[~recalled])
        self.res['ts'] = ts.reshape(len(self.freqs)+2, -1)
        self.res['ps'] = ps.reshape(len(self.freqs)+2, -1)

        # compute all the average stats that we also compute SubjectSME
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()
        self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()
        sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        self.res['contig_freq_inds_pos'] = contig_pos
        sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        self.res['contig_freq_inds_neg'] = contig_neg

        # store the slopes, intercepts, and residuals as well
        self.res['slopes'] = slopes
        self.res['intercepts'] = intercepts
        self.res['resids'] = resids




