"""
"""
import os
import pdb
import ram_data_helpers
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import xarray as xray
from scipy.stats.mstats import zscore, zmap
from copy import deepcopy
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from SubjectLevel.subject_analysis import SubjectAnalysis
# plt.style.use('/home1/jfm2/python/RAM_classify/myplotstyle.mplstyle')


class SubjectSME(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode. Specifically,
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSME, self).__init__(task=task, subject=subject)
        self.task_phase = ['enc']  # ['enc'] or ['rec']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'sme.p'

    def run(self):
        """
        Convenience function to run all the steps for SubjectSME analysis.
        """
        if self.feat_type != 'power':
            print('%s: .feat_type must be set to power for this analysis to run.' % self.subj)
            return

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

            # Step 3A: fit model
            print('%s: Running SME.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency using a two sample ttest.

        .res will have the keys 'ts' and 'ps'.
        """

        # Get recalled or not labels
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # reshape the power data to be events x features and normalize
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[recalled], X[~recalled])

        # store results
        res = {}

        # store the t-stats and p values for each electrode and freq. Reshape back to frequencies x electrodes.
        res['ts'] = ts.reshape(len(self.freqs), -1)
        res['ps'] = ps.reshape(len(self.freqs), -1)
        self.res = res

    # Lab meeting topic: how does the SME manifest in terms of changes in the power spectrum?
    # Show example trials with fit line. Find examples of different kinds: tilt, shift, oscillation

    # def plot_spectra_average(self):
    #     plt.plot(np.log10(s.subject_data.frequency), s.subject_data[recalled, :, 20].mean('events'), c='#8c564b')
    #     plt.plot(np.log10(s.subject_data.frequency), s.subject_data[~recalled, :, 20].mean('events'), c='#1f77b4')

    def sme_by_region(self, region):
        """

        """
        loc_dict = ram_data_helpers.bin_elec_locs(self.subject_data.attrs['loc_tag'],
                                                  self.subject_data.attrs['anat_region'],
                                                  self.subject_data.attrs['chan_tags'])

        inds = loc_dict[region]
        return np.nanmean(self.res['ts'][:, inds], axis=1)

    def sme_by_elec(self, region):
        loc_dict = ram_data_helpers.bin_elec_locs(self.subject_data.attrs['loc_tag'],
                                                  self.subject_data.attrs['anat_region'],
                                                  self.subject_data.attrs['chan_tags'])

        inds = loc_dict[region]

    def normalize_power(self, X):
        """
        Normalizes (zscores) each column in X. If rows of comprised of different task phases, each task phase is
        normalized to itself

        returns normalized X
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            X[sess_event_mask] = zscore(X[sess_event_mask], axis=0)
        return X

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = 'sme_%s_%s' %(self.recall_filter_func.__name__, self.task_phase[0])
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

