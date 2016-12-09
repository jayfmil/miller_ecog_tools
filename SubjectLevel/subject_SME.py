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
from copy import deepcopy
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn import linear_model
from subject_data import SubjectData
plt.style.use('/home1/jfm2/python/RAM_classify/myplotstyle.mplstyle')


class SubjectSME(SubjectData):
    """
    Subclass of SubjectData with methods to analyze power spectrum of each electrode. More details..
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSME, self).__init__(task=task, subject=subject)

        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None

        self.task_phase = ['enc']  # ['enc'] or ['rec']

        # put a check on this, has to be power
        self.feat_type = 'power'

        self.load_res_if_file_exists = False
        self.save_res = True

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

            # Step 3A: fit model
            print('%s: Running SME.' % self.subj)
            self.sme()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def make_res_dir(self):
        """
        Create directory where classifier data will be saved/loaded if it needs to be created. This also will define
        self.res_dir and self.res_save_file
        """

        self.res_dir = self._generate_res_save_path()
        self.res_save_file = os.path.join(self.res_dir, self.subj + '_robust_reg.p')
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
            print('Slopes/intercept data must be loaded or computed before saving. Use .load_data() or .model()')
            return

        # write pickle file
        with open(self.res_save_file, 'wb') as f:
            pickle.dump(self.res, f, protocol=-1)

    def sme(self):
        """

        """

        # Get recalled or not labels
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # x var is frequency of the power spectrum
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)
        # X = np.reshape(X, self.subject_data.shape)
        ts, ps, = ttest_ind(X[recalled], X[~recalled])

        # store results
        res = {}

        # store the slopes and intercepts
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

