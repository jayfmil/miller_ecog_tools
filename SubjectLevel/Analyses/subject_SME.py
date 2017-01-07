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
from operator import itemgetter
from itertools import groupby
from scipy.stats.mstats import zscore, zmap
from copy import deepcopy
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from SubjectLevel.subject_analysis import SubjectAnalysis
# plt.style.use('/home1/jfm2/python/RAM_classify/myplotstyle.mplstyle')


class SubjectSME(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode.
    """

    def __init__(self, task=None, subject=None):
        super(SubjectSME, self).__init__(task=task, subject=subject)
        self.task_phase_to_use = ['enc']  # ['enc'] or ['rec']
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

            # Step 4A: compute subsequenct memory effect at each electrode
            print('%s: Running SME.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency using a two sample ttest.

        .res will have the keys:
                                 'ts'
                                 'ps'
                                 'regions'
                                 'ts_region'
                                 'sme_count_pos'
                                 'sme_count_neg'
                                 'elec_n'
                                 'contig_freq_inds_pos'
                                 'contig_freq_inds_neg'

        """

        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # reshape the power data to be events x features and normalize
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[recalled], X[~recalled])

        # store results.
        self.res = {}

        # store the t-stats and p values for each electrode and freq. Reshape back to frequencies x electrodes.
        self.res['ts'] = ts.reshape(len(self.freqs), -1)
        self.res['ps'] = ps.reshape(len(self.freqs), -1)

        # make a binned version of t-stats that is frequency x brain region. Calling this from within .analysis() for
        # convenience because I know the data is loaded now, which we need to have access to the electrode locations.
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()

        # also counts of positive SME electrodes and negative SME electrodes by region
        self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()

        # also, for each electrode, find ranges of neighboring frequencies that are significant for both postive and
        # negative effecst
        sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        self.res['contig_freq_inds_pos'] = contig_pos

        sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        self.res['contig_freq_inds_neg'] = contig_neg

    def plot_spectra_average(self, elec):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for recalled (red) and not-recalled (blue) items. Bottom panel is t-stat at each frequency comparing the
        recalled and not recalled distributions, with shaded areas indicating p<.05.

        elec (int): electrode number that you wish to plot.
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        with plt.style.context('myplotstyle.mplstyle'):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            x = np.log10(self.subject_data.frequency)
            ax1.plot(x, self.subject_data[recalled, :, elec].mean('events'), c='#8c564b', label='Good Memory', linewidth=4)
            ax1.plot(x, self.subject_data[~recalled, :, elec].mean('events'), c='#1f77b4', label='Bad Memory', linewidth=4)
            ax1.set_ylabel('log(power)')
            ax1.yaxis.label.set_fontsize(24)
            l = ax1.legend()

            y = self.res['ts'][:, elec]
            p = self.res['ps'][:, elec]
            ax2.plot(x, y, '-k', linewidth=4)
            ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
            ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=-1)

            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y > 0), facecolor='#8c564b', edgecolor='#8c564b')
            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y < 0), facecolor='#1f77b4', edgecolor='#1f77b4')
            ax2.set_ylabel('t-stat')
            ax2.yaxis.label.set_fontsize(24)

            plt.xlabel('Frequency', fontsize=24)
            _ = plt.xticks(x[::4], np.round(self.freqs[::4] * 10) / 10, rotation=-45)

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax1.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec+1, chan_tag, anat_region, loc))

        return f

    def find_continuous_ranges(self, data):
        """
        Given an array of integers, finds continuous ranges. Similar in concept to 1d version of bwlabel in matlab on a
        boolean vector. This method is really clever, it subtracts the index of each entry from the value and then
        groups all those with the same difference.

        Credit: http://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
        """

        ranges = []
        for k, g in groupby(enumerate(data), lambda (i, x): i - x):
            group = map(itemgetter(1), g)
            ranges.append((group[0], group[-1]))
        return ranges

    def sme_by_region(self):
        """
        Bin (average) res['ts'] by brain region. Return array that is freqs x region, and return region strings.
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        # average all the elecs within each region. Iterate over the sorted keys because I don't know if dictionary
        # keys are always returned in the same order?
        regions = np.array(sorted(self.elec_locs.keys()))
        t_array = np.stack([np.nanmean(self.res['ts'][:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
        return t_array, regions

    def sme_by_region_counts(self):
        """
        Count of significant electrodes by region
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        regions = np.array(sorted(self.elec_locs.keys()))
        ts = self.res['ts']
        ps = self.res['ps']

        # counts of significant positive SMEs
        count_pos = [np.nansum((ts[:, self.elec_locs[x]] > 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_pos = np.stack(count_pos, axis=1)

        # counts of significant negative SMEs
        count_neg = [np.nansum((ts[:, self.elec_locs[x]] < 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_neg = np.stack(count_neg, axis=1)

        # count of electrodes by region
        n = np.array([np.nansum(self.elec_locs[x]) for x in regions])
        return count_pos, count_neg, n

    def normalize_power(self, X):
        """
        Normalizes (zscores) each column in X. If rows of comprised of different task phases, each task phase is
        normalized to itself

        returns normalized X
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in self.task_phase_to_use:
                task_mask = self.task_phase == phase
                X[sess_event_mask & task_mask] = zscore(X[sess_event_mask & task_mask], axis=0)
        return X

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = 'sme_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

