"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency and time bine, compare correctly and
incorrectly recalled items using a t-test.
"""
import os
import pdb
import ram_data_helpers
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from scipy.stats.mstats import zscore, zmap
from copy import deepcopy
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from SubjectLevel.subject_analysis import SubjectAnalysis
import matplotlib.cm as cmx
import matplotlib.colors as clrs

try:

    # this will fail is we don't have an x server
    disp = os.environ['DISPLAY']
    from surfer import Surface, Brain
    from mayavi import mlab
    import platform
    if platform.system() == 'Darwin':
        os.environ['SUBJECTS_DIR'] = '/Users/jmiller/data/eeg/freesurfer/subjects/'
    else:
        os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'
except (ImportError, KeyError):
    print('Brain plotting not supported')


class SubjectSMETime(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode. Differs from SubjectSME in
    that this analysis includes multiple time bins
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectSMETime, self).__init__(task=task, subject=subject, montage=montage)
        self.task_phase_to_use = ['enc']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'sme_time.p'

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
            print('%s: Running SME with multiple time windows.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency and time using a two sample ttest.

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
                                 MORE UPDATE THIS

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

        # for convenience, also compute within power averaged bands for low freq and high freq
        lfa_inds = self.freqs <= 10
        lfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, lfa_inds, :, :], axis=1)
        lfa_ts, lfa_ps = ttest_ind(lfa_pow[recalled], lfa_pow[~recalled])

        # high freq
        hfa_inds = self.freqs >= 60
        hfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, hfa_inds, :, :], axis=1)
        hfa_ts, hfa_ps = ttest_ind(hfa_pow[recalled], hfa_pow[~recalled])

        # store results.
        self.res = {}
        self.res['ts_lfa'] = lfa_ts
        self.res['ps_lfa'] = lfa_ps
        self.res['ts_hfa'] = hfa_ts
        self.res['ps_hfa'] = hfa_ps

        # store the t-stats and p values for each electrode and freq. Reshape back to frequencies x electrodes.
        self.res['ts'] = ts.reshape(self.subject_data.shape[1:])
        self.res['ps'] = ps.reshape(self.subject_data.shape[1:])

        self.res['time_bins'] = self.subject_data['time'].data
        # make a binned version of t-stats that is frequency x brain region. Calling this from within .analysis() for
        # convenience because I know the data is loaded now, which we need to have access to the electrode locations.
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()

        # also counts of positive SME electrodes and negative SME electrodes by region
        # self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()

        # also, for each electrode, find ranges of neighboring frequencies that are significant for both postive and
        # negative effecst
        # sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        # contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        # self.res['contig_freq_inds_pos'] = contig_pos

        # sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        # contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        # self.res['contig_freq_inds_neg'] = contig_neg

    def plot_time_by_freq(self, elec):

        plot_data = self.res['ts'][:, elec, :]
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(self.res['time_bins']))[::3], self.res['time_bins'][::3], fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs), range(len(self.freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.gca().invert_yaxis()
            plt.grid()

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec + 1, chan_tag, anat_region, loc))

    def plot_time_by_freq_region(self, region):

        ind = self.res['regions'] == region
        plot_data = self.res['ts_region'][:, ind, :]
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(self.res['time_bins']))[::3], self.res['time_bins'][::3], fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs), range(len(self.freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.gca().invert_yaxis()
            plt.grid()

            # chan_tag = self.subject_data.attrs['chan_tags'][elec]
            # anat_region = self.subject_data.attrs['anat_region'][elec]
            # loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax.set_title('%s - %s' % (self.subj, region))

    def sme_by_region(self, res_key='ts'):
        """
        Bin (average) res['ts'] by brain region. Return array that is freqs x region x time, and return region strings.
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
        t_array = np.stack([np.nanmean(self.res[res_key][:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
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

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in self.task_phase_to_use:
                task_mask = self.task_phase == phase

                m = np.mean(X[sess_event_mask & task_mask], axis=1)
                m = np.mean(m, axis=0)
                s = np.std(X[sess_event_mask & task_mask], axis=1)
                s = np.mean(s, axis=0)
                X[sess_event_mask & task_mask] = (X[sess_event_mask & task_mask] - m) / s
        return X

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = 'sme_time_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

