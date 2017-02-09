"""
"""
# from __future__ import print_function
import os
import pdb
import ram_data_helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, sem
from copy import deepcopy
from SubjectLevel.subject_analysis import SubjectAnalysis
from scipy.stats.mstats import zscore
from SubjectLevel.par_funcs import par_find_peaks


class SubjectPeaks(SubjectAnalysis):
    """
    Version of SubjectSME that, instead of performing the stats on normalized power, first fits a robust regression line
    to the power spectra, and then does stats on the residuals, the slope, and the offset.
    """

    def __init__(self, task=None, subject=None):
        super(SubjectPeaks, self).__init__(task=task, subject=subject)
        self.task_phase_to_use = ['enc']  # ['enc'] or ['rec']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        self.rec_thresh = None

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'peaks.p'

    def run(self):
        """
        Convenience function to run all the steps for SubjectPeaks analysis.
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
            print('%s: Finding spectral peaks.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Fits a robust regression model to the power spectrum of each electrode
        """

        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # noramlize power spectra
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        mean_p_spect = np.mean(p_spect, axis=0)

        # x var is frequency of the power spectrum
        x = np.expand_dims(np.log10(self.subject_data.frequency.data), axis=1)
        x_rep = np.tile(x, mean_p_spect.shape[1]).T

        # find peak frequencies for each electrode
        if self.pool is None:
            elec_res = map(par_find_peaks, zip(mean_p_spect.T, x_rep))
        else:
            elec_res = self.pool.map(par_find_peaks, zip(mean_p_spect.T, x_rep))
        peak_freqs = np.stack([foo for foo in elec_res])

        # reshape the power data to be events x features and normalize
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[recalled], X[~recalled])

        # store results
        self.res = {}
        self.res['ts'] = ts.reshape(len(self.freqs), -1)
        self.res['ps'] = ps.reshape(len(self.freqs), -1)
        self.res['peak_freqs'] = peak_freqs.T
        self.res['peaks_mean_region'], self.res['peaks_count_region'], self.res['elec_n'] = self.peaks_by_region_counts()
        self.res['regions'] = np.array(sorted(self.elec_locs.keys()))

        # run ttest comparing good and bad memory at each feature
        # ts, ps, = ttest_ind(X[recalled], X[~recalled])
        # self.res['ts'] = ts.reshape(len(self.freqs)+2, -1)
        # self.res['ps'] = ps.reshape(len(self.freqs)+2, -1)

        # compute all the average stats that we also compute SubjectSME
        # self.res['ts_region'], self.res['regions'] = self.sme_by_region()
        # self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()
        # sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        # contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        # self.res['contig_freq_inds_pos'] = contig_pos
        # sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        # contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        # self.res['contig_freq_inds_neg'] = contig_neg

        # store the slopes, intercepts, and residuals as well
        # self.res['slopes'] = slopes
        # self.res['bband_power'] = bband_power
        # self.res['intercepts'] = intercepts
        # self.res['resids'] = resids

    def peaks_by_region_counts(self):
        """

        """
        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        regions = np.array(sorted(self.elec_locs.keys()))
        peaks = self.res['peak_freqs']
        #     ts = self.res['ts']
        #     ps = self.res['ps']

        mean_peaks = np.stack([np.mean(peaks[:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
        count_peaks = np.stack([np.sum(peaks[:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
        n = np.array([np.nansum(self.elec_locs[x]) for x in regions])
        return mean_peaks, count_peaks, n

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
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        with plt.style.context('myplotstyle.mplstyle'):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            x = np.log10(self.subject_data.frequency)
            y = np.mean(p_spect[:, :, elec], axis=0)
            ax1.plot(x, y, c='#8c564b', linewidth=4)
            ax1.set_ylabel('Normalized log(power)')
            ax1.yaxis.label.set_fontsize(24)
            l = ax1.legend()

            peaks = self.res['peak_freqs'][:, elec]
            ax1.scatter(x[peaks], y[peaks], s=100, c='k', zorder=10)

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
            new_x = self.compute_pow_two_series()
            ax2.xaxis.set_ticks(np.log10(new_x))
            ax2.xaxis.set_ticklabels(new_x, rotation=0)
            # _ = plt.xticks(x[::4], np.round(self.freqs[::4] * 10) / 10, rotation=-45)

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax1.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec+1, chan_tag, anat_region, loc))

        return f

    def plot_sme_only_peaks(self):

        peaks = self.res['peak_freqs']
        ts = self.res['ts']
        ts[~peaks] = np.nan
        # plt.plot(np.nanmean(ts, axis=1))

        with plt.style.context('myplotstyle.mplstyle'):
            f, ax = plt.subplots(2, 1, sharex=True)

            ax.scatter(np.log10(self.freqs), np.nanmean(ts, axis=1))
            new_x = self.compute_pow_two_series()
            ax.xaxis.set_ticks(np.log10(new_x))
            ax.xaxis.set_ticklabels(new_x, rotation=0)

    def plot_peaks(self):
        """

        """

        if not self.res:
            print('%s: must run .analysis() before plotting peaks.' % self.subj)
            return
        # pdb.set_trace()
        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)

            clim = np.max(np.abs([np.nanmin(self.res['ts']), np.nanmax(self.res['ts'])])) + 2
            im1 = plt.imshow(self.res['ts'], interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim,
                             aspect='auto')

            p2 = np.ma.masked_where(self.res['ps'] < .05, self.res['ps'])

            plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=1)

            p3 = np.ma.masked_where(~self.res['peak_freqs'], self.res['peak_freqs'])
            plt.imshow(p3 > 0, interpolation='nearest', cmap='gray', aspect='auto', alpha=.5)

            #         im2 = plt.imshow(self.res['peak_freqs'].T, interpolation='nearest', cmap='binary', aspect='auto')

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs), range(len(self.freqs)))

            ax.yaxis.set_ticks(new_y)
            ax.yaxis.set_ticklabels(new_freqs, rotation=0)

            # _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Electrode', fontsize=24)
            ax.xaxis.set_ticks(np.arange(self.res['peak_freqs'].shape[1])[::10])
            ax.xaxis.set_ticklabels(np.arange(self.res['peak_freqs'].shape[1])[::10] + 1)

            plt.gca().invert_yaxis()

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))

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

        dir_str = 'peaks_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

















