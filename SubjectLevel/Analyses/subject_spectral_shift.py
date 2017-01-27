"""
"""
# from __future__ import print_function
import os
import pdb
import ram_data_helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec
from sklearn import linear_model
import statsmodels.api as sm
from scipy.stats import ttest_ind, sem
from copy import deepcopy
# from SubjectLevel.subject_analysis import SubjectAnalysis
# from SubjectLevel.Analyses import subject_SME
from SubjectLevel.Analyses.subject_SME import SubjectSME as SME
from SubjectLevel.par_funcs import par_robust_reg


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

        # noramlize power spectra
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        # x var is frequency of the power spectrum
        x = np.expand_dims(np.log10(self.subject_data.frequency.data), axis=1)
        x_rep = np.tile(x, p_spect.shape[0]).T

        # run robust regression for each event and elec in order to get the residuals, slope, and offset
        print('%s: Running robust regression for %d elecs and %d events.' % (self.subj, p_spect.shape[2], p_spect.shape[0]))
        if self.pool is None:
            elec_res = map(par_robust_reg, zip(p_spect, x_rep))
        else:
            elec_res = self.pool.map(par_robust_reg, zip(p_spect, x_rep))

        intercepts = np.stack([foo[0] for foo in elec_res])
        slopes = np.stack([foo[1] for foo in elec_res])
        resids = np.stack([foo[2] for foo in elec_res])
        bband_power = np.stack([foo[3] for foo in elec_res])

        # make a new array that is the concatenation of residuals, slopes, broadband power, intercepts
        # shape is num events x (num freqs + 2) x num elecs. Reshape to be num events x whatever so we can do a ttest
        # comparing recalled and now recalled by columns
        X = np.concatenate([resids, np.expand_dims(slopes, axis=1), np.expand_dims(bband_power, axis=1)], axis=1)
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
        self.res['bband_power'] = bband_power
        self.res['intercepts'] = intercepts
        self.res['resids'] = resids

    def plot_spectra_average(self, elec):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for recalled (red) and not-recalled (blue) items. Bottom panel is t-stat at each frequency comparing the
        recalled and not recalled distributions, with shaded areas indicating p<.05.

        Plotting code is sometimes so ugly sorry.

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
            f = plt.figure()
            gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[3, 1], wspace=.35)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[0, 1])
            ax4 = plt.subplot(gs[1, 1])

            x = np.log10(self.subject_data.frequency)
            ax1.plot(x, self.res['resids'][recalled, :, elec].mean(axis=0), c='#8c564b', label='Good Memory', linewidth=4)
            ax1.plot(x, self.res['resids'][~recalled, :, elec].mean(axis=0), c='#1f77b4', label='Bad Memory', linewidth=4)
            ax1.set_ylabel('Residual')
            ax1.yaxis.label.set_fontsize(24)
            l = ax1.legend(loc=4)

            y = self.res['ts'][:-2, elec]
            p = self.res['ps'][:-2, elec]
            ax2.plot(x, y, '-k', linewidth=4)
            ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
            ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=-1)

            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y > 0), facecolor='#8c564b', edgecolor='#8c564b')
            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y < 0), facecolor='#1f77b4', edgecolor='#1f77b4')
            ax2.set_ylabel('t-stat')
            ax2.yaxis.label.set_fontsize(24)

            ax2.set_xlabel('Frequency')
            ax2.xaxis.label.set_fontsize(24)

            # ax1.xaxis.set_ticks(x[::4])
            # ax1.xaxis.set_ticklabels('')
            # ax2.xaxis.set_ticks(x[::4])
            # ax2.xaxis.set_ticklabels(np.round(self.freqs[::4] * 10) / 10, rotation=-45)

            new_x = self.compute_pow_two_series()
            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.xaxis.set_ticklabels('')
            ax2.xaxis.set_ticks(np.log10(new_x))
            ax2.xaxis.set_ticklabels(new_x, rotation=0)

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            ttl = ax1.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec + 1, chan_tag, anat_region, loc))
            ttl.set_position([.5, 1.05])

            slopes = [self.res['slopes'][recalled, elec].mean(axis=0), self.res['slopes'][~recalled, elec].mean(axis=0)]
            slopes_e = [sem(self.res['slopes'][recalled, elec], axis=0) * 1.96,
                        sem(self.res['slopes'][~recalled, elec], axis=0) * 1.96]

            offsets = [self.res['intercepts'][recalled, elec].mean(axis=0),
                       self.res['intercepts'][~recalled, elec].mean(axis=0)]
            offsets_e = [sem(self.res['intercepts'][recalled, elec], axis=0) * 1.96,
                         sem(self.res['intercepts'][~recalled, elec], axis=0) * 1.96]

            ax3.bar([.75, 1.25], slopes, .35, alpha=1,
                    yerr=slopes_e, zorder=4, color=['#8c564b', '#1f77b4'],
                    error_kw={'zorder': 10, 'ecolor': 'k'})
            ax3.set_ylabel('Slope', fontsize=24)
            # ax3.set_xlim(.5, 1.75)
            # ax3.set_ylim(top=-1.5)
            _ = ax3.xaxis.set_ticklabels('')

            ax4.bar([.75, 1.25], offsets, .35, alpha=1,
                             yerr=offsets_e, zorder=4, color=['#8c564b', '#1f77b4'],
                             error_kw={'zorder': 10, 'ecolor': 'k'})
            ax4.set_ylabel('Offset', fontsize=24)
            _ = ax4.xaxis.set_ticklabels('')
            # ax4.set_xlim(.5, 1.75)
            # ax4.set_ylim(bottom=7)

        return f

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


























