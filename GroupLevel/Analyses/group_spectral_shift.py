from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem
import pdb
import numpy as np
import pandas as pd
import matplotlib.gridspec
import matplotlib.pyplot as plt


class GroupSpectralShift(Group):
    """
    Subclass of Group. Used to run subject_spectral_shift.
    """

    def __init__(self, analysis='spectral_shift_enc', subject_settings='default', open_pool=False, n_jobs=100,
                 **kwargs):
        super(GroupSpectralShift, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                                 open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupSpectralShift, self).process()

    def plot_tstat_sme(self, region=None):
        """
        Plots mean t-statistics, across subjects, comparing remembered and not remembered items as a function of
        frequency. Very similar to the same function in group_SME, but modified to because the last two elements in
        res['ts'] here are slope and offset.
        """

        regions = self.subject_objs[0].res['regions']

        # Use all electrodes if region is None. Mean within each subject.
        if region is None:
            ts = np.stack([x.res['ts'].mean(axis=1) for x in self.subject_objs], axis=0)
            region = 'All'
        else:

            # otherwise, just pull out the electrodes in the ROI. NB: These have already been averaged within subject.
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return
            ts = np.stack([x.res['ts_region'][:, region_ind].flatten() for x in self.subject_objs], axis=0)

        # t, p = ttest_1samp(ts, 0, axis=0, nan_policy='omit')

        # stats on stats
        t_mean = np.nanmean(ts, axis=0)
        t_sem = sem(ts, axis=0, nan_policy='omit') * 1.96

        # as a function of frequency
        y_mean = t_mean[:-2]
        y_sem = t_sem[:-2]

        x = np.log10(self.subject_objs[0].freqs)

        # also compute the mean slope and offsets
        y_slope_off_mean = t_mean[-2:]
        y_slope_off_sem = t_sem[-2:]

        with plt.style.context('myplotstyle.mplstyle'):

            f = plt.figure()
            ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((2, 5), (0, 4), colspan=1)

            # plot with a shaded 95% CI
            ax1.plot(x, y_mean, '-k', linewidth=4, zorder=6)
            ax1.fill_between(x, y_mean - y_sem, y_mean + y_sem, facecolor=[.5, .5, .5, .5], edgecolor=[.5, .5, .5, .5], zorder=5)
            ax1.plot([x[0], x[-1]], [0, 0], '-k', linewidth=2)

            # relabel x-axis to be powers of two
            new_x = self.compute_pow_two_series()
            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.xaxis.set_ticklabels(new_x, rotation=0)
            ax1.set_ylim(-1, 1)

            ax1.set_xlabel('Frequency', fontsize=24)
            ax1.set_ylabel('Average t-stat', fontsize=24)
            ax1.set_title('%s SME, N=%d' % (region, np.sum(~np.isnan(ts), axis=0)[0]))

            # also plot bar for average slope and offset t-stat
            ax2.bar([.75, 1.25], y_slope_off_mean, .35, alpha=1, yerr=y_slope_off_sem, zorder=4, color=[.5, .5, .5, .5],
                    edgecolor='k',
                    error_kw={'zorder': 10, 'ecolor': 'k'})
            ax2.xaxis.set_ticks([.75 + .175, 1.25 + .175])
            ax2.plot(ax2.get_xlim(), [0, 0], '--k', lw=2, zorder=3)
            ax2.xaxis.set_ticklabels(['Slopes', 'Offets'], rotation=-90)
            ax2.set_ylim(-1, 1)
            ax2.set_xlim(.6, 1.4 + .35)
            ax2.set_yticklabels('')

    def plot_count_sme(self, region=None):
        """
        Plot proportion of electrodes that are signifcant at a given frequency across all electrodes in the entire
        dataset, seperately for singificantly negative and sig. positive.

        In additional to plotting at each frequency, also plots slopes and offsets.
        """

        regions = self.subject_objs[0].res['regions']
        if region is None:

            # get the counts for the frquency features
            sme_pos_freq = np.stack([np.sum((x.res['ts'][:-2] > 0) & (x.res['ps'][:-2] < .05), axis=1) for x in self.subject_objs],
                               axis=0)
            sme_neg_freq = np.stack([np.sum((x.res['ts'][:-2] < 0) & (x.res['ps'][:-2] < .05), axis=1) for x in self.subject_objs],
                               axis=0)

            # also get the counts for the slope and offset features
            sme_pos_slope_offset = np.stack(
                [np.sum((x.res['ts'][-2:] > 0) & (x.res['ps'][-2:] < .05), axis=1) for x in self.subject_objs],
                axis=0)

            sme_neg_slope_offset = np.stack(
                [np.sum((x.res['ts'][-2:] < 0) & (x.res['ps'][-2:] < .05), axis=1) for x in self.subject_objs],
                axis=0)

            n = np.stack([x.res['ts'].shape[1] for x in self.subject_objs], axis=0)
            region = 'All'
        else:
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return

            sme_pos_freq = np.stack([x.res['sme_count_pos'][:-2, region_ind].flatten() for x in self.subject_objs], axis=0)
            sme_neg_freq = np.stack([x.res['sme_count_neg'][:-2, region_ind].flatten() for x in self.subject_objs], axis=0)
            sme_pos_slope_offset = np.stack([x.res['sme_count_pos'][-2:, region_ind].flatten() for x in self.subject_objs], axis=0)
            sme_neg_slope_offset = np.stack([x.res['sme_count_neg'][-2:, region_ind].flatten() for x in self.subject_objs], axis=0)
            n = np.stack([x.res['elec_n'][region_ind].flatten() for x in self.subject_objs], axis=0)

        n = float(n.sum())
        x = np.log10(self.subject_objs[0].freqs)
        x_label = np.round(self.subject_objs[0].freqs * 10) / 10
        with plt.style.context('myplotstyle.mplstyle'):
            f = plt.figure()
            ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((2, 5), (0, 4), colspan=1)

            ax1.plot(x, sme_pos_freq.sum(axis=0) / n * 100, linewidth=4, c='#8c564b', label='Good Memory', zorder=4)
            ax1.plot(x, sme_neg_freq.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4', label='Bad Memory', zorder=5)
            l = ax1.legend(loc=0)

            new_x = self.compute_pow_two_series()
            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [5, 5], '--k', lw=2, zorder=3)
            ax1.xaxis.set_ticklabels(new_x, rotation=0)
            ax1.set_xlabel('Frequency', fontsize=24)
            ax1.set_ylabel('Percent Sig. Electrodes', fontsize=24)
            ax1.set_title('%s: %d electrodes' % (region, int(n)))

            ax2.bar([.15, 1.35], sme_pos_slope_offset.sum(axis=0) / n * 100, .5, alpha=1, zorder=4, color=np.array([140, 86, 75])/255.,
                    edgecolor='k', align='center',
                    error_kw={'zorder': 10, 'ecolor': 'k'})
            ax2.bar([.65, 1.85], sme_neg_slope_offset.sum(axis=0) / n * 100, .5, alpha=1, zorder=4, color=np.array([31, 119, 180])/255.,
                    edgecolor='k', align='center',
                    error_kw={'zorder': 10, 'ecolor': 'k'})

            ax2.xaxis.set_ticks([.4, 1.6])
            ax2.plot(ax2.get_xlim(), [5, 5], '--k', lw=2, zorder=3)
            max_lim = np.max([np.max(ax1.get_ylim()), np.max(ax2.get_ylim())])
            ax1.set_ylim(0, max_lim)
            ax2.set_ylim(0, max_lim)
            _ = ax2.set_yticklabels('')
            _ = ax2.set_xticklabels(['Slopes', 'Offsets'], rotation=-90)

