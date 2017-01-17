from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem

import pdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class GroupSME(Group):
    """
    Subclass of Group. Used to run subject_SME.
    """

    def __init__(self, analysis='sme_enc', subject_settings='default', open_pool=False, n_jobs=100, **kwargs):
        super(GroupSME, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                       open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupSME, self).process()

    def plot_sme_map(self):
        pass

    def plot_tstat_sme(self, region=None):
        """
        Plots mean t-statistics, across subjects, comparing remembered and not remembered items as a function of
        frequency.
        """

        regions = self.subject_objs[0].res['regions']
        if region is None:
            ts = np.stack([x.res['ts'].mean(axis=1) for x in self.subject_objs], axis=0)
            region = 'All'
        else:
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return
            ts = np.stack([x.res['ts_region'][:, region_ind].flatten() for x in self.subject_objs], axis=0)

        t, p = ttest_1samp(ts, 0, axis=0, nan_policy='omit')

        y_mean = np.nanmean(ts, axis=0)
        y_sem = sem(ts, axis=0, nan_policy='omit') * 1.96

        x = np.log10(self.subject_objs[0].freqs)
        x_label = np.round(self.subject_objs[0].freqs * 10) / 10

        with plt.style.context('myplotstyle.mplstyle'):

            # fig, ax = plt.subplots()
            fig = plt.figure()
            ax = plt.subplot2grid((2, 5), (0, 0), colspan=5)
            ax.plot(x, y_mean, '-k', linewidth=4, zorder=6)
            ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, facecolor=[.5, .5, .5, .5], edgecolor=[.5, .5, .5, .5], zorder=5)
            ax.plot([x[0], x[-1]], [0, 0], '-k', linewidth=2)

            new_x = self.compute_pow_two_series()
            ax.xaxis.set_ticks(np.log10(new_x))
            ax.xaxis.set_ticklabels(new_x, rotation=0)
            plt.ylim(-1, 1)

            ax.set_xlabel('Frequency', fontsize=24)
            ax.set_ylabel('Average t-stat', fontsize=24)

            # ax.fill_between(x, [0]*50, y_mean, where=(p<.05)&(t>0),facecolor='#8c564b', edgecolor='#8c564b')
            # ax.fill_between(x, [0]*50, y_mean, where=(p<.05)&(t<0),facecolor='#1f77b4', edgecolor='#1f77b4')

            plt.title('%s SME, N=%d' % (region, np.sum(~np.isnan(ts), axis=0)[0]))

    def plot_count_sme(self, region=None):
        """
        Plot proportion of electrodes that are signifcant at a given frequency across all electrodes in the entire
        dataset, seperately for singificantly negative and sig. positive.
        """

        regions = self.subject_objs[0].res['regions']
        if region is None:
            sme_pos = np.stack([np.sum((x.res['ts'] > 0) & (x.res['ps'] < .05), axis=1) for x in self.subject_objs],
                               axis=0)
            sme_neg = np.stack([np.sum((x.res['ts'] < 0) & (x.res['ps'] < .05), axis=1) for x in self.subject_objs],
                               axis=0)
            n = np.stack([x.res['ts'].shape[1] for x in self.subject_objs], axis=0)
            region = 'All'
        else:
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return

            sme_pos = np.stack([x.res['sme_count_pos'][:, region_ind].flatten() for x in self.subject_objs], axis=0)
            sme_neg = np.stack([x.res['sme_count_neg'][:, region_ind].flatten() for x in self.subject_objs], axis=0)
            n = np.stack([x.res['elec_n'][region_ind].flatten() for x in self.subject_objs], axis=0)

        n = float(n.sum())
        x = np.log10(self.subject_objs[0].freqs)
        x_label = np.round(self.subject_objs[0].freqs * 10) / 10
        with plt.style.context('myplotstyle.mplstyle'):

            fig = plt.figure()
            ax = plt.subplot2grid((2, 5), (0, 0), colspan=5)
            plt.plot(x, sme_pos.sum(axis=0) / n * 100, linewidth=4, c='#8c564b', label='Good Memory')
            plt.plot(x, sme_neg.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4', label='Bad Memory')
            l = plt.legend()

            new_x = self.compute_pow_two_series()
            ax.xaxis.set_ticks(np.log10(new_x))
            ax.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [5, 5], '--k', lw=2, zorder=3)
            ax.xaxis.set_ticklabels(new_x, rotation=0)

            plt.xlabel('Frequency', fontsize=24)
            plt.ylabel('Percent Sig. Electrodes', fontsize=24)
            plt.title('%s: %d electrodes' % (region, int(n)))

    def plot_feature_map(self):
        """
        Makes a heatmap style plot of average SME tstats as a function of brain region.
        """

        # stack all the subject means
        region_mean = np.stack([x.res['ts_region'] for x in self.subject_objs], axis=0)

        # reorder to group the regions in a way that visually makes more sense
        regions = np.array(['IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC'])
        key_order = self.subject_objs[0].res['regions']
        new_order = np.searchsorted(key_order, np.array(regions))
        region_mean = region_mean[:, :, new_order]

        # mean across subjects, that is what we will plot
        plot_data = np.nanmean(region_mean, axis=0)
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))

        # also create a mask of significant region/frequency bins
        t, p = ttest_1samp(region_mean, 0, axis=0, nan_policy='omit')
        p2 = np.ma.masked_where(p < .05, p)

        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='mean(t-stat)', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(regions)), regions, fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.subject_objs[0].freqs),
                              range(len(self.subject_objs[0].freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)

            # overlay mask
            plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=.6)
            plt.gca().invert_yaxis()
            plt.grid()
