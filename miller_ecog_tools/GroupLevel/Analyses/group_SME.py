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
            ax.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [2.5, 2.5], '--k', lw=2, zorder=3)
            ax.xaxis.set_ticklabels(new_x, rotation=0)

            plt.xlabel('Frequency', fontsize=24)
            plt.ylabel('Percent Sig. Electrodes', fontsize=24)
            plt.title('%s: %d electrodes' % (region, int(n)))

    def plot_feature_map(self, do_outline=True, alpha=.6, plot_res_key='ts',
                         stat_res_key='ts', clim=None, hemi='both', cb_label='mean(t-stat)'):
        """
        Makes a heatmap style plot of average SME tstats as a function of brain region.
        """

        # stack all the subject means
        regions = np.array(['IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC'])
        plot_region_mean = []
        stat_region_mean = []
        for region in regions:
            if hemi == 'both':
                plot_region_mean.append(np.stack(
                    [np.nanmean(x.res[plot_res_key][:, x.elec_locs[region]], axis=1) for x in self.subject_objs],
                    axis=0))
                stat_region_mean.append(np.stack(
                    [np.nanmean(x.res[stat_res_key][:, x.elec_locs[region]], axis=1) for x in self.subject_objs],
                    axis=0))
            elif hemi == 'l':
                plot_region_mean.append(np.stack(
                    [np.nanmean(x.res[plot_res_key][:, (x.elec_locs[region]) & (~x.elec_locs['is_right'])], axis=1) for
                     x in self.subject_objs], axis=0))
                stat_region_mean.append(np.stack(
                    [np.nanmean(x.res[stat_res_key][:, (x.elec_locs[region]) & (~x.elec_locs['is_right'])], axis=1) for
                     x in self.subject_objs], axis=0))
            elif hemi == 'r':
                plot_region_mean.append(np.stack(
                    [np.nanmean(x.res[plot_res_key][:, (x.elec_locs[region]) & (x.elec_locs['is_right'])], axis=1) for x
                     in self.subject_objs], axis=0))
                stat_region_mean.append(np.stack(
                    [np.nanmean(x.res[stat_res_key][:, (x.elec_locs[region]) & (x.elec_locs['is_right'])], axis=1) for x
                     in self.subject_objs], axis=0))

        plot_region_mean = np.stack(plot_region_mean, -1)
        stat_region_mean = np.stack(stat_region_mean, -1)

        # mean across subjects, that is what we will plot
        plot_data = np.nanmean(plot_region_mean, axis=0)
        if clim is None:
            clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))

        # also create a mask of significant region/frequency bins
        t, p = ttest_1samp(stat_region_mean, 0, axis=0, nan_policy='omit')
        p2 = np.ma.masked_where(p < .05, p)

        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label=cb_label, size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(regions)), regions, fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.subject_objs[0].freqs),
                              range(len(self.subject_objs[0].freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)

            # overlay mask
            if do_outline:
                self.plot_mask_outline(p2)
            plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=alpha)
            plt.gca().invert_yaxis()
            plt.grid()
        return plot_region_mean, fig, ax, cb

    def plot_mask_outline(self, p_mat, lw=3):
        """
        Plot outline of significant regions.
        Credit: http://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
        """
        mapimg = ~p_mat.mask

        # a vertical line segment is needed, when the pixels next to each other horizontally
        #   belong to diffferent groups (one is part of the mask, the other isn't)
        # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
        ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

        # the same is repeated for horizontal segments
        hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

        # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
        #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
        # in order to draw a discountinuous line, we add Nones in between segments
        l = []
        for p in zip(*hor_seg):
            l.append((p[1], p[0] + 1))
            l.append((p[1] + 1, p[0] + 1))
            l.append((np.nan, np.nan))

        # and the same for vertical segments
        for p in zip(*ver_seg):
            l.append((p[1] + 1, p[0]))
            l.append((p[1] + 1, p[0] + 1))
            l.append((np.nan, np.nan))

        # now we transform the list into a numpy array of Nx2 shape
        segments = np.array(l)
        segments[:, 0] = p_mat.shape[1] * segments[:, 0] / mapimg.shape[1] - .5
        segments[:, 1] = p_mat.shape[0] * segments[:, 1] / mapimg.shape[0] - .5

        # and now there isn't anything else to do than plot it
        plt.plot(segments[:, 0], segments[:, 1], color='k', linewidth=lw)
