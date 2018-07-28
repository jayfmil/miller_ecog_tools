from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem, binom_test, binom
import pdb
import numpy as np
import pandas as pd
import matplotlib.gridspec
import matplotlib.pyplot as plt
from copy import deepcopy


class GroupSpectralShift(Group):
    """
    Subclass of Group. Used to run subject_spectral_shift.
    """

    def __init__(self, analysis='spectral_shift_enc', subject_settings='default', open_pool=False, n_jobs=50,
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
            ax2.xaxis.set_ticklabels(['Slopes', 'BB Pow'], rotation=-90)
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
        p_corr = 0.05 / (len(x) + 2)

        with plt.style.context('myplotstyle.mplstyle'):
            f = plt.figure()
            ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((2, 5), (0, 4), colspan=1)

            ax1.plot(x, sme_pos_freq.sum(axis=0) / n * 100, linewidth=4, c='#8c564b', label='Good Memory', zorder=4)
            p_pos = np.array(map(lambda x: binom_test(x, n, .025), sme_pos_freq.sum(axis=0)))
            # ax1.plot(x[p_pos < p_corr], (sme_pos_freq.sum(axis=0) / n * 100)[p_pos < p_corr], 'o', c='#8c564b', lw=0,
            #          markersize=10)

            ax1.plot(x, sme_neg_freq.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4', label='Bad Memory', zorder=5)
            p_neg = np.array(map(lambda x: binom_test(x, n, .025), sme_neg_freq.sum(axis=0)))
            # ax1.plot(x[p_neg < p_corr], (sme_neg_freq.sum(axis=0) / n * 100)[p_neg < p_corr], 'o', c='#1f77b4', lw=0,
            #          markersize=10)

            crit_perc = np.where(1 - binom.cdf(np.arange(n), n, .025) < p_corr)[0][0] / n * 100.
            # ax1.plot(x, [crit_perc]*len(x), '--k', linewidth=2, zorder=3)

            l = ax1.legend(loc=0)

            new_x = self.compute_pow_two_series()
            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [2.5, 2.5], '--k', lw=2, zorder=3)
            # ax1.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [crit_perc, crit_perc], '--k', lw=2, zorder=3)
            ax1.xaxis.set_ticklabels(new_x, rotation=0)
            ax1.set_xlabel('Frequency', fontsize=24)
            ax1.set_ylabel('Percent Sig. Electrodes', fontsize=24)
            ax1.set_title('%s: %d electrodes' % (region, int(n)))

            ax2.bar([.15, 1.35], sme_pos_slope_offset.sum(axis=0) / n * 100, .5, alpha=1, zorder=4, color=np.array([140, 86, 75])/255.,
                    edgecolor='k', align='center',
                    error_kw={'zorder': 10, 'ecolor': 'k'})
            p_pos_slopes = np.array(map(lambda x: binom_test(x, n, .025), sme_pos_slope_offset.sum(axis=0)))

            ax2.bar([.65, 1.85], sme_neg_slope_offset.sum(axis=0) / n * 100, .5, alpha=1, zorder=4, color=np.array([31, 119, 180])/255.,
                    edgecolor='k', align='center',
                    error_kw={'zorder': 10, 'ecolor': 'k'})
            p_neg_slopes = np.array(map(lambda x: binom_test(x, n, .025), sme_neg_slope_offset.sum(axis=0)))

            ax2.xaxis.set_ticks([.4, 1.6])
            # ax2.plot(ax2.get_xlim(), [crit_perc, crit_perc], '--k', lw=2, zorder=3)
            ax2.plot(ax2.get_xlim(), [2.5, 2.5], '--k', lw=2, zorder=3)
            max_lim = np.max([np.max(ax1.get_ylim()), np.max(ax2.get_ylim())])

            ax1.set_ylim(0, max_lim)
            ax2.set_ylim(0, max_lim)
            _ = ax2.set_yticklabels('')
            _ = ax2.set_xticklabels(['Slopes', 'BB Pow'], rotation=-90)

    def plot_feature_map(self, do_overlay=True, alpha=.6):
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
            if do_overlay:
                plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=alpha)
            plt.gca().invert_yaxis()
            plt.grid()
        return fig, ax, cb

    def plot_peak_sme(self, region=None):
        """
        Percent of electrodes showing a significant difference between recalled and not recalled
        items in the robust fit residuals. Separate lines for electrodes with peaks at a given
        frequency and without peaks.
        """

        # to hold analysis at peaks.
        peaks = []
        peaks_sig = []

        # to hold analysis at non-peaks
        npeaks = []
        npeaks_sigs = []

        for subj_obj in self.subject_objs:

            # boolean of freq x elec indicating peaks
            if region is None:
                elec_inds = np.ones(subj_obj.res['peak_freqs'].shape[1], dtype=bool)
            else:
                elec_inds = subj_obj.elec_locs[region]
            peaks_subj = subj_obj.res['peak_freqs'][:, elec_inds]

            # copy p-vals, nan out non-peak features
            ps_subj = deepcopy(subj_obj.res['ps'][:-2, elec_inds])
            ps_subj[~peaks_subj] = np.nan

            # number of peaks vs freq, number of sig vs freq
            peak_counts = np.sum(~np.isnan(ps_subj), axis=1)
            sig_counts = np.sum(ps_subj < .05, axis=1)
            peaks.append(peak_counts)
            peaks_sig.append(sig_counts)

            # repeat, but now nan out peaks
            ps_subj = deepcopy(subj_obj.res['ps'][:-2, elec_inds])
            ps_subj[peaks_subj] = np.nan
            peak_counts = np.sum(~np.isnan(ps_subj), axis=1)
            sig_counts = np.sum(ps_subj < .05, axis=1)
            npeaks.append(peak_counts)
            npeaks_sigs.append(sig_counts)

        # freqs by subjs
        peaks = np.stack(peaks, axis=-1)
        peaks_sig = np.stack(peaks_sig, axis=-1)
        npeaks = np.stack(npeaks, axis=-1)
        npeaks_sigs = np.stack(npeaks_sigs, axis=-1)

        # set up plot axes
        with plt.style.context('myplotstyle.mplstyle'):
            f = plt.figure()
            ax2 = plt.subplot2grid((10, 1), (0, 0), rowspan=1)
            ax1 = plt.subplot2grid((10, 1), (2, 0), rowspan=8)

            new_x = self.compute_pow_two_series()
            ax1.set_xlim(np.log10(self.subject_objs[0].freqs)[0], np.log10(self.subject_objs[0].freqs)[1])
            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.xaxis.set_ticklabels(new_x, rotation=0)

            ax2.set_xlim(np.log10(self.subject_objs[0].freqs)[0], np.log10(self.subject_objs[0].freqs)[1])
            ax2.xaxis.set_ticks(np.log10(new_x))
            ax2.xaxis.set_ticklabels('', rotation=0)
            ax2.yaxis.set_ticks([0, peaks.sum(axis=1).max()])
            ax2.plot(np.log10(self.subject_objs[0].freqs), peaks.sum(axis=1))

            # plot sig peak
            y = np.divide(peaks_sig.sum(axis=1).astype(float), peaks.sum(axis=1)) * 100
            plt.plot(np.log10(self.subject_objs[0].freqs), y, linewidth=4, label='Peak')
            plt.scatter(np.log10(self.subject_objs[0].freqs), y)

            # plot sig non peak
            y = np.divide(npeaks_sigs.sum(axis=1).astype(float), npeaks.sum(axis=1)) * 100
            plt.plot(np.log10(self.subject_objs[0].freqs), y, linewidth=4, label='No Peak')
            plt.legend()

            plt.xlabel('Frequency (Hz)', fontsize=20)
            plt.ylabel('Percent', fontsize=20)


