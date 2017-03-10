from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem
import pdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class GroupSMETimebins(Group):
    """
    Subclass of Group. Used to run subject_SME_timebins.
    """

    def __init__(self, analysis='sme_enc_timebins', subject_settings='default_50_freqs_timebins', open_pool=False,
                 n_jobs=50, **kwargs):
        super(GroupSMETimebins, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                               open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupSMETimebins, self).process()

    def plot_feature_map(self, do_overlay=True, alpha=.6, region=None, clim=None, res_key='ts'):
        """
        Makes a heatmap style plot of average SME tstats as a function of brain region.
        """

        regions = self.subject_objs[0].res['regions']
        if region is None:
            print('Please enter one of: %s.' % ', '.join(regions))
            return
        else:
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return
            region_mean = np.squeeze(
                np.stack([x.res[res_key][:, region_ind, :] for x in self.subject_objs], axis=0))
            region_mean = np.stack([np.nanmean(x.res[res_key][:, x.elec_locs[region]], axis=1) for x in self.subject_objs], axis=0)

            # mean across subjects, that is what we will plot
        plot_data = np.nanmean(region_mean, axis=0)
        if clim is None:
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

            plt.xticks(range(len(self.subject_objs[0].res['time_bins']))[::3],
                       self.subject_objs[0].res['time_bins'][::3], fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.subject_objs[0].freqs),
                              range(len(self.subject_objs[0].freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)

            # overlay mask
            if do_overlay:
                plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=alpha)
            plt.gca().invert_yaxis()
            plt.grid()

            # plot zero line
            t0 = np.interp(0, self.subject_objs[0].res['time_bins'],
                           range(self.subject_objs[0].res['time_bins'].shape[0]))
            plt.plot([t0, t0], [-.5, len(self.subject_objs[0].freqs)-.5], '--k', lw=4)

        return fig, ax, cb