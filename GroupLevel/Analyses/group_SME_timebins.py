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

    def plot_feature_map(self, do_outline=True, alpha=.6, region=None, clim=None, res_key='ts'):
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
            # region_mean = np.squeeze(
            #     np.stack([x.res[res_key][:, region_ind, :] for x in self.subject_objs], axis=0))
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

            stop = np.floor(self.subject_objs[0].res['time_bins'][-1] * 100 / 25) * .25
            start = np.ceil(self.subject_objs[0].res['time_bins'][0] * 100 / 25) * .25
            x_inds = np.arange(start, stop + .1, .25)
            new_x = np.interp(x_inds, self.subject_objs[0].res['time_bins'],
                              range(self.subject_objs[0].res['time_bins'].shape[0]))
            _ = plt.xticks(new_x, np.arange(start, stop + .1, .25), fontsize=20, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.subject_objs[0].freqs),
                              range(len(self.subject_objs[0].freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)

            # overlay mask
            if do_outline:
                self.plot_mask_outline(p2)
            plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=alpha)
            plt.gca().invert_yaxis()
            plt.grid()

            # plot zero line
            t0 = new_x[x_inds == 0]
            plt.plot([t0, t0], [-.5, len(self.subject_objs[0].freqs)-.5], '--k', lw=3)

        return fig, ax, cb

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
