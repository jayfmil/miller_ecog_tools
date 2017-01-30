from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem

import pdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class GroupFindSpectralPeaks(Group):
    """
    Subclass of Group. Used to run subject_SME.
    """

    def __init__(self, analysis='find_peaks_enc', subject_settings='default_50_freqs', open_pool=False, n_jobs=100,
                 **kwargs):
        super(GroupFindSpectralPeaks, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                                     open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupFindSpectralPeaks, self).process()

    def plot_peak_count(self, region=None):
        """
        Plot proportion of electrodes that are signifcant at a given frequency across all electrodes in the entire
        dataset, seperately for singificantly negative and sig. positive.
        """

        regions = self.subject_objs[0].res['regions']
        if region is None:
            peaks = np.stack([np.sum((x.res['peak_freqs']), axis=1) for x in self.subject_objs],
                             axis=0)
            n = np.stack([x.res['peak_freqs'].shape[1] for x in self.subject_objs], axis=0)
            region = 'All'
        else:
            region_ind = regions == region
            if ~np.any(region_ind):
                print('Invalid region, please use: %s.' % ', '.join(regions))
                return

            peaks = np.stack([x.res['peaks_count_region'][:, region_ind].flatten() for x in self.subject_objs], axis=0)
            n = np.stack([x.res['elec_n'][region_ind].flatten() for x in self.subject_objs], axis=0)

        n = float(n.sum())
        x = np.log10(self.subject_objs[0].freqs)
        x_label = np.round(self.subject_objs[0].freqs * 10) / 10
        with plt.style.context('myplotstyle.mplstyle'):

            fig = plt.figure()
            ax = plt.subplot2grid((2, 5), (0, 0), colspan=5)
            plt.plot(x, peaks.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4')
            #         plt.plot(x, sme_neg.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4', label='Bad Memory')
            l = plt.legend()

            new_x = self.compute_pow_two_series()
            ax.xaxis.set_ticks(np.log10(new_x))
            #         ax.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [5, 5], '--k', lw=2, zorder=3)
            ax.xaxis.set_ticklabels(new_x, rotation=0)

            plt.xlabel('Frequency', fontsize=24)
            plt.ylabel('Elecs w/Peaks (%)', fontsize=24)
            plt.title('%s: %d electrodes' % (region, int(n)))
