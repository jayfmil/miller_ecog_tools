from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby


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

        # After processing the subjects, this will be a dataframe of summary data
        # self.summary_table = None

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupSME, self).process()

        # also make a summary table
        # data = np.array([[x.res['auc'], x.res['loso'], x.skew] for x in self.subject_objs])
        # subjs = [x.subj for x in self.subject_objs]
        # self.summary_table = pd.DataFrame(data=data, index=subjs, columns=['AUC', 'LOSO', 'Skew'])

    # def find_contiguous_sig_freqs(self):
    #
    #
        # http://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
        # ranges = []
        # for k, g in groupby(enumerate(data), lambda (i, x): i - x):
        #     group = map(itemgetter(1), g)
        #     if len(group) > 1:
        #         ranges.append((group[0], group[-1]))

    def plot_sme_map(self):
        pass

    def plot_count_sme(self, region):
        """

        """

        regions = self.subject_objs[0].res['regions']
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
            plt.plot(x, sme_pos.sum(axis=0) / n, linewidth=4, c='#8c564b', label='Good Memory')
            plt.plot(x, sme_neg.sum(axis=0) / n, linewidth=4, c='#1f77b4', label='Bad Memory')
            l = plt.legend()
            plt.xticks(x[::3], x_label[::3], rotation=-45)
            plt.xlabel('Frequency', fontsize=24)
            plt.ylabel('Percent Sig. Electrodes', fontsize=24)
            plt.title('%s: %d electrodes' % (region, int(n)))

