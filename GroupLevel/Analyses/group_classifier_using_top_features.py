from GroupLevel.group import Group
from scipy.stats import sem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GroupClassifier(Group):
    """
    Subclass of Group. Used to run subject_classify_using_top_features. Has methods for plotting.
    """

    def __init__(self, analysis='classify_enc_top_elecs', subject_settings='default', open_pool=False, n_jobs=100,
                 **kwargs):
        super(GroupClassifier, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                              open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to run the classifier for each subject.
        """
        super(GroupClassifier, self).process()

    def plot_average_by_num_features(self):
        """
        Plots auc as a function of number of electrodes (or frequencies) inlcuded, averaged across all subjects.
        """

        if self.subject_objs is None:
            print('Must run .process() first before plotting the results.')
            return

        # get the results from each subject and stack.
        N = 50 if self.subject_objs[0].do_top_elecs else len(self.subject_objs[0].freqs)
        aucs = np.stack([np.nanmean(x.res['aucs_by_n_feats'][:, :N], axis=0) for x in self.subject_objs], axis=0)
        err = sem(aucs, axis=0)

        with plt.style.context('myplotstyle.mplstyle'):
            x = np.arange(aucs.shape[1]) + 1
            y = aucs.mean(axis=0)
            plt.plot(x, y, linewidth=4)
            plt.fill_between(x, y - err, y + err, facecolor=[.5, .5, .5, .5], edgecolor=[.5, .5, .5, .5], zorder=0)
            plt.ylabel('AUC', fontsize=20)
            plt.xlabel('# Electrodes' if self.subject_objs[0].do_top_elecs else '# Frequencies', fontsize=20)
            plt.title('N: %d' % aucs.shape[0], fontsize=20)


