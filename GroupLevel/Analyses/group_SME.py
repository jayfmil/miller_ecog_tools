from GroupLevel.group import Group
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

    def plot_terciles(self):
        pass

    def plot_sme_map(self):
        pass

    def plot_count_sme(self, region):
        """

        """

        regions = self.subject_objs.res['regions']
        region_ind = regions == region
        if ~np.any(region_ind):
            print('Invalid region, please use: %s.' % ', '.join(regions))
            return

        x = np.log10(self.subject_objs[0].freqs)
        x_label = np.round(self.subject_objs[0].freqs * 10)/10
        # with plt.style.context('../../myplotstyle.mplstyle'):


