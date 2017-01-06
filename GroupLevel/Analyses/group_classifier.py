from GroupLevel.group import Group
from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GroupClassifier(Group):
    """
    Subclass of Group. Used to run subject_classify for a specific experiment and classification settings.
    """

    def __init__(self, analysis='classify_enc', subject_settings='default', open_pool=False, n_jobs=100, **kwargs):
        super(GroupClassifier, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                              open_pool=open_pool, n_jobs=n_jobs, **kwargs)

        # After processing the subjects, this will be a dataframe of summary data
        self.summary_table = None

    def process(self):
        """
        Call Group.process() to run the classifier for each subject. Then make a summary dataframe based on the results.
        """
        super(GroupClassifier, self).process()

        # also make a summary table
        data = np.array([[x.res['auc'], x.res['loso'], x.skew] for x in self.subject_objs])
        subjs = [x.subj for x in self.subject_objs]
        self.summary_table = pd.DataFrame(data=data, index=subjs, columns=['AUC', 'LOSO', 'Skew'])

    def plot_terciles(self):
        pass

    def plot_feature_map(self):
        pass

    def plot_auc_hist(self):
        """
        Plot histogram of AUC values.
        """
        with plt.style.context('myplotstyle.mplstyle'):
            self.summary_table.hist(column='AUC', bins=20, zorder=5)
            plt.xlim(.2, .8)
            plt.ylabel('Count', fontsize=24)
            plt.xlabel('AUC', fontsize=24)
            plt.plot([.5, .5], [plt.ylim()[0], plt.ylim()[1] + 1], '--k')

            t, p = ttest_1samp(self.summary_table['AUC'], .5)
            _ = plt.title(r'Mean AUC: %.3f, $t(%d) = %.2f, p < 10^{%s}$' % (self.summary_table['AUC'].mean(),
                                                                            self.summary_table.shape[0],
                                                                            t, np.ceil(np.log10(p))))
