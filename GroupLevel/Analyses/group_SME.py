from GroupLevel.group import Group
import numpy as np
import pandas as pd


class GroupSME(Group):
    """
    Subclass of Group. Used to run subject_classify for a specific experiment and classification settings.
    """

    def __init__(self, analysis='sme_enc', subject_settings='default', open_pool=False, n_jobs=100, **kwargs):
        super(GroupSME, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                       open_pool=open_pool, n_jobs=n_jobs, **kwargs)

        # After processing the subjects, this will be a dataframe of summary data
        self.summary_table = None

    def process(self):
        """
        Call Group.process() to run the classifier for each subject. Then make a summary dataframe based on the results.
        """
        super(GroupSME, self).process()

        # also make a summary table
        # data = np.array([[x.res['auc'], x.res['loso'], x.skew] for x in self.subject_objs])
        # subjs = [x.subj for x in self.subject_objs]
        # self.summary_table = pd.DataFrame(data=data, index=subjs, columns=['AUC', 'LOSO', 'Skew'])

    def plot_terciles(self):
        pass

    def plot_feature_map(self):
        pass

