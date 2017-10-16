from GroupLevel.group import Group
from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


class GroupClassifier(Group):
    """
    Subclass of Group. Used to run subject_classify for a specific experiment and classification settings.
    """

    def __init__(self, analysis='classify_inner_cv', subject_settings='THR1_inner_cv', open_pool=False, n_jobs=50, **kwargs):
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
        data = np.array([[x.res['auc'], x.res['loso']] for x in self.subject_objs])
        subjs = [x.subj for x in self.subject_objs]
        self.summary_table = pd.DataFrame(data=data, index=subjs, columns=['AUC', 'LOSO'])