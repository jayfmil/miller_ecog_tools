from GroupLevel.group import Group
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

