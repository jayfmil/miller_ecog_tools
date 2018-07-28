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

    def __init__(self, analysis='classify_enc_triangle', subject_settings='triangle', open_pool=False, n_jobs=50, **kwargs):
        super(GroupClassifier, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                              open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to run the classifier for each subject. Then make a summary dataframe based on the results.
        """
        super(GroupClassifier, self).process()

