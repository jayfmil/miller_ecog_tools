from GroupLevel.group import Group
from operator import itemgetter
from itertools import groupby
from scipy.stats import ttest_1samp, sem
import pdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class GroupElecCluster(Group):
    """
    Subclass of Group. Used to run subject_SME.
    """

    def __init__(self, analysis='traveling', subject_settings='traveling_FR1', open_pool=False, n_jobs=25, **kwargs):
        super(GroupElecCluster, self).__init__(analysis=analysis, subject_settings=subject_settings,
                                               open_pool=open_pool, n_jobs=n_jobs, **kwargs)

    def process(self):
        """
        Call Group.process() to compute the subsequent memory effect for each subject.
        """
        super(GroupElecCluster, self).process()
