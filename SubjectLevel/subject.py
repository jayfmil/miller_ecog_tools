import numpy as np
import matplotlib.pyplot as plt
import ram_data_helpers
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind


class Subject(object):
    """
    Base class upon which data and Analyses are built. Defines subject name and experiment, and forces them to be valid.
    """
    valid_tasks = ['RAM_TH1', 'RAM_TH3', 'RAM_YC1', 'RAM_YC2', 'RAM_FR1', 'RAM_FR2', 'RAM_FR3']

    def __init__(self, task=None, subject=None):

        # these are checked to be valid tasks and subjects
        self.task = task
        self.subj = subject

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, t):
        if t in self.valid_tasks:
            self._task = t
        else:
            self._task = None
            print('Invalid task, must be one of %s.' % ', '.join(self.valid_tasks))

    @property
    def subj(self):
        return self._subj

    @subj.setter
    def subj(self, s):
        if self.task is not None:
            valid_subjs = ram_data_helpers.get_subjs(self.task)
            if s in valid_subjs:
                self._subj = s
            else:
                self._subj = None
                print('Invalid subject for %s, must be one of %s.' % (self.task, ', '.join(valid_subjs)))
        else:
            print('Must set valid task.')
            self._subj = None





















