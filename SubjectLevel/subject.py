import numpy as np
import matplotlib.pyplot as plt
import ram_data_helpers
from subject_classifier import SubjectClassifier
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind


class Subject(SubjectClassifier):
    """
    Subject class that inherits from SubjectClassifier. Methods for visualizing a subject's data and classifier results.

    This is kind of weird. think about organization
    """
    valid_tasks = ['RAM_TH1', 'RAM_TH3', 'RAM_YC1', 'RAM_YC2', 'RAM_FR1', 'RAM_FR2', 'RAM_FR3']

    def __init__(self, task, subject):
        super(Subject, self).__init__()

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
            print 'Invalid task, must be one of %s.' % ', '.join(self.valid_tasks)

    @property
    def subj(self):
        return self._subj

    @subj.setter
    def subj(self, s):
        if self.task is not None:
            valid_subjs = ram_data_helpers.get_subjs(self.task)
            if s in valid_subjs:
                self._subj = s
            elif s is not None:
                print s
                self._subj = None
                print 'Invalid subject for %s, must be one of %s.' % (self.task, ', '.join(valid_subjs))
        else:
            print 'Must set valid task.'

    def plot_classifier_terciles(self):
        """
        Plot change in subject recall rate as a function of three bins of classifier probaility outputs.
        """
        if not self.class_res:
            print('Classifier data must be loaded or computed.')
            return

        tercile_delta_rec = self.compute_terciles()
        plt.bar(range(3), tercile_delta_rec, align='center', color=[.5, .5, .5], linewidth=2)

    def compute_terciles(self):
        """
        Compute change in subject recall rate as a function of three bins of classifier probability outputs.
        """
        if not self.class_res:
            print('Classifier data must be loaded or computed.')
            return

        binned_data = binned_statistic(self.class_res['probs'], self.class_res['Y'], statistic='mean',
                                       bins=np.percentile(self.class_res['probs'], [0, 33, 67, 100]))
        tercile_delta_rec = (binned_data[0] - np.mean(self.class_res['Y'])) / np.mean(self.class_res['Y']) * 100
        return tercile_delta_rec




    def compute_forward_model(self):
        """

        """
        if not self.class_res and not self.subject_data:
            print('Both classifier data and subject data must be loaded to compute forward model.')
            return

        # reshape data to events x number of features
        X = self.subject_data.data.reshape(self.subject_data.shape[0], -1)

        # normalize data by session if the features are oscillatory power
        if self.feat_type == 'power':
            X = self.normalize_power(X)

        probs_log = np.log(self.class_res['probs'] / (1 - self.class_res['probs']))
        covx = np.cov(X.T)
        covs = np.cov(probs_log)
        W = self.class_res['model'].coef_
        A = np.dot(covx, W.T) / covs
        return A
            # ts, ps = ttest_ind(feat_mat[recalls], feat_mat[~recalls])






















