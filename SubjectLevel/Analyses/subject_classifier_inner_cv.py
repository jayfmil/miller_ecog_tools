import pdb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import ttest_ind
from SubjectLevel.Analyses.subject_classifier import SubjectClassifier as SC
from sklearn.metrics import roc_auc_score, roc_curve

class SubjectClassifier(SC):
    """

    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectClassifier, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

        # string to use when saving results files
        self.res_str = 'classify_inner_cv.p'

        self.do_compute_forward_model = False
        self.cv_dim = 'time'

        # If True, will classify based on best electrodes. If False, will classify based on top frequencies.
        # self.do_top_elecs = do_top_elecs
        #
        # # Setting to divide the data into random halves. .analysis() will use one half to pick the features to use and
        # # to train the classifier, and will test on the other half. This is repeated n_perms times. This analsysis
        # # doesn't really make sense if do_random_half_cv is set to False, so don't do that.
        # self.do_random_half_cv = True
        # self.n_perms = 100
        # self.do_compute_forward_model = False

    def analysis(self, permute=False):
        """

        """

        # make a copy of the data so we can restore to original after we are done
        subject_data = deepcopy(self.subject_data)
        full_cv_dict = self.cross_val_dict.copy()
        task_phase = self.task_phase.copy()
        cv_axis = subject_data.get_axis_num(self.cv_dim)

        probs = []
        Ys = []
        AUCs = []
        for cv_num, cv in enumerate(full_cv_dict.keys()):

            # sjould filter to cv events here
            inner_aucs = np.zeros(len(subject_data[self.cv_dim].data))
            for inner_cv_num, inner_cv_label in enumerate(subject_data[self.cv_dim].data):
                self.subject_data = subject_data[full_cv_dict[cv]['train_bool']].loc[{self.cv_dim: inner_cv_label}]
                self.task_phase = task_phase[full_cv_dict[cv]['train_bool']]
                self.make_cross_val_labels()
                super(SubjectClassifier, self).analysis()
                inner_aucs[inner_cv_num] = self.res['auc']

            best_param = subject_data[self.cv_dim].data[np.argmax(inner_aucs)]
            self.cross_val_dict = {cv: full_cv_dict[cv]}
            self.task_phase = task_phase
            self.subject_data = subject_data.loc[{self.cv_dim: best_param}]
            super(SubjectClassifier, self).analysis()
            probs.extend(self.res['probs'])
            Ys.extend(self.res['Y'])
            AUCs.append(self.res['auc'])

        # make subject data be the full dataset again
        self.subject_data = subject_data

        # finally, store results
        self.res = {}
        # pdb.set_trace()
        if len(np.unique(self.subject_data.events.data['session'])) == 1:
            self.res['auc'] = roc_auc_score(Ys, probs)
            self.res['loso'] = 0
        else:
            self.res['auc'] = np.mean(AUCs)
            self.res['loso'] = 1

    def plot_triangle(self):

        full_time_axis = np.round(self.time_bins * 1000) / 1000
        uniq_full_time_axis = np.unique(full_time_axis)

        windows = np.round(np.ptp(self.time_bins, axis=1) * 1000) / 1000
        uniq_windows = np.unique(windows)

        aucs = np.empty((uniq_windows.shape[0], uniq_full_time_axis.shape[0]))
        aucs[:] = np.nan

        for row, window in enumerate(uniq_windows):
            for col, t_bin in enumerate(uniq_full_time_axis):
                bin = (windows == window) & (full_time_axis == t_bin)
                if np.any(bin):
                    aucs[row, col] = self.res['aucs'][bin]

        fig, ax = plt.subplots(1, 1)
        plt.imshow(aucs, interpolation='nearest', cmap='RdBu_r', aspect='auto', zorder=2)
        plt.xticks(range(uniq_full_time_axis.shape[0])[::3], uniq_full_time_axis[::3], rotation=45, fontsize=16)
        plt.xlabel('Bin Center (s)', fontsize=16)

        plt.yticks(range(uniq_windows.shape[0])[::3], uniq_windows[::3], fontsize=16)
        plt.ylabel('Window Size (s)', fontsize=16)
        plt.gca().invert_yaxis()

        clim = np.abs(np.array([np.nanmin(aucs), np.nanmax(aucs)]) - .5).max()
        plt.clim(.5 - clim, .5 + clim)

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)

        ax.set_axisbelow(True)
        ax.yaxis.grid()
        ax.xaxis.grid()

        row, col = np.unravel_index(np.nanargmax(aucs), aucs.shape)
        best_time = uniq_full_time_axis[col]
        best_window = uniq_windows[row]
        return aucs, best_time, best_window