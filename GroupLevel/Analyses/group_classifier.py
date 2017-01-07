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
        """
        TO DO
        """
        pass

    def plot_feature_map(self):
        """
        Makes a heatmap style plot of average forward model transformed classifier weight as a function of brain
        region. This will shows which brain regions are important for predicting good vs bad memory.
        """

        # stack all the subject means
        region_mean = np.stack([x.res['forward_model_by_region'] for x in self.subject_objs], axis=0)

        # reorder to group the regions in a way that visually makes more sense
        regions = np.array(['IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC'])
        key_order = self.subject_objs[0].res['regions']
        new_order = np.searchsorted(key_order, np.array(regions))
        region_mean = region_mean[:, :, new_order]

        # mean across subjects, that is what we will plot
        plot_data = np.nanmean(region_mean, axis=0)
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))

        # also create a mask of significant region/frequency bins
        t, p = ttest_1samp(region_mean, 0, axis=0, nan_policy='omit')
        p2 = np.ma.masked_where(p < .05, p)

        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='Feature Importance', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(regions)), regions, fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.subject_objs[0].freqs),
                              range(len(self.subject_objs[0].freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)

            # overlay mask
            plt.imshow(p2 > 0, interpolation='nearest', cmap='gray_r', aspect='auto', alpha=.6)
            plt.gca().invert_yaxis()
            plt.grid()

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
                                                                            self.summary_table.shape[0]-1,
                                                                            t, int(np.ceil(np.log10(p)))))

