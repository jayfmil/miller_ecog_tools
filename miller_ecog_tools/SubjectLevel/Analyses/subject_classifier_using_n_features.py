"""
Uses a penalized logistic regression classifier to predict single-trial memory success or failure based on spectral
power features.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import zscore, zmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from miller_ecog_tools.SubjectLevel.Analyses.subject_classifier import SubjectClassifierAnalysis


class SubjectClassifierNFeaturesAnalysis(SubjectClassifierAnalysis):
    """
    Modified SubjectClassifierAnalysis that iteratively computes classification performance as a function of number of
    electrodes, from 1 .. total number.

    Electrodes will be included in order of decreasing strength univariate t-test vs good and bad memory. In other
    words, for the iteration with only 1 electrode, it will be the "best" electrode. This is Figure 7 of Miller et al.,
    2018. Big picture: this can test if the signals at different electrodes are redundant.
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectClassifierNFeaturesAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'classifier_n_elecs.p'
        # whether to compute a null distribution of classifier performance
        self.compute_null_dist = False
        self.num_iters = 100

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def _make_cross_val_labels(self):
        """
        Unlike SubjectClassifierAnalysis, we here use a split-half CV. One half will be used to identify the features to
        use using a univariate t-test. The other half will
        """

        # create folds based on either sessions or trials within a session
        sessions = self.subject_data.event.data['session']
        if len(np.unique(self.subject_data.event.data['session'])) > 1:
            folds = sessions
            is_multi_sess = True
        else:
            folds = self.subject_data.event.data[self.trial_field]
            is_multi_sess = False

        # make dictionary to hold booleans for training and test indices for each fold
        cv_dict = {}
        uniq_folds = np.unique(folds)
        for fold in uniq_folds:
            cv_dict[fold] = {}
            cv_dict[fold]['train_bool'] = folds != fold
            cv_dict[fold]['test_bool'] = folds == fold
        return cv_dict, is_multi_sess

    def analysis(self, permute=False):
        """
        Performs the classification analysis through leave-one-trial-out or leave-one-session-out cross validation. For
        each training set, learn the relationship electrode x spectral power features and behavioral success or failure,
        then test on on each test set behavior. Iterate over all folds to compute average classifier performance.

        Optional parameter
        ------------------
        permute: bool (default False)
            If True, will randomize the behavioral outcomes across trials. This can be used to build a null distribution
            of classifier performance to which the true classifier performance may be compared.
        """
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s classifier: please provide a .recall_filter_func function.' % self.subject)
        y = self.recall_filter_func(self.subject_data)

        # zscore the data by session, and reshape to obs x features
        x = self.zscore_data().reshape(self.subject_data.shape[0], -1)

        # create the classifier
        classifier = LogisticRegression(C=self.C, penalty=self.norm, solver='liblinear')

        # compute the cross validatin folds
        cv_dict, is_multi_sess = self._make_cross_val_labels()

        # do the actual classification
        auc, probs = self.do_cv(cv_dict, is_multi_sess, classifier, x, y, permute=False)

        # store results
        self.res['auc'] = auc
        self.res['is_multi_sess'] = is_multi_sess
        self.res['probs'] = probs
        self.res['y'] = y

        # also run the model on the full dataset and store
        model = self.do_fit_model(classifier, x, y)
        self.res['model'] = model

        # transform model weights into a more easily interpretable result
        forward_model = self.compute_forward_model(x, probs, model.coef_)
        self.res['forward_model'] = forward_model.reshape(self.subject_data.shape[1], -1)

        # compute null distribution if desired
        if self.compute_null_dist:
            auc_null = [self.do_cv(cv_dict, is_multi_sess, classifier, x, y, True)[0] for _ in range(self.num_iters)]
            self.res['auc_null'] = np.array(auc_null)
            self.res['p_val'] = np.mean(self.res['auc'] < auc_null)

    @staticmethod
    def do_cv(cv_dict, is_multi_sess, classifier, x, y, permute=False):
        """
        Loop over all cross validation folds, return area under the curve (AUC) and class probabilities.
        """

        # permute distribution of behavior if desired. Should this be done with each fold?
        if permute:
            y = np.random.permutation(y)

        # if leave-one-session-out cross validation, this will hold area under the curve for each hold out
        fold_aucs = np.empty(shape=(len(cv_dict)), dtype=np.float)

        # will hold the predicted class probability for all the test data
        probs = np.empty(shape=y.shape, dtype=np.float)

        # now loop over all the cross validation folds
        for cv_num, cv in enumerate(cv_dict.keys()):

            # Training data for fold
            x_train = x[cv_dict[cv]['train_bool']]
            y_train = y[cv_dict[cv]['train_bool']]

            # Test data for fold
            x_test = x[cv_dict[cv]['test_bool']]
            y_test = y[cv_dict[cv]['test_bool']]

            # normalize the train data, and then normalize the test data by the mean and sd of the train data
            # this is a little silly because the data are already zscored by session, but it could presumably
            # have an effect for within-session leave-out-trial-out cross validation. The main point is that train
            # and test data should be scaled the same
            x_train = zscore(x_train, axis=0)
            x_test = zmap(x_test, x_train, axis=0)

            # fit the model for this fold
            classifier = SubjectClassifierAnalysis.do_fit_model(classifier, x_train, y_train)

            # now predict class probability of test data
            test_probs = classifier.predict_proba(x_test)[:, 1]
            probs[cv_dict[cv]['test_bool']] = test_probs

            # if session level CV, compute the area under the curve for this fold and store
            if is_multi_sess:
                fold_aucs[cv_num] = roc_auc_score(y_test, test_probs)

        # compute AUC based on all CVs, either as the average of the session-level AUCs, or all the cross-validated
        # predictions of the within session CVs
        auc = fold_aucs.mean() if is_multi_sess else roc_auc_score(y, probs)
        return auc, probs

    @staticmethod
    def do_fit_model(classifier, x_train, y_train):
        """
        Do the model fit, return the fitted classifier object.
        """

        # weight observations by number of positive and negative class
        y_ind = y_train.astype(int)
        recip_freq = 1. / np.bincount(y_ind)
        recip_freq /= np.mean(recip_freq)
        weights = recip_freq[y_ind]
        return classifier.fit(x_train, y_train, sample_weight=weights)

    @staticmethod
    def compute_forward_model(x, probs, model_coef):
        """
        Compute "forward model" to make the classifier model weights interpretable. Based on Haufe et al, 2014 - On the
        interpretation of weight vectors of linear models in multivariate neuroimaging, Neuroimage
        """

        # compute forward model
        probs_log = np.log(probs / (1 - probs))
        covx = np.cov(x.T)
        covs = np.cov(probs_log)
        A = np.dot(covx, model_coef.T) / covs
        return A

    ######################
    # PLOTTING FUNCTIONS #
    ######################
    def plot_roc(self):
        """
        Plot receiver operating charactistic curve for this subject's classifier.
        """

        fpr, tpr, _ = roc_curve(self.res['y'], self.res['probs'])
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                plt.plot(fpr, tpr, lw=4, label='ROC curve (AUC = %0.2f)' % self.res['auc'])
                plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='_nolegend_')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=24)
                plt.ylabel('True Positive Rate', fontsize=24)
                plt.legend(loc="lower right")
                if 'p_val' not in self.res:
                    title = 'ROC (AUC: {0:.3f})'.format(self.res['auc'])
                else:
                    p = self.res['p_val']
                    if p == 0:
                        p_str = '< {0:.2f}'.format(1 / self.num_iters)
                    else:
                        p_str = '= {0:.3f}'.format(p)
                    title = 'ROC (AUC: {0:.3f}, p{1})'.format(self.res['auc'], p_str)
                plt.title(title)
        plt.gcf().set_size_inches(12, 9)

    def plot_elec_heat_map(self, sortby_column1='', sortby_column2=''):
        """
        Frequency by electrode SME visualization.

        Plots a channel x frequency visualization of the subject's data
        sortby_column1: if given, will sort the data by this column and then plot
        sortby_column2: secondary column for sorting
        """

        # group the electrodes by region, if we have the info
        do_region = True
        if sortby_column1 and sortby_column2:
            regions = self.elec_info[sortby_column1].fillna(self.elec_info[sortby_column2]).fillna(value='')
            elec_order = np.argsort(regions)
            groups = np.unique(regions)
        elif sortby_column1:
            regions = self.elec_info[sortby_column1].fillna(value='')
            elec_order = np.argsort(regions)
            groups = np.unique(regions)
        else:
            elec_order = np.arange(self.elec_info.shape[0])
            do_region = False

        # make dataframe of results for easier plotting
        df = pd.DataFrame(self.res['forward_model'], index=self.freqs,
                          columns=self.subject_data.channel)
        df = df.T.iloc[elec_order].T

        # make figure. Add axes for colorbar
        with mpl.rc_context({'ytick.labelsize': 14,
                             'xtick.labelsize': 14,
                             'axes.labelsize': 20}):
            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.1)

            # plot heatmap
            plt.gcf().set_size_inches(18, 12)
            clim = np.max(np.abs(self.res['forward_model']))
            sns.heatmap(df, cmap='RdBu_r', linewidths=.5,
                        yticklabels=df.index.values.round(2), ax=ax,
                        cbar_ax=cax, vmin=-clim, vmax=clim, cbar_kws={'label': 'Forward model weight'})
            ax.set_xlabel('Channel', fontsize=24)
            ax.set_ylabel('Frequency (Hz)', fontsize=24)
            ax.invert_yaxis()

            # if plotting region info
            if do_region:
                ax2 = divider.append_axes('top', size='3%', pad=0)
                for i, this_group in enumerate(groups):
                    x = np.where(regions[elec_order] == this_group)[0]
                    ax2.plot([x[0] + .5, x[-1] + .5], [0, 0], '-', color=[.7, .7, .7])
                    if len(x) > 1:
                        if ' ' in this_group:
                            this_group = this_group.split()[0]+' '+''.join([x[0].upper() for x in this_group.split()[1:]])
                        else:
                            this_group = this_group[:12] + '.'
                        plt.text(np.mean([x[0] + .5, x[-1] + .5]), 0.05, this_group,
                                 fontsize=14,
                                 horizontalalignment='center',
                                 verticalalignment='bottom', rotation=90)
                ax2.set_xlim(ax.get_xlim())
                ax2.set_yticks([])
                ax2.set_xticks([])
                ax2.axis('off')

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
