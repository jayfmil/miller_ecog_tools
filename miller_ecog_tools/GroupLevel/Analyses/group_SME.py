# from miller_ecog_tools.GroupLevel.group import Group

from scipy.stats import ttest_1samp, sem

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class GroupSMEAnalysis(object):
    """

    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe
        self.group_df = self.create_res_df()

    def create_res_df(self):
        """
        Create one dataframe with the t-statistics for every electrode, subject, frequency. Now you can do awewsome
        things like average by subject, region, frequency in one line like:

        df.groupby(['subject', 'regions', 'frequency']).mean().groupby(['regions', 'frequency']).mean()

        ------
        Returns dataframe with columns 'label', 'subject', 'regions', 'hemi', 'frequency', 't-stat'
        """

        # for each subject
        dfs = []
        for subj in self.analysis_objects:
            region_key = 'stein.region' if 'stein.region' in subj.elec_info else 'ind.region'
            hemi_key = 'ind.x'
            if subj.elec_info[hemi_key].iloc[0] == 'NaN':
                hemi_key = 'tal.x'
            regions = subj.bin_electrodes_by_region(elec_column1=region_key, x_coord_column=hemi_key)

            # make a dataframe
            df = pd.DataFrame(data=subj.res['ts'].T, columns=subj.freqs)
            df['label'] = regions['label']
            df['regions'] = regions['region']
            df['hemi'] = regions['hemi']
            df['subject'] = subj.subject

            # melt it so that there is a row for every electrode and freqency
            df = df.melt(value_vars=subj.freqs, var_name='frequency', value_name='t-stat',
                         id_vars=['label', 'subject', 'regions', 'hemi'])

            # append to list
            dfs.append(df)

        # make group df
        df = pd.concat(dfs)
        return df

    def plot_region_heatmap(self):
        """

        Plots a frequency x region heatmap of mean t-statistics.

        """

        # mean t-stat within subject by region and frequency, then mean across subjects
        mean_df = self.group_df.groupby(['subject', 'regions', 'frequency']).mean().groupby(['regions', 'frequency']).mean()
        mean_df = mean_df.reset_index()

        # ignore data without a region
        mean_df['regions'].replace('', np.nan, inplace=True)
        mean_df = mean_df.dropna(subset=['regions'])

        # reshape it for easier plotting with seaborn
        mean_df = mean_df.pivot_table(index='frequency', columns='regions', values='t-stat')

        # center the colormap and plot
        clim = np.max(np.abs(mean_df.values))
        with sns.plotting_context("talk"):
            sns.heatmap(mean_df, cmap='RdBu_r',
                        yticklabels=mean_df.index.values.round(2),
                        vmin=-clim,
                        vmax=clim,
                        cbar_kws={'label': 't-stat'})

        plt.gcf().set_size_inches(12, 9)

    def plot_tstat_sme(self, region=None):
        """
        Plots mean t-statistics, across subjects, comparing remembered and not remembered items as a function of
        frequency.
        """

        # mean within subject (and within region if region is not None)
        if region is not None:
            mean_within_subj = self.group_df.groupby(['subject', 'regions', 'frequency']).mean().reset_index()
            data = mean_within_subj[mean_within_subj['regions'] == region].pivot_table(index='subject',
                                                                                       columns='frequency',
                                                                                       values='t-stat')
        else:
            mean_within_subj = self.group_df.groupby(['subject', 'frequency']).mean().reset_index()
            data = mean_within_subj.pivot_table(index='subject', columns='frequency', values='t-stat')

        x = np.log10(data.columns)
        y = data.mean()
        err = data.sem() * 1.96

        fig, ax = plt.subplots()
        with sns.plotting_context("talk"):
            ax.plot(x, y)
            ax.fill_between(x, y - err, y + err, facecolor=[.5, .5, .5, .5], edgecolor=[.5, .5, .5, .5], zorder=5)
            ax.plot([x[0], x[-1]], [0, 0], '-k', linewidth=2)

            new_x = self.compute_pow_two_series(data.columns)
            ax.xaxis.set_ticks(np.log10(new_x))
            ax.xaxis.set_ticklabels(new_x, rotation=0)
            plt.ylim(-1, 1)

            ax.set_xlabel('Frequency', fontsize=24)
            ax.set_ylabel('Average t-stat', fontsize=24)

            plt.title('%s SME, N=%d' % (region, data.shape[0]))
            fig.set_size_inches(12, 9)

    @staticmethod
    def compute_pow_two_series(freqs):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))

    #  def plot_count_sme(self, region=None):
    #      """
    #      Plot proportion of electrodes that are signifcant at a given frequency across all electrodes in the entire
    #      dataset, seperately for singificantly negative and sig. positive.
    #      """
    #
    #      regions = self.subject_objs[0].res['regions']
    #      if region is None:
    #          sme_pos = np.stack([np.sum((x.res['ts'] > 0) & (x.res['ps'] < .05), axis=1) for x in self.subject_objs],
    #                             axis=0)
    #          sme_neg = np.stack([np.sum((x.res['ts'] < 0) & (x.res['ps'] < .05), axis=1) for x in self.subject_objs],
    #                             axis=0)
    #          n = np.stack([x.res['ts'].shape[1] for x in self.subject_objs], axis=0)
    #          region = 'All'
    #      else:
    #          region_ind = regions == region
    #          if ~np.any(region_ind):
    #              print('Invalid region, please use: %s.' % ', '.join(regions))
    #              return
    #
    #          sme_pos = np.stack([x.res['sme_count_pos'][:, region_ind].flatten() for x in self.subject_objs], axis=0)
    #          sme_neg = np.stack([x.res['sme_count_neg'][:, region_ind].flatten() for x in self.subject_objs], axis=0)
    #          n = np.stack([x.res['elec_n'][region_ind].flatten() for x in self.subject_objs], axis=0)
    #
    #      n = float(n.sum())
    #      x = np.log10(self.subject_objs[0].freqs)
    #      x_label = np.round(self.subject_objs[0].freqs * 10) / 10
    #      with plt.style.context('myplotstyle.mplstyle'):
    #
    #          fig = plt.figure()
    #          ax = plt.subplot2grid((2, 5), (0, 0), colspan=5)
    #          plt.plot(x, sme_pos.sum(axis=0) / n * 100, linewidth=4, c='#8c564b', label='Good Memory')
    #          plt.plot(x, sme_neg.sum(axis=0) / n * 100, linewidth=4, c='#1f77b4', label='Bad Memory')
    #          l = plt.legend()
    #
    #          new_x = self.compute_pow_two_series()
    #          ax.xaxis.set_ticks(np.log10(new_x))
    #          ax.plot([np.log10(new_x)[0], np.log10(new_x)[-1]], [2.5, 2.5], '--k', lw=2, zorder=3)
    #          ax.xaxis.set_ticklabels(new_x, rotation=0)
    #
    #          plt.xlabel('Frequency', fontsize=24)
    #          plt.ylabel('Percent Sig. Electrodes', fontsize=24)
    #          plt.title('%s: %d electrodes' % (region, int(n)))
    #
    #