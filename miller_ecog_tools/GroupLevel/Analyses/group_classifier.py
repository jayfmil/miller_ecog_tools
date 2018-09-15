import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class GroupClassifierAnalysis(object):
    """
    Group helpers for aggregation of SubjectClassifierAnalysis data.

    Creates .group_df, which is a dataframe of all the t-statistics with a row for each electrode.
    Also provides plotting methods:
        plot_region_heatmap()
        plot_tstat_sme()
    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe for electrode level
        self.elec_df = self.create_elec_df()

        # also summarize subject level
        self.subject_df = pd.DataFrame([(x.subject, x.res['auc'], x.res['is_multi_sess']) for x in analysis_objects],
                                       columns=['subject', 'auc', 'is_multi_sess'])

    def create_elec_df(self):
        """
        Create one dataframe with the model weights for every electrode, subject, frequency. Now you can do awewsome
        things like average by subject, region, frequency in one line like:

        df.groupby(['subject', 'regions', 'frequency']).mean().groupby(['regions', 'frequency']).mean()

        ------
        Returns dataframe with columns 'label', 'subject', 'regions', 'hemi', 'frequency', 'model-weight', 'avg.x',
        'avg.y', 'avg.z'
        """

        # for each subject
        dfs = []
        for subj in self.analysis_objects:
            region_key = 'stein.region' if 'stein.region' in subj.elec_info else 'ind.region'
            hemi_key = 'ind.x'
            if subj.elec_info[hemi_key].iloc[0] == 'NaN':
                hemi_key = 'tal.x'
            regions = subj.bin_electrodes_by_region(elec_column1=region_key, x_coord_column=hemi_key)

            # get xyz from average brain
            xyz = subj.elec_info[['avg.x', 'avg.y', 'avg.z']]

            # make a dataframe
            df = pd.DataFrame(data=subj.res['forward_model'].T, columns=subj.freqs)
            df['label'] = regions['label']
            df['regions'] = regions['region']
            df['hemi'] = regions['hemi']
            df['subject'] = subj.subject
            df = pd.concat([df, xyz], axis=1)

            # melt it so that there is a row for every electrode and freqency
            df = df.melt(value_vars=subj.freqs, var_name='frequency', value_name='model-weight',
                         id_vars=['label', 'subject', 'regions', 'hemi', 'avg.x', 'avg.y', 'avg.z'])

            # append to list
            dfs.append(df)

        # make group df
        df = pd.concat(dfs)
        return df

    def plot_region_heatmap(self):
        """

        Plots a frequency x region heatmap of mean classifier weights.

        """

        # mean t-stat within subject by region and frequency, then mean across subjects
        mean_df = self.elec_df.groupby(['subject', 'regions', 'frequency']).mean().groupby(['regions', 'frequency']).mean()
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
            plt.gca().invert_yaxis()
            plt.ylabel('Frequency')
            plt.xlabel('')

        plt.gcf().set_size_inches(12, 9)

    @staticmethod
    def compute_pow_two_series(freqs):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))
