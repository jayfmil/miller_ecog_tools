import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


class BRINoveltyAnalysis(object):
    """

    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe for electrode level
        self.elec_df_z = self.create_elec_df(do_t_not_z=False)
        self.elec_df_t = self.create_elec_df(do_t_not_z=True)


    def create_elec_df(self, do_t_not_z=False):
        """

        """
        res_key = 'delta_t' if do_t_not_z else 'delta_z'
        # for each subject
        dfs = []
        for subj in self.analysis_objects:
            for k, v in subj.res.items():
                time = v[res_key].columns.values
                elec_df = v[res_key].copy().reset_index(drop=True)
                elec_df['subj'] = subj.subject
                elec_df['hemi'] = v['hemi']
                elec_df['region'] = v['region']
                elec_df['label'] = k
                dfs.append(elec_df.melt(value_vars=time, var_name='time', value_name='stat',
                                             id_vars=['subj', 'region', 'hemi', 'label']))

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
