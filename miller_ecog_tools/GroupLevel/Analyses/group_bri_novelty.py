import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


class GroupNoveltyAnalysis(object):
    """

    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe for subeject level aggregation
        print('Creating group dataframe 1 of 2 for BRI novelty analysis.')
        self.df_z = self.create_subj_df(do_t_not_z=False)
        print('Creating group dataframe 2 of 2 for BRI novelty analysis.')
        self.df_t = self.create_subj_df(do_t_not_z=True)

    def create_subj_df(self, do_t_not_z=False):
        """

        """
        res_key = 'delta_t' if do_t_not_z else 'delta_z'
        # for each subject
        dfs = []
        time = None
        for subj in tqdm(self.analysis_objects):
            subj_df = []
            for k, v in subj.res.items():
                if time is None:
                    time = np.round(v[res_key].columns.values, 3)
                elec_df = v[res_key].copy()
                if len(elec_df.columns) > len(time):
                    elec_df = elec_df.iloc[:, 1:len(time)+1]
                elec_df.columns = time
                elec_df = elec_df.reset_index(drop=False)
                elec_df['subj'] = subj.subject
                elec_df['hemi'] = v['hemi']
                elec_df['region'] = v['region']
                elec_df['label'] = k
                elec_df = elec_df.melt(value_vars=time, var_name='time', value_name='stat',
                                       id_vars=['subj', 'region', 'hemi', 'frequency', 'label'])
                subj_df.append(elec_df)
            subj_df = pd.concat(subj_df)
            subj_df = subj_df.groupby(['subj', 'hemi', 'region', 'frequency', 'time']).mean()
            dfs.append(subj_df)

        # make group df
        df = pd.concat(dfs)
        return df

    def plot_region_heatmap(self, hemi, region, do_t_not_z=False):

        df = self.df_z if not do_t_not_z else self.df_t
        region_means = df.groupby(['hemi', 'region', 'frequency', 'time']).mean()
        data = region_means.loc[hemi, region].pivot_table(index='frequency', columns='time', values='stat')
        time = data.columns.values
        freqs = data.index.values
        clim = np.max(np.abs(data.values))
        cbar_label = 'mean(Z)' if not do_t_not_z else 'mean(t)'

        with plt.style.context('seaborn-white'):
            with mpl.rc_context({'ytick.labelsize': 22,
                                 'xtick.labelsize': 22}):
                # make the initial plot
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 10)
                im = ax.imshow(data.values,
                               aspect='auto', interpolation='bicubic', cmap='RdBu_r', vmax=clim, vmin=-clim)
                ax.invert_yaxis()

                # set the x values to be specific timepoints
                x_vals = np.array([-500, -250, 0, 250, 500, 750, 1000]) / 1000
                new_xticks = np.round(np.interp(x_vals, time, np.arange(len(time))))
                ax.set_xticks(new_xticks)
                ax.set_xticklabels([x for x in x_vals], fontsize=22)
                ax.set_xlabel('Time (s)', fontsize=24)

                # now the y
                ax.set_yticks(np.arange(len(freqs))[::5])
                ax.set_yticklabels(np.round(freqs[::5], 2), fontsize=20)
                ax.set_ylabel('Frequency (Hz)', fontsize=24)

                # add colorbar
                cbar = plt.colorbar(im)
                ticklabels = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(ticklabels, fontsize=16)
                cbar.ax.set_ylabel(cbar_label, fontsize=20)

                title_str = '{} - {}'.format(hemi, region)
                ax.set_title(title_str, fontsize=20)

    @staticmethod
    def compute_pow_two_series(freqs):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))
