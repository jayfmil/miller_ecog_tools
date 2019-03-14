import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.stats import ttest_1samp, sem


class GroupNoveltyAnalysis(object):
    """

    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe for subeject level aggregation
        print('Creating group dataframe 1 of 2 of LFP data for BRI novelty analysis.')
        # self.df_z = self.create_subj_df_lfp(do_t_not_z=False)
        print('Creating group dataframe 2 of 2 of LFP data for BRI novelty analysis.')
        # self.df_t = self.create_subj_df_lfp(do_t_not_z=True)

        self.df_rayleigh_all = self.create_subj_df_rayleigh(k='phase_stats_all_events')
        self.df_rayleigh_resp_events = self.create_subj_df_rayleigh(k='phase_stats_resp_events')
        self.df_rayleigh_resp_items = self.create_subj_df_rayleigh(k='phase_stats_resp_items')
        # self.df_rayleigh_resp_events_inv = self.create_subj_df_rayleigh(k='phase_stats_resp_events_inv')
        # self.df_rayleigh_resp_item_inv = self.create_subj_df_rayleigh(k='phase_stats_resp_items_inv')

    def create_subj_df_lfp(self, do_t_not_z=False):
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

    def create_subj_df_rayleigh(self, k='phase_stats_all_events'):

        df = []
        if 'all' not in k:
            lfp_power_key = 'delta_z_' + '_'.join(k.split('_')[-2:])
            do_all_events = False
        else:
            do_all_events = True

        for subj in self.analysis_objects:
            for channel_key, channel in subj.res.items():
                for cluster_key in channel['firing_rates']:

                    if ~np.any(np.isnan(channel['firing_rates'][cluster_key][k]['z_novel'])):
                        rayleigh_novel_z = channel['firing_rates'][cluster_key][k]['z_novel']
                        rayleigh_rep_z = channel['firing_rates'][cluster_key][k]['z_rep']
                        z_diff = rayleigh_novel_z - rayleigh_rep_z

                        if do_all_events:
                            z_diff_lfp = channel['delta_z']
                        else:
                            z_diff_lfp = channel['firing_rates'][cluster_key][lfp_power_key]
                        z_diff_lfp = z_diff_lfp.T.loc[subj.phase_bin_start:subj.phase_bin_stop].mean().values

                        subj_df = pd.DataFrame(data=[z_diff, z_diff_lfp, subj.power_freqs]).T
                        subj_df.columns = ['z_rayleigh_diff', 'z_power_diff', 'frequency']
                        subj_df['region'] = channel['region']
                        subj_df['subject'] = subj.subject
                        subj_df['hemi'] = channel['hemi']
                        subj_df['id'] = channel_key + '/' + cluster_key

                        df.append(subj_df)

        df = pd.concat(df)
        return df

    def compute_rayleigh_z_diff(self, hemi, region, df, ylim=None, ylim2=None):

        if hemi is not None:
            # aggregated at electrode level
            elec_df = df.groupby(['id', 'region', 'hemi', 'frequency']).mean()
            elec_df_region = elec_df.loc[pd.IndexSlice[:, region, hemi], :].pivot_table(index='id', columns='frequency')

            # aggregated at subject level by meaning all electrodes within subject region hemi
            subj_df = df.groupby(['subject', 'region', 'hemi', 'frequency']).mean()
            subj_df_region = subj_df.loc[pd.IndexSlice[:, region, hemi], :].pivot_table(index='subject',
                                                                                        columns='frequency')
        else:
            # aggregated at electrode level
            elec_df = df.groupby(['id', 'region', 'frequency']).mean()
            elec_df_region = elec_df.loc[pd.IndexSlice[:, region], :].pivot_table(index='id', columns='frequency')

            # aggregated at subject level by meaning all electrodes within subject region hemi
            subj_df = df.groupby(['subject', 'region', 'frequency']).mean()
            subj_df_region = subj_df.loc[pd.IndexSlice[:, region], :].pivot_table(index='subject', columns='frequency')

        with plt.style.context('seaborn-white'):
            with mpl.rc_context({'ytick.labelsize': 18,
                                 'xtick.labelsize': 18}):
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
                for plot_data in zip([elec_df_region, subj_df_region], [ax1, ax2],
                                     ['{} - {}: {} Units', '{} - {}: {} Subjects']):

                    plot_df = plot_data[0]
                    ax = plot_data[1]

                    freqs = plot_df.loc[:, 'z_rayleigh_diff'].columns.get_level_values(0).values
                    new_x = np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))
                    y_mean = plot_df.loc[:, 'z_rayleigh_diff'].values.mean(axis=0)
                    y_sem = sem(plot_df.loc[:, 'z_rayleigh_diff'].values, axis=0)
                    ts, ps = ttest_1samp(plot_df.loc[:, 'z_rayleigh_diff'].values, 0)

                    ax.plot(np.log10(freqs), y_mean, '-k', lw=2)
                    ax.plot(np.log10(freqs), [0] * len(freqs), '-k', lw=1, zorder=-1)
                    ax.fill_between(np.log10(freqs), y_mean - y_sem,
                                    y_mean + y_sem, alpha=.5)
                    ax.xaxis.set_ticks(np.log10(new_x))
                    ax.xaxis.set_ticklabels(new_x, rotation=0)

                    if ylim is None:
                        ylim = ax.get_ylim()
                    ax.set_ylim([-np.max(np.abs(ylim)), np.max(np.abs(ylim))])
                    if np.any(ps < 0.05):
                        height = np.ptp(ylim) * .05
                        top = np.max(np.abs(ylim))
                        for sig in np.log10(freqs)[ps < 0.05]:
                            ax.plot([sig, sig], [top - height, top], '-', c=[.5, .5, .5], lw=10,
                                    solid_capstyle='butt', alpha=.5)
                            ax.plot([sig, sig], [-top + height, -top], '-', c=[.5, .5, .5], lw=10,
                                    solid_capstyle='butt', alpha=.5)

                    divider = make_axes_locatable(ax)
                    ax2 = divider.append_axes("bottom", size="100%", pad="15%")
                    #                 ax2 = ax.twinx()
                    y_mean = plot_df.loc[:, 'z_power_diff'].values.mean(axis=0)
                    y_sem = sem(plot_df.loc[:, 'z_power_diff'].values, axis=0)
                    ts, ps = ttest_1samp(plot_df.loc[:, 'z_power_diff'].values, 0)

                    ax2.plot(np.log10(freqs), y_mean, '-k', lw=2)
                    ax2.plot(np.log10(freqs), [0] * len(freqs), '-k', lw=1, zorder=-1)
                    ax2.fill_between(np.log10(freqs), y_mean - y_sem,
                                     y_mean + y_sem, alpha=.5)
                    ax2.xaxis.set_ticks(np.log10(new_x))
                    ax2.xaxis.set_ticklabels(new_x, rotation=0)

                    if ylim2 is None:
                        ylim2 = ax2.get_ylim()
                    ax2.set_ylim([-np.max(np.abs(ylim2)), np.max(np.abs(ylim2))])
                    if np.any(ps < 0.05):
                        height = np.ptp(ylim2) * .05
                        top = np.max(np.abs(ylim2))
                        for sig in np.log10(freqs)[ps < 0.05]:
                            ax2.plot([sig, sig], [top - height, top], '-', c=[.5, .5, .5], lw=10,
                                     solid_capstyle='butt', alpha=.5)
                            ax2.plot([sig, sig], [-top + height, -top], '-', c=[.5, .5, .5], lw=10,
                                     solid_capstyle='butt', alpha=.5)

                    ax2.set_xlabel('Frequency (Hz)', fontsize=20)
                    ax.set_ylabel('$\Delta$ Rayleigh (Z)', fontsize=20)
                    ax2.set_ylabel('$\Delta$ Power (Z)', fontsize=20)
                    ax.set_title(plot_data[2].format(hemi, region, plot_df.shape[0]), fontsize=18)

                fig.set_size_inches(10, 16)

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
