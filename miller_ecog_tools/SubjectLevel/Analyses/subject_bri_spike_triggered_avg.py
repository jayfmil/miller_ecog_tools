"""
Spike-triggered averages from the brain research institute data.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, ttest_ind

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_BRI_data import SubjectBRIData


class SubjectSTA(SubjectAnalysisBase, SubjectBRIData):
    """
    Computes spike triggered averages.
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectSTA, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'sta.p'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        Just averages the data.
        """
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # loop over sessions
        sta_list = []
        p_spect_list = []
        region_list = []
        hemi_list = []
        session_list = []
        channel_list = []
        unit_list = []

        for session_name, session_dict in self.subject_data.items():

            # and channels
            channels = session_dict.keys()
            for channel in channels:
                data = session_dict[channel]['ST_eeg']

                # and units within a channel
                chan_units = np.unique(data.event.data['cluster_num'])
                for this_unit in chan_units:
                    unit_data = data[data.event.data['cluster_num'] == this_unit]

                    # compute spike-triggered average and store, along with attributes
                    sta_list.append(unit_data.mean(axis=0))
                    session_list.append(session_name)
                    channel_list.append(channel)
                    unit_list.append(this_unit)
                    region_list.append(np.unique(unit_data.event.data['region'])[0])

                    # also store spike triggered average
                    p_spect_list.append(session_dict[channel]['power_spectra'][this_unit])

        # create multi indexed dataframe
        names = ['session', 'channel', 'unit', 'region', 'hemi']
        index = pd.MultiIndex.from_arrays([session_list, channel_list, unit_list, region_list, hemi_list], names=names)
        sta_df = pd.DataFrame(data=np.squeeze(np.stack(sta_list, 0)), index=index, columns=unit_data.time)

        # and power spectra
        p_spect_df = pd.DataFrame(data=np.squeeze(np.stack(p_spect_list, 0)), index=index, columns=self.freqs)

        # store results.
        self.res['sta'] = sta_df
        self.res['p_spect'] = p_spect_df

    def plot_spectra_average(self, elec_label='', region_column='', loc_tag_column=''):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for recalled (red) and not-recalled (blue) items. Bottom panel is t-stat at each frequency comparing the
        recalled and not recalled distributions, with shaded areas indicating p<.05.

        elec_label: electrode label that you wish to plot.
        region_column: column of the elec_info dataframe that will be used to label the plot
        loc_tag_column: another of the elec_info dataframe that will be used to label the plot
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subject)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subject)
            return

        # get the index into the data for this electrode
        elec_ind = self.subject_data.channel == elec_label
        if ~np.any(elec_ind):
            print('%s: must enter a valid electrode label, as found in self.subject_data.channel' % self.subject)
            return

        # normalize spectra
        recalled = self.res['recalled']
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        # create axis
        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

                # will plot in log space
                x = np.log10(self.subject_data.frequency)

                ###############
                ## Top panel ##
                ###############
                # recalled mean and err
                rec_mean = np.mean(p_spect[recalled, :, elec_ind], axis=0)
                rec_sem = sem(p_spect[recalled, :, elec_ind], axis=0)
                ax1.plot(x, rec_mean, c='#8c564b', label='Good Memory', linewidth=2)
                ax1.fill_between(x, rec_mean + rec_sem, rec_mean - rec_sem, color='#8c564b', alpha=.5)

                # not recalled mean and err
                nrec_mean = np.mean(p_spect[~recalled, :, elec_ind], axis=0)
                nrec_sem = sem(p_spect[~recalled, :, elec_ind], axis=0)
                ax1.plot(x, nrec_mean, color='#1f77b4', label='Bad Memory', linewidth=2)
                ax1.fill_between(x, nrec_mean + nrec_sem, nrec_mean - nrec_sem, color='#1f77b4', alpha=.5)

                # y labels and y ticks
                ax1.set_ylabel('Normalized log(power)')
                ax1.yaxis.label.set_fontsize(24)
                ax1.yaxis.set_ticks([-2, -1, 0, 1, 2])
                ax1.set_ylim([-2, 2])

                # make legend
                l = ax1.legend()
                frame = l.get_frame()
                frame.set_facecolor('w')
                for legobj in l.legendHandles:
                    legobj.set_linewidth(5)

                ##################
                ## Bottom panel ##
                ##################
                y = np.squeeze(self.res['ts'][:, elec_ind])
                p = np.squeeze(self.res['ps'][:, elec_ind])
                ax2.plot(x, y, '-k', linewidth=4)
                ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
                ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=1)
                ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y > 0), facecolor='#8c564b', edgecolor='#8c564b')
                ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y < 0), facecolor='#1f77b4', edgecolor='#1f77b4')
                ax2.set_ylabel('t-stat')
                plt.xlabel('Frequency', fontsize=24)
                ax2.yaxis.label.set_fontsize(24)
                ax2.yaxis.set_ticks([-2, 0, 2])

                # put powers of two on the x-axis for both panels
                new_x = self.compute_pow_two_series()
                ax2.xaxis.set_ticks(np.log10(new_x))
                ax2.xaxis.set_ticklabels(new_x, rotation=0)
                ax1.xaxis.set_ticks(np.log10(new_x))
                ax1.xaxis.set_ticklabels('')

                # get some localization info for the title
                elec_info_chan = self.elec_info[self.elec_info.label == elec_label]
                title_str = ''
                for col in [region_column, loc_tag_column]:
                    if col:
                        title_str += ' ' + str(elec_info_chan[col].values) + ' '

                _ = ax1.set_title('%s - %s' % (self.subject, elec_label) + title_str)
                plt.gcf().set_size_inches(12, 9)

        return plt.gcf()

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
        df = pd.DataFrame(self.res['ts'], index=self.freqs,
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
            clim = np.max(np.abs(self.res['ts']))
            sns.heatmap(df, cmap='RdBu_r', linewidths=.5,
                        yticklabels=df.index.values.round(2), ax=ax,
                        cbar_ax=cax, vmin=-clim, vmax=clim, cbar_kws={'label': 't-stat'})
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

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.event.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.event.data['session'] == sess)
            m = np.mean(X[sess_event_mask], axis=1)
            m = np.mean(m, axis=0)
            s = np.std(X[sess_event_mask], axis=1)
            s = np.mean(s, axis=0)
            X[sess_event_mask] = (X[sess_event_mask] - m) / s
        return X

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
