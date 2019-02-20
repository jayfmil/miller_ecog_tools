import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


class GroupBRIRetrievalAnalysis(object):
    """

    """

    def __init__(self, analysis_objects):
        self.analysis_objects = analysis_objects

        # make group level dataframe for subeject level aggregation
        print('Creating group dataframe for BRI retrieval analysis.')
        self.group_df, self.time = self.create_subj_df()

    def create_subj_df(self):
        """

        """
        res_key = 'erp_by_lag'

        # for each subject
        dfs = []
        time = None
        for subj in tqdm(self.analysis_objects):
            subj_df = []
            for k, v in subj.res.items():
                if time is None:
                    time = np.round(v[res_key].index, 3)
                elec_df = v[res_key].copy()
                elec_df = elec_df.reset_index(drop=False)

                # instead of time in ms, convert it to sample number starting at 0. This way differences in the
                # float values from subject to subject are ignored
                _, time_to_order = np.unique(elec_df.time.values, return_inverse=True)
                elec_df.time = time_to_order
                elec_df['subj'] = subj.subject
                elec_df['hemi'] = v['hemi']
                elec_df['region'] = v['region']
                elec_df['label'] = k
                subj_df.append(elec_df)
            subj_df = pd.concat(subj_df)
            subj_df = subj_df.groupby(['subj', 'hemi', 'region', 'time', 'lag']).mean()
            dfs.append(subj_df)

        # make group df
        df = pd.concat(dfs)
        return df, np.unique(time)

    def plot_region_erp(self, hemi, region, lags=['0', '1', '4-8', '16-32'], only_correct=False):

        y_str = 'y' if not only_correct else 'y_correct'

        # aggregate the data
        region_means = self.group_df.groupby(['hemi', 'region', 'time', 'lag']).mean()
        region_sems = self.group_df.groupby(['hemi', 'region', 'time', 'lag']).sem()
        data = region_means.loc[hemi, region].pivot_table(index='lag', columns='time', values=y_str)
        data_sems = region_sems.loc[hemi, region].pivot_table(index='lag', columns='time', values=y_str)
        data.columns = np.unique(self.time)
        data_sems.columns = np.unique(self.time)

        # this is the actual data to plot, using only the specific lags
        data_to_plot = data.loc[lags].T
        sems_to_plot = data_sems.loc[lags].T

        with plt.style.context('seaborn-white'):
            with mpl.rc_context({'ytick.labelsize': 22,
                                 'xtick.labelsize': 22,
                                 'legend.fontsize': 18}):
                # make the initial axis
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 10)

                # plot the lines
                data_to_plot.plot(ax=ax, colormap='Set2', lw=2)

                # and the error regions
                for i, this_lag in enumerate(lags):
                    y = data_to_plot[this_lag]
                    y1 = data_to_plot[this_lag] - sems_to_plot[this_lag]
                    y2 = data_to_plot[this_lag] + sems_to_plot[this_lag]
                    c = ax.lines[i].get_color()
                    ax.fill_between(data_to_plot.index.values, y1.values, y2.values, alpha=.3,
                                    facecolor=c, edgecolor=None, lw=0)

                    # set labels
                ax.set_xlabel('Time (s)', fontsize=24)
                ax.set_ylabel('Voltage', fontsize=24)

                # and title
                ax.legend().set_title('')
                title_str = '{} - {}'.format(hemi, region)
                ax.set_title(title_str, fontsize=20)

        return ax

    @staticmethod
    def compute_pow_two_series(freqs):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))
