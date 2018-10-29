import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmx
import nibabel as nib
import ipyvolume.pylab as p3
import ipyvolume as ipv

from scipy.stats import ttest_1samp
from joblib import Parallel, delayed


class GroupFitSpectraAnalysis(object):
    """
    Group helpers for aggregation of SubjectFitSpectraAnalysis data.

    Creates .group_df, which is a dataframe of all the t-statistics with a row for each electrode.
    Also provides plotting methods:
        plot_region_heatmap()
        plot_tstat_sme()
    """

    def __init__(self, analysis_objects, res_key='ts_resid'):
        self.analysis_objects = analysis_objects

        # make group level dataframe
        self.group_df = self.create_res_df(res_key)

    def create_res_df(self, res_key):
        """
        Create one dataframe with the t-statistics for every electrode, subject, frequency. Now you can do awewsome
        things like average by subject, region, frequency in one line like:

        df.groupby(['subject', 'regions', 'frequency']).mean().groupby(['regions', 'frequency']).mean()

        ------
        Returns dataframe with columns 'label', 'subject', 'regions', 'hemi', 'frequency', 't-stat', 'avg.x', 'avg.y',
        'avg.z'
        """

        # for each subject
        df_resids = []

        for subj in self.analysis_objects:
            if ~np.any(np.isnan(subj.res[res_key])):
                if 'stein.region' in subj.elec_info:
                    region_key1 = 'stein.region'
                elif 'locTag' in subj.elec_info:
                    region_key1 = 'locTag'
                else:
                    region_key1 = ''

                if 'ind.region' in subj.elec_info:
                    region_key2 = 'ind.region'
                else:
                    region_key2 = 'indivSurf.anatRegion'

                hemi_key = 'ind.x' if 'ind.x' in subj.elec_info else 'indivSurf.x'
                if subj.elec_info[hemi_key].iloc[0] == 'NaN':
                    hemi_key = 'tal.x'
                regions = subj.bin_electrodes_by_region(elec_column1=region_key1 if region_key1 else region_key2,
                                                        elec_column2=region_key2,
                                                        x_coord_column=hemi_key)

                # get xyz from average brain
                coord_str = 'avg' if 'avg.x' in subj.elec_info else 'avgSurf'
                xyz = subj.elec_info[[coord_str + '.{}'.format(i) for i in ['x', 'y', 'z']]]

                # make a dataframe for the t-stats based on residuals
                df = pd.DataFrame(data=subj.res[res_key].T, columns=subj.freqs)
                df['label'] = regions['label']
                df['regions'] = regions['region']
                df['hemi'] = regions['hemi']

                extra_columns = []
                if 'ts_slopes' in subj.res:
                    df['slope'] = subj.res['ts_slopes']
                    df['offset'] = subj.res['ts_offsets']
                    extra_columns = ['slope', 'offset']
                df['subject'] = subj.subject
                df = pd.concat([df, xyz], axis=1)

                # melt it so that there is a row for every electrode and freqency
                df = df.melt(value_vars=subj.freqs, var_name='frequency', value_name='t-stat',
                             id_vars=['label', 'subject', 'regions', 'hemi', 'avg.x', 'avg.y', 'avg.z']+extra_columns)

                # append to list
                df_resids.append(df)

        # make group df
        df = pd.concat(df_resids)
        return df

    def plot_region_heatmap(self, clim=None):
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
        if clim is None:
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

    def plot_tstat_sme(self, region=None, ylim=None):
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

        with plt.style.context('fivethirtyeight'):
            with mpl.rc_context({'ytick.labelsize': 16,
                                 'xtick.labelsize': 16}):
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.fill_between(x, y - err, y + err, facecolor=[.5, .5, .5, .5], edgecolor=[.5, .5, .5, .5], zorder=5)
                ax.plot([x[0], x[-1]], [0, 0], '-k', linewidth=2)

                new_x = self.compute_pow_two_series(data.columns)
                ax.xaxis.set_ticks(np.log10(new_x))
                ax.xaxis.set_ticklabels(new_x, rotation=0)
                if ylim is None:
                    ylim = 1
                plt.ylim(-ylim, ylim)

                ax.set_xlabel('Frequency', fontsize=24)
                ax.set_ylabel('Average t-stat', fontsize=24)

                plt.title('%s SME, N=%d' % (region if region is not None else 'All', data.shape[0]))
                fig.set_size_inches(12, 9)

    def plot_group_brain_activation(self, radius=12.5, freq_range=(40, 200), clim=None, cmap='RdBu_r', n_perms=100,
                                    min_n=5, res_key='t-stat'):
        """
        Plots brain surface based on the mean activity across the group. Uses ipyvolume to plot

        Parameters
        ----------
        radius: float
            Maximum distance between an electrode and a vertex to be counted as data for that vertex
        freq_range: list
            2 element list defining minimum and maximum frequncies to average.
        clim: float
            Maximum/minumum value of the colormap. Will be centered at zero. If not given, will use the maximum absolute
            value of the data.
        cmap: str
            matplotlib colormap to use
        n_perms: int
            Number of permutations to do when computing our t-statistic significance thresholds
        min_n: int
            Vertices with less than this number of subjects will be plotted in gray, regardless of the significance val
        res_key: str
            Column name of dataframe to use as the metric

        Returns
        -------
        left hemisphere and right hemisphere activation maps
        """

        # load brain mesh
        l_coords, l_faces, r_coords, r_faces = self.load_brain_mesh()

        # compute mean activation. First get vertex x subject arrays
        l_vert_vals, r_vert_vals = self.compute_surface_map(radius, freq_range, res_key)

        # we will actually be plotting t-statistics, so compute those
        l_ts, l_ps = ttest_1samp(l_vert_vals, 0, axis=1, nan_policy='omit')
        r_ts, r_ps = ttest_1samp(r_vert_vals, 0, axis=1, nan_policy='omit')

        # not let's compute our significance thresholds via non-parametric permutation procedure
        sig_thresh = self.compute_permute_dist_par(l_vert_vals, r_vert_vals, n_perms=n_perms)

        # define colormap range
        if clim is None:
            clim = np.max([np.nanmax(np.abs(l_ts)), np.nanmax(np.abs(r_ts))])
        c_norm = plt.Normalize(vmin=-clim, vmax=clim)
        c_mappable = cmx.ScalarMappable(norm=c_norm, cmap=plt.get_cmap(cmap))

        # compute surface colors for left
        valid_l_inds = ~np.isnan(l_ts) & (np.sum(np.isnan(l_vert_vals), axis=1) >= min_n)
        l_colors = np.full((l_ts.shape[0], 4), 0.)
        l_colors[valid_l_inds] = c_mappable.to_rgba(l_ts[valid_l_inds])

        # and right
        valid_r_inds = ~np.isnan(r_ts) & (np.sum(np.isnan(r_vert_vals), axis=1) >= min_n)
        r_colors = np.full((r_ts.shape[0], 4), 0.)
        r_colors[valid_r_inds] = c_mappable.to_rgba(r_ts[valid_r_inds])

        # lastly, mask out vertices that do not meet our significance thresh
        sig_l = (l_ts < sig_thresh[0]) | (l_ts > sig_thresh[1])
        sig_r = (r_ts < sig_thresh[0]) | (r_ts > sig_thresh[1])
        l_colors[~sig_l] = [.7, .7, .7, 0.]
        r_colors[~sig_r] = [.7, .7, .7, 0.]

        # plot it!
        fig = p3.figure(width=800, height=800, lighting=True)
        brain_l = p3.plot_trisurf(l_coords[:, 0], l_coords[:, 1], l_coords[:, 2], triangles=l_faces, color=l_colors)
        brain_r = p3.plot_trisurf(r_coords[:, 0], r_coords[:, 1], r_coords[:, 2], triangles=r_faces, color=r_colors)

        # turn off axis and make square
        ipv.squarelim()
        ipv.style.box_off()
        ipv.style.axes_off()
        p3.show()

        return fig

    def compute_surface_map(self, radius=12.5, freq_range=(40, 200), res_key='t-stat'):
        """
        Returns brain surface activation maps based on averaging all subject t-statistics in a given frequency range.

        Parameters
        ----------
        radius: float
            Maximum distance between an electrode and a vertex to be counted as data for that vertex
        freq_range: list
            2 element list defining minimum and maximum frequncies to average.
        res_key: str
            Column name of dataframe to use as the metric

        Returns
        -------
        left hemisphere and right hemisphere activation maps (vertices x subjects)
        """

        # load average brain pial surface mesh
        l_coords, l_faces, r_coords, r_faces = self.load_brain_mesh('average')

        # mean t-stats over frequency range of interest
        freq_inds = (self.group_df.frequency >= freq_range[0]) & (self.group_df.frequency <= freq_range[1])
        mean_val_df = self.group_df[freq_inds].groupby(['subject', 'label', 'avg.x', 'avg.y', 'avg.z']).mean()
        mean_val_df = mean_val_df.reset_index()

        # initialize array to count whether a given face is near an electrode for each subject
        subjs = mean_val_df.subject.unique()
        n_subjs = len(subjs)
        l_vert_mean = np.full((l_coords.shape[0], n_subjs), np.nan)
        r_vert_mean = np.full((r_coords.shape[0], n_subjs), np.nan)

        # loop over each subject.
        for i, subj in enumerate(subjs):
            subj_res = mean_val_df[mean_val_df.subject == subj]
            for col in ['avg.x', 'avg.y', 'avg.z']:
                subj_res[col] = subj_res[col].astype('float')

            if ('avg.x' not in subj_res) or np.isnan(subj_res['avg.x'].iloc[0]):
                print('Skipping {}, missing electrode coordinates.'.format(subj))
                continue

            # make subject specific surface
            l_subj_verts = np.full((l_coords.shape[0], subj_res.shape[0]), np.nan)
            r_subj_verts = np.full((r_coords.shape[0], subj_res.shape[0]), np.nan)

            print('%s: finding valid vertices.' % subj)
            for e_num, (index, elec) in enumerate(subj_res[['avg.x', 'avg.y', 'avg.z', res_key]].iterrows()):
                l_subj_verts[np.linalg.norm(l_coords - elec.values[:3], axis=1) < radius, e_num] = elec[res_key]
                r_subj_verts[np.linalg.norm(r_coords - elec.values[:3], axis=1) < radius, e_num] = elec[res_key]

            l_vert_mean[:, i] = np.nanmean(l_subj_verts, axis=1)
            r_vert_mean[:, i] = np.nanmean(r_subj_verts, axis=1)
        return l_vert_mean, r_vert_mean

    @staticmethod
    def compute_permute_dist_par(l_vert_vals, r_vert_vals, n_perms=100):
        """
        Parameters
        ----------
        l_vert_vals: np.ndarray
            vertices x subject array of vals for the left hemisphere
        r_vert_vals: np.ndarray
            vertices x subject array of vals for the right hemisphere
        n_perms: int
            number of permutations to do. Diminishing returns after 100 or so..

        Returns
        -------
        Two element list for lower and upper sig. thresholds, based on 2.5 and 97.5th percentiles of permuted data
        """
        f = _par_compute_single_perm
        res = Parallel(n_jobs=12, verbose=5)(delayed(f)(x[0], x[1]) for x in [[l_vert_vals, r_vert_vals]] * n_perms)
        return np.nanpercentile(np.concatenate(res), [2.5, 97.5])

    @staticmethod
    def load_brain_mesh(subj='average', datadir='/data/eeg/freesurfer/subjects/{}/surf'):
        """

        Parameters
        ----------
        subj: str
            directory name containing freesurfer brain mesh (default: average)
        datadir: str
            path to freesurfer surface data

        returns
        -------
        l_coords: numpy.ndarray
            n x 3 array of 3d coordinates, specifying vertex locations for left hemisphere
        l_faces: numpy.ndarray
            n x 3 array of indices defining the triangles array
        r_coords: numpy.ndarray
            same as l_coords, for right hemisphere
        r_faces: numpy.ndarray
            same as l_faces, for right hemisphere
        """

        # load average brain pial surface mesh
        l_coords, l_faces = nib.freesurfer.read_geometry('{}/lh.pial'.format(datadir.format(subj)))
        r_coords, r_faces = nib.freesurfer.read_geometry('{}/rh.pial'.format(datadir.format(subj)))
        return l_coords, l_faces, r_coords, r_faces

    @staticmethod
    def compute_pow_two_series(freqs):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(freqs[-1]) - 1).bit_length())) + 1))


def _par_compute_single_perm(l_vert_vals, r_vert_vals):
    """
    Compute a single permutation of our procedure to find brain activation significance thresholds.

    This procedure randomly sign flips half the subjects and recomputes the t-statistics at each vertex.
    """

    # choose random half of subjects to sign flip
    flipped_subjs = np.random.rand(l_vert_vals.shape[1]) < .5

    # flip the values for those subjects left
    tmp_l = l_vert_vals.copy()
    tmp_l[:, flipped_subjs] = -tmp_l[:, flipped_subjs]

    # and right
    tmp_r = r_vert_vals.copy()
    tmp_r[:, flipped_subjs] = -tmp_r[:, flipped_subjs]

    # ttest this set of vertices with half subjects flipped against 0
    l_ts_perm_tmp, l_ps_perm_tmp = ttest_1samp(tmp_l, 0, axis=1, nan_policy='omit')
    r_ts_perm_tmp, r_ps_perm_tmp = ttest_1samp(tmp_r, 0, axis=1, nan_policy='omit')

    # return the concatenation of left and right
    return np.concatenate([l_ts_perm_tmp, r_ts_perm_tmp])