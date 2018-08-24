import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmx
import nibabel as nib
import ipyvolume.pylab as p3
import ipyvolume as ipv


class GroupSMEAnalysis(object):
    """
    Group helpers for aggregation of SubjectSMEAnalysis data.

    Creates .group_df, which is a dataframe of all the t-statistics with a row for each electrode.
    Also provides plotting methods:
        plot_region_heatmap()
        plot_tstat_sme()
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
        Returns dataframe with columns 'label', 'subject', 'regions', 'hemi', 'frequency', 't-stat', 'avg.x', 'avg.y',
        'avg.z'
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
            df = pd.DataFrame(data=subj.res['ts'].T, columns=subj.freqs)
            df['label'] = regions['label']
            df['regions'] = regions['region']
            df['hemi'] = regions['hemi']
            df['subject'] = subj.subject
            df = pd.concat([df, xyz], axis=1)

            # melt it so that there is a row for every electrode and freqency
            df = df.melt(value_vars=subj.freqs, var_name='frequency', value_name='t-stat',
                         id_vars=['label', 'subject', 'regions', 'hemi', 'avg.x', 'avg.y', 'avg.z'])

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
            plt.gca().invert_yaxis()
            plt.ylabel('Frequency')
            plt.xlabel('')

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
                plt.ylim(-1, 1)

                ax.set_xlabel('Frequency', fontsize=24)
                ax.set_ylabel('Average t-stat', fontsize=24)

                plt.title('%s SME, N=%d' % (region if region is not None else 'All', data.shape[0]))
                fig.set_size_inches(12, 9)

    def plot_group_brain_activation(self, radius=12.5, freq_range=(40, 200), clim=None, cmap='RdBu_r'):
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

        Returns
        -------
        left hemisphere and right hemisphere activation maps
        """

        # load brain mesh
        l_coords, l_faces, r_coords, r_faces = self.load_brain_mesh()

        # compute mean activation
        l_vert_mean, r_vert_mean = self.compute_surface_map(radius, freq_range)

        # define colormap range
        if clim is None:
            clim = np.max([np.nanmax(np.abs(l_vert_mean)), np.nanmax(np.abs(r_vert_mean))])
        c_norm = plt.Normalize(vmin=-clim, vmax=clim)
        c_mappable = cmx.ScalarMappable(norm=c_norm, cmap=plt.get_cmap(cmap))

        # compute surface colors for left
        valid_l_inds = ~np.isnan(l_vert_mean)
        l_colors = np.full((l_vert_mean.shape[0], 4), 0.)
        l_colors[valid_l_inds] = c_mappable.to_rgba(l_vert_mean[valid_l_inds])

        # and right
        valid_r_inds = ~np.isnan(r_vert_mean)
        r_colors = np.full((r_vert_mean.shape[0], 4), 0.)
        r_colors[valid_r_inds] = c_mappable.to_rgba(r_vert_mean[valid_r_inds])

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

    def compute_surface_map(self, radius=12.5, freq_range=(40, 200)):
        """
        Returns brain surface activation maps based on averaging all subject t-statistics in a given frequency range.

        Parameters
        ----------
        radius: float
            Maximum distance between an electrode and a vertex to be counted as data for that vertex
        freq_range: list
            2 element list defining minimum and maximum frequncies to average.

        Returns
        -------
        left hemisphere and right hemisphere activation maps
        """

        # load average brain pial surface mesh
        l_coords, l_faces, r_coords, r_faces = self.load_brain_mesh('average')

        # initialize array to count whether a given face is near an electrode for each subject
        subjs = self.group_df.subject.unique()
        n_subjs = len(subjs)
        l_vert_mean = np.full((l_coords.shape[0], n_subjs), np.nan)
        r_vert_mean = np.full((r_coords.shape[0], n_subjs), np.nan)

        # mean t-stats over frequency range of interest
        freq_inds = (self.group_df.frequency >= freq_range[0]) & (self.group_df.frequency <= freq_range[1])
        mean_val_df = self.group_df[freq_inds].groupby(['subject', 'label', 'avg.x', 'avg.y', 'avg.z']).mean()
        mean_val_df = mean_val_df.reset_index()

        # loop over each subject.
        for i, subj in enumerate(subjs):
            subj_res = mean_val_df[mean_val_df.subject == subj]
            for col in ['avg.x', 'avg.y', 'avg.z']:
                subj_res[col] = subj_res[col].astype('float')

            if np.isnan(subj_res['avg.x'].iloc[0]):
                print('Skipping {}, missing electrode coordinates.'.format(subj))
                continue

            # make subject specific surface
            l_subj_verts = np.full((l_coords.shape[0], subj_res.shape[0]), np.nan)
            r_subj_verts = np.full((r_coords.shape[0], subj_res.shape[0]), np.nan)

            print('%s: finding valid vertices.' % subj)
            for e_num, (index, elec) in enumerate(subj_res[['avg.x', 'avg.y', 'avg.z', 't-stat']].iterrows()):
                l_subj_verts[np.linalg.norm(l_coords - elec.values[:3], axis=1) < radius, e_num] = elec['t-stat']
                r_subj_verts[np.linalg.norm(r_coords - elec.values[:3], axis=1) < radius, e_num] = elec['t-stat']

            l_vert_mean[:, i] = np.nanmean(l_subj_verts, axis=1)
            r_vert_mean[:, i] = np.nanmean(r_subj_verts, axis=1)
        return np.nanmean(l_vert_mean, axis=1), np.nanmean(r_vert_mean, axis=1)

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