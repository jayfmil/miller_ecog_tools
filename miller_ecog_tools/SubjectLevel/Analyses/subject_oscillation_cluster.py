import os
import numpy as np
import pandas as pd

from tarjan import tarjan
from scipy.spatial.distance import pdist, squareform

# bunch of matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for brain plotting
import nilearn.plotting as ni_plot

from miller_ecog_tools.SubjectLevel.par_funcs import par_find_peaks_by_chan, my_local_max
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_eeg_data import SubjectEEGData


class SubjectOscillationClusterAnalysis(SubjectAnalysisBase, SubjectEEGData):
    """
    Subclass of SubjectAnalysis and SubjectEEGData that identifies clusters of electrodes in a subject that
    exhibit peaks in the power spectrum at similar frequencies.
    """

    res_str_tmp = 'elec_cluster_%d_mm_%d_elec_min_%s_elec_type_%s_sep_hemis_%.2f_cluster_range.p'
    attrs_in_res_str = ['elec_types_allowed', 'min_elec_dist', 'min_num_elecs', 'separate_hemis', 'cluster_freq_range']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectOscillationClusterAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = SubjectOscillationClusterAnalysis.res_str_tmp

        # default frequency settings for identifying peaks
        self.freqs = np.logspace(np.log10(3), np.log10(40), 200)
        self.bipolar = False
        self.start_time = 0
        self.end_time = 1600
        self.mono_avg_ref = True

        # standard deviation theshhold for detecting peaks
        self.std_thresh = 0.5

        # window size to find clusters (in Hz)
        self.cluster_freq_range = 2.

        # D: depths, G: grids, S: strips
        self.elec_types_allowed = ['D', 'G', 'S']

        # spatial distance considered near (mm)
        self.min_elec_dist = 20.

        # If True, osciallation clusters can't cross hemispheres
        self.separate_hemis = True

        # number of electrodes needed to be considered a cluster
        self.min_num_elecs = 4

        # elec_info column from which to extract x,y,z coordinates
        self.elec_pos_column = 'ind.'

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        Identifies clusters of electrodes that have oscillations at similar frequencies, by:

        1. Determining the distance between all pairs of electrodes
        2. Computing the mean power spectra of each electrode
        3. Identifying at which each electrode has peaks over the 1/f background power
        4. Identifying clusters based on being spatially contiguous, as defined by spatial distance (.min_elec_dist),
           electrode type (.elec_types_allowed), hemisphere (.separate_hemis), and frequency contiguous
           (.cluster_freq_range). This uses the "targan" algorithm.
        """

        # make sure we have data
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
            return

        # make distance matrix for all electrodes. If separating the hemispheres, move the hemispheres far apart
        xyz = self._get_elec_xyz()
        if self.separate_hemis:
            xyz[xyz[:, 0] < 0, 0] -= 100
        elec_dists = squareform(pdist(xyz))

        # figure out which pairs of electrodes are closer than the threshold
        near_adj_matr = (elec_dists < self.min_elec_dist) & (elec_dists > 0.)
        allowed_elecs = np.array([e in self.elec_types_allowed for e in self.elec_info['type']])

        # normalized power spectra
        p_spect = self.normalize_power_spectrum()

        # Compute mean power spectra across events, and then find where each electrode has peaks
        mean_p_spect = np.mean(p_spect, axis=self.subject_data.get_axis_num('event'))
        peaks = par_find_peaks_by_chan([mean_p_spect, self.freqs, self.std_thresh])

        # now that we know at which each electrode has peaks, compute clusters of electrodes that exhibit peaks at
        # similar frequencies and are close enough together
        self.res['clusters'] = self.find_clusters_from_peaks(peaks, near_adj_matr, allowed_elecs)

    def find_clusters_from_peaks(self, peaks, near_adj_matr, allowed_elecs):
        """
        Given a a frequency by channel array, use the tarjan algorithm to identify clusters of electrodes.

        Parameters
        ----------
        peaks: numpy.ndarray
            frequency x channel boolean array
        near_adj_matr: numpy.ndarray
            square boolean array indicating whether any two electrodes are considered to be near each other
        allowed_elecs: numpy.array
            boolean array the same length as the number of electrodes, indicating whether an electrode can be included
            or should be automatically excluded

        Returns
        -------
        pandas.DataFrame with a row for each electrode and a olumn for each cluster, named cluster1, cluster2, ...
        The value indicates the frequency of the peak for that electrode. NaN means no peak/not in cluster.
        """

        # compute frequency bins
        window_centers = np.arange(self.freqs[0], self.freqs[-1] + .001, 1)
        windows = [(x - self.cluster_freq_range / 2., x + self.cluster_freq_range / 2.) for x in window_centers]
        window_bins = np.stack([(self.freqs >= x[0]) & (self.freqs <= x[1]) for x in windows], axis=0)

        # make sure only electrodes of allowed types are included
        peaks[:, ~allowed_elecs] = False

        # bin peaks, count them up, and find the peaks (of the peaks...)xw
        binned_peaks = np.stack([np.any(peaks[x], axis=0) for x in window_bins], axis=0)
        peak_freqs = my_local_max(binned_peaks.sum(axis=1))

        # for each peak frequency, identify clusters
        cluster_count = 0
        df_list = []
        for this_peak_freq in peak_freqs:
            near_this_peak_freq = near_adj_matr.copy()

            # This is leaving in only electrodes with a peak at this freq?
            near_this_peak_freq[~binned_peaks[this_peak_freq]] = False
            near_this_peak_freq[:, ~binned_peaks[this_peak_freq]] = False

            # use targan algorithm to find the clusters
            graph = {}
            for elec, row in enumerate(near_this_peak_freq):
                graph[elec] = np.where(row)[0]
            clusters = tarjan(graph)

            # only keep clusters with enough electrodes
            good_clusters = np.array([len(x) for x in clusters]) >= self.min_num_elecs
            for good_cluster in np.where(good_clusters)[0]:
                cluster_count += 1

                # store all electrodes in the cluster
                col_name = 'cluster{}'.format(cluster_count)
                cluster_df = pd.DataFrame(data=np.full(shape=(peaks.shape[1]), fill_value=np.nan), columns=[col_name])

                # find mean frequency of cluster, first taking the mean freq within each electrode and then across
                mean_freqs = []
                for elec in peaks[window_bins[this_peak_freq]][:, clusters[good_cluster]].T:
                    mean_freqs.append(np.mean(self.freqs[window_bins[this_peak_freq]][elec]))
                cluster_df.iloc[clusters[good_cluster], 0] = mean_freqs
                df_list.append(cluster_df)

        # also add some useful info to the table. x,y,z and electrode name
        df = None
        if df_list:
            df = pd.concat(df_list, axis='columns')
            x, y, z = self._get_elec_xyz().T
            df['x'] = x
            df['y'] = y
            df['z'] = z
            df['label'] = self.elec_info['label']

        # return df with column for each cluster
        return df

    def find_clusters_from_peaks2(self, peaks, near_adj_matr, allowed_elecs):
        """
        Honghui's more advanced clustering algo.
        """

        steplength = 5
        windowLength = 15
        distance = 25
        CLUSTERS = []
        # allowed_elecs = np.array([e in self.elec_types_allowed for e in self.e_type])
        # xyz_tmp = np.stack(self.elec_xyz_avg[allowed_elecs])
        # elec_dists = squareform(pdist(xyz_tmp))
        # near_adj_matr = (elec_dists < 25) & (elec_dists > 0)
        peaks[:, ~allowed_elecs] = False
        peakid = peaks
        # peak,peakid=par_find_peaks_by_ev(self.subject_data[:,:,allowed_elecs].mean(dim='events'))
        windows = np.stack([[i, windowLength + i] for i in range(0, 201 - steplength, steplength)])
        for ire in range(10):
            peak_counts = np.sum(peakid, axis=1)
            window_counts = np.array([sum(peak_counts[w[0]:w[1]]) for w in windows])
            peak_window = np.argmax(window_counts)
            near_this_ev = near_adj_matr.copy()
            peak_within_window = np.any(peakid[windows[peak_window, 0]:windows[peak_window, 1]], axis=0)
            near_this_ev[~peak_within_window, :] = False
            near_this_ev[:, ~peak_within_window] = False
            # use targan algorithm to find the clusters
            graph = {}
            for elec, row in enumerate(near_this_ev):
                graph[elec] = np.where(row)[0]
            groups = tarjan(graph)
            # choose the connected componet with most electrodes as seed
            clusterSeed = sorted(sorted(groups, key=lambda a: -len(a))[0])
            # make it a dictionary
            window_true = np.zeros((200), dtype=bool)
            window_true[windows[peak_window, 0]:windows[peak_window, 1]] = True
            cluster = {}
            if len(clusterSeed) > 1:
                for i in clusterSeed:
                    peak_freq = np.squeeze(np.where(np.logical_and(peakid[:, i], window_true))[0][0])
                    cluster[i] = peak_freq
                    peakid[peak_freq, i] = False
            if len(cluster) > 1:
                for ire2 in range(10):
                    for i in range(len(near_adj_matr)):
                        if i not in cluster:
                            near_freqS = np.squeeze(list(cluster.values()))[near_adj_matr[i, list(cluster.keys())]]
                            if len(near_freqS) > 1:
                                window_true = np.zeros((200), dtype=bool)
                                window_true[windows[peak_window, 0]:windows[peak_window, 1]] = True
                                near_freq = int(np.median(near_freqS))
                                electrode_frequency = np.where(peakid[:, i])[0]
                                if np.any(np.abs(electrode_frequency - near_freq) < 15):
                                    peak_freq = np.array(min(electrode_frequency, key=lambda x: abs(x - near_freq)))
                                    cluster[i] = peak_freq
                                    peakid[peak_freq, i] = False
                CLUSTERS.append(cluster)
        for i in CLUSTERS:
            for j in i:
                i[j] = self.freqs[i[j]]
        res = {}
        i = 0
        while i < (len(CLUSTERS)):
            if len(CLUSTERS[i]) > 3:
                res[i] = CLUSTERS[i]
            i += 1
        cluster_count = 0
        df_list = []

        for i in res.keys():
            cluster_count += 1
            col_name = 'cluster{}'.format(cluster_count)
            cluster_df = pd.DataFrame(data=np.full(shape=(peaks.shape[1]), fill_value=np.nan), columns=[col_name])
            for j in res[i]:
                cluster_df.iloc[j] = res[i][j]
            df_list.append(cluster_df)
        df = None
        if df_list:
            df = pd.concat(df_list, axis='columns')
            x, y, z = self._get_elec_xyz().T
            df['x'] = x
            df['y'] = y
            df['z'] = z
            df['label'] = self.elec_info['label']
        return df

    def plot_cluster_freqs_on_brain(self, cluster_name, xyz, colormap='viridis', vmin=None, vmax=None, do_3d=False):
        """
        Plot the frequencies of single electrode cluster on either a 2d or interactive 2d brain.

        Parameters
        ----------
        cluster_name: str
            Name of column in self.res['clusters']
        xyz: np.ndarray
            3 x n array of electrode locations. Should be in MNI space.
        colormap: str
            matplotlib colormap name
        vmin: float
            lower limit of colormap values. If not given, lowest value in frequency column will be used
        vmax: float
            upper limit of colormap values. If not given, highest value in frequency column will be used
        do_3d:
            Whether to plot an interactive 3d brain, or a 2d brain

        Returns
        -------
        If 2d, returns the matplotlib figure. If 3d, returns the html used to render the brain.
        """

        # get frequecies for this cluster
        freqs = self.res['clusters'][cluster_name].values

        # get color for each frequency. Start with all black
        colors = np.stack([[0., 0., 0., 0.]] * len(freqs))

        # fill in colors of electrodes with defined frequencies
        cm = plt.get_cmap(colormap)
        cNorm = clrs.Normalize(vmin=np.nanmin(freqs) if vmin is None else vmin,
                               vmax=np.nanmax(freqs) if vmax is None else vmax)
        colors[~np.isnan(freqs)] = cmx.ScalarMappable(norm=cNorm, cmap=cm).to_rgba(freqs[~np.isnan(freqs)])

        # if plotting 2d, use the nilearn glass brain
        if not do_3d:
            fig, ax = plt.subplots()
            ni_plot.plot_connectome(np.eye(xyz.shape[0]), xyz,
                                    node_kwargs={'alpha': 0.7, 'edgecolors': None},
                                    node_size=60, node_color=colors, display_mode='lzr',
                                    axes=ax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='4%', pad=0)
            cb1 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                            norm=cNorm,
                                            orientation='horizontal')
            cb1.set_label('Frequency', fontsize=20)
            cb1.ax.tick_params(labelsize=14)
            fig.set_size_inches(15, 10)
            return fig

        # if plotting 3d use the nilearn 3d brain. Unfortunately this doesn't add a colorbar.
        else:
            # you need to return the html. If in jupyter, it will automatically render
            return ni_plot.view_markers(xyz, colors=colors, marker_size=6)

    @staticmethod
    def tal2mni(xyz):
        """
        Converts coordinates in talairach space to MNI space.

        Parameters
        ----------
        xyz: np.ndarray
            3 x n array of electrode locations.

        Returns
        -------
        3 x n array of electrodes locations in MNI space

        Credit: nibabel.affines.apply_affine
        """

        def transform(affine, xyz):
            shape = xyz.shape
            xyz = xyz.reshape((-1, xyz.shape[-1]))
            rzs = affine[:-1, :-1]
            trans = affine[:-1, -1]
            res = np.dot(xyz, rzs.T) + trans[None, :]
            return res.reshape(shape)

        pos_z = np.array([[0.9900, 0, 0, 0],
                          [0, 0.9688, 0.0460, 0],
                          [0, -0.0485, 0.9189, 0],
                          [0, 0, 0, 1.0000]])
        neg_z = np.array([[0.9900, 0, 0, 0],
                          [0, 0.9688, 0.0420, 0],
                          [0, -0.0485, 0.8390, 0],
                          [0, 0, 0, 1.0000]])

        mni_coords = np.zeros(xyz.shape)
        mni_coords[xyz[:, 2] > 0] = transform(np.linalg.inv(pos_z), xyz[xyz[:, 2] > 0])
        mni_coords[xyz[:, 2] <= 0] = transform(np.linalg.inv(neg_z), xyz[xyz[:, 2] <= 0])
        return mni_coords

    def _get_elec_xyz(self):
        if '{}{}'.format(self.elec_pos_column, 'x') in self.elec_info:
            xyz = self.elec_info[['{}{}'.format(self.elec_pos_column, coord) for coord in ['x', 'y', 'z']]].values
        else:
            print('{}: {} column not in elec_locs, defaulting to x, y, and z.'.format(self.subject, self.elec_pos_column))
            xyz = self.elec_info[[coord for coord in ['x', 'y', 'z']]].values
        return xyz

    # automatically set the .res_str based on the class attributes
    @property
    def min_elec_dist(self):
        return self._min_elec_dist

    @min_elec_dist.setter
    def min_elec_dist(self, t):
        self._min_elec_dist = t
        self.set_res_str()

    @property
    def elec_types_allowed(self):
        return self._elec_types_allowed

    @elec_types_allowed.setter
    def elec_types_allowed(self, t):
        self._elec_types_allowed = t
        self.set_res_str()

    @property
    def min_num_elecs(self):
        return self._min_num_elecs

    @min_num_elecs.setter
    def min_num_elecs(self, t):
        self._min_num_elecs = t
        self.set_res_str()

    @property
    def separate_hemis(self):
        return self._separate_hemis

    @separate_hemis.setter
    def separate_hemis(self, t):
        self._separate_hemis = t
        self.set_res_str()

    @property
    def cluster_freq_range(self):
        return self._cluster_freq_range

    @cluster_freq_range.setter
    def cluster_freq_range(self, t):
        self._cluster_freq_range = t
        self.set_res_str()

    def set_res_str(self):
        if np.all([hasattr(self, x) for x in SubjectOscillationClusterAnalysis.attrs_in_res_str]):
            self.res_str = SubjectOscillationClusterAnalysis.res_str_tmp % (self.min_elec_dist,
                                                                            self.min_num_elecs,
                                                                            '_'.join(self.elec_types_allowed),
                                                                            self.separate_hemis,
                                                                            self.cluster_freq_range)