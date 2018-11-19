import os
import numpy as np
import pandas as pd

from tarjan import tarjan
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert

# bunch of matplotlib stuff
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for brain plotting
import nilearn.plotting as ni_plot

from miller_ecog_tools.SubjectLevel.par_funcs import par_find_peaks_by_chan, my_local_max
from ptsa.data.filters import MorletWaveletFilter
from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_ram_eeg_data import SubjectRamEEGData


class SubjectTravelingWaveAnalysis(SubjectAnalysisBase, SubjectRamEEGData):
    """
    Subclass of SubjectAnalysis and SubjectRamEEGData.

    Meant to be run as a SubjectAnalysisPipeline with SubjectOscillationClusterAnalysis as the first step.
    Uses the results of SubjectOscillationClusterAnalysis to compute traveling wave statistics on the clusters.
    """

    res_str_tmp = 'elec_cluster_%d_mm_%d_elec_min_%s_elec_type_%s_sep_hemis_%.2f_cluster_range.p'
    attrs_in_res_str = ['elec_types_allowed', 'min_elec_dist', 'min_num_elecs', 'separate_hemis', 'cluster_freq_range']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectTravelingWaveAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        ## UPDATE
        self.res_str = SubjectTravelingWaveAnalysis.res_str_tmp

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """

        """

        # make sure we have data
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
            return

        # we must have 'clusters' in self.res
        if 'clusters' in self.res:
            # do stuff
            pass
        else:
            print('{}: self.res must have a clusters entry before running.'.format(self.subject))









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
        if np.all([hasattr(self, x) for x in SubjectTravelingWaveAnalysis.attrs_in_res_str]):
            self.res_str = SubjectTravelingWaveAnalysis.res_str_tmp % (self.min_elec_dist,
                                                                            self.min_num_elecs,
                                                                            '_'.join(self.elec_types_allowed),
                                                                            self.separate_hemis,
                                                                            self.cluster_freq_range)