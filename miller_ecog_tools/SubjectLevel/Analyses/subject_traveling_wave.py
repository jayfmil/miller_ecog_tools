import os
import re
import numpy as np
import pycircstat
import numexpr
import pandas as pd

from scipy.signal import hilbert
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

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

    # when band passing the EEG, don't allow the frequency range to go below this value
    LOWER_MIN_FREQ = 0.5

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectTravelingWaveAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'trav_waves.p'

        # when computing bandpass, plus/minus this number of frequencies
        self.hilbert_half_range = 1.5

        # time period on which to compute cluster statistics
        self.cluster_stat_start_time = 0
        self.cluster_stat_end_time = 1600

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__ + '_res')

    def analysis(self):
        """

        """

        # make sure we have data
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
            return

        # we must have 'clusters' in self.res
        if 'clusters' in self.res:
            self.res['traveling_waves'] = {}

            # get cluster names from dataframe columns
            cluster_names = list(filter(re.compile('cluster[0-9]+').match, self.res['clusters'].columns))

            # get circular-linear regression parameters
            theta_r, params = self.compute_grid_parameters()

            # compute cluster stats for each cluster
            for this_cluster_name in cluster_names:
                cluster_res = {}

                # get the names of the channels in this cluster
                cluster_elecs = self.res['clusters'][self.res['clusters'][this_cluster_name].notna()]['label']

                # for the channels in this cluster, bandpass and then hilbert to get the phase info
                phase_data, cluster_mean_freq = self.compute_hilbert_for_cluster(this_cluster_name)

                # reduce to only time inverval of interest
                time_inds = (phase_data.time >= self.cluster_stat_start_time) & (
                        phase_data.time <= self.cluster_stat_end_time)
                phase_data = phase_data[:, :, time_inds]

                # get electrode coordinates in 2d
                norm_coords = self.compute_2d_elec_coords(this_cluster_name)

                # run the cluster stats for time-averaged data
                mean_rel_phase = pycircstat.mean(phase_data.data, axis=2)
                mean_cluster_wave_ang, mean_cluster_wave_freq, mean_cluster_r2_adj = \
                    circ_lin_regress(mean_rel_phase.T, norm_coords, theta_r, params)
                cluster_res['mean_cluster_wave_ang'] = mean_cluster_wave_ang
                cluster_res['mean_cluster_wave_freq'] = mean_cluster_wave_freq
                cluster_res['mean_cluster_r2_adj'] = mean_cluster_r2_adj

                # and run it for each time point
                num_times = phase_data.shape[-1]
                data_as_list = zip(phase_data.T, [norm_coords] * num_times, [theta_r] * num_times, [params] * num_times)
                res_as_list = Parallel(n_jobs=12, verbose=5)(delayed(circ_lin_regress)(x[0].data, x[1], x[2], x[3])
                                                             for x in data_as_list)
                cluster_res['cluster_wave_ang'] = np.stack([x[0] for x in res_as_list], axis=0).astype('float32')
                cluster_res['cluster_wave_freq'] = np.stack([x[1] for x in res_as_list], axis=0).astype('float32')
                cluster_res['cluster_r2_adj'] = np.stack([x[2] for x in res_as_list], axis=0).astype('float32')
                cluster_res['mean_freq'] = cluster_mean_freq
                cluster_res['channels'] = cluster_elecs.values
                cluster_res['time'] = phase_data.time.data
                cluster_res['phase_data'] = pycircstat.mean(phase_data, axis=1).astype('float32')
                self.res['traveling_waves'][this_cluster_name] = cluster_res

        else:
            print('{}: self.res must have a clusters entry before running.'.format(self.subject))
            return

    def compute_grid_parameters(self):
        """
        Angle and phase offsets over which to compute the traveling wave statistics. Consider making these
        modifiable.
        """

        thetas = np.radians(np.arange(0, 356, 5))
        rs = np.radians(np.arange(0, 18.1, .5))
        theta_r = np.stack([(x, y) for x in thetas for y in rs])
        params = np.stack([theta_r[:, 1] * np.cos(theta_r[:, 0]), theta_r[:, 1] * np.sin(theta_r[:, 0])], -1)
        return theta_r, params

    def compute_hilbert_for_cluster(self, this_cluster_name):

        # first, get the eeg for just channels in cluster
        cluster_rows = self.res['clusters'][this_cluster_name].notna()
        cluster_elec_labels = self.res['clusters'][cluster_rows]['label']
        cluster_eeg = self.subject_data[:, np.in1d(self.subject_data.channel, cluster_elec_labels)]

        # bandpass eeg at the mean frequency, making sure the lower frequency isn't too low
        cluster_mean_freq = self.res['clusters'][cluster_rows][this_cluster_name].mean()
        cluster_freq_range = [cluster_mean_freq - self.hilbert_half_range, cluster_mean_freq + self.hilbert_half_range]
        if cluster_freq_range[0] < SubjectTravelingWaveAnalysis.LOWER_MIN_FREQ:
            cluster_freq_range[0] = SubjectTravelingWaveAnalysis.LOWER_MIN_FREQ
        phase_data = RAM_helpers.band_pass_eeg(cluster_eeg, cluster_freq_range)
        phase_data = phase_data.transpose('channel', 'event', 'time')
        phase_data.data = np.angle(hilbert(phase_data.data, N=phase_data.shape[-1], axis=-1))

        # compute mean phase and phase difference between ref phase and each electrode phase
        ref_phase = pycircstat.mean(phase_data.data, axis=0)
        phase_data.data = pycircstat.cdiff(phase_data.data, ref_phase)
        return phase_data, cluster_mean_freq

    def compute_2d_elec_coords(self, this_cluster_name):

        # compute PCA of 3d electrode coords to get 2d coords
        cluster_rows = self.res['clusters'][this_cluster_name].notna()
        xyz = self.res['clusters'][cluster_rows][['x', 'y', 'z']].values
        xyz -= np.mean(xyz, axis=0)
        pca = PCA(n_components=3)
        norm_coords = pca.fit_transform(xyz)[:, :2]
        return norm_coords


def circ_lin_regress(phases, coords, theta_r, params):
    """

    """

    n = phases.shape[1]
    pos_x = np.expand_dims(coords[:, 0], 1)
    pos_y = np.expand_dims(coords[:, 1], 1)

    # compute predicted phases for angle and phase offset
    x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y

    # Compute resultant vector length. This is faster than calling pycircstat.resultant_vector_length
    x1 = numexpr.evaluate('sum(cos(x) / n, axis=1)')
    x1 = numexpr.evaluate('x1 ** 2')
    x2 = numexpr.evaluate('sum(sin(x) / n, axis=1)')
    x2 = numexpr.evaluate('x2 ** 2')
    Rs = numexpr.evaluate('-sqrt(x1 + x2)')

    # for each time and event, find the parameters with the smallest -R
    min_vals = theta_r[np.argmin(Rs, axis=1)]

    sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
    offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
                      np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)

    # compute circular correlation coefficient between actual phases and predicited phases
    circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)

    # compute adjusted r square
    r2_adj = circ_corr_coef ** 2

    wave_ang = min_vals[:, 0]
    wave_freq = min_vals[:, 1]

    return wave_ang, wave_freq, r2_adj

