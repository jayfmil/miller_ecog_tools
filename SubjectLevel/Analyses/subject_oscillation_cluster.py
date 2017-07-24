"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec
import ram_data_helpers
from sklearn import linear_model
import statsmodels.api as sm
from scipy.stats import ttest_ind, sem
from copy import deepcopy
from SubjectLevel.subject_analysis import SubjectAnalysis
from tarjan import tarjan
import itertools
# from SubjectLevel.Analyses import subject_SME
# from SubjectLevel.Analyses.subject_SME import SubjectSME as SME
from SubjectLevel.par_funcs import par_find_peaks_by_ev
from scipy.spatial.distance import pdist, squareform
from scipy.signal import argrelmax, hilbert
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import pdb
import pycircstat
import numexpr




class SubjectElecCluster(SubjectAnalysis):
    """

    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectAnalysis, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

        self.task_phase_to_use = ['enc']  # ['enc'] or ['rec']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled_norm
        self.rec_thresh = None

        # string to use when saving results files
        self.res_str = 'elec_cluster.p'

        # default frequency settings
        self.feat_type = 'power'
        self.freqs = np.logspace(np.log10(2), np.log10(32), 129)

        # window size to find clusters (in Hz)
        self.cluster_freq_range = 2.

        # spatial distance considered near
        self.near_dist = 15.

        # number of electrodes needed to be considered a clust
        self.min_num_elecs = 4

        self.clusters = {}


    def analysis(self):
        """

        """



        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # compute frequency bins
        window_centers = np.arange(self.freqs[0], self.freqs[-1] + .001, 1)
        windows = [(x - self.cluster_freq_range / 2., x + self.cluster_freq_range / 2.) for x in window_centers]
        window_bins = np.stack([(self.freqs >= x[0]) & (self.freqs <= x[1]) for x in windows], axis=0)

        # distance matrix for all electrodes
        elec_dists = squareform(pdist(np.stack(self.elec_xyz_indiv)))
        near_adj_matr = (elec_dists < 15.) & (elec_dists > 0.)

        # noramlize power spectra
        p_spect = deepcopy(self.subject_data)
        p_spect = self.normalize_spectra(p_spect)

        # find cluters using mean power spectra
        mean_p_spect = p_spect.mean(dim='events')
        peaks = par_find_peaks_by_ev(mean_p_spect)
        self.clusters = self.find_clusters_from_peaks([peaks], near_adj_matr, window_bins, window_centers)

        thetas = np.radians(np.arange(0, 356, 5))
        rs = np.radians(np.arange(0, 18.1, .5))
        theta_r = np.stack([(x, y) for x in thetas for y in rs])
        params = np.stack([theta_r[:, 1] * np.cos(theta_r[:, 0]), theta_r[:, 1] * np.sin(theta_r[:, 0])], -1)

        # for electrode in each cluster, compute phase at the mean cluster frequency
        for freq in np.sort(self.clusters.keys()):
            for cluster_count, cluster_elecs in enumerate(self.clusters[freq]['elecs']):
                cluster_freq = self.clusters[freq]['mean_freqs'][cluster_count]

                cluster_elecs_mono = np.unique(
                    list((itertools.chain(*self.subject_data['bipolar_pairs'][cluster_elecs].data))))
                cluster_elecs_bipol = np.array(self.subject_data['bipolar_pairs'][cluster_elecs].data,
                                               dtype=[('ch0', '|S3'), ('ch1', '|S3')]).view(np.recarray)

                # make plust minus range a parameter. make buffer a parameter
                cluster_ts = self.load_eeg(self.subject_data.events.data.view(np.recarray), cluster_elecs_mono,
                                           cluster_elecs_bipol, self.start_time[0], self.end_time[0], 1.0,
                                           pass_band=[cluster_freq-1.5, cluster_freq+1.5])
                cluster_ts = cluster_ts.resampled(250)
                cluster_ts.data = np.angle(hilbert(cluster_ts, N=cluster_ts.shape[-1], axis=-1))
                cluster_ts = cluster_ts.remove_buffer(1.0)

                xyz = np.stack(self.elec_xyz_indiv[cluster_elecs], 0)
                xyz -= np.mean(xyz, axis=0)
                pca = PCA(n_components=3)
                norm_coords = pca.fit_transform(xyz)[:, :2]

                # for each timepoint, for each event
                # remove this loop too?????? Nope too much memory
                self.cluster_ts = cluster_ts
                self.norm_coords = norm_coords
                self.theta_r = theta_r
                self.params = params
                # return
                for cluster_this_time in tqdm(cluster_ts.T):
                    wave_ang, wave_freq, r2_adj = self.circ_lin_regress(cluster_this_time.data, norm_coords, theta_r,
                                                                        params)
                    # for cluster_this_ev in cluster_this_time:
                    #     wave_ang, wave_freq, r2_adj = self.circ_lin_regress(cluster_this_ev.data, norm_coords, theta_r, params)
                pdb.set_trace()






        # rec_p_spect = p_spect[recalled].mean(dim='events')
        # rec_peaks = par_find_peaks_by_ev(rec_p_spect)
        # self.rec_clusters = self.find_clusters_from_peaks([rec_peaks], near_adj_matr, window_bins, window_centers)

        # nrec_p_spect = p_spect.mean(dim='events')
        # nrec_peaks = par_find_peaks_by_ev(nrec_p_spect)
        # self.nrec_clusters = self.find_clusters_from_peaks([nrec_peaks], near_adj_matr, window_bins, window_centers)

        #
        # print('%s: finding peaks for %d events and %d electrodes.' % (self.subj, self.subject_data.shape[0], self.subject_data.shape[2]))
        #
        # if self.pool is None:
        #     peaks = map(par_find_peaks_by_ev, tqdm(self.subject_data))
        # else:
        #     peaks = self.pool.map(par_find_peaks_by_ev, self.subject_data)
        # peaks = np.stack(peaks, axis=0)
        #
        #
        #
        # # loop over each event and 1) for each electrode, determine if there is a peak in a given bin; 2) after
        # # computing the binned peaks for all electrodes, count the number of electrodes with peaks in each bin and find
        # # local maxima; 3) for each frequency at a local max, find clusters using .near_dist threshold and tarjan's
        # # algo.
        #
        # # binned_peak_by_elec = np.zeros((peaks.shape[0], len(windows), peaks.shape[2])).astype(bool)
        # # peak_count_by_freqs = np.zeros((peaks.shape[0], len(windows)))
        #
        # # what analyses do we want to do.
        # # simplest, is there a cluster for an event and frequency bin
        # clust_count =  np.zeros((peaks.shape[0], len(windows)))
        # self.find_clusters_from_peaks(peaks, near_adj_matr, window_bins)
        # self.clust_count = clust_count

    def circ_lin_regress(self, phases, coords, theta_r, params):
        """
        Performs 2D circular linear regression.

        This is ported from Honghui's matlab code. To be honest, I'm not totally sure what it's doing but whatever..

        :param phases:
        :param coords:
        :return:
        """

        n = phases.shape[1]
        pos_x = np.expand_dims(coords[:, 0], 1)
        pos_y = np.expand_dims(coords[:, 1], 1)

        x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y
        # Rs_test = -np.sqrt(np.square(np.sum(np.cos(x) / n, axis=1)) + np.square(np.sum(np.sin(x) / n, axis=1)))

        # x = np.expand_dims(phases, 2)
        # y = params[:, 0] * pos_x - params[:, 1] * pos_y
        # x = numexpr.evaluate('x - y')
        x1 = numexpr.evaluate('sum(cos(x) / n, axis=1)')
        x1 = numexpr.evaluate('x1 ** 2')
        x2 = numexpr.evaluate('sum(sin(x) / n, axis=1)')
        x2 = numexpr.evaluate('x2 ** 2')
        Rs = numexpr.evaluate('-sqrt(x1 + x2)')

        min_vals = theta_r[np.argmin(Rs, axis=1)]

        sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
        offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
                          np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
        pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)

        # compute circular correlation coefficient between actual phases and predicited phases
        circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)

        # compute adjusted r square
        r2_adj = 1 - ((1 - circ_corr_coef ** 2) * (n - 1)) / (n - 4)
        wave_ang = min_vals[:, 0]
        wave_freq = min_vals[:, 1]

        # phase_mean = np.mod(np.angle(np.sum(np.exp(1j * phases)) / len(phases)), 2 * np.pi)
        # pos_circ_mean = np.mod(np.angle(np.sum(np.exp(1j * pos_circ)) / len(phases)), 2 * np.pi)

        # cc = np.sum(np.sin(phases - phase_mean) * np.sin(pos_circ - pos_circ_mean)) / \
        #      np.sqrt(np.sum(np.sin(phases - phase_mean) ** 2) * np.sum(np.sin(pos_circ - pos_circ_mean) ** 2))
        return wave_ang, wave_freq, r2_adj







    def find_clusters_from_peaks(self, peaks, near_adj_matr, window_bins, window_centers):

        all_clusters = {k: {'elecs': [], 'mean_freqs': []} for k in window_centers}
        for i, ev in enumerate(peaks):

            # bin peaks, count them up, and find the peaks (of the peaks...)
            binned_peaks = np.stack([np.any(ev[x], axis=0) for x in window_bins], axis=0)
            peak_freqs = argrelmax(binned_peaks.sum(axis=1))[0]

            # for each peak frequency, identify clusters
            for this_peak_freq in peak_freqs:
                near_this_ev = near_adj_matr.copy()
                near_this_ev[~binned_peaks[this_peak_freq]] = False
                near_this_ev[:, ~binned_peaks[this_peak_freq]] = False

                # use targan algorithm to find the clusters
                graph = {}
                for elec, row in enumerate(near_this_ev):
                    graph[elec] = np.where(row)[0]
                clusters = tarjan(graph)

                # only keep clusters with enough electrodes
                good_clusters = np.array([len(x) for x in clusters]) >= self.min_num_elecs
                for good_cluster in np.where(good_clusters)[0]:

                    # store all eelctrodes in the cluster
                    all_clusters[window_centers[this_peak_freq]]['elecs'].append(clusters[good_cluster])

                    # find mean frequency of cluster, first taking the mean freq within each electrode and then across
                    mean_freqs = []
                    for elec in ev[window_bins[this_peak_freq]][:, clusters[good_cluster]].T:
                        mean_freqs.append(np.mean(self.freqs[window_bins[this_peak_freq]][elec]))
                    all_clusters[window_centers[this_peak_freq]]['mean_freqs'].append(np.mean(mean_freqs))

        return dict((k, v) for k, v in all_clusters.items() if all_clusters[k]['elecs'])

    # def

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in self.task_phase_to_use:
                task_mask = self.task_phase == phase

                m = np.mean(X[sess_event_mask & task_mask], axis=1)
                m = np.mean(m, axis=0)
                s = np.std(X[sess_event_mask & task_mask], axis=1)
                s = np.mean(s, axis=0)
                X[sess_event_mask & task_mask] = (X[sess_event_mask & task_mask] - m) / s
        return X













