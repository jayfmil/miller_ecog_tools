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
from scipy.stats.mstats import zscore
from copy import deepcopy
from SubjectLevel.subject_analysis import SubjectAnalysis
from tarjan import tarjan
import itertools
# from SubjectLevel.Analyses import subject_SME
# from SubjectLevel.Analyses.subject_SME import SubjectSME as SME
from xarray import concat
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
import os



class SubjectElecCluster(SubjectAnalysis):
    """

    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectElecCluster, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

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

        # plus/minus this value when computer hilbert phase
        self.hilbert_half_range = 1.5

        # number of electrodes needed to be considered a clust
        self.min_num_elecs = 5

        # number of permutations to compute null r-square distribution.
        # NOT CURRENTLY IMPLEMENTED
        self.num_perms = 100

        # dictionary will hold the cluter results
        self.res = {}

        self.do_compute_sme = False
        self.sme_bands = [[3, 8], [10, 14], [44, 100]]

        # should have an option to copmute the electrodes that comprise a traveling wave based on just the recalled
        # (or not recalled) trials? Or look for both?

    def run(self):
        """
        Convenience function to run all the steps for .
        """
        if self.feat_type != 'power':
            print('%s: .feat_type must be set to power for this analysis to run.' % self.subj)
            return

        # Step 1: load data
        if self.subject_data is None:
            self.load_data()

        # Step 2: create (if needed) directory to save/load
        self.make_res_dir()

        # Step 3: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 4: if not loaded ...
        if not self.res:

            # Step 4A: compute subsequenct memory effect at each electrode
            print('%s: Running oscillation cluster statistics.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """

        """



        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        eeg = None
        self.res['clusters'] = {}

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
        self.res['clusters'] = self.find_clusters_from_peaks([peaks], near_adj_matr, window_bins, window_centers)

        thetas = np.radians(np.arange(0, 356, 5))
        rs = np.radians(np.arange(0, 18.1, .5))
        theta_r = np.stack([(x, y) for x in thetas for y in rs])
        params = np.stack([theta_r[:, 1] * np.cos(theta_r[:, 0]), theta_r[:, 1] * np.sin(theta_r[:, 0])], -1)

        # for each frequency with clusters
        for freq in np.sort(self.res['clusters'].keys()):
        # for freq in [9.0]:
            self.res['clusters'][freq]['cluster_wave_ang'] = []
            self.res['clusters'][freq]['cluster_wave_freq'] = []
            self.res['clusters'][freq]['cluster_r2_adj'] = []
            self.res['clusters'][freq]['mean_cluster_wave_ang'] = []
            self.res['clusters'][freq]['mean_cluster_wave_freq'] = []
            self.res['clusters'][freq]['mean_cluster_r2_adj'] = []
            self.res['clusters'][freq]['coords'] = []
            self.res['clusters'][freq]['phase_ts'] = []

            # for each cluster at this frequency
            for cluster_count, cluster_elecs in enumerate(self.res['clusters'][freq]['elecs']):
                cluster_freq = self.res['clusters'][freq]['mean_freqs'][cluster_count]

                # only compute for electrodes in this cluster
                if self.bipolar:
                    cluster_elecs_mono = np.unique(
                        list((itertools.chain(*self.subject_data['bipolar_pairs'][cluster_elecs].data))))
                    cluster_elecs_bipol = np.array(self.subject_data['bipolar_pairs'][cluster_elecs].data,
                                                   dtype=[('ch0', '|S3'), ('ch1', '|S3')]).view(np.recarray)

                    # load eeg and filter in the frequency band of interest
                    cluster_ts = self.load_eeg(self.subject_data.events.data.view(np.recarray), cluster_elecs_mono,
                                               cluster_elecs_bipol, self.start_time[0], self.end_time[0], 1.0,
                                               pass_band=[cluster_freq-self.hilbert_half_range, cluster_freq+self.hilbert_half_range])
                    cluster_ts = cluster_ts.resampled(250)

                # if doing monopolar, which apparently is much better for this analysis, we are going to load eeg for
                # all channels first (not just cluster channels) and do an average re-reference.
                else:

                    if eeg is None:
                        print('%s: Loading EEG.' % self.subj)
                        eeg = []
                        uniq_sessions = np.unique(self.subject_data.events.data['session'])
                        for s, session in enumerate(uniq_sessions):
                            print('%s: Loading EEG session %d of %d.' % (self.subj, s+1, len(uniq_sessions)))
                            sess_inds = self.subject_data.events.data['session'] == session

                            sess_eeg = self.load_eeg(self.subject_data.events.data.view(np.recarray)[sess_inds],
                                                     self.subject_data['channels'].data,
                                                     None, self.start_time[0], self.end_time[0], 1.0)
                            sess_eeg = sess_eeg.resampled(250.)
                            eeg.append(sess_eeg)
                        eeg = concat(eeg, dim='events')

                    print('Band pass EEG')
                    cluster_ts = self.band_pass_eeg(eeg[cluster_elecs], [cluster_freq - self.hilbert_half_range,
                                                    cluster_freq + self.hilbert_half_range])

                print('Hilbert')
                cluster_ts.data = np.angle(hilbert(cluster_ts, N=cluster_ts.shape[-1], axis=-1))
                cluster_ts = cluster_ts.remove_buffer(1.0)
                print('Done Hilbert')
                self.res['time_ax'] = cluster_ts['time'].data

                # compute mean phase and phase difference between ref phase and each electrode phase
                ref_phase = pycircstat.mean(cluster_ts.data, axis=0)
                cluster_ts.data = pycircstat.cdiff(cluster_ts.data, ref_phase)

                # compute PCA of 3d electrode coords to get 2d coords
                xyz = np.stack(self.elec_xyz_indiv[cluster_elecs], 0)
                xyz -= np.mean(xyz, axis=0)
                pca = PCA(n_components=3)
                norm_coords = pca.fit_transform(xyz)[:, :2]
                self.res['clusters'][freq]['coords'].append(norm_coords)

                # compute mean cluster statistics
                mean_rel_phase = pycircstat.mean(cluster_ts.data, axis=2)
                mean_cluster_wave_ang, mean_cluster_wave_freq, mean_cluster_r2_adj = \
                    circ_lin_regress(mean_rel_phase.T, norm_coords, theta_r, params)
                self.res['clusters'][freq]['mean_cluster_wave_ang'].append(mean_cluster_wave_ang)
                self.res['clusters'][freq]['mean_cluster_wave_freq'].append(mean_cluster_wave_freq)
                self.res['clusters'][freq]['mean_cluster_r2_adj'].append(mean_cluster_r2_adj)

                # cluster_wave_ang = np.empty(cluster_ts.T.shape[:2])
                # cluster_wave_freq = np.empty(cluster_ts.T.shape[:2])
                # cluster_r2_adj = np.empty(cluster_ts.T.shape[:2])

                num_iters = cluster_ts.T.shape[0]
                data_as_list = zip(cluster_ts.T, [norm_coords]*num_iters, [theta_r]*num_iters, [params]*num_iters)
                res_as_list = Parallel(n_jobs=12, verbose=5)(delayed(circ_lin_regress)(x[0].data, x[1], x[2], x[3]) for x in tqdm(data_as_list))

                self.res['clusters'][freq]['cluster_wave_ang'].append(np.stack([x[0] for x in res_as_list], axis=0))
                self.res['clusters'][freq]['cluster_wave_freq'].append(np.stack([x[1] for x in res_as_list], axis=0))
                self.res['clusters'][freq]['cluster_r2_adj'].append(np.stack([x[2] for x in res_as_list], axis=0))
                self.res['clusters'][freq]['phase_ts'].append(cluster_ts.T.data)

                # for t, cluster_this_time in enumerate(tqdm(cluster_ts.T)):
                #     wave_ang, wave_freq, r2_adj = self.circ_lin_regress(cluster_this_time.data, norm_coords, theta_r,
                #                                                         params)
                #     cluster_wave_ang[t] = wave_ang
                #     cluster_wave_freq[t] = wave_freq
                #     cluster_r2_adj[t] = r2_adj
                # self.res['clusters'][freq]['cluster_wave_ang'].append(cluster_wave_ang)
                # self.res['clusters'][freq]['cluster_wave_freq'].append(cluster_wave_freq)
                # self.res['clusters'][freq]['cluster_r2_adj'].append(cluster_r2_adj)

                #### BIPOLAR DOES IT MATTER???????

                # compute shuffled r-square values. Perform on the mean relative phase across time
                # mean_rel_phase =  pycircstat.mean(cluster_ts.data, axis=2)
                # norm_coords_shuf = norm_coords.copy()
                # cluster_r2_adj_shufs = np.zeros(cluster_r2_adj.shape[-1])
                # for shuf in range(self.num_perms):
                #     np.random.shuffle(norm_coords_shuf)
                # mean_cluster_wave_ang, mean_cluster_wave_freq, mean_r2_adj = self.circ_lin_regress(mean_rel_phase,
                #                                                                                    norm_coords,
                #                                                                                    theta_r, params)
                #     pdb.set_trace()

        if self.res['clusters'] and self.do_compute_sme:
            # this will only work for monopolar for now.. maybe remove bipolar support from this code entirely
            ts = []
            ps = []
            for freq_range in self.sme_bands:
                band_eeg = self.band_pass_eeg(eeg, freq_range)
                band_eeg.data = np.log10(np.abs(hilbert(band_eeg, N=band_eeg.shape[-1], axis=-1)) ** 2)
                band_eeg = band_eeg.remove_buffer(1.0)

                X = band_eeg.mean(dim='time').data.T                
                X = self.normalize_power(X)
                ts_freq, ps_freq, = ttest_ind(X[recalled], X[~recalled])
                ts.append(ts_freq)
                ps.append(ps_freq)
            self.res['ts'] = np.stack(ts, -1)
            self.res['ps'] = np.stack(ps, -1)




        # rec_p_spect = p_spect[recalled].mean(dim='events')
        # rec_peaks = par_find_peaks_by_ev(rec_p_spect)
        # self.rec_clusters = self.find_clusters_from_peaks([rec_peaks], near_adj_matr, window_bins, window_centers)

        # nrec_p_spect = p_spect.mean(dim='events')
        # nrec_peaks = par_find_peaks_by_ev(nrec_p_spect)
        # self.nrec_clusters = self.find_clusters_from_peaks([nrec_peaks], near_adj_matr, window_bins, window_centers)


    # @staticmethod
    # def circ_lin_regress(phases, coords, theta_r, params):
    #     """
    #     Performs 2D circular linear regression.
    #
    #     This is ported from Honghui's matlab code. To be honest, I'm not totally sure what it's doing but whatever..
    #
    #     :param phases:
    #     :param coords:
    #     :return:
    #     """
    #
    #     n = phases.shape[1]
    #     pos_x = np.expand_dims(coords[:, 0], 1)
    #     pos_y = np.expand_dims(coords[:, 1], 1)
    #
    #     # compute predicted phases for angle and phase offset
    #     x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y
    #
    #     # Compute resultant vector length. This is faster than calling pycircstat.resultant_vector_length
    #     # now = time.time()
    #     x1 = numexpr.evaluate('sum(cos(x) / n, axis=1)')
    #     x1 = numexpr.evaluate('x1 ** 2')
    #     x2 = numexpr.evaluate('sum(sin(x) / n, axis=1)')
    #     x2 = numexpr.evaluate('x2 ** 2')
    #     Rs = numexpr.evaluate('-sqrt(x1 + x2)')
    #     # print(time.time() - now)
    #
    #     # this is slower
    #     # now = time.time()
    #     # Rs_new = -pycircstat.resultant_vector_length(x, axis=1)
    #     # tmp = np.abs(((np.exp(1j * x)).sum(axis=1) / n))
    #     # print(time.time() - now)
    #
    #     # this is basically the same as method 1
    #     # now = time.time()
    #     # tmp = numexpr.evaluate('sum(exp(1j * x), axis=1)')
    #     # tmp = numexpr.evaluate('abs(tmp) / n')
    #     # print(time.time() - now)
    #
    #     # for each time and event, find the parameters with the smallest -R (why are we taking the negative..)
    #     min_vals = theta_r[np.argmin(Rs, axis=1)]
    #
    #     sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
    #     offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
    #                       np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    #     pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)
    #
    #     # compute circular correlation coefficient between actual phases and predicited phases
    #     circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)
    #
    #     # compute adjusted r square
    #     r2_adj = circ_corr_coef ** 2
    #     # r2_adj = 1 - ((1 - circ_corr_coef ** 2) * (n - 1)) / (n - 4)
    #
    #     wave_ang = min_vals[:, 0]
    #     wave_freq = min_vals[:, 1]
    #
    #     # phase_mean = np.mod(np.angle(np.sum(np.exp(1j * phases)) / len(phases)), 2 * np.pi)
    #     # pos_circ_mean = np.mod(np.angle(np.sum(np.exp(1j * pos_circ)) / len(phases)), 2 * np.pi)
    #
    #     # cc = np.sum(np.sin(phases - phase_mean) * np.sin(pos_circ - pos_circ_mean)) / \
    #     #      np.sqrt(np.sum(np.sin(phases - phase_mean) ** 2) * np.sum(np.sin(pos_circ - pos_circ_mean) ** 2))
    #     return wave_ang, wave_freq, r2_adj
    #
    #
    #
    #
    #


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

    def plot_cluster_features_by_rec(self):

        edges = np.arange(0, 2 * np.pi + .2, np.pi / 10)
        angs = np.mean(np.stack([edges[1:], edges[:-1]]), axis=0)

        red = '#8c564b'
        blue = '#1f77b4'
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        for k in self.res['clusters'].keys():
            for clust_num in range(len(self.res['clusters'][k]['elecs'])):

                #             fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((2, 4), (0, 1), rowspan=2)
                ax3 = plt.subplot2grid((2, 4), (0, 2), polar=True)
                ax4 = plt.subplot2grid((2, 4), (1, 2), polar=True)
                ax5 = plt.subplot2grid((2, 4), (0, 3), polar=True)
                ax6 = plt.subplot2grid((2, 4), (1, 3), polar=True)
                #             plt.gcf().subplots_adjust(wspace=0.9)

                plt.suptitle('Freq: %.3f Hz, %d electrodes' % (
                self.res['clusters'][k]['mean_freqs'][clust_num], len(self.res['clusters'][k]['elecs'][clust_num])),
                             y=1.1)
                plt.tight_layout()

                # left panel, adjusted R^2
                r2_rec = self.res['clusters'][k]['mean_cluster_r2_adj'][clust_num][recalled]
                r2_nrec = self.res['clusters'][k]['mean_cluster_r2_adj'][clust_num][~recalled]
                t, p = ttest_ind(r2_rec, r2_nrec, nan_policy='omit')
                m = [np.nanmean(r2_rec), np.nanmean(r2_nrec)]
                e = [sem(r2_rec, nan_policy='omit'), sem(r2_nrec, nan_policy='omit')]

                ax1.bar([.2], m[0], .35, color=red, linewidth=3.5, yerr=e[0],
                        error_kw=dict(elinewidth=3, ecolor='k', capsize=8, capthick=3))
                ax1.bar([.8], m[1], .35, color=blue, linewidth=3.5, yerr=e[1],
                        error_kw=dict(elinewidth=3, ecolor='k', capsize=8, capthick=3))
                ax1.set_xticks([.2, .8])
                ax1.set_xticklabels(['Rec', 'NRec'])
                ax1.set_ylabel('Adjusted ${R^2}$')
                ax1.set_title('p: %.3f' % p, fontdict={'fontsize': 14})
                ax1.set_axisbelow(True)

                # middle panel, wave direction
                counts_rec = np.histogram(self.res['clusters'][k]['mean_cluster_wave_ang'][clust_num][recalled], bins=edges)[0]
                bars = ax3.bar(angs, counts_rec, width=np.pi / 10, bottom=0.0, zorder=10)
                #             ax2.set_yticklabels('')
                ax3.set_xticklabels('')
                #             print(dir(ax2))
                for r, bar in zip(counts_rec, bars):
                    bar.set_edgecolor('k')
                    bar.set_facecolor(red)
                    bar.set_alpha(0.8)

                counts_nrec = np.histogram(self.res['clusters'][k]['mean_cluster_wave_ang'][clust_num][~recalled], bins=edges)[
                    0]
                bars = ax4.bar(angs, counts_nrec, width=np.pi / 10, bottom=0.0, zorder=10)
                ax4.set_xticklabels('')
                ax4.set_xlabel('Angle V1')
                for r, bar in zip(counts_nrec, bars):
                    bar.set_edgecolor('k')
                    bar.set_facecolor(blue)
                    bar.set_alpha(0.8)

                pval, P = pycircstat.cmtest(self.res['clusters'][k]['mean_cluster_wave_ang'][clust_num][recalled],
                                            self.res['clusters'][k]['mean_cluster_wave_ang'][clust_num][~recalled])
                ax3.set_title('p: %.3f' % pval, fontdict={'fontsize': 14})

                # middle panel, wave direction
                counts_rec = \
                np.histogram(pycircstat.mean(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[recalled],
                             bins=edges)[0]
                bars = ax5.bar(angs, counts_rec, width=np.pi / 10, bottom=0.0, zorder=10)
                #             ax2.set_yticklabels('')
                ax5.set_xticklabels('')
                for r, bar in zip(counts_rec, bars):
                    bar.set_edgecolor('k')
                    bar.set_facecolor(red)
                    bar.set_alpha(0.8)

                counts_nrec = \
                np.histogram(pycircstat.mean(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[~recalled],
                             bins=edges)[0]
                bars = ax6.bar(angs, counts_nrec, width=np.pi / 10, bottom=0.0, zorder=10)
                ax6.set_xticklabels('')
                ax6.set_xlabel('Angle V2')
                for r, bar in zip(counts_nrec, bars):
                    bar.set_edgecolor('k')
                    bar.set_facecolor(blue)
                    bar.set_alpha(0.8)

                pval, P = pycircstat.cmtest(
                    pycircstat.mean(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[recalled],
                    pycircstat.mean(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[~recalled])
                ax5.set_title('p: %.3f' % pval, fontdict={'fontsize': 14})

                rvl_rec = pycircstat.resultant_vector_length(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[
                    recalled]
                rvl_nrec = pycircstat.resultant_vector_length(self.res['clusters'][k]['cluster_wave_ang'][clust_num], axis=0)[
                    ~recalled]
                t, p = ttest_ind(rvl_rec, rvl_nrec, nan_policy='omit')
                m = [np.nanmean(rvl_rec), np.nanmean(rvl_nrec)]
                e = [sem(rvl_rec, nan_policy='omit'), sem(rvl_nrec, nan_policy='omit')]

                ax2.bar([.2], m[0], .35, color=red, linewidth=3.5, yerr=e[0],
                        error_kw=dict(elinewidth=3, ecolor='k', capsize=8, capthick=3))
                ax2.bar([.8], m[1], .35, color=blue, linewidth=3.5, yerr=e[1],
                        error_kw=dict(elinewidth=3, ecolor='k', capsize=8, capthick=3))
                ax2.set_xticks([.2, .8])
                ax2.set_xticklabels(['Rec', 'NRec'])
                ax2.set_ylabel('RVL')
                ax2.set_title('p: %.3f' % p, fontdict={'fontsize': 14})
                ax2.set_axisbelow(True)
                plt.show()

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

    def normalize_power(self, X):
        """
        Normalizes (zscores) each column in X. If rows of comprised of different task phases, each task phase is
        normalized to itself

        returns normalized X
        """
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            for phase in self.task_phase_to_use:
                task_mask = self.task_phase == phase
                X[sess_event_mask & task_mask] = zscore(X[sess_event_mask & task_mask], axis=0)
        return X

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = 'traveling_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)


def circ_lin_regress(phases, coords, theta_r, params):
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

    # compute predicted phases for angle and phase offset
    x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y

    # Compute resultant vector length. This is faster than calling pycircstat.resultant_vector_length
    # now = time.time()
    x1 = numexpr.evaluate('sum(cos(x) / n, axis=1)')
    x1 = numexpr.evaluate('x1 ** 2')
    x2 = numexpr.evaluate('sum(sin(x) / n, axis=1)')
    x2 = numexpr.evaluate('x2 ** 2')
    Rs = numexpr.evaluate('-sqrt(x1 + x2)')
    # print(time.time() - now)

    # this is slower
    # now = time.time()
    # Rs_new = -pycircstat.resultant_vector_length(x, axis=1)
    # tmp = np.abs(((np.exp(1j * x)).sum(axis=1) / n))
    # print(time.time() - now)

    # this is basically the same as method 1
    # now = time.time()
    # tmp = numexpr.evaluate('sum(exp(1j * x), axis=1)')
    # tmp = numexpr.evaluate('abs(tmp) / n')
    # print(time.time() - now)

    # for each time and event, find the parameters with the smallest -R (why are we taking the negative..)
    min_vals = theta_r[np.argmin(Rs, axis=1)]

    sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
    offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
                      np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)

    # compute circular correlation coefficient between actual phases and predicited phases
    circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)

    # compute adjusted r square
    # pdb.set_trace()
    r2_adj = circ_corr_coef ** 2
    # r2_adj = 1 - ((1 - circ_corr_coef ** 2) * (n - 1)) / (n - 4)

    wave_ang = min_vals[:, 0]
    wave_freq = min_vals[:, 1]

    # phase_mean = np.mod(np.angle(np.sum(np.exp(1j * phases)) / len(phases)), 2 * np.pi)
    # pos_circ_mean = np.mod(np.angle(np.sum(np.exp(1j * pos_circ)) / len(phases)), 2 * np.pi)

    # cc = np.sum(np.sin(phases - phase_mean) * np.sin(pos_circ - pos_circ_mean)) / \
    #      np.sqrt(np.sum(np.sin(phases - phase_mean) ** 2) * np.sum(np.sin(pos_circ - pos_circ_mean) ** 2))
    return wave_ang, wave_freq, r2_adj










