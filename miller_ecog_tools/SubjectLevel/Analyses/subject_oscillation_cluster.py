"""
This code is a mess, clean up..
"""
import numpy as np
import numexpr
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ram_data_helpers
import RAM_helpers
import pdb

from ptsa.data.filters import MorletWaveletFilter
from scipy.stats import ttest_ind, pearsonr
from scipy.stats.mstats import zscore
from copy import deepcopy


from tarjan import tarjan
from xarray import concat
# from SubjectLevel.par_funcs import par_find_peaks_by_ev, my_local_max
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from joblib import Parallel, delayed

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_eeg_data import SubjectEEGData
from miller_ecog_tools.SubjectLevel.par_funcs import par_find_peaks_by_ev, my_local_max

# SubjectOscillationClusterAnalysis also runs SubjectSMEAnalysis
from miller_ecog_tools.SubjectLevel.Analyses.subject_SME import SubjectSMEAnalysis


class SubjectOscillationClusterAnalysis(SubjectAnalysisBase, SubjectEEGData):
    """

    """

    res_str_tmp = 'elec_cluster_%d_mm_%d_elec_min_%s_elec_type_%s_sep_hemis_%.2f_cluster_range.p'
    attrs_in_res_str = ['elec_types_allowed', 'min_elec_dist', 'min_num_elecs', 'separate_hemis', 'cluster_freq_range']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectOscillationClusterAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # Set recall_filter_func to be a function that takes in events and returns bool of recalled items
        self.recall_filter_func = None

        # string to use when saving results files
        self.res_str = SubjectOscillationClusterAnalysis.res_str_tmp

        # default frequency settings. These are what are passed to load_data(), and this is used when identifying
        # the peak frequency at each electrode and the electrode clusters. For settings related to computing the
        # subsequent memory effect, use the SME_* attributes below
        self.freqs = np.logspace(np.log10(2), np.log10(32), 129)
        self.bipolar = False
        self.start_time = 0
        self.end_time = 1600

        # settings for computing SME
        # asfafa
        self.sme_freqs = np.logspace(np.log10(1), np.log10(200), 50)
        self.sme_start_time = 0.0
        self.sme_end_time = 1.6

        # window size to find clusters (in Hz)
        self.cluster_freq_range = 2.

        # D: depths, G: grids, S: strips
        self.elec_types_allowed = ['D', 'G', 'S']

        # spatial distance considered near
        self.min_elec_dist = 15.

        # If True, osciallation clusters can't cross hemispheres
        self.separate_hemis = True

        # number of electrodes needed to be considered a clust
        self.min_num_elecs = 4

        # dictionary will hold the cluter results
        self.res = {}

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

    def analysis(self):
        """
        Does a lot. Explain please.
        """

        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s SME: please provide a .recall_filter_func function.' % self.subject)
        recalled = self.recall_filter_func(self.subject_data)

        # initialize eeg and res
        self.res['clusters'] = {}

        # compute frequency bins
        window_centers = np.arange(self.freqs[0], self.freqs[-1] + .001, 1)
        windows = [(x - self.cluster_freq_range / 2., x + self.cluster_freq_range / 2.) for x in window_centers]
        window_bins = np.stack([(self.freqs >= x[0]) & (self.freqs <= x[1]) for x in windows], axis=0)

        # distance matrix for all electrodes. If separating the hemispheres, move the hemispheres far apart
        xyz_tmp = np.stack(self.elec_xyz_indiv)
        if self.separate_hemis:
            xyz_tmp[xyz_tmp[:, 0] < 0, 0] -= 100
        elec_dists = squareform(pdist(xyz_tmp))

        # figure out which pairs of electodes are closer than the threshold
        near_adj_matr = (elec_dists < self.min_elec_dist) & (elec_dists > 0.)
        allowed_elecs = np.array([e in self.elec_types_allowed for e in self.e_type])

        # noramlize power spectra
        p_spect = deepcopy(self.subject_data)
        p_spect = self.normalize_spectra(p_spect)

        # Compute mean power spectra across events, and then find where each electrode has peaks
        mean_p_spect = p_spect.mean(dim='events')
        peaks = par_find_peaks_by_ev(mean_p_spect)
        self.res['clusters'] = self.find_clusters_from_peaks([peaks], near_adj_matr, allowed_elecs,
                                                             window_bins, window_centers)

        # use the subject_SME class to compute/load the sme for this subject
        subject_sme = subject_SME.SubjectSME(task=self.task,
                                             montage=self.montage,
                                             subject=self.subj)
        subject_sme.task_phase_to_use = ['enc']
        subject_sme.start_time = self.sme_start_time
        subject_sme.end_time = self.sme_end_time
        subject_sme.freqs = self.sme_freqs
        subject_sme.bipolar = self.bipolar
        subject_sme.load_data_if_file_exists = True
        subject_sme.load_res_if_file_exists = True
        subject_sme.run()
        self.res['sme'] = subject_sme.res

        if not os.path.exists(subject_sme.save_file):
            subject_sme.save_data()
        self.res['sme'] = subject_sme.res

        # also compute sme at the freqs used to compute peaks
        subject_sme = subject_SME.SubjectSME(task=self.task,
                                             montage=self.montage,
                                             subject=self.subj)
        subject_sme.task_phase_to_use = ['enc']
        subject_sme.start_time = self.start_time
        subject_sme.end_time = self.end_time
        subject_sme.freqs = self.freqs
        subject_sme.bipolar = self.bipolar
        subject_sme.load_data_if_file_exists = True
        subject_sme.load_res_if_file_exists = True
        subject_sme.run()
        if not os.path.exists(subject_sme.save_file):
            subject_sme.save_data()
        self.res['sme_low_freqs'] = subject_sme.res

        # finally, compute SME at the precise frequency of the peak for each electrode
        # loading eeg for all channels first so that we can do an average reference
        if len(self.res['clusters']) > 0:
            eeg = self.load_eeg_all_chans()

        for freq in np.sort(list(self.res['clusters'].keys())):
            self.res['clusters'][freq]['elec_ts'] = []
            self.res['clusters'][freq]['cluster_region'] = []

            for cluster_count, cluster_elecs in enumerate(self.res['clusters'][freq]['elecs']):
                elec_freqs = self.res['clusters'][freq]['elec_freqs'][cluster_count]

                ts_cluster = []
                for elec_info in zip(cluster_elecs, elec_freqs):
                    this_elec_num = elec_info[0]
                    this_elec_freq = np.array([elec_info[1]])
                    elec_pow, _ = MorletWaveletFilterCpp(time_series=eeg[this_elec_num], freqs=this_elec_freq,
                                                         output='power', width=5, cpus=10, verbose=False).filter()
                    data = elec_pow.data
                    elec_pow.data = numexpr.evaluate('log10(data)')
                    elec_pow.remove_buffer(1.6)
                    elec_pow = elec_pow.mean(dim='time')
                    elec_pow = RAM_helpers.make_events_first_dim(elec_pow)
                    elec_pow.data = RAM_helpers.zscore_by_session(elec_pow)
                    ts, ps = ttest_ind(elec_pow[recalled], elec_pow[~recalled])
                    ts_cluster.append(ts[0])
                self.res['clusters'][freq]['elec_ts'].append(ts_cluster)

                # label the cluster by the region iwth the most electrodes
                keys = [x for x in self.elec_locs.keys() if x != 'is_right']
                r = keys[np.argmax([np.sum(self.elec_locs[x][cluster_elecs]) for x in keys])]
                self.res['clusters'][freq]['cluster_region'].append(r)


    def load_eeg_all_chans(self):

        eeg = []
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        events_as_recarray = self.subject_data.events.data.view(np.recarray)
        # load by session and channel to avoid using to much memory
        for s, session in enumerate(uniq_sessions):
            print('%s: Loading EEG session %d of %d.' % (self.subj, s+1, len(uniq_sessions)))

            sess_inds = self.subject_data.events.data['session'] == session
            chan_eegs = []
            # loop over each channel
            for channel in tqdm(self.subject_data['channels'].data):
                chan_eegs.append(RAM_helpers.load_eeg(events_as_recarray[sess_inds],
                                          np.array([channel]), self.sme_start_time,
                                          self.sme_end_time, 1.6))

            # create timeseries object for session because concatt doesn't work over the channel dim
            chan_dim = chan_eegs[0].get_axis_num('channels')
            elecs = np.concatenate([x[x.dims[chan_dim]].data for x in chan_eegs])
            chan_eegs_data = np.concatenate([x.data for x in chan_eegs], axis=chan_dim)
            coords = chan_eegs[0].coords
            coords['channels'] = elecs
            sess_eeg = TimeSeriesX(data=chan_eegs_data, coords=coords, dims=chan_eegs[0].dims)
            sess_eeg = sess_eeg.transpose('channels', 'events', 'time')
            sess_eeg -= sess_eeg.mean(dim='channels')

            # hold all session events
            eeg.append(sess_eeg)

        # concat all session evenets
        # make sure all the time samples are the same from each session. Can differ if the sessions were
        # recorded at different sampling rates, even though we are downsampling to the same rate
        if len(eeg) > 1:
            if ~np.all([np.array_equal(eeg[0].time.data, eeg[x].time.data) for x in range(1, len(eeg))]):
                print('%s: not all time samples equal. Setting to values from first session.' % self.subj)
                for x in range(1, len(eeg)):
                    eeg[x] = eeg[x][:, :, :eeg[0].shape[2]]
                    eeg[x].time.data = eeg[0].time.data

        eeg = concat(eeg, dim='events')
        eeg['events'] = self.subject_data.events

        return eeg

    def find_clusters_from_peaks(self, peaks, near_adj_matr, allowed_elecs, window_bins, window_centers):
        """
        Finds oscillation clusters from the peaks in the power spectra. This is where the spatial smoothing and tarjan
        algorithm are implemented. Returns a dictionary with info about each cluster.

        :param peaks:
        :param near_adj_matr:
        :param allowed_elecs:
        :param window_bins:
        :param window_centers:
        :return:
        """

        all_clusters = {k: {'elecs': [], 'mean_freqs': [], 'elec_freqs': []} for k in window_centers}
        for i, ev in enumerate(peaks):

            # make sure only electrodes of allowed types are included
            ev[:, ~allowed_elecs] = False

            # bin peaks, count them up, and find the peaks (of the peaks...)
            binned_peaks = np.stack([np.any(ev[x], axis=0) for x in window_bins], axis=0)
            # peak_freqs = argrelmax(binned_peaks.sum(axis=1))[0]
            peak_freqs = my_local_max(binned_peaks.sum(axis=1))

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
                    all_clusters[window_centers[this_peak_freq]]['elec_freqs'].append(mean_freqs)
                    all_clusters[window_centers[this_peak_freq]]['mean_freqs'].append(np.mean(mean_freqs))

        return dict((k, v) for k, v in all_clusters.items() if all_clusters[k]['elecs'])

    def plot_cluster_freq_ts_by_corr(self):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.set(style='ticks', palette='Set2')
        sns.despine()
        sns.set_context("talk", font_scale=1.4)
        lfa_inds = (self.freqs >= 1) & (self.freqs <= 10)
        lfa_ts = self.res['sme_low_freqs']['ts'][lfa_inds].mean(axis=0)

        for cluster_freq in self.res['clusters']:
            clusters = self.res['clusters'][cluster_freq]
            num_clusters = len(self.res['clusters'][cluster_freq]['elec_freqs'])
            for cluster_num in range(num_clusters):

                # plot correlation
                ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=3, rowspan=1)
                x = np.array(clusters['elec_freqs'][cluster_num])
                y = np.array(clusters['elec_ts'][cluster_num])
                ymax = np.max(np.abs([np.min(y), np.max(y)])) + .5
                cluster_freq = clusters['mean_freqs'][cluster_num]

                ax1 = sns.regplot(x=x, y=y, ci=None, fit_reg=True, truncate=True,
                                  scatter_kws={'s': 130, 'edgecolor': np.array(sns.color_palette()[0]) * .5, 'lw': 1},
                                  line_kws={'lw': 3, 'color': 'k', 'zorder': -1}, ax=ax1)
                _ = ax1.set_ylabel('t-stat (peak)')
                #         _ = ax1.set_xlabel('Frequency (Hz)')
                _ = ax1.set_ylim(-ymax, ymax)
                r, p = pearsonr(x, y)
                if p < 0.01:
                    p = np.power(10, np.ceil(np.log10(p)))
                cluster_region = clusters['cluster_region'][cluster_num]
                _ = ax1.set_title('{0} ({1}, {2:.3f} Hz): r={3:.3f}, p{4}{5:.3f}'.format(self.subj, cluster_region,
                                                                                         cluster_freq, r,
                                                                                         '<' if p <= 0.01 else '=', p))

                ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=3, rowspan=1)
                x = np.array(clusters['elec_freqs'][cluster_num])
                y = lfa_ts[clusters['elecs'][cluster_num]]
                ymax2 = np.max(np.abs([np.min(y), np.max(y)])) + .5
                cluster_freq = clusters['mean_freqs'][cluster_num]

                ax4 = sns.regplot(x=x, y=y, ci=None, fit_reg=True, truncate=True,
                                  scatter_kws={'s': 130, 'edgecolor': np.array(sns.color_palette()[0]) * .5, 'lw': 1},
                                  line_kws={'lw': 3, 'color': 'k', 'zorder': -1}, ax=ax4)
                _ = ax4.set_ylabel('t-stat (1-10 Hz)')
                _ = ax4.set_xlabel('Frequency (Hz)')
                _ = ax4.set_ylim(-ymax2, ymax2)
                r, p = pearsonr(x, y)
                if p < 0.01:
                    p = np.power(10, np.ceil(np.log10(p)))
                _ = ax4.set_title('{0} ({1}, {2:.3f} Hz): r={3:.3f}, p{4}{5:.3f}'.format(self.subj, cluster_region,
                                                                                         cluster_freq, r,
                                                                                         '<' if p <= 0.01 else '=', p))

                # plot mean SME
                ax2 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=1)
                x = np.log10(self.sme_freqs)
                elec_ts = self.res['sme']['ts'][:, clusters['elecs'][cluster_num]].T
                new_x = np.power(2, range(int(np.log2(2 ** (int(self.sme_freqs[-1]) - 1).bit_length())) + 1))
                m = np.mean(elec_ts, axis=0)
                e = np.std(elec_ts, axis=0) / np.sqrt(elec_ts.shape[0] - 1)
                ci = (m - e, m + e)
                _ = ax2.fill_between(x, ci[0], ci[1], alpha=0.2)
                _ = ax2.plot(x, m, lw=4)
                _ = ax2.set_xticks(np.log10(new_x))
                _ = ax2.set_xticklabels(new_x)
                _ = ax2.plot(x, [0] * len(x), '-k', zorder=-1, lw=2)
                _ = ax2.set_ylim(-ymax, ymax)

                # plot mean SME low
                ax2 = plt.subplot2grid((2, 5), (1, 3), colspan=2, rowspan=1)
                x = np.log10(self.freqs)
                elec_ts = self.res['sme_low_freqs']['ts'][:, clusters['elecs'][cluster_num]].T
                new_x = np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
                m = np.mean(elec_ts, axis=0)
                e = np.std(elec_ts, axis=0) / np.sqrt(elec_ts.shape[0] - 1)
                ci = (m - e, m + e)
                _ = ax2.fill_between(x, ci[0], ci[1], alpha=0.2)
                _ = ax2.plot(x, m, lw=4)
                _ = ax2.set_xticks(np.log10(new_x))
                _ = ax2.set_xticklabels(new_x)
                _ = ax2.plot(x, [0] * len(x), '-k', zorder=-1, lw=2)
                _ = ax2.set_ylim(-ymax, ymax)

                plt.tight_layout()
                plt.gcf().set_size_inches(15, 8)
                plt.show()
        return

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


