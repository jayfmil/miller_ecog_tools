"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency and time bine, compare correctly and
incorrectly recalled items using a t-test.
"""
import os
import pdb
import ram_data_helpers
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from scipy.stats.mstats import zscore, zmap
from copy import deepcopy
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from SubjectLevel.subject_analysis import SubjectAnalysis
import matplotlib.cm as cmx
import matplotlib.colors as clrs

try:

    # this will fail is we don't have an x server
    disp = os.environ['DISPLAY']
    from surfer import Surface, Brain
    from mayavi import mlab
    import platform
    if platform.system() == 'Darwin':
        os.environ['SUBJECTS_DIR'] = '/Users/jmiller/data/eeg/freesurfer/subjects/'
    else:
        os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'
except (ImportError, KeyError):
    print('Brain plotting not supported')


class SubjectSMETime(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode. Differs from SubjectSME in
    that this analysis includes multiple time bins
    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectSMETime, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)
        self.task_phase_to_use = ['enc']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None

        # do we want to create a 'ts' array that is freq x elec x time x permutations
        self.make_perm_array = False
        self.n_iters = 1000

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'sme_time.p'

    def run(self):
        """
        Convenience function to run all the steps for SubjectSME analysis.
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
            print('%s: Running SME with multiple time windows.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency and time using a two sample ttest.

        .res will have the keys:
                                 'ts'
                                 'ps'
                                 'regions'
                                 'ts_region'
                                 'sme_count_pos'
                                 'sme_count_neg'
                                 'elec_n'
                                 'contig_freq_inds_pos'
                                 'contig_freq_inds_neg'
                                 MORE UPDATE THIS

        """

        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # reshape the power data to be events x features and normalize
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)

        # for every frequency, electrode, timebin, subtract mean recalled from mean non-recalled zpower
        delta_z = np.nanmean(X[recalled], axis=0) - np.nanmean(X[~recalled], axis=0)
        delta_z = delta_z.reshape(self.subject_data.shape[1:])

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[recalled], X[~recalled])

        # for convenience, also compute within power averaged bands for low freq and high freq
        lfa_inds = self.freqs <= 10
        lfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, lfa_inds, :, :], axis=1)
        lfa_ts, lfa_ps = ttest_ind(lfa_pow[recalled], lfa_pow[~recalled])

        # high freq
        hfa_inds = self.freqs >= 60
        hfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, hfa_inds, :, :], axis=1)
        hfa_ts, hfa_ps = ttest_ind(hfa_pow[recalled], hfa_pow[~recalled])

        # store results.
        self.res = {}
        self.res['zs'] = delta_z
        self.res['ts_lfa'] = lfa_ts
        self.res['ps_lfa'] = lfa_ps
        self.res['ts_hfa'] = hfa_ts
        self.res['ps_hfa'] = hfa_ps

        # store the t-stats and p values for each electrode and freq. Reshape back to frequencies x electrodes.
        self.res['ts'] = ts.reshape(self.subject_data.shape[1:])
        self.res['ps'] = ps.reshape(self.subject_data.shape[1:])

        self.res['time_bins'] = self.subject_data['time'].data
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()
        self.res['zs_region'], _ = self.sme_by_region(res_key='zs')

        if self.make_perm_array:
            recalled = deepcopy(recalled)
            ts_perm = np.zeros((self.n_iters,) + self.subject_data.shape[1:])
            for i in range(self.n_iters):
                np.random.shuffle(recalled)
                ts_perm[i] = ttest_ind(X[recalled], X[~recalled])[0].reshape(self.subject_data.shape[1:])
            self.res['ts_perm'] = ts_perm

    def plot_time_by_freq(self, elec, res_key='ts'):
        """
        Plot time x frequency spectrogram for a specific electrode.
        """

        did_load = False
        if self.subject_data is None:
            did_load = True
            self.load_data()

        plot_data = self.res[res_key][:, elec, :]
        p = np.ma.masked_where(self.res['ps'] < .05, self.res['ps'])
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(self.res['time_bins']))[::3], self.res['time_bins'][::3], fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs), range(len(self.freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            self.plot_mask_outline(p)
            plt.gca().invert_yaxis()
            plt.grid()

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec + 1, chan_tag, anat_region, loc))

        if did_load:
            self.subject_data = None

    def plot_mask_outline(self, p_mat, lw=3):
        """
        Plot outline of significant regions.
        Credit: http://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
        """
        mapimg = ~p_mat.mask

        # a vertical line segment is needed, when the pixels next to each other horizontally
        #   belong to diffferent groups (one is part of the mask, the other isn't)
        # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
        ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

        # the same is repeated for horizontal segments
        hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

        # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
        #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
        # in order to draw a discountinuous line, we add Nones in between segments
        l = []
        for p in zip(*hor_seg):
            l.append((p[1], p[0] + 1))
            l.append((p[1] + 1, p[0] + 1))
            l.append((np.nan, np.nan))

        # and the same for vertical segments
        for p in zip(*ver_seg):
            l.append((p[1] + 1, p[0]))
            l.append((p[1] + 1, p[0] + 1))
            l.append((np.nan, np.nan))

        # now we transform the list into a numpy array of Nx2 shape
        segments = np.array(l)
        segments[:, 0] = p_mat.shape[1] * segments[:, 0] / mapimg.shape[1] - .5
        segments[:, 1] = p_mat.shape[0] * segments[:, 1] / mapimg.shape[0] - .5

        # and now there isn't anything else to do than plot it
        plt.plot(segments[:, 0], segments[:, 1], color='k', linewidth=lw)

    def plot_time_by_freq_region(self, region, res_key='ts_region'):

        ind = self.res['regions'] == region
        plot_data = self.res[res_key][:, ind, :]
        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            plt.xticks(range(len(self.res['time_bins']))[::3], self.res['time_bins'][::3], fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs), range(len(self.freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.gca().invert_yaxis()
            plt.grid()

            # chan_tag = self.subject_data.attrs['chan_tags'][elec]
            # anat_region = self.subject_data.attrs['anat_region'][elec]
            # loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax.set_title('%s - %s' % (self.subj, region))

    def sme_by_region(self, res_key='ts'):
        """
        Bin (average) res['ts'] by brain region. Return array that is freqs x region x time, and return region strings.
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        # average all the elecs within each region. Iterate over the sorted keys because I don't know if dictionary
        # keys are always returned in the same order?
        regions = np.array(sorted(self.elec_locs.keys()))
        t_array = np.stack([np.nanmean(self.res[res_key][:, self.elec_locs[x]], axis=1) for x in regions], axis=1)
        return t_array, regions

    def sme_by_region_counts(self):
        """
        Count of significant electrodes by region
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        regions = np.array(sorted(self.elec_locs.keys()))
        ts = self.res['ts']
        ps = self.res['ps']

        # counts of significant positive SMEs
        count_pos = [np.nansum((ts[:, self.elec_locs[x]] > 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_pos = np.stack(count_pos, axis=1)

        # counts of significant negative SMEs
        count_neg = [np.nansum((ts[:, self.elec_locs[x]] < 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_neg = np.stack(count_neg, axis=1)

        # count of electrodes by region
        n = np.array([np.nansum(self.elec_locs[x]) for x in regions])
        return count_pos, count_neg, n

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

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))

    def _generate_res_save_path(self):
        """
        Build path to where results should be saved (or loaded from). Return string.
        """

        dir_str = 'sme_time_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

