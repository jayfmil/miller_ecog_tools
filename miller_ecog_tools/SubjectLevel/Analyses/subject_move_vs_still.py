"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency, compare correctly and incorrectly is_move
items using a t-test.
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


class SubjectMoveStill(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode.
    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectMoveStill, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)


        # put a check on this
        self.feat_phase = ['move']
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'move_still.p'

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
            print('%s: Running SME.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """

        """

        # reshape the power data to be events x features and normalize
        X = deepcopy(self.subject_data.data)
        X = X.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)

        is_move = self.subject_data.events.data['type'] == 'move'

        # for every frequency, electrode, timebin, subtract mean is_move from mean non-is_move zpower
        delta_z = np.nanmean(X[is_move], axis=0) - np.nanmean(X[~is_move], axis=0)
        delta_z = delta_z.reshape(self.subject_data.shape[1:])

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(X[is_move], X[~is_move])
        sessions = self.subject_data.events.data['session']
        ts_by_sess = []
        ps_by_sess = []
        for sess in np.unique(sessions):
            sess_ind = sessions == sess
            ts_sess, ps_sess = ttest_ind(X[is_move & sess_ind], X[~is_move & sess_ind])
            ts_by_sess.append(ts_sess.reshape(len(self.freqs), -1))
            ps_by_sess.append(ps_sess.reshape(len(self.freqs), -1))

        # for convenience, also compute within power averaged bands for low freq and high freq
        lfa_inds = self.freqs <= 10
        lfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, lfa_inds, :], axis=1)
        lfa_ts, lfa_ps = ttest_ind(lfa_pow[is_move], lfa_pow[~is_move])

        # high freq
        hfa_inds = self.freqs >= 60
        hfa_pow = np.mean(X.reshape(self.subject_data.shape)[:, hfa_inds, :], axis=1)
        hfa_ts, hfa_ps = ttest_ind(hfa_pow[is_move], hfa_pow[~is_move])

        # store results.
        self.res = {}
        self.res['zs'] = delta_z
        self.res['p_recall'] = np.mean(is_move)
        self.res['ts_lfa'] = np.expand_dims(lfa_ts, axis=0)
        self.res['ps_lfa'] = np.expand_dims(lfa_ps, axis=0)
        self.res['ts_hfa'] = np.expand_dims(hfa_ts, axis=0)
        self.res['ps_hfa'] = np.expand_dims(hfa_ps, axis=0)
        self.res['ts_sess'] = ts_by_sess
        self.res['ps_sess'] = ps_by_sess

        # store the t-stats and p values for each electrode and freq. Reshape back to frequencies x electrodes.
        self.res['ts'] = ts.reshape(len(self.freqs), -1)
        self.res['ps'] = ps.reshape(len(self.freqs), -1)

        # make a binned version of t-stats that is frequency x brain region. Calling this from within .analysis() for
        # convenience because I know the data is loaded now, which we need to have access to the electrode locations.
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()

        # also counts of positive SME electrodes and negative SME electrodes by region
        self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()

        # also, for each electrode, find ranges of neighboring frequencies that are significant for both postive and
        # negative effecst
        # sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        # contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        # self.res['contig_freq_inds_pos'] = contig_pos

        # sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        # contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        # self.res['contig_freq_inds_neg'] = contig_neg

    def plot_spectra_average(self, elec):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for is_move (red) and not-is_move (blue) items. Bottom panel is t-stat at each frequency comparing the
        is_move and not is_move distributions, with shaded areas indicating p<.05.

        elec (int): electrode number that you wish to plot.
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        self.filter_data_to_task_phases(self.task_phase_to_use)
        is_move = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        with plt.style.context('myplotstyle.mplstyle'):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            x = np.log10(self.subject_data.frequency)
            # ax1.plot(x, self.subject_data[is_move, :, elec].mean('events'), c='#8c564b', label='Good Memory', linewidth=4)
            # ax1.plot(x, self.subject_data[~is_move, :, elec].mean('events'), c='#1f77b4', label='Bad Memory', linewidth=4)
            ax1.plot(x, np.mean(p_spect[is_move, :, elec], axis=0), c='#8c564b', label='Good Memory',
                     linewidth=4)
            ax1.plot(x, np.mean(p_spect[~is_move, :, elec], axis=0), c='#1f77b4', label='Bad Memory',
                     linewidth=4)
            ax1.set_ylabel('Normalized log(power)')
            ax1.yaxis.label.set_fontsize(24)
            l = ax1.legend()

            y = self.res['ts'][:, elec]
            p = self.res['ps'][:, elec]
            ax2.plot(x, y, '-k', linewidth=4)
            ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
            ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=-1)

            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y > 0), facecolor='#8c564b', edgecolor='#8c564b')
            ax2.fill_between(x, [0] * len(x), y, where=(p < .05) & (y < 0), facecolor='#1f77b4', edgecolor='#1f77b4')
            ax2.set_ylabel('t-stat')
            ax2.yaxis.label.set_fontsize(24)

            plt.xlabel('Frequency', fontsize=24)
            new_x = self.compute_pow_two_series()
            ax2.xaxis.set_ticks(np.log10(new_x))
            ax2.xaxis.set_ticklabels(new_x, rotation=0)
            # _ = plt.xticks(x[::4], np.round(self.freqs[::4] * 10) / 10, rotation=-45)

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax1.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec+1, chan_tag, anat_region, loc))

        return f

    def plot_sme_on_brain(self, do_lfa=True, only_sig=False):
        """
        Render the average brain and plot the electrodes. Color code by t-statistic of SME.

        Returns brain object. Useful if you want to do brain.save_image().
        """

        # render brain
        brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast', background='white',
                      offscreen=False)

        # change opacity
        brain.brain_matrix[0][0]._geo_surf.actor.property.opacity = .5
        brain.brain_matrix[0][1]._geo_surf.actor.property.opacity = .5

        # values to be plotted
        # sme_by_elec = np.mean(self.res['ts'][freq_inds, :], axis=0)
        sme_by_elec = self.res['ts_lfa'].T if do_lfa else self.res['ts_hfa'].T
        ps = self.res['ps_lfa'].T if do_lfa else self.res['ps_hfa'].T

        # plot limits defined by range of t-stats
        clim = np.max(np.abs([np.nanmax(sme_by_elec), np.nanmin(sme_by_elec)]))

        # compute color for each electrode based on stats
        cm = plt.get_cmap('RdBu_r')
        cNorm = clrs.Normalize(vmin=-clim, vmax=clim)
        scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        colors = scalarmap.to_rgba(sme_by_elec) * 255
        if only_sig:
            colors[ps > .05] = [0, 0, 0, 255]

        x, y, z = self.elec_xyz_avg.T
        scalars = np.arange(colors.shape[0])

        brain.pts = mlab.points3d(x, y, z, scalars, scale_factor=(10. * .4), opacity=1,
                              scale_mode='none')
        brain.pts.glyph.color_mode = 'color_by_scalar'
        brain.pts.module_manager.scalar_lut_manager.lut.table = colors
        return brain

    def plot_sme_specificity_on_brain(self, do_lfa=True, only_sig=False, radius=12.5):
        """
        Render the average brain and plot the electrodes. Color code by t-statistic of SME.

        Returns brain object. Useful if you want to do brain.save_image().
        """

        # render brain
        brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast', background='white',
                      offscreen=False)

        # change opacity
        brain.brain_matrix[0][0]._geo_surf.actor.property.opacity = .5
        brain.brain_matrix[0][1]._geo_surf.actor.property.opacity = .5

        # values to be plotted
        # sme_by_elec = np.mean(self.res['ts'][freq_inds, :], axis=0)
        sme_by_elec = self.res['ts_lfa'] if do_lfa else self.res['ts_hfa']
        sme_by_elec = np.abs(sme_by_elec)
        ps = self.res['ps_lfa'] if do_lfa else self.res['ps_hfa']

        sme_normed = np.zeros(self.elec_xyz_avg.shape[0])
        sme_normed[:] = np.nan
        for elec, xyz in enumerate(self.elec_xyz_avg):
            near_elecs = np.linalg.norm(xyz - self.elec_xyz_avg, axis=1) < radius
            if np.sum(near_elecs) > 3:
                sme_normed[elec] = sme_by_elec[elec] / np.sum(sme_by_elec[near_elecs])

        # plot limits defined by range of t-stats
        print(np.nanmax(sme_normed))
        clim = 0.5

        # compute color for each electrode based on stats
        cm = plt.get_cmap('Reds')
        cNorm = clrs.Normalize(vmin=0, vmax=.5)
        scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        colors = scalarmap.to_rgba(sme_normed) * 255
        colors[np.isnan(sme_normed)] = [0, 0, 0, 255]
        if only_sig:
            colors[ps > .05] = [0, 0, 0, 255]

        x, y, z = self.elec_xyz_avg.T
        scalars = np.arange(colors.shape[0])

        brain.pts = mlab.points3d(x, y, z, scalars, scale_factor=(10. * .4), opacity=1,
                                  scale_mode='none')
        brain.pts.glyph.color_mode = 'color_by_scalar'
        brain.pts.module_manager.scalar_lut_manager.lut.table = colors
        return brain

    def plot_elec_heat_map(self):
        """
        Frequency by electrode SME visualization.
        """

        clim = np.max(np.abs([np.min(self.res['ts']), np.max(self.res['ts'])]))
        ant_post_order = np.argsort(self.elec_xyz_avg[:, 1])
        left_elecs = (self.elec_xyz_avg[:, 0] < 0)[ant_post_order]
        left_ts = self.res['ts'][:, ant_post_order[left_elecs]]
        right_ts = self.res['ts'][:, ant_post_order[~left_elecs]]

        with plt.style.context('myplotstyle.mplstyle'):
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(self.res['ts'], interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)  # ,rotation=90)
            cb.ax.tick_params(labelsize=12)

            # plt.xticks(range(len(regions)), regions, fontsize=24, rotation=-45)

            new_freqs = self.compute_pow_two_series()
            new_y = np.interp(np.log10(new_freqs[:-1]), np.log10(self.freqs),
                              range(len(self.freqs)))
            _ = plt.yticks(new_y, new_freqs[:-1], fontsize=20)
            plt.ylabel('Frequency', fontsize=24)
            plt.gca().invert_yaxis()
            plt.grid()

    # def find_continuous_ranges(self, data):
    #     """
    #     Given an array of integers, finds continuous ranges. Similar in concept to 1d version of bwlabel in matlab on a
    #     boolean vector. This method is really clever, it subtracts the index of each entry from the value and then
    #     groups all those with the same difference.
    #
    #     Credit: http://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
    #     """
    #
    #     ranges = []
    #     for k, g in groupby(enumerate(data), lambda (i, x): i - x):
    #         group = map(itemgetter(1), g)
    #         ranges.append((group[0], group[-1]))
    #     return ranges

    def sme_by_region(self, res_key='ts'):
        """
        Bin (average) res['ts'] by brain region. Return array that is freqs x region, and return region strings.
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

        dir_str = 'move_still'
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)
