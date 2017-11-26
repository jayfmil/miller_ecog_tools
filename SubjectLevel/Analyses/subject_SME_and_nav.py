"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency, compare correctly and incorrectly recalled
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

class SubjectSME(SubjectAnalysis):
    """
    Subclass of SubjectAnalysis with methods to analyze power spectrum of each electrode.
    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectSME, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)
        self.task_phase_to_use = ['enc']  # ['enc'] or ['rec']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled        
        self.rec_thresh = None

        # put a check on this, has to be power
        self.feat_type = 'power'

        # string to use when saving results files
        self.res_str = 'sme_nav.p'

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
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency using a two sample ttest.

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

        # self.filter_data_to_task_phases(self.task_phase_to_use)
        # recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

        # hardcoded for the TH task. Sorry. Trying to get a paper published.
        # get indices of item presentation events and label recall and not recalled
        chest_inds = (self.subject_data.events.data['type'] == 'CHEST') & (self.subject_data.events.data['confidence'] >= 0)
        chest_data = self.subject_data[chest_inds]
        not_low_conf = chest_data.events.data['confidence'] > 0
        thresh = np.median(chest_data.events.data['norm_err'])
        radius = chest_data.events.data['radius_size'][0]
        correct = chest_data.events.data['distErr'] < radius
        not_far_dist = chest_data.events.data['norm_err'] < thresh
        recalled = not_low_conf & (not_far_dist | correct)

        # get indices of move and still events
        still_inds = self.subject_data.events.data['type'] == 'STILL'
        move_inds = self.subject_data.events.data['type'] == 'MOVE'

        X = deepcopy(self.subject_data.data)
        uniq_sessions = np.unique(self.subject_data.events.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            X[sess_event_mask] = zscore(X[sess_event_mask], axis=0)
        # X = X.reshape(self.subject_data.shape[0], -1)

        # for every frequency, electrode, timebin, subtract mean recalled from mean non-recalled zpower
        rec_pow_mean = np.nanmean(X[chest_inds][recalled], axis=0)
        nrec_pow_mean = np.nanmean(X[chest_inds][~recalled], axis=0)
        delta_z = rec_pow_mean - nrec_pow_mean

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts_sme, ps_sme, = ttest_ind(X[chest_inds][recalled], X[chest_inds][~recalled], axis=0, nan_policy='omit')

        # ditto for move vs still
        move_pow_mean = np.nanmean(X[move_inds], axis=0)
        still_pow_mean = np.nanmean(X[still_inds], axis=0)
        delta_z_move = move_pow_mean - still_pow_mean

        # run ttest at each frequency and electrode comparing still and move
        ts_move, ps_move, = ttest_ind(X[move_inds], X[still_inds], axis=0, nan_policy='omit')

        # finally do nav vs baseline
        nav_inds = self.subject_data.events.data['type'] == 'NAV'
        baseline_inds = self.subject_data.events.data['type'] == 'BASELINE'
        nav_pow_mean = np.nanmean(X[nav_inds], axis=0)
        baseline_pow_mean = np.nanmean(X[baseline_inds], axis=0)
        delta_z_nav = nav_pow_mean - baseline_pow_mean
        ts_nav, ps_nav, = ttest_ind(X[nav_inds], X[baseline_inds], axis=0, nan_policy='omit')

        # store results.
        self.res = {}
        self.res['delta_z_sme'] = delta_z
        self.res['rec_pow_mean'] = rec_pow_mean
        self.res['nrec_pow_mean'] = nrec_pow_mean
        self.res['ts_sme'] = ts_sme
        self.res['ps_sme'] = ps_sme

        self.res['delta_z_move'] = delta_z_move
        self.res['move_pow_mean'] = move_pow_mean
        self.res['still_pow_mean'] = still_pow_mean
        self.res['ts_move'] = ts_move
        self.res['ps_move'] = ps_move

        self.res['delta_z_nav'] = delta_z_nav
        self.res['nav_pow_mean'] = nav_pow_mean
        self.res['baseline_pow_mean'] = baseline_pow_mean
        self.res['ts_nav'] = ts_nav
        self.res['ps_nav'] = ps_nav

        self.res['p_recall'] = np.mean(recalled)

        p_spect = deepcopy(self.subject_data.data)
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.events.data['session'] == sess)
            m = np.mean(p_spect[sess_event_mask], axis=1)
            m = np.mean(m, axis=0)
            s = np.std(p_spect[sess_event_mask], axis=1)
            s = np.mean(s, axis=0)
            p_spect[sess_event_mask] = (p_spect[sess_event_mask] - m) / s

        self.res['mean_rec_pspect'] = np.nanmean(p_spect[chest_inds][recalled], axis=0)
        self.res['mean_nrec_pspect'] = np.nanmean(p_spect[chest_inds][~recalled], axis=0)
        self.res['mean_move_pspect'] = np.nanmean(p_spect[move_inds], axis=0)
        self.res['mean_still_pspect'] = np.nanmean(p_spect[still_inds], axis=0)

        if self.task == 'RAM_TH1':
            rec_continuous = 1 - self.subject_data[chest_inds].events.data['norm_err']
            rs = np.array([np.corrcoef(x, rec_continuous)[0, 1] for x in X[chest_inds].reshape(np.sum(chest_inds), -1).T])
            self.res['rs'] = rs.reshape(len(self.freqs), -1)
            # self.res['med_dist'] = np.median(self.subject_data.events.data['distErr'])
            # self.res['skew'] = self.res['med_dist'] - np.mean(self.subject_data.events.data['distErr'])
            self.res['med_dist'] = np.median(rec_continuous)
            self.res['mean_dist'] = np.mean(rec_continuous)
            self.res['skew'] = self.res['med_dist'] - self.res['mean_dist']
            self.res['rs_region'], self.res['regions'] = self.sme_by_region(res_key='rs')

        # make a binned version of t-stats that is frequency x brain region. Calling this from within .analysis() for
        # convenience because I know the data is loaded now, which we need to have access to the electrode locations.
        self.res['ts_region'], self.res['regions'] = self.sme_by_region()

        # also counts of positive SME electrodes and negative SME electrodes by region
        self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()

        # # also, for each electrode, find ranges of neighboring frequencies that are significant for both postive and
        # # negative effecst
        # sig_pos = (self.res['ps'] < .05) & (self.res['ts'] > 0)
        # contig_pos = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_pos.T.tolist())
        # self.res['contig_freq_inds_pos'] = contig_pos
        #
        # sig_neg = (self.res['ps'] < .05) & (self.res['ts'] < 0)
        # contig_neg = map(lambda x: self.find_continuous_ranges(np.where(x)[0]), sig_neg.T.tolist())
        # self.res['contig_freq_inds_neg'] = contig_neg

    def plot_spectra_average(self, elec):
        """
        Create a two panel figure with shared x-axis. Top panel is log(power) as a function of frequency, seperately
        plotted for recalled (red) and not-recalled (blue) items. Bottom panel is t-stat at each frequency comparing the
        recalled and not recalled distributions, with shaded areas indicating p<.05.

        elec (int): electrode number that you wish to plot.
        """
        if self.subject_data is None:
            print('%s: data must be loaded before computing SME by region. Use .load_data().' % self.subj)
            return

        if not self.res:
            print('%s: must run .analysis() before computing SME by region' % self.subj)
            return

        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)
        p_spect = deepcopy(self.subject_data.data)
        p_spect = self.normalize_spectra(p_spect)

        with plt.style.context('myplotstyle.mplstyle'):
            #         f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            #                     wspace=None, hspace=0.01)
            x = np.log10(self.subject_data.frequency)
            # ax1.plot(x, self.subject_data[recalled, :, elec].mean('events'), c='#8c564b', label='Good Memory', linewidth=4)
            # ax1.plot(x, self.subject_data[~recalled, :, elec].mean('events'), c='#1f77b4', label='Bad Memory', linewidth=4)
            rec_mean = np.mean(p_spect[recalled, :, elec], axis=0)
            rec_sem = sem(p_spect[recalled, :, elec], axis=0)
            ax1.plot(x, rec_mean, c='#8c564b', label='Good Memory',
                     linewidth=2)
            ax1.fill_between(x, rec_mean + rec_sem, rec_mean - rec_sem, color='#8c564b', alpha=.5)

            nrec_mean = np.mean(p_spect[~recalled, :, elec], axis=0)
            nrec_sem = sem(p_spect[~recalled, :, elec], axis=0)
            ax1.plot(x, nrec_mean, color='#1f77b4', label='Bad Memory',
                     linewidth=2)
            ax1.fill_between(x, nrec_mean + nrec_sem, nrec_mean - nrec_sem, color='#1f77b4', alpha=.5)

            ax1.set_ylabel('Normalized log(power)')
            ax1.yaxis.label.set_fontsize(24)
            ax1.yaxis.set_ticks([-2, -1, 0, 1, 2])
            ax1.set_ylim([-2,2])
            # ax1.yaxis.set_ticks([3.5, 4.5, 5.5, 6.5])
            # ax1.yaxis.set_ticks([4, 5, 6])

            l = ax1.legend()
            frame = l.get_frame()
            frame.set_facecolor('w')
            for legobj in l.legendHandles:
                legobj.set_linewidth(5)

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
            ax2.yaxis.set_ticks([-2, 0, 2])

            ax1.xaxis.set_ticks(np.log10(new_x))
            ax1.xaxis.set_ticklabels('')

            ax1.spines['left'].set_linewidth(2)
            ax1.spines['bottom'].set_linewidth(2)
            # ax1.spines['bottom'].set_color([.5, .5, .5])
            ax2.spines['left'].set_linewidth(2)
            ax2.spines['bottom'].set_linewidth(2)
            # _ = plt.xticks(x[::4], np.round(self.freqs[::4] * 10) / 10, rotation=-45)

            chan_tag = self.subject_data.attrs['chan_tags'][elec]
            anat_region = self.subject_data.attrs['anat_region'][elec]
            loc = self.subject_data.attrs['loc_tag'][elec]
            _ = ax1.set_title('%s - elec %d: %s, %s, %s' % (self.subj, elec + 1, chan_tag, anat_region, loc))

        ax_list = plt.gcf().axes
        for ax in ax_list:
            ax.set_axisbelow(True)
            ax.set_axis_bgcolor('w')
            ax.grid(color=(.5, .5, .5))

        return plt.gcf()

    def plot_sme_on_brain(self, do_lfa=True, only_sig=False, no_colors=False):
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

        colors = scalarmap.to_rgba(np.squeeze(sme_by_elec)) * 255
        if only_sig:
            colors[ps > .05] = [0, 0, 0, 255]
        if no_colors:
            colors[:] = [0, 0, 0, 255]

        x, y, z = np.stack(self.elec_xyz_avg).T
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

    def find_continuous_ranges(self, data):
        """
        Given an array of integers, finds continuous ranges. Similar in concept to 1d version of bwlabel in matlab on a
        boolean vector. This method is really clever, it subtracts the index of each entry from the value and then
        groups all those with the same difference.

        Credit: http://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
        """

        ranges = []
        for k, g in groupby(enumerate(data), lambda (i, x): i - x):
            group = map(itemgetter(1), g)
            ranges.append((group[0], group[-1]))
        return ranges

    def sme_by_region(self, res_key='ts_sme'):
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
        ts = self.res['ts_sme']
        ps = self.res['ps_sme']

        # counts of significant positive SMEs
        count_pos = [np.nansum((ts[:, self.elec_locs[x]] > 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_pos = np.stack(count_pos, axis=1)

        # counts of significant negative SMEs
        count_neg = [np.nansum((ts[:, self.elec_locs[x]] < 0) & (ps[:, self.elec_locs[x]] < .05), axis=1) for x in regions]
        count_neg = np.stack(count_neg, axis=1)

        # count of electrodes by region
        n = np.array([np.nansum(self.elec_locs[x]) for x in regions])
        return count_pos, count_neg, n


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

        dir_str = 'sme_%s_%s' % (self.recall_filter_func.__name__, '_'.join(self.task_phase_to_use))
        if self.save_dir is None:
            save_dir = self._generate_save_path(self.base_dir)
        else:
            save_dir = self.save_dir

        return os.path.join(os.path.split(save_dir)[0], dir_str)

