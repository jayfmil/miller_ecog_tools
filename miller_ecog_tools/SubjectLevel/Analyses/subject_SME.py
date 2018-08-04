"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency, compare correctly and incorrectly recalled
items using a t-test.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clrs

from copy import deepcopy
from scipy.stats import sem, ttest_ind

from miller_ecog_tools.SubjectLevel.subject_data import SubjectEEGData
from miller_ecog_tools.SubjectLevel.Analyses.subject_analysis import SubjectAnalysisBase


class SubjectSMEAnalysis(SubjectAnalysisBase, SubjectEEGData):
    """
    Subclass of SubjectAnalysis and SubjectEEGData with methods to compute the Subsequent Memory Effect for each
    electrode. This compares recalled items to not recalled items using t-test.

    The user must define the .recall_filter_func attribute of this class. This should be a function that, given a set
    of events, returns a boolean array of recalled (True) and not recalled (False) items.
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectSMEAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'sme.p'

        # The SME analysis is a contract between two conditions (recalled and not recalled items). Set
        # recall_filter_func to be a function that takes in events and returns indices of recalled items
        self.recall_filter_func = None

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        Performs the subsequent memory analysis by comparing the distribution of remembered and not remembered items
        at each electrode and frequency using a two sample ttest.
        """
        if self.subject_data is None:
            print('%s: compute of load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s SME: please provide a .recall_filter_func function.' % self.subject)
        recalled = self.recall_filter_func(self.subject_data)

        # zscore the data by session
        z_data = self.zscore_data()

        # for every frequency, electrode, timebin, subtract mean recalled from mean non-recalled zpower
        delta_z = np.nanmean(z_data[recalled], axis=0) - np.nanmean(z_data[~recalled], axis=0)
        delta_z = delta_z.reshape(self.subject_data.shape[1:])

        # run ttest at each frequency and electrode comparing remembered and not remembered events
        ts, ps, = ttest_ind(z_data[recalled], z_data[~recalled])

        # also do this by session
        sessions = self.subject_data.event.data['session']
        ts_by_sess = []
        ps_by_sess = []
        for sess in np.unique(sessions):
            sess_ind = sessions == sess
            ts_sess, ps_sess = ttest_ind(z_data[recalled & sess_ind], z_data[~recalled & sess_ind])
            ts_by_sess.append(ts_sess.reshape(len(self.freqs), -1))
            ps_by_sess.append(ps_sess.reshape(len(self.freqs), -1))

        # store results.
        self.res = {}
        self.res['zs'] = delta_z
        self.res['p_recall'] = np.mean(recalled)
        self.res['ts_sess'] = ts_by_sess
        self.res['ps_sess'] = ps_by_sess
        self.res['ts'] = ts
        self.res['ps'] = ps
        self.res[recalled] = recalled

        # make a binned version of t-stats that is frequency x brain region. Calling this from within .analysis() for
        # convenience because I know the data is loaded now, which we need to have access to the electrode locations.
        # self.res['ts_region'], self.res['regions'] = self.sme_by_region()

        # also counts of positive SME electrodes and negative SME electrodes by region
        # self.res['sme_count_pos'], self.res['sme_count_neg'], self.res['elec_n'] = self.sme_by_region_counts()

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

        recalled = self.recall_filter_func(self.task, self.subject_data.event.data, self.rec_thresh)
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

            chan_tag = self.tag_name[elec]
            anat_region = self.anat_region[elec]
            loc = self.loc_tag[elec]
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
    #     for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
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

    def normalize_spectra(self, X):
        """
        Normalize the power spectra by session.
        """
        uniq_sessions = np.unique(self.subject_data.event.data['session'])
        for sess in uniq_sessions:
            sess_event_mask = (self.subject_data.event.data['session'] == sess)
            m = np.mean(X[sess_event_mask], axis=1)
            m = np.mean(m, axis=0)
            s = np.std(X[sess_event_mask], axis=1)
            s = np.mean(s, axis=0)
            X[sess_event_mask] = (X[sess_event_mask] - m) / s
        return X

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.freqs[-1]) - 1).bit_length())) + 1))
