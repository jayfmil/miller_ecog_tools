"""
This code is a mess, clean up..
"""
import numpy as np
import pycircstat
import numexpr
import os
import matplotlib.pyplot as plt
import ram_data_helpers
import RAM_helpers
import pdb

from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp
from scipy.stats import ttest_ind, ttest_1samp, sem
from scipy.stats.mstats import zscore
from copy import deepcopy
from SubjectLevel.subject_analysis import SubjectAnalysis
from SubjectLevel.Analyses import subject_SME
from tarjan import tarjan
from xarray import concat
from SubjectLevel.par_funcs import par_find_peaks_by_ev, my_local_max
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import Parallel, delayed

try:

    # this will fail is we don't have an x server
    disp = os.environ['DISPLAY']
    from surfer import Surface, Brain
    from mayavi import mlab
    from tvtk.tools import visual
    import platform
    if platform.system() == 'Darwin':
        os.environ['SUBJECTS_DIR'] = '/Users/jmiller/data/eeg/freesurfer/subjects/'
    else:
        os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'
except (ImportError, KeyError):
    print('Brain plotting not supported')



class SubjectElecCluster(SubjectAnalysis):
    """

    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectElecCluster, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

        self.task_phase_to_use = ['enc']
        self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        self.rec_thresh = None

        # string to use when saving results files
        self.res_str = 'peaks.p'

        # default frequency settings. These are what are passed to load_data(), and this is used when identifying
        # the peak frequency at each electrode and the electrode clusters. For settings related to computing the
        # subsequent memory effect, use the SME_* attributes below
        self.feat_type = 'power'
        self.freqs = np.logspace(np.log10(2), np.log10(32), 129)
        self.bipolar = False
        self.start_time = [0.0]
        self.end_time = [1.6]

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

    def run(self):
        """
        Convenience function to run all the steps
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
            print('%s: Finding oscillation clusters.' % self.subj)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        Does a lot. Explain please.
        """

        # Get recalled or not labels
        self.filter_data_to_task_phases(self.task_phase_to_use)
        recalled = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

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
        if not os.path.exists(subject_sme.save_file):
            subject_sme.save_data()

        # finally, compute SME at the precise frequency of the peak for each electrode
        # loading eeg for all channels first so that we can do an average reference
        if len(self.res['clusters']) > 0:
            eeg = self.load_eeg_all_chans()
            self.eeg = eeg

        for freq in np.sort(list(self.res['clusters'].keys())):
            self.res['clusters'][freq]['elec_ts'] = []

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

    # can we reduce these three highly similar brain plotting functions.
    def plot_cluster_on_brain(self, timepoint=None, use_rel_phase=True, save_dir=None):
        """
        Plots electrodes on brain, colored by phase.
        """

        fig_dict = {'left': [], 'right': [], 'inf': [], 'sup': [], 'freq': [], 'r2': [], 'n': [],
                    'clus_freq': [], 'clus_ind': []}

        # get electode locations
        x, y, z = np.stack(self.elec_xyz_avg).T

        reset_timepoint = True if timepoint is None else False

        # loop over each cluster
        for clus_freq in self.res['clusters'].keys():
            clusters = self.res['clusters'][clus_freq]
            for i, cluster_elecs in enumerate(clusters['elecs']):

                # sorry
                try:
                    mlab.close()
                except:
                    pass

                # render brain
                brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast', background='white',
                              offscreen=False, show_toolbar=True)

                # change opacity
                brain.brain_matrix[0][0]._geo_surf.actor.property.opacity = .2
                brain.brain_matrix[0][1]._geo_surf.actor.property.opacity = .2

                # get phases to plot
                if timepoint is None:
                    timepoint = np.argmax(np.nanmean(clusters['cluster_r2_adj'][i], axis=1))

                if use_rel_phase:
                    rel_phase = pycircstat.mean(clusters['phase_ts'][i][timepoint], axis=0)
                    rel_phase = (rel_phase + np.pi) % (2*np.pi) - np.pi
                    rel_phase *= 180./np.pi
                    rel_phase -= rel_phase.min() - 1
                    # rel_phase[rel_phase > np.pi] -= 2 * np.pi
                    # rel_phase[rel_phase < -np.pi] += 2 * np.pi
                    # rel_phase[rel_phase < np.pi / 2] += 2 * np.pi
                    # rel_phase -= rel_phase.min()
                    # rel_phase = np.ceil(rel_phase * 180. / np.pi + 0.01)
                    # rel_phase = rel_phase - rel_phase.min() + 1
                else:
                    pass

                # convert phases to colors
                # cm = plt.get_cmap('jet')
                # cNorm = clrs.Normalize(vmin=1, vmax=np.max(rel_phase))
                # scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                # colors = scalarmap.to_rgba(np.squeeze(rel_phase)) * 255

                # create array of colors, including black for the number cluster electrodes
                # in_cluster_elecs = cluster_elecs
                # all_elec_colors = np.array([[0, 0, 0, 255]] * x.shape[0])
                # all_elec_colors[in_cluster_elecs] = colors
                # scalars = np.arange(all_elec_colors.shape[0])

                brain.pts = mlab.points3d(x[cluster_elecs], y[cluster_elecs], z[cluster_elecs], rel_phase,
                                          opacity=1, scale_factor=(10. * .4),
                                          scale_mode='vector', name='phase_elecs')
                brain.pts.glyph.color_mode = 'color_by_scalar'
                brain.pts.module_manager.scalar_lut_manager.lut_mode = 'jet'
                brain.pts.glyph.scale_mode = 'scale_by_vector'
                # brain.pts.mlab_source.vectors = np.tile(np.random.random((len(cluster_elecs),)), (3, 1)).T

                hipp_scaling = np.ones((len(cluster_elecs), 3)) * .6
                hipp_scaling[self.elec_locs['Hipp'][cluster_elecs]] = .85
                # print self.elec_locs['Hipp'][cluster_elecs]
                brain.pts.mlab_source.vectors = hipp_scaling

                not_cluster_elecs = np.setdiff1d(np.arange(len(x)), cluster_elecs)
                mlab.points3d(x[not_cluster_elecs], y[not_cluster_elecs], z[not_cluster_elecs], scale_factor=(10. * .4),
                              opacity=1,
                              scale_mode='none', name='not_phase_elecs', color=(0, 0, 0))

                # time_s = clusters['phase_ts'][i][timepoint].time.data
                time_s = np.arange(self.hilbert_start_time, self.hilbert_end_time, 1 / 250.)[timepoint]
                colorbar = mlab.colorbar(brain.pts,  title='Phase, t=%.2f s' % time_s,
                                         orientation='horizontal', label_fmt='%.1f')
                colorbar.scalar_bar_representation.position = [0.1, 0.9]
                colorbar.scalar_bar_representation.position2 = [0.8, 0.1]
                brain.pts.module_manager.scalar_lut_manager.label_text_property.bold = True
                brain.pts.module_manager.scalar_lut_manager.label_text_property.color = (.4, .4, .4)
                brain.pts.module_manager.scalar_lut_manager.label_text_property.font_size = 10
                brain.pts.module_manager.scalar_lut_manager.label_text_property.italic = False
                brain.pts.module_manager.scalar_lut_manager.title_text_property.color = (0, 0, 0)
                brain.pts.module_manager.scalar_lut_manager.title_text_property.color = (0, 0, 0)
                brain.pts.module_manager.scalar_lut_manager.title_text_property.opacity = 1.0
                brain.pts.module_manager.scalar_lut_manager.title_text_property.italic = False

                xyz = np.stack([x,y,z]).T[cluster_elecs]
                xyz -= np.mean(xyz, axis=0)
                pca = PCA(n_components=3)
                pca.fit_transform(xyz)
                mean_ang = pycircstat.mean(clusters['cluster_wave_ang'][i][timepoint])
                ad = np.cos(mean_ang) * pca.components_[0] + np.sin(mean_ang) * pca.components_[1]
                ad = 10 * ad / np.linalg.norm(ad)

                start = np.stack(self.elec_xyz_avg[cluster_elecs], 0).mean(axis=0) - ad
                stop = np.stack(self.elec_xyz_avg[cluster_elecs], 0).mean(axis=0) + ad

                visual.set_viewer(mlab.gcf())
                ar1 = visual.arrow(x=start[0], y=start[1], z=start[2])
                ar1.length_cone = 0.4
                arrow_length = np.linalg.norm(stop-start)
                ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
                ar1.pos = ar1.pos/arrow_length
                ar1.axis = stop-start
                ar1.color = (1., 0, 0)

                # some tweaks to the lighting
                mlab.gcf().scene.light_manager.light_mode = 'vtk'
                mlab.gcf().scene.light_manager.lights[0].activate = True
                mlab.gcf().scene.light_manager.lights[1].activate = True

                if save_dir is not None:
                    r2 = np.nanmean(clusters['mean_cluster_r2_adj'][i])
                    fig_dict['r2'].append(r2)
                    freq = clusters['mean_freqs'][i]
                    fig_dict['freq'].append(freq)
                    fig_dict['clus_freq'].append(clus_freq)
                    fig_dict['clus_ind'].append(i)
                    n_elecs = len(cluster_elecs)
                    fig_dict['n'].append(n_elecs)

                    # left
                    # pdb.set_trace()
                    # if np.any(x[cluster_elecs] < 0):
                    mlab.view(azimuth=180, distance=500)
                    fpath = os.path.join(save_dir,
                                         '%s_freq_%.3f_%d_elecs_r2_%.2f_left_t_%.3d.png' % (self.subj, freq, n_elecs,
                                                                                            r2, timepoint))
                    fig_dict['left'].append(fpath)
                    fig_dict
                    brain.save_image(fpath)

                    # right
                    # if np.any(x[cluster_elecs] > 0):
                    mlab.view(azimuth=0, distance=500)
                    fpath = os.path.join(save_dir,
                                         '%s_freq_%.3f_%d_elecs_r2_%.2f_right_t_%.3d.png' % (self.subj, freq, n_elecs,
                                                                                             r2, timepoint))
                    brain.save_image(fpath)
                    fig_dict['right'].append(fpath)

                    # inf
                    mlab.view(azimuth=0, elevation=180, distance=500)
                    fpath = os.path.join(save_dir,
                                         '%s_freq_%.3f_%d_elecs_r2_%.2f_inf_t_%.3d.png' % (self.subj, freq, n_elecs,
                                                                                           r2, timepoint))
                    brain.save_image(fpath)
                    fig_dict['inf'].append(fpath)

                    # sup
                    mlab.view(azimuth=0, elevation=0, distance=500)
                    fpath = os.path.join(save_dir,
                                         '%s_freq_%.3f_%d_elecs_r2_%.2f_sup_t_%.3d.png' % (self.subj, freq, n_elecs,
                                                                                           r2, timepoint))
                    brain.save_image(fpath)
                    fig_dict['sup'].append(fpath)

                if reset_timepoint:
                    timepoint = None
        return brain, fig_dict

    def plot_sme_on_brain(self, do_activation=False, clim=None, save_dir=None):

        # get electrode locations
        x, y, z = np.stack(self.elec_xyz_avg).T

        reset_clim = True if clim is None else False
        fig_dict = {'left': [], 'right': [], 'inf': [], 'sup': []}
        res_key = 'ts_item' if do_activation else 'ts_sme'

        # loop over each freq range
        for i, sme_freq in enumerate(self.res[res_key].T):

            # sorry
            try:
                mlab.close()
            except:
                pass

            if clim is None:
                clim = np.max(np.abs(sme_freq))
            # render brain
            brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast', background='white',
                          offscreen=False, show_toolbar=True)

            # change opacity
            brain.brain_matrix[0][0]._geo_surf.actor.property.opacity = .3
            brain.brain_matrix[0][1]._geo_surf.actor.property.opacity = .3

            brain.pts = mlab.points3d(x, y, z, sme_freq, scale_factor=(10. * .4), opacity=1,
                                      scale_mode='none', name='ts')
            brain.pts.glyph.color_mode = 'color_by_scalar'
            brain.pts.module_manager.scalar_lut_manager.lut_mode = 'RdBu'

            colorbar = mlab.colorbar(brain.pts, title='t-stat', orientation='horizontal', label_fmt='%.1f')
            colorbar.scalar_bar_representation.position = [0.1, 0.9]
            colorbar.scalar_bar_representation.position2 = [0.8, 0.1]
            brain.pts.module_manager.scalar_lut_manager.label_text_property.bold = True
            brain.pts.module_manager.scalar_lut_manager.label_text_property.color = (.4, .4, .4)
            brain.pts.module_manager.scalar_lut_manager.label_text_property.font_size = 10
            brain.pts.module_manager.scalar_lut_manager.label_text_property.italic = False
            brain.pts.module_manager.scalar_lut_manager.title_text_property.color = (0, 0, 0)
            brain.pts.module_manager.scalar_lut_manager.title_text_property.opacity = 1.0
            brain.pts.module_manager.scalar_lut_manager.title_text_property.italic = False

            brain.pts.module_manager.scalar_lut_manager.lut.table_range = [-clim, clim]
            lut = brain.pts.module_manager.scalar_lut_manager.lut.table.to_array()
            lut = lut[::-1]
            brain.pts.module_manager.scalar_lut_manager.lut.table = lut

            # some tweaks to the lighting
            mlab.gcf().scene.light_manager.light_mode = 'vtk'
            mlab.gcf().scene.light_manager.lights[0].activate = True
            mlab.gcf().scene.light_manager.lights[1].activate = True

            if save_dir is not None:
                freq_range = self.sme_bands[i]

                # left
                mlab.view(azimuth=180, distance=500)
                fpath = os.path.join(save_dir,
                                     '%s_%s_%.2f-%.2f_left.png' % (self.subj, res_key, freq_range[0], freq_range[-1]))
                fig_dict['left'].append(fpath)
                brain.save_image(fpath)

                # right
                mlab.view(azimuth=0, distance=500)
                fpath = os.path.join(save_dir,
                                     '%s_%s_%.2f-%.2f_right.png' % (self.subj, res_key, freq_range[0], freq_range[-1]))
                fig_dict['right'].append(fpath)
                brain.save_image(fpath)

                # inf
                mlab.view(azimuth=0, elevation=180, distance=500)
                fpath = os.path.join(save_dir,
                                     '%s_%s_%.2f-%.2f_inf.png' % (self.subj, res_key, freq_range[0], freq_range[-1]))
                fig_dict['inf'].append(fpath)
                brain.save_image(fpath)

                # sup
                mlab.view(azimuth=0, elevation=0, distance=500)
                fpath = os.path.join(save_dir,
                                     '%s_%s_%.2f-%.2f_sup.png' % (self.subj, res_key, freq_range[0], freq_range[-1]))
                fig_dict['sup'].append(fpath)
                brain.save_image(fpath)

            if reset_clim:
                clim = None

        return brain, fig_dict

    def plot_clusters_on_brain(self, save_dir=None):
        """
        Plots electodes on brain, colors by frequency.
        """

        # get electode locations
        x, y, z = np.stack(self.elec_xyz_avg).T
        fig_dict = {'left': [], 'right': [], 'inf': [], 'sup': []}

        # loop over each cluster
        elecs = []
        freqs = []
        for clus_freq in self.res['clusters'].keys():
            clusters = self.res['clusters'][clus_freq]
            elecs.extend([item for sublist in clusters['elecs'] for item in sublist])
            freqs.extend([item for sublist in clusters['elec_freqs'] for item in sublist])
        elecs = np.array(elecs)
        freqs = np.array(freqs)

        # repeats = np.array([True if x in set(elecs[:i]) else False for i, x in enumerate(elecs)])
        #     elecs = [elecs[~repeats], elecs[repeats]]
        #     freqs = [freqs[~repeats], freqs[repeats]]
        #     return elecs, freqs

        # print elecs.shape
        # print np.unique(elecs).shape
        #     print freqs

        # sorry
        try:
            mlab.close()
        except:
            pass

        # render brain
        brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast', background='white',
                      offscreen=False, show_toolbar=True)

        # change opacity
        brain.brain_matrix[0][0]._geo_surf.actor.property.opacity = .3
        brain.brain_matrix[0][1]._geo_surf.actor.property.opacity = .3
        brain.pts = mlab.points3d(x[elecs], y[elecs], z[elecs], freqs,
                                  scale_factor=(10. * .4), opacity=1,
                                  scale_mode='none', name='freqs_elecs')
        brain.pts.glyph.color_mode = 'color_by_scalar'
        brain.pts.module_manager.scalar_lut_manager.lut_mode = 'viridis'

        not_cluster_elecs = np.setdiff1d(np.arange(len(x)), elecs)
        mlab.points3d(x[not_cluster_elecs], y[not_cluster_elecs], z[not_cluster_elecs], scale_factor=(10. * .4),
                      opacity=1,
                      scale_mode='none', name='not_freq_elecs', color=(0, 0, 0))

        colorbar = mlab.colorbar(brain.pts, title='Frequency (Hz)', orientation='horizontal', label_fmt='%.1f')
        colorbar.scalar_bar_representation.position = [0.1, 0.9]
        colorbar.scalar_bar_representation.position2 = [0.8, 0.1]
        brain.pts.module_manager.scalar_lut_manager.label_text_property.bold = True
        brain.pts.module_manager.scalar_lut_manager.label_text_property.color = (.2, .2, .2)
        brain.pts.module_manager.scalar_lut_manager.label_text_property.font_size = 10
        brain.pts.module_manager.scalar_lut_manager.label_text_property.italic = False
        brain.pts.module_manager.scalar_lut_manager.title_text_property.color = (0, 0, 0)
        brain.pts.module_manager.scalar_lut_manager.title_text_property.opacity = 1.0
        brain.pts.module_manager.scalar_lut_manager.title_text_property.italic = False

        # some tweaks to the lighting
        mlab.gcf().scene.light_manager.light_mode = 'vtk'
        mlab.gcf().scene.light_manager.lights[0].activate = True
        mlab.gcf().scene.light_manager.lights[1].activate = True

        if save_dir is not None:

            # left
            mlab.view(azimuth=180, distance=500)
            fpath = os.path.join(save_dir, '%s_peak_freqs_left.png' % self.subj)
            fig_dict['left'].append(fpath)
            brain.save_image(fpath)

            # right
            mlab.view(azimuth=0, distance=500)
            fpath = os.path.join(save_dir, '%s_peak_freqs_right.png' % self.subj)
            fig_dict['right'].append(fpath)
            brain.save_image(fpath)

            # inf
            mlab.view(azimuth=0, elevation=180, distance=500)
            fpath = os.path.join(save_dir, '%s_peak_freqs_inf.png' % self.subj)
            fig_dict['inf'].append(fpath)
            brain.save_image(fpath)

            # sup
            mlab.view(azimuth=0, elevation=0, distance=500)
            fpath = os.path.join(save_dir, '%s_peak_freqs_sup.png' % self.subj)
            fig_dict['sup'].append(fpath)
            brain.save_image(fpath)

        return brain, fig_dict

    def plot_cluster_features_by_rec(self):
        """
        Plots some cluster metrics for good and bad memory
        """

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

    This is ported from Honghui's matlab code.

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
    # pdb.set_trace()

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

    # for each time and event, find the parameters with the smallest -R
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


def circ_lin_regress1d(phases, coords, theta_r, params):
    """
    Performs 2D circular linear regression.

    This is ported from Honghui's matlab code.

    :param phases:
    :param coords:
    :return:
    """


    n = phases.shape[1]
    pos_x = np.expand_dims(coords, 1)

    # compute predicted phases for angle and phase offset

    x = np.expand_dims(phases, 2) - params[:, 0] * pos_x
    # pdb.set_trace()
    # pos_slopes = np.unique(theta_r[:, 1])
    # x = np.expand_dims(phases, 2) - pos_slopes * pos_x

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

    # for each time and event, find the parameters with the smallest -R
    # sl = pos_slopes[np.argmin(Rs, axis=1)]


    min_vals = theta_r[np.argmin(Rs, axis=1)]
    sl = min_vals[:, 1] * np.cos(min_vals[:, 0])

    # sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])

    offs = np.arctan2(np.sum(np.sin(phases.T - sl * pos_x), axis=0), np.sum(np.cos(phases.T - sl * pos_x), axis=0))


    # offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
    #                   np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    pos_circ = np.mod(sl * pos_x + offs, 2 * np.pi)

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






