import numpy as np
import re
import os
from glob import glob
import pdb
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ranksums, ttest_1samp, ttest_ind
import cPickle as pickle
import cluster_helper.cluster
import ram_data_helpers
from TH_load_features import load_features, compute_subj_dir_path
import matplotlib
import matplotlib.pyplot as plt
from xray import concat
from sklearn.preprocessing import LabelEncoder


class SME:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    data_path = '/data/eeg'
    save_dir = '/scratch/jfm2/python/TH'
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', train_phase='enc',  bipolar=True,
                 freqs=None, start_time=-1.2, end_time=0.5, time_bins=None,
                 ROIs=None, recall_filter_func=None, rec_thresh=None, pool=None):

        # if subjects not given, get list from /data/events/ directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task


        # task phase to train on ('enc', 'rec', or 'both')
        self.train_phase = train_phase

        # this is stupid but don't let it precess some subjects. R1132C didn't use the confident judgements so we can't
        # work with the data.
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1219C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1227T']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar
        self.ROIs = ROIs

        # these are the frequencies where we will compute power
        self.freqs = freqs

        # start and end time to use, relative to events. If train_phase is 'both', enter a list of two times for start
        # and end (encoding time first)
        self.start_time = start_time
        self.end_time = end_time

        # An array of time bins to average. If None, will just average from start_time to end_time
        self.time_bins = time_bins

        if recall_filter_func is None:
            self.recall_filter_func = ram_data_helpers.filter_events_to_recalled
        else:
            self.recall_filter_func = recall_filter_func
        self.rec_thresh = rec_thresh

        self.pool = pool

        # holds the output from all subjects
        self.res = None

    def run_SME_for_all_subjs(self):

        SMEs_all = []
        for subj in self.subjs:
            print 'Processing %s.' % subj
            try:
                subj_res = self.run_SME_for_single_subj(subj)
                SMEs_all.append(subj_res)
            except:
                print 'Error processing %s' % subj
        self.res = SMEs_all

    def run_SME_for_single_subj(self, subj):

        # freq x elecs x events x time bins
        subj_features = load_features(subj, self.task, self.train_phase, self.start_time, self.end_time,
                                      self.time_bins, self.freqs, None, None,
                                      None, self.bipolar, 'power', False,
                                      False, '/scratch/jfm2/python/', self.ROIs, self.pool)

        # events x elecs x freqs x time bins
        dims = subj_features.dims
        subj_features = subj_features.transpose(dims[2], dims[1], dims[0], dims[3])

        recalls = self.recall_filter_func(self.task, subj_features.events.data, self.rec_thresh)
        sessions = subj_features.events.data['session']
        uniq_sessions = np.unique(sessions)

        # turn into matrix for easier zscoring
        subj_data = subj_features.data.reshape(subj_features.data.shape[0], -1)
        for sess in uniq_sessions:
            sess_event_mask = (sessions == sess)
            subj_data[sess_event_mask] = zscore(subj_data[sess_event_mask], axis=0)

        # reshape back
        subj_data = subj_data.reshape(subj_features.data.shape)

        # ttest at each  elec, freq, time
        ts, ps = ttest_ind(subj_data[recalls], subj_data[~recalls])

        subj_res = {}
        subj_res['subj'] = subj
        subj_res['time_bins'] = self.time_bins
        subj_res['time_mean'] = self.time_bins.mean(axis=1)
        subj_res['ts'] = ts
        subj_res['ps'] = ps
        subj_res['loc_tag'] = subj_features.attrs['loc_tag']
        subj_res['anat_region'] = subj_features.attrs['anat_region']
        subj_res['chan_tags'] = subj_features.attrs['chan_tags']
        subj_res['cv_type'] = 'loso' if len(np.unique(sessions)) > 1 else 'lolo'
        subj_res['channels'] = subj_features.channels.data if not self.bipolar else subj_features.bipolar_pairs.data
        return subj_res

    def compute_feature_heatmap(self, subjs=None, hemi='both', cv_type=('loso', 'lolo'), do_thresh=False,
                                regions=('IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC')):
        if subjs is None:
            subjs = self.subjs

        SME_arrays = np.zeros((len(self.freqs), self.time_bins.shape[0], len(self.res), len(regions)))
        SME_arrays[:] = np.nan
        for s_count, subj_res in enumerate(self.res):
            if subj_res['subj'] in subjs and (subj_res['cv_type'] in cv_type):
                loc_dict = ram_data_helpers.bin_elec_locs(subj_res['loc_tag'], subj_res['anat_region'],subj_res['chan_tags'])

                hemi_bool = np.array([True] * len(loc_dict['is_right']))
                if hemi == 'right':
                    hemi_bool = loc_dict['is_right']
                elif hemi == 'left':
                    hemi_bool = ~loc_dict['is_right']

                for r_count, region in enumerate(regions):
                    if not np.any(np.isnan(subj_res['ts'])):
                        data = subj_res['ts'][loc_dict[region] & hemi_bool].mean(axis=0)
                        SME_arrays[:, :, s_count, r_count] = data

        for r in range(len(regions)):

            plot_data, p = ttest_1samp(SME_arrays[:, :, :, r], 0, axis=2, nan_policy='omit')

            if do_thresh:
                plot_data[p > .05] = np.nan

            clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
            fig, ax = plt.subplots(1, 1)
            im = plt.imshow(plot_data, interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')

            time_mean = self.time_bins.mean(axis=1)
            plt.xticks(range(len(time_mean))[3::5], np.round(time_mean[3::5] * 10) / 10, fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)

            plt.yticks(range(len(self.freqs))[::3], (np.round(self.freqs * 10) / 10)[::3],
                       fontsize=16)
            plt.ylabel('Frequency', fontsize=16)

            cb = plt.colorbar()
            cb.set_label(label='t-stat', size=16)
            cb.ax.tick_params(labelsize=12)

            plt.title(regions[r], fontsize=16)

            plt.gca().invert_yaxis()
            fig.set_size_inches(18, 6)











