import numpy as np
import os
import pdb
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ranksums, ttest_1samp, ttest_ind
from ptsa.data.readers.TalReader import TalReader
import cPickle as pickle
import cluster_helper.cluster
import ram_data_helpers
import matplotlib
import matplotlib.pyplot as plt
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter
from scipy.signal import hilbert


class PacAna:
    # default paths
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', bipolar=True, freqs=None, freq_bands=None, hilbert_phase_band=None,
                 num_phase_bins=None, start_time=-1.2, end_time=0.5,  ROIs=None, pool=None):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs
        print self.subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task

        # this is stupid but don't let it precess R1132C - this subject didn't use the confident judgements so we can't
        #  work with the data
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']
            self.subjs = [subj for subj in self.subjs if subj != 'R1219C']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute power
        self.freqs = freqs
        self.freq_bands = freq_bands
        self.hilbert_phase_band = hilbert_phase_band
        self.num_phase_bins = num_phase_bins

        # time bin to use
        self.start_time = start_time
        self.end_time = end_time

        self.ROIs = ROIs

        # if doing parallel jobs, pool with be a cluster_helper object, otherwise None
        self.pool = pool

        # where to save data
        self.base_dir = os.path.join(PacAna.base_dir, task)

        ####
        self.res = None

    def run_pac_for_all_subjs(self):
        # class_data_all_subjs = []
        for subj in self.subjs:
            print 'Processing %s.' % subj

            # define base directory for subject
            f1 = self.freqs[0]
            f2 = self.freqs[-1]
            bipol_str = 'bipol' if self.bipolar else 'mono'
            subj_base_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f_%s' % (len(self.freqs), f1, f2, bipol_str),
                                         'enc_start_%.1f_stop_%.1f_rec_%.1f_stop_%.1f' %
                                         (self.start_time[0], self.end_time[0], self.start_time[1], self.end_time[1]),
                                         'pac', subj)

            # sub directories hold electrode data, feature data, and classifier output
            subj_elec_dir = os.path.join(subj_base_dir, 'elec_data')

            try:
                events = ram_data_helpers.load_subj_events(self.task, subj)
            except (ValueError, AttributeError, IOError):
                print 'Error processing %s. Could not load events.' % subj

            # make directory if missing
            if not os.path.exists(subj_base_dir):
                try:
                    os.makedirs(subj_base_dir)
                except OSError:
                    pass

            # run classifier for subject
            try:
                subj_classify = self.run_pac_for_single_subj(subj, subj_base_dir)
            except (ValueError, AttributeError, IOError):
               print 'Error processing %s.' % subj

    def run_pac_for_single_subj(self, subj, subj_base_dir):

        # get electrode numbers and events
        elecs_bipol, elecs_monopol = ram_data_helpers.load_subj_elecs(subj)
        events = ram_data_helpers.load_subj_events(self.task, subj, 'enc', None, False if self.bipolar else True)

        # construct input to main prcoessing function
        elecs = elecs_bipol if self.bipolar else elecs_monopol

        # filter ROIs
        loc_tag, anat_region, chan_tags = ram_data_helpers.load_subj_elec_locs(subj, self.bipolar)
        if self.ROIs is not None:
            loc_dict = ram_data_helpers.bin_elec_locs(loc_tag, anat_region)
            roi_elecs = np.array([False] * len(loc_tag))
            if isinstance(self.ROIs, str):
                ROIs = [self.ROIs]
            for roi in ROIs:
                roi_elecs = roi_elecs | loc_dict[roi]
            elecs = elecs[roi_elecs]
            # loc_tag = loc_tag[roi_elecs]
            # anat_region = anat_region[roi_elecs]
            # chan_tags = chan_tags[roi_elecs]
        if len(elecs) == 0:
            print('%s no electrodes in %s' % (subj, ', '.join(ROIs)))
            return

        recalls = ram_data_helpers.filter_events_to_recalled_just_median(self.task, events)
        for elec in elecs:
            pow_by_phase_bin, pow_double_binned = self.run_pac_for_elec(events, elec)

    def run_pac_for_elec(self, events, elec):

        # if doing bipolar, load in pair of channels
        if self.bipolar:

            # load eeg for channels in this bipolar_pair
            eeg_reader = EEGReader(events=events, channels=np.array(list(elec)), start_time=self.start_time,
                                   end_time=self.end_time, buffer_time=3.0)
            eegs = eeg_reader.read()

            # convert to bipolar
            bipolar_recarray = np.recarray(shape=1, dtype=[('ch0', '|S3'), ('ch1', '|S3')])
            bipolar_recarray[0] = elec
            m2b = MonopolarToBipolarMapper(time_series=eegs, bipolar_pairs=bipolar_recarray)
            eegs = m2b.filter()
        else:

            # load eeg for for single channe;
            eeg_reader = EEGReader(events=events, channels=np.array([elec]),
                                   start_time=self.start_time, end_time=self.end_time, buffer_time=3.0)
            eegs = eeg_reader.read()

        # filter 60 Hz line noise
        b_filter = ButterworthFilter(time_series=eegs, freq_range=[58., 62.], filt_type='stop', order=4)
        eegs_filtered = b_filter.filter()

        # filter within band and run hilbert transform
        hb_vals = hilbert(eegs_filtered.filtered(self.hilbert_phase_band, filt_type='pass', order=2),
                          axis=eegs_filtered.get_axis_num('time'))

        # copy eeg timeseries (to keep metadata) and replace data with hilbert phase
        hb_ts = eegs_filtered.copy(deep=True)
        # hb_ts.data = (np.angle(hb_vals) + 5*np.pi/4) % 2*np.pi
        hb_ts.data = np.angle(hb_vals)
        hb_ts = hb_ts.remove_buffer(duration=3.0)

        # now compute power at freqs
        wf = MorletWaveletFilter(time_series=eegs_filtered, freqs=self.freqs)
        pow_elec, phase_elec = wf.filter()
        pow_elec = pow_elec.remove_buffer(duration=3.0)

        # log transform
        np.log10(pow_elec.data, out=pow_elec.data)

        # subtract 1/f
        x = np.log10(self.freqs)
        A = np.vstack([x, np.ones(len(x))]).T
        y = np.mean(np.mean(pow_elec.data, axis=3), axis=2)
        m, c = np.linalg.lstsq(A, y)[0]
        fit = m * x + c
        pow_elec.data = np.transpose(pow_elec.data.T - fit)

        uniq_sessions = np.unique(events.session)
        # for sess in uniq_sessions:
        #     sess_event_mask = (sessions == sess)
        #     feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

        # bin pow based on phase of hilbert transformed signal
        num_bins = self.num_phase_bins
        # bins = np.linspace(0, 2 * np.pi, num_bins+1)
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        # inds = np.digitize(hb_ts, bins)
        inds = np.digitize(-hb_ts, bins)

        # only include time points when mean power is above treshhold
        # m = np.mean(pow_elec.data, axis=2).flatten().mean()
        # s = np.mean(pow_elec.data, axis=2).flatten().std()
        # thresh = m + 2*s
        # pow_elec.data[pow_elec.data < thresh] = np.nan

        # hold the binned values. There is surely a better way to do this
        pow_by_phase_bin = np.empty(shape=(pow_elec.shape[0], pow_elec.shape[1], pow_elec.shape[2], num_bins))
        for bin in range(1, num_bins + 1):
            bin_inds = inds == bin
            for ev in range(len(pow_elec.events)):
                pow_by_phase_bin[:, :, ev, bin - 1] = np.nanmean(pow_elec[:, :, ev, bin_inds[:, ev, :].flatten()],
                                                                 axis=2)
                # for sess in uniq_sessions:
                #     sess_event_mask = (events.session == sess)
                #     pow_by_phase_bin[:, :, sess_event_mask, bin - 1] = zscore(pow_by_phase_bin[:, :, sess_event_mask, bin - 1], axis=2)
        # pdb.set_trace()
        # lastly, bin power by freq_bands
        num_bands = np.shape(self.freq_bands)[0]
        pow_double_binned = np.empty(shape=(num_bands, pow_by_phase_bin.shape[1], pow_by_phase_bin.shape[2],
                                            pow_by_phase_bin.shape[3]))
        bands = self.freq_bands if self.freq_bands.ndim > 1 else [self.freq_bands]
        band_array = np.recarray(shape=num_bands, dtype=[('bin_start', 'float'), ('bin_end', 'float')])
        for i, band in enumerate(bands):
            band_array[i] = band
            band_inds = (self.freqs >= band[0]) & (self.freqs <= band[1])
            pow_double_binned[i, :, :, :] = np.nanmean(pow_by_phase_bin[band_inds], axis=0)

        return pow_by_phase_bin, pow_double_binned
