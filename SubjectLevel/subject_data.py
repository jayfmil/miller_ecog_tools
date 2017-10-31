import cPickle as pickle
import joblib
import os
import numpy as np
import numexpr
from xray import concat
import ram_data_helpers
from TH_load_features import load_features
from subject import Subject
import pdb

from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
# from ptsa.data.filters import ButterworthFilter
# from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter
from ptsa.data.TimeSeriesX import TimeSeriesX
from ptsa.data.filters.MorletWaveletFilterCpp import MorletWaveletFilterCpp

class SubjectData(Subject):
    """
    Data class contains default data settings and handles raw(ish) data IO.
    """

    def __init__(self, task=None, subject=None, montage=0,  use_json=True):
        super(SubjectData, self).__init__(task=task, subject=subject, montage=montage,  use_json=use_json)
        self.feat_phase = ['enc']
        self.feat_type = 'power'
        self.start_time = [-1.2]
        self.end_time = [0.5]
        self.bipolar = True # FALSE CURRENTLY UNSUPPORTED
        self.freqs = np.logspace(np.log10(1), np.log10(200), 8)
        self.hilbert_phase_band = None
        self.freq_bands = None
        self.mean_pow = False
        self.num_phase_bins = None
        self.time_bins = None
        self.ROIs = None
        self.pool = None

        # this will hold the subject data after load_data() is called
        self.subject_data = None

        # this will hold the a dictionary of electrode locations after load_data() is called
        self.elec_locs = {}
        self.e_type = None
        self.elec_xyz_avg = None
        self.elec_xyz_indiv = None
        self.elec_xyz_mni = None
        self.elec_xyz_tal = None

        # For each entry in .subject_data, will be either 'enc' or 'rec'
        self.task_phase = None

        # if data already exists on disk, just load it. If False, will recompute if do_not_compute is False
        self.load_data_if_file_exists = True
        self.do_not_compute = False

        # base directory to save data
        self.base_dir = '/scratch/jfm2/python'

        # location of save data will be defined after save() is called
        self.save_dir = None
        self.save_file = None

    def load_data(self):
        """
        Loads features for each feature type in self.feat_phase and concats along events dimension.
        """
        if self.subj is None:
            print('Attribute subj must be set before loading data.')
            return

        # define the location where data will be saved and load if already exists and not recomputing
        self.save_dir = self._generate_save_path(self.base_dir)
        self.save_file = os.path.join(self.save_dir, self.subj + '_features.p')

        # if events have been modified since the data was saved, recompute
        force_recompute = False
        event_mtime = ram_data_helpers.get_event_mtime(self.task, self.subj, self.montage, self.use_json)
        if os.path.exists(self.save_file):
            data_mtime = os.path.getmtime(self.save_file)
            if event_mtime > data_mtime:
                if not self.do_not_compute:
                    force_recompute = True
                    print('%s: Events have been modified since data created, recomputing.' % self.subj)

        elif self.do_not_compute:
            print('%s: subject_data does not exist, not computing.' % self.subj)
            return

        if not force_recompute and self.load_data_if_file_exists and os.path.exists(self.save_file):
            print('%s: Input data already exists, loading.' % self.subj)
            self.subject_data = joblib.load(self.save_file)
            # with open(self.save_file, 'rb') as f:
            #     self.subject_data = pickle.load(f)

        # otherwise compute
        else:
            subj_features = []
            # loop over all task phases
            for s_time, e_time, phase in zip(self.start_time if isinstance(self.start_time, list) else [self.start_time],
                                             self.end_time if isinstance(self.end_time, list) else [self.end_time],
                                             self.feat_phase):

                subj_features.append(load_features(self.subj, self.task, self.montage, self.use_json, phase,
                                                   s_time, e_time, self.time_bins, self.freqs, self.freq_bands,
                                                   self.hilbert_phase_band, self.num_phase_bins, self.bipolar,
                                                   self.feat_type, self.mean_pow, False, '',
                                                   self.ROIs, self.pool))
            if len(subj_features) > 1:
                subj_features = concat(subj_features, dim='events')
            else:
                subj_features = subj_features[0]

            # make sure the events dimension is the first dimension
            if subj_features.dims[0] != 'events':
                ev_dim = np.where(np.array(subj_features.dims) == 'events')[0]
                new_dim_order = np.hstack([ev_dim, np.setdiff1d(range(subj_features.ndim), ev_dim)])
                subj_features = subj_features.transpose(*np.array(subj_features.dims)[new_dim_order])

            # store as self.subject_data
            self.subject_data = subj_features
            self.subject_data.data = self.subject_data.data.astype('float32')
            # self.compute_power()

        self.add_loc_info()

        # lastly, create task_phase array that is standardized regardless of experiment
        self.task_phase = self.subject_data.events.data['type']
        if 'RAM_YC' in self.task:
            enc_str = 'NAV_LEARN'
            rec_str = 'NAV_TEST'
        elif 'RAM_TH' in self.task:
            enc_str = 'CHEST'
            rec_str = 'REC'
            # if 'RAM_THR' in self.task:
            #     rec_str = 'REC_EVENT'
            #     rec_str = 'PROBE'
            if 'move' in self.feat_phase:
                self.task_phase[self.task_phase == 'move'] = 'enc'
                self.task_phase[self.task_phase == 'still'] = 'enc'
        elif 'RAM_PAL' in self.task:
            enc_str = 'STUDY_PAIR'
            rec_str = 'TEST_PROBE'
        else:
            enc_str = 'WORD'
            rec_str = 'REC_WORD'

        self.task_phase[self.task_phase == enc_str] = 'enc'
        self.task_phase[self.task_phase == rec_str] = 'rec'

    def add_loc_info(self):

        # also create the elctrode location dictionary
        tal = ram_data_helpers.load_tal(self.subj, self.montage, self.bipolar, self.use_json)
        self.elec_locs = ram_data_helpers.bin_elec_locs(tal['loc_tag'],
                                                        tal['anat_region'],
                                                        np.stack(tal['xyz_indiv']))
        self.e_type = tal['e_type']
        self.elec_xyz_avg = tal['xyz_avg']
        self.elec_xyz_indiv = tal['xyz_indiv']
        self.elec_xyz_mni = tal['xyz_mni']
        self.elec_xyz_tal = tal['xyz_tal']

    def compute_power(self):

        # get electrodes
        elecs_bipol, elecs_monopol = ram_data_helpers.load_subj_elecs(self.subj, self.montage, self.use_json)

        # for each task_phase, get events
        full_pow_mat = None
        for s_time, e_time, phase in zip(self.start_time if isinstance(self.start_time, list) else [self.start_time],
                                         self.end_time if isinstance(self.end_time, list) else [self.end_time],
                                         self.feat_phase):
            events = ram_data_helpers.load_subj_events(self.task, self.subj, self.montage, phase, None,
                                                       False if self.bipolar else True, self.use_json)

            # create list for start and end times for power calc
            if callable(s_time) or callable(e_time):
                if s_time != e_time:
                    print('start_time and end_time functions must be the same.')
                    return
                s_times, e_times = s_time(events)
            else:
                s_times = e_times = None

            # compute power by session. This should help avoid using too much memory
            task_phase_pow_mat = None
            sessions = np.unique(events.session)
            for sess in sessions:
                print('sess: %d' % sess)

                sess_events = events[events.session == sess]

                # if we only have one start time and end time, then we can load all the evnets at once
                if s_times is None:
                    eeg_info = [[sess_events, s_time, e_time]]
                else:
                    eeg_info = zip(sess_events, s_times[events.session == sess], e_times[events.session == sess])

                ev_pow_mat = None
                for this_eeg_info in eeg_info:

                    # load eeg
                    eeg_reader = EEGReader(events=this_eeg_info[0], channels=elecs_monopol, start_time=this_eeg_info[1],
                                           end_time=this_eeg_info[2])
                    eeg = eeg_reader.read()

                    # add buffer
                    buf_dur = e_time - s_time - .01
                    if buf_dur > 2.0:
                        buf_dur = 2.0
                    eeg = eeg.add_mirror_buffer(duration=buf_dur)

                    # convert to bipolar
                    eeg = MonopolarToBipolarMapper(time_series=eeg, bipolar_pairs=elecs_bipol.view(np.recarray)).filter()

                    # filter line noise
                    eeg = eeg.filtered(freq_range=[58., 62.], filt_type='stop', order=4)

                    # downsample to conserve memory a bit
                    # eeg = eeg.resampled(500)
                    # eeg['samplerate'] = 500.

                    # compute power, in chunks if necessary.
                    all_chunk_pow = None
                    chunk_len = len([i for i in range(eeg.shape[0]) if i * np.prod(eeg.shape[1:]) * self.freqs.shape[0] < 1e9/2])
                    chunk_vals = zip(np.arange(0, eeg.shape[0], chunk_len), np.append(np.arange(0, eeg.shape[0], chunk_len)[1:], eeg.shape[0]))
                    print(len(chunk_vals))
                    for chunk in chunk_vals:
                        print(chunk)
                        chunk_pow_mat, _ = MorletWaveletFilterCpp(time_series=eeg[chunk[0]:chunk[1]], freqs=self.freqs,
                                                                  output='power', width=5, cpus=25).filter()
                        dims = chunk_pow_mat.dims

                        # remove buffer and log transform
                        chunk_pow_mat = chunk_pow_mat.remove_buffer(buf_dur)
                        data = chunk_pow_mat.data
                        chunk_pow_mat.data = numexpr.evaluate('log10(data)')
                        dim_str = chunk_pow_mat.dims[1]
                        coord = chunk_pow_mat.coords[dim_str]
                        ev = chunk_pow_mat.events
                        freqs = chunk_pow_mat.frequency
                        # np.log10(chunk_pow_mat.data, out=chunk_pow_mat.data)

                        # mean power over time or time bins
                        if self.time_bins is None:
                            chunk_pow_mat = chunk_pow_mat.mean(axis=3)
                        else:
                            pow_list = []
                            # pdb.set_trace()
                            for t, tbin in enumerate(self.time_bins):
                                # print(t)
                                # tmp2 = [np.mean(chunk_pow_mat.data[:, :, :, inds], axis=3) for inds in tmp]
                                # tmp = [(chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1]) for tbin in self.time_bins]
                                # tmp = [np.where((chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1]))[0] for tbin in self.time_bins]
                                # tmp2 = np.expand_dims(np.stack(tmp, 0), 0)
                                # chunk_pow_mat.data.T[tmp3].mean(axis=1).T
                                inds = (chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1])
                                pow_list.append(np.mean(chunk_pow_mat.data[:, :, :, inds], axis=3))
                            chunk_pow_mat = np.stack(pow_list, axis=3)
                            chunk_pow_mat = TimeSeriesX(data=chunk_pow_mat,
                                                        dims=['frequency', dim_str, 'events', 'time'],
                                                        coords={'frequency': freqs,
                                                                dim_str: coord,
                                                                'events': ev,
                                                                'time': self.time_bins.mean(axis=1)})

                        all_chunk_pow = chunk_pow_mat if all_chunk_pow is None else concat([all_chunk_pow,
                                                                                           chunk_pow_mat],
                                                                                           dim=dims[1])
                    ev_pow_mat = all_chunk_pow if ev_pow_mat is None else concat((ev_pow_mat, all_chunk_pow), dim='events')
                task_phase_pow_mat = ev_pow_mat if task_phase_pow_mat is None else concat((task_phase_pow_mat, ev_pow_mat), dim='events')
            full_pow_mat = task_phase_pow_mat if full_pow_mat is None else concat((full_pow_mat, task_phase_pow_mat), dim='events')

        # make sure events is the first dim and store as self.subject_data
        ev_dim = np.where(np.array(full_pow_mat.dims) == 'events')[0]
        new_dim_order = np.hstack([ev_dim, np.setdiff1d(range(full_pow_mat.ndim), ev_dim)])
        self.subject_data = full_pow_mat.transpose(*np.array(full_pow_mat.dims)[new_dim_order])

    def save_data(self):
        """
        Saves self.data as a pickle to location defined by _generate_save_path.
        """
        if self.subject_data is None:
            print('Data must be loaded before saving. Use .load_data()')
            return

        # make directories if missing
        if not os.path.exists(os.path.split(self.save_dir)[0]):
            try:
                os.makedirs(os.path.split(self.save_dir)[0])
            except OSError:
                pass
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                pass

        # pickle file
        joblib.dump(self.subject_data, self.save_file)
        # with open(self.save_file, 'wb') as f:
        #     pickle.dump(self.subject_data, f, protocol=-1)

    def _generate_save_path(self, base_dir):
        """
        Define save directory based on settings so things stay reasonably organized on disk. Return string.
        """

        f1 = self.freqs[0]
        f2 = self.freqs[-1]
        bipol_str = 'bipol' if self.bipolar else 'mono'
        tbin_str = '1_bin' if self.time_bins is None else str(self.time_bins.shape[0]) + '_bins'

        if callable(self.start_time):
            start_stop_zip = zip(self.feat_phase, [self.start_time.__name__])
            start_stop_str = '_'.join(['%s_%s' % (x[0], x[1]) for x in start_stop_zip])
        else:
            start_stop_zip = zip(self.feat_phase,
                                 self.start_time if isinstance(self.start_time, list) else [self.start_time],
                                 self.end_time if isinstance(self.end_time, list) else [self.end_time])
            start_stop_str = '_'.join(['%s_start_%.1f_stop_%.1f' % (x[0], x[1], x[2]) for x in start_stop_zip])

        base_dir = os.path.join(base_dir,
                                self.task,
                                '%d_freqs_%.1f_%.1f_%s' % (len(self.freqs), f1, f2, bipol_str),
                                start_stop_str,
                                tbin_str,
                                self.subj,
                                str(self.montage),
                                self.feat_type)

        return base_dir
