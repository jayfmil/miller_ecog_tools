import cPickle as pickle
import os
import numpy as np
from xray import concat
import ram_data_helpers
from TH_load_features import load_features
from subject import Subject


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
        self.bipolar = True
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
                force_recompute = True
                print('%s: Events have been modified since data created, recomputing.' % self.subj)

        elif self.do_not_compute:
            print('%s: subject_data does not exist, not computing.' % self.subj)
            return

        if not force_recompute and self.load_data_if_file_exists and os.path.exists(self.save_file):
            print('%s: Input data already exists, loading.' % self.subj)
            with open(self.save_file, 'rb') as f:
                subj_features = pickle.load(f)

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

        # store as self.data
        self.subject_data = subj_features

        # also create the elctrode location dictionary
        self.elec_locs = ram_data_helpers.bin_elec_locs(self.subject_data.attrs['loc_tag'],
                                                        self.subject_data.attrs['anat_region'],
                                                        self.subject_data.attrs['chan_tags'])
        self.e_type = self.subject_data.attrs['e_type']
        self.elec_xyz_avg = self.subject_data.attrs['xyz_avg']
        self.elec_xyz_indiv = self.subject_data.attrs['xyz_indiv']

        # lastly, create task_phase array that is standardized regardless of experiment
        self.task_phase = self.subject_data.events.data['type']

        if 'RAM_YC' in self.task:
            enc_str = 'NAV_LEARN'
            rec_str = 'NAV_TEST'
        elif 'RAM_TH' in self.task:
            enc_str = 'CHEST'
            rec_str = 'REC'
        elif 'RAM_PAL' in self.task:
            enc_str = 'STUDY_PAIR'
            rec_str = 'TEST_PROBE'
        else:
            enc_str = 'WORD'
            rec_str = 'REC_WORD'
        # enc_str = 'CHEST' if 'RAM_TH' in self.task else 'WORD'
        # rec_str = 'REC' if 'RAM_TH' in self.task else 'REC_WORD'
        self.task_phase[self.task_phase == enc_str] = 'enc'
        self.task_phase[self.task_phase == rec_str] = 'rec'

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
        with open(self.save_file, 'wb') as f:
            pickle.dump(self.subject_data, f, protocol=-1)

    def _generate_save_path(self, base_dir):
        """
        Define save directory based on settings so things stay reasonably organized on disk. Return string.
        """

        f1 = self.freqs[0]
        f2 = self.freqs[-1]
        bipol_str = 'bipol' if self.bipolar else 'mono'
        tbin_str = '1_bin' if self.time_bins is None else str(self.time_bins.shape[0]) + '_bins'

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
