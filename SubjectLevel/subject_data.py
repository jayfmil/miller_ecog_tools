import os
import pdb
import matplotlib
import ram_data_helpers
import cPickle as pickle
import numpy as np
from TH_load_features import load_features
from xray import concat
from subject import Subject


class SubjectData(Subject):
    """
    Data class contains default data settings and handles raw(ish) data IO.
    """

    def __init__(self, task=None, subject=None):
        super(SubjectData, self).__init__(task=task, subject=subject)
        self.task = 'RAM_TH1'
        self.subj = None
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

        # if data already exists on disk, just load it. If False, will recompute
        self.load_data_if_file_exists = True

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
        if self.load_data_if_file_exists & os.path.exists(self.save_file):
            print('Feature data already exists for %s, loading.' % self.subj)
            with open(self.save_file, 'rb') as f:
                subj_features = pickle.load(f)

        # otherwise compute
        else:
            subj_features = []
            # loop over all task phases
            for s_time, e_time, phase in zip(self.start_time if isinstance(self.start_time, list) else [self.start_time],
                                             self.end_time if isinstance(self.end_time, list) else [self.end_time],
                                             self.feat_phase):

                subj_features.append(load_features(self.subj, self.task, phase, s_time, e_time,
                                                   self.time_bins, self.freqs, self.freq_bands,
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
                                self.feat_type)

        return base_dir
