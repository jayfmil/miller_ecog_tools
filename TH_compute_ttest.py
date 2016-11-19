import numpy as np
import re
import os
from glob import glob
from scipy.stats import ttest_ind
import pdb
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from ptsa.data.readers import EEGReader
from ptsa.data.filters import MonopolarToBipolarMapper
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters.MorletWaveletFilter import MorletWaveletFilter
from ptsa.data.TimeSeriesX import TimeSeriesX
import cPickle as pickle
import cluster_helper.cluster

class ComputeTTest:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    power_dir = '/scratch/jfm2/python_power/TH/50_freqs/window_size_1000_step_size_10'

    def __init__(self, subjs=None, do_par=False):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = self.get_th_subjs()
        self.subjs = subjs

    def compute_ttest_at_each_time(self, subj):
        save_file = os.path.join(ComputeTTest.power_dir, subj + '.p')
        if not os.path.exists(save_file):
            return

        with open(save_file, 'rb') as f:
            subj_data = pickle.load(f)

        recalled = self.filter_events_to_recalled(subj_data)
        t, p = ttest_ind(subj_data[:, :, recalled, :], subj_data[:, :, ~recalled, :], axis=2)
        return t, p, subj_data.frequency.data

    def filter_events_to_recalled(self, subj_data):
        not_low_conf = subj_data.events.data['confidence'] > 0
        not_far_dist = subj_data.events.data['distErr'] < np.median(subj_data.events.data['distErr'])
        recalled = not_low_conf & not_far_dist
        return recalled

    @classmethod
    def get_th_subjs(cls):
        """Returns list of subjects who performed TH1."""
        subjs = glob(os.path.join(cls.event_path, 'R*_events.mat'))
        subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
        subjs.sort()
        return subjs
