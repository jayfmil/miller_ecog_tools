import numpy as np
from numpy.lib.recfunctions import append_fields
from ptsa.data.readers import BaseEventReader
from scipy.stats import binned_statistic_2d
import re
import os
from glob import glob
import pdb

class TaskBehavior:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    x_min = 359.9
    x_max = 409.9
    y_min = 318.0
    y_max = 399.3

    def __init__(self, subjs=None, num_bins=15):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = self.get_th_subjs()
        self.subjs = subjs

        self.x_edges = np.linspace(TaskBehavior.x_min, TaskBehavior.x_max, num_bins+1)
        self.y_edges = np.linspace(TaskBehavior.y_min, TaskBehavior.y_max, num_bins+1)

    def beh_ana_for_all_subjs(self):
        stat = []
        stat_non_flipped = []
        stat_flipped = []
        for subj in self.subjs:
            print 'Processing %s.' % subj
            subj_res = self.beh_ana_for_single_subj(subj)
            stat.append(subj_res[0])
            stat_non_flipped.append(subj_res[1])
            stat_flipped.append(subj_res[2])
        return stat, stat_non_flipped, stat_flipped

    def beh_ana_for_single_subj(self, subj):

        # load events for subject
        events = self.load_subj_events(subj)

        # add field for error percentile (performance factor)
        error_percentiles = self.calc_norm_dist_error(events.locationX, events.locationY, events.distErr)
        events = append_fields(events, 'norm_err', error_percentiles, dtypes=float, usemask=False, asrecarray=True)

        # bin error by x and y
        x = events.locationX
        y = events.locationY
        err = events.distErr
        stat = binned_statistic_2d(x, y, err, bins=[self.x_edges, self.y_edges])[0]

        # bin only for non-flipped trials
        stat_non_flipped = binned_statistic_2d(x[events.isRecFromStartSide == 1],
                                               y[events.isRecFromStartSide == 1],
                                               err[events.isRecFromStartSide == 1], bins=[self.x_edges, self.y_edges])[0]

        # bin only for flipped trials
        stat_flipped = binned_statistic_2d(x[events.isRecFromStartSide == 0],
                                           y[events.isRecFromStartSide == 0],
                                           err[events.isRecFromStartSide == 0], bins=[self.x_edges, self.y_edges])[0]
        return stat, stat_non_flipped, stat_flipped

    @classmethod
    def load_subj_events(cls, subj):
        """Returns subject event structure."""
        subj_ev_path = os.path.join(cls.event_path, subj + '_events.mat')
        e_reader = BaseEventReader(filename=subj_ev_path, eliminate_events_with_no_eeg=True)
        events = e_reader.read()

        # change the item field name to item_name to not cause issues with item()
        events.dtype.names = ['item_name' if i == 'item' else i for i in events.dtype.names]
        ev_order = np.argsort(events, order=('session', 'trial', 'mstime'))
        events = events[ev_order]

        # filter to just chest events, including empty
        events = events[(events.type == 'CHEST') & (events.confidence >= 0)]
        return events

    @classmethod
    def get_th_subjs(cls):
        """Returns list of subjects who performed TH1."""
        subjs = glob(os.path.join(cls.event_path, 'R*_events.mat'))
        subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
        subjs.sort()
        return subjs

    @classmethod
    def calc_norm_dist_error(cls, x_pos, y_pos, act_errs):
        rand_x = np.random.uniform(359.9, 409.9, 100000)
        rand_y = np.random.uniform(318.0, 399.3, 100000)

        error_percentiles = np.zeros(np.shape(act_errs), dtype=float)
        for i, this_item in enumerate(zip(x_pos, y_pos, act_errs)):
            if np.isnan(this_item[2]):
                error_percentiles[i] = np.nan
            else:
                possible_errors = np.sqrt((rand_x - this_item[0]) ** 2 + (rand_y - this_item[1]) ** 2)
                error_percentiles[i] = np.mean(possible_errors < this_item[2])
        return error_percentiles
