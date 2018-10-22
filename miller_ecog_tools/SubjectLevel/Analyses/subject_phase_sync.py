"""
Basic Subsequent Memory Effect Analysis. For every electrode and frequency, compare correctly and incorrectly recalled
items using a t-test.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pycircstat
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, ttest_ind
from scipy.signal import hilbert
from itertools import combinations

from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_events_data import SubjectEventsRAMData


class SubjectPhaseSyncAnalysis(SubjectAnalysisBase, SubjectEventsRAMData):
    """
    Subclass of SubjectAnalysis and SubjectEventsRAMData

    The user must define the .recall_filter_func attribute of this class. This should be a function that, given a set
    of events, returns a boolean array of recalled (True) and not recalled (False) items.
    """

    res_str_tmp = 'elec_cluster_%d_mm_%d_elec_min_%s_elec_type_%s_sep_hemis_%.2f_cluster_range.p'
    attrs_in_res_str = ['elec_types_allowed', 'min_elec_dist', 'min_num_elecs', 'separate_hemis', 'cluster_freq_range']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectPhaseSyncAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = SubjectPhaseSyncAnalysis.res_str_tmp

        # The SME analysis is a contract between two conditions (recalled and not recalled items). Set
        # recall_filter_func to be a function that takes in events and returns bool of recalled items
        self.recall_filter_func = None

        # a list of lists defining ROIs. Each sublist will be treated as a single ROI. Append `left-` and `right-` to
        # each label you input
        self.roi_list = [['left-IFG'], ['left-Hipp', 'right-Hipp']]

        self.start_time = -500
        self.end_time = 1500
        self.wave_num = 5
        self.buf_ms = 2000
        self.noise_freq = [58., 62.]
        self.resample_freq = 250.
        self.hilbert_band_pass_range = [1, 4]
        self.log_power = True

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """

        """

        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)

        # Get recalled or not labels
        if self.recall_filter_func is None:
            print('%s SME: please provide a .recall_filter_func function.' % self.subject)
        recalled = self.recall_filter_func(self.subject_data)

        # filter to electrodes in ROIs. First get broad electrode region labels
        region_df = self.bin_electrodes_by_region()
        region_df['merged_col'] = region_df['hemi'] + '-' + region_df['region']

        # make sure we have electrodes in each unique region
        for roi in self.roi_list:
            for label in roi:
                if ~np.any(region_df.merged_col == label):
                    print('{}: no {} electrodes, cannot compute synchrony.'.format(self.subject, label))
                    return

        # then filter into just to ROIs defined above
        elecs_to_use = region_df.merged_col.isin([item for sublist in self.roi_list for item in sublist])
        elec_scheme = self.elec_info.copy(deep=True)
        elec_scheme['ROI'] = region_df.merged_col[elecs_to_use]
        elec_scheme = elec_scheme[elecs_to_use].reset_index()

        # load eeg with pass band
        phase_data = RAM_helpers.load_eeg(self.subject_data,
                                          self.start_time,
                                          self.end_time,
                                          buf_ms=self.buf_ms,
                                          elec_scheme=elec_scheme,
                                          noise_freq=self.noise_freq,
                                          resample_freq=self.resample_freq,
                                          pass_band=self.hilbert_band_pass_range)

        # get phase at each timepoint
        phase_data.data = np.angle(hilbert(phase_data, N=phase_data.shape[-1], axis=-1))

        # remove the buffer
        phase_data = phase_data.remove_buffer(self.buf_ms / 1000.)

        # so now we have event x elec x time phase values. What to do?
        # define the pairs
        elec_label_pairs = []
        elec_region_pairs = []
        elec_pair_pvals = []
        elec_pair_zs = []
        elec_pair_pvals_rec = []
        elec_pair_zs_rec = []
        elec_pair_pvals_nrec = []
        elec_pair_zs_nrec = []

        # loop over each pair of ROIs
        for region_pair in combinations(self.roi_list, 2):
            elecs_region_1 = np.where(elec_scheme.ROI.isin(region_pair[0]))[0]
            elecs_region_2 = np.where(elec_scheme.ROI.isin(region_pair[1]))[0]

            # loop over all pairs of electrodes in the ROIs
            for elec_1 in elecs_region_1:
                for elec_2 in elecs_region_2:
                    elec_label_pairs.append([elec_scheme.iloc[elec_1].label, elec_scheme.iloc[elec_2].label])
                    elec_region_pairs.append(region_pair)

                    # and take the difference in phase values for this electrode pair
                    elec_pair_diff = pycircstat.cdiff(phase_data[:, elec_1], phase_data[:, elec_2])

                    # compute rayleigh on the phase difference
                    elec_pair_pval, elec_pair_z = pycircstat.rayleigh(elec_pair_diff, axis=0)
                    elec_pair_pvals.append(elec_pair_pval)
                    elec_pair_zs.append(elec_pair_z)

                    # also compute for recalled and not recalled items
                    elec_pair_pval_rec, elec_pair_z_rec = pycircstat.rayleigh(elec_pair_diff[recalled], axis=0)
                    elec_pair_pvals_rec.append(elec_pair_pval_rec)
                    elec_pair_zs_rec.append(elec_pair_z_rec)

                    elec_pair_pval_nrec, elec_pair_z_nrec = pycircstat.rayleigh(elec_pair_diff[~recalled], axis=0)
                    elec_pair_pvals_nrec.append(elec_pair_pval_nrec)
                    elec_pair_zs_nrec.append(elec_pair_z_nrec)

                    # do some shuffling here. Probably pull this whole section out into different function




    def bin_eloctrodes_into_rois(self):
        """

        Returns
        -------

        """

        # figure out the column names to use. Can very depending on where the electrode info came from
        if 'stein.region' in self.elec_info:
            region_key1 = 'stein.region'
        elif 'locTag' in self.elec_info:
            region_key1 = 'locTag'
        else:
            region_key1 = ''

        if 'ind.region' in self.elec_info:
            region_key2 = 'ind.region'
        else:
            region_key2 = 'indivSurf.anatRegion'
        hemi_key = 'ind.x' if 'ind.x' in self.elec_info else 'indivSurf.x'
        if self.elec_info[hemi_key].iloc[0] == 'NaN':
            hemi_key = 'tal.x'

        # hardcoding this dictionary mapping electrode labels to regions
        roi_dict = {'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                             'Right CA3', 'Right DG', 'Right Sub'],
                    'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC'],
                    'IFG': ['parsopercularis', 'parsorbitalis', 'parstriangularis'],
                    'MFG': ['caudalmiddlefrontal', 'rostralmiddlefrontal'],
                    'SFG': ['superiorfrontal'],
                    'Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal'],
                    'Parietal': ['inferiorparietal', 'supramarginal', 'superiorparietal', 'precuneus'],
                    'Occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']}

        regions = self.bin_electrodes_by_region(elec_column1=region_key1 if region_key1 else region_key2,
                                                elec_column2=region_key2,
                                                x_coord_column=hemi_key,
                                                roi_dict=roi_dict)
        return regions
