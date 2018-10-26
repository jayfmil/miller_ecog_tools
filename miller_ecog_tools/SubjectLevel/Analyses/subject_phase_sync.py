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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from scipy.signal import hilbert
from itertools import combinations
from joblib import Parallel, delayed
from copy import deepcopy

from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_ram_eeg_data import SubjectRamEEGData


class SubjectPhaseSyncAnalysis(SubjectAnalysisBase, SubjectRamEEGData):
    """
    Subclass of SubjectAnalysis and SubjectRamEEGData

    The user must define the .recall_filter_func attribute of this class. This should be a function that, given a set
    of events, returns a boolean array of recalled (True) and not recalled (False) items.
    """

    res_str_tmp = 'phase_sync_{0}_start_{1}_stop_{2}_range_{3}_bipolar_{4}.p'
    attrs_in_res_str = ['start_time', 'end_time', 'hilbert_band_pass_range', 'roi_list', 'bipolar']

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

        self.hilbert_band_pass_range = [1, 4]

        self.do_perm_test = False
        self.n_perms = 500

        self.include_phase_diffs_in_res = True

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
        region_df = self.bin_eloctrodes_into_rois()
        region_df['merged_col'] = region_df['hemi'] + '-' + region_df['region']

        # make sure we have electrodes in each unique region
        for roi in self.roi_list:
            has_elecs = []
            for label in roi:
                if np.any(region_df.merged_col == label):
                    has_elecs.append(True)
            if ~np.any(has_elecs):
                print('{}: no {} electrodes, cannot compute synchrony.'.format(self.subject, roi))
                return

        # then filter into just to ROIs defined above
        elecs_to_use = region_df.merged_col.isin([item for sublist in self.roi_list for item in sublist])
        elec_scheme = self.elec_info.copy(deep=True)
        elec_scheme['ROI'] = region_df.merged_col[elecs_to_use]
        elec_scheme = elec_scheme[elecs_to_use].reset_index()

        # load eeg with pass band
        # phase_data = RAM_helpers.load_eeg(self.subject_data,
        #                                   self.start_time,
        #                                   self.end_time,
        #                                   buf_ms=self.buf_ms,
        #                                   elec_scheme=elec_scheme,
        #                                   noise_freq=self.noise_freq,
        #                                   resample_freq=self.resample_freq,
        #                                   pass_band=self.hilbert_band_pass_range)

        # get phase at each timepoint
        phase_data = deepcopy(self.subject_data[:, elecs_to_use])
        phase_data.data = np.angle(hilbert(phase_data.data, N=phase_data.shape[-1], axis=-1))

        # remove the buffer
        phase_data = phase_data.remove_buffer(self.buf_ms / 1000.)

        # loop over each pair of ROIs
        for region_pair in combinations(self.roi_list, 2):
            elecs_region_1 = np.where(elec_scheme.ROI.isin(region_pair[0]))[0]
            elecs_region_2 = np.where(elec_scheme.ROI.isin(region_pair[1]))[0]

            elec_label_pairs = []
            elec_pair_pvals = []
            elec_pair_zs = []
            elec_pair_rvls = []
            elec_pair_pvals_rec = []
            elec_pair_zs_rec = []
            elec_pair_rvls_rec = []
            elec_pair_pvals_nrec = []
            elec_pair_zs_nrec = []
            elec_pair_rvls_nrec = []
            delta_mem_rayleigh_zscores = []
            delta_mem_rvl_zscores = []

            elec_pair_phase_diffs = []

            # loop over all pairs of electrodes in the ROIs
            for elec_1 in elecs_region_1:
                for elec_2 in elecs_region_2:
                    elec_label_pairs.append([elec_scheme.iloc[elec_1].label, elec_scheme.iloc[elec_2].label])

                    # and take the difference in phase values for this electrode pair
                    elec_pair_phase_diff = pycircstat.cdiff(phase_data[:, elec_1], phase_data[:, elec_2])
                    if self.include_phase_diffs_in_res:
                        elec_pair_phase_diffs.append(elec_pair_phase_diff)

                    # compute the circular stats
                    elec_pair_stats = calc_circ_stats(elec_pair_phase_diff, recalled, do_perm=False)
                    elec_pair_pvals.append(elec_pair_stats['elec_pair_pval'])
                    elec_pair_zs.append(elec_pair_stats['elec_pair_z'])
                    elec_pair_rvls.append(elec_pair_stats['elec_pair_rvl'])
                    elec_pair_pvals_rec.append(elec_pair_stats['elec_pair_pval_rec'])
                    elec_pair_zs_rec.append(elec_pair_stats['elec_pair_z_rec'])
                    elec_pair_pvals_nrec.append(elec_pair_stats['elec_pair_pval_nrec'])
                    elec_pair_zs_nrec.append(elec_pair_stats['elec_pair_z_nrec'])
                    elec_pair_rvls_rec.append(elec_pair_stats['elec_pair_rvl_rec'])
                    elec_pair_rvls_nrec.append(elec_pair_stats['elec_pair_rvl_nrec'])

                    # compute null distributions for the memory stats
                    if self.do_perm_test:
                        delta_mem_rayleigh_zscore, delta_mem_rvl_zscore = self.compute_null_stats(elec_pair_phase_diff,
                                                                                                  recalled,
                                                                                                  elec_pair_stats)
                        delta_mem_rayleigh_zscores.append(delta_mem_rayleigh_zscore)
                        delta_mem_rvl_zscores.append(delta_mem_rvl_zscore)

            region_pair_key = '+'.join(['-'.join(r) for r in region_pair])
            self.res[region_pair_key] = {}
            self.res[region_pair_key]['elec_label_pairs'] = elec_label_pairs
            self.res[region_pair_key]['elec_pair_pvals'] = np.stack(elec_pair_pvals, 0)
            self.res[region_pair_key]['elec_pair_zs'] = np.stack(elec_pair_zs, 0)
            self.res[region_pair_key]['elec_pair_rvls'] = np.stack(elec_pair_rvls, 0)
            self.res[region_pair_key]['elec_pair_pvals_rec'] = np.stack(elec_pair_pvals_rec, 0)
            self.res[region_pair_key]['elec_pair_zs_rec'] = np.stack(elec_pair_zs_rec, 0)
            self.res[region_pair_key]['elec_pair_pvals_nrec'] = np.stack(elec_pair_pvals_nrec, 0)
            self.res[region_pair_key]['elec_pair_zs_nrec'] = np.stack(elec_pair_zs_nrec, 0)
            self.res[region_pair_key]['elec_pair_rvls_rec'] = np.stack(elec_pair_rvls_rec, 0)
            self.res[region_pair_key]['elec_pair_rvls_nrec'] = np.stack(elec_pair_rvls_nrec, 0)
            if self.do_perm_test:
                self.res[region_pair_key]['delta_mem_rayleigh_zscores'] = np.stack(delta_mem_rayleigh_zscores, 0)
                self.res[region_pair_key]['delta_mem_rvl_zscores'] = np.stack(delta_mem_rvl_zscores, 0)
            if self.include_phase_diffs_in_res:
                self.res[region_pair_key]['elec_pair_phase_diffs'] = np.stack(elec_pair_phase_diffs, -1)
            self.res[region_pair_key]['time'] = phase_data.time.data
            self.res[region_pair_key]['recalled'] = recalled

    def compute_null_stats(self, elec_pair_phase_diff, recalled, elec_pair_stats):

        res = Parallel(n_jobs=12, verbose=5)(delayed(calc_circ_stats)(elec_pair_phase_diff, recalled, True)
                                             for _ in range(self.n_perms))

        # for the rayleigh z and the resultant vector length, compute the actual difference between good and bad
        # memory at each timepoint. Then compute a null distribution from shuffled data. Then compute the rank of the
        # real data compared to the shuffled at each timepoint. Convert rank to z-score and return
        null_elec_pair_zs_rec = np.stack([x['elec_pair_z_rec'] for x in res], 0)
        null_elec_pair_zs_nrec = np.stack([x['elec_pair_z_nrec'] for x in res], 0)
        null_delta_mem_zs = null_elec_pair_zs_rec - null_elec_pair_zs_nrec
        real_delta_mem_zs = elec_pair_stats['elec_pair_z_rec'] - elec_pair_stats['elec_pair_z_nrec']
        delta_mem_zs_rank = np.mean(real_delta_mem_zs > null_delta_mem_zs, axis=0)
        delta_mem_zs_rank[delta_mem_zs_rank == 0] += 1/self.n_perms
        delta_mem_zs_rank[delta_mem_zs_rank == 1] -= 1 / self.n_perms

        null_elec_pair_rvls_rec = np.stack([x['elec_pair_rvl_rec'] for x in res], 0)
        null_elec_pair_rvls_nrec = np.stack([x['elec_pair_rvl_nrec'] for x in res], 0)
        null_delta_mem_rvls = null_elec_pair_rvls_rec - null_elec_pair_rvls_nrec
        real_delta_mem_rvls = elec_pair_stats['elec_pair_rvl_rec'] - elec_pair_stats['elec_pair_rvl_nrec']
        delta_mem_rvls_rank = np.mean(real_delta_mem_rvls > null_delta_mem_rvls, axis=0)
        delta_mem_rvls_rank[delta_mem_rvls_rank == 0] += 1/self.n_perms
        delta_mem_rvls_rank[delta_mem_rvls_rank == 1] -= 1 / self.n_perms

        return norm.ppf(delta_mem_zs_rank), norm.ppf(delta_mem_rvls_rank)

    def bin_eloctrodes_into_rois(self):
        """
        Bin electrode into broader ROIs.
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

    # the following properties and setters automatically change the res_str so the saved files will have useful names
    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, t):
        self._start_time = t
        self.set_res_str()

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, t):
        self._end_time = t
        self.set_res_str()

    @property
    def hilbert_band_pass_range(self):
        return self._hilbert_band_pass_range

    @hilbert_band_pass_range.setter
    def hilbert_band_pass_range(self, t):
        self._hilbert_band_pass_range = t
        self.set_res_str()

    @property
    def roi_list(self):
        return self._roi_list

    @roi_list.setter
    def roi_list(self, t):
        self._roi_list = t
        self.set_res_str()

    @property
    def bipolar(self):
        return self._bipolar

    @bipolar.setter
    def bipolar(self, t):
        self._bipolar = t
        self.set_res_str()

    def set_res_str(self):
        if np.all([hasattr(self, x) for x in SubjectPhaseSyncAnalysis.attrs_in_res_str]):
            self.res_str = SubjectPhaseSyncAnalysis.res_str_tmp.format(self.start_time, self.end_time,
                                                                       '-'.join([str(x) for x in self.hilbert_band_pass_range]),
                                                                       '+'.join(['-'.join(r) for r in self.roi_list]),
                                                                       self.bipolar)


def calc_circ_stats(elec_pair_phase_diff, recalled, do_perm=False):
    if do_perm:
        recalled = np.random.permutation(recalled)

    # compute rayleigh on the phase difference
    elec_pair_pval, elec_pair_z = pycircstat.rayleigh(elec_pair_phase_diff, axis=0)

    # compute rvl on the phase difference
    elec_pair_rvl = pycircstat.resultant_vector_length(elec_pair_phase_diff, axis=0)

    # also compute for recalled and not recalled items
    elec_pair_pval_rec, elec_pair_z_rec = pycircstat.rayleigh(elec_pair_phase_diff[recalled], axis=0)
    elec_pair_pval_nrec, elec_pair_z_nrec = pycircstat.rayleigh(elec_pair_phase_diff[~recalled], axis=0)

    # and also compute resultant vector length
    elec_pair_rvl_rec = pycircstat.resultant_vector_length(elec_pair_phase_diff[recalled], axis=0)
    elec_pair_rvl_nrec = pycircstat.resultant_vector_length(elec_pair_phase_diff[~recalled], axis=0)

    return {'elec_pair_pval': elec_pair_pval,
            'elec_pair_z': elec_pair_z,
            'elec_pair_rvl': elec_pair_rvl,
            'elec_pair_pval_rec': elec_pair_pval_rec,
            'elec_pair_z_rec': elec_pair_z_rec,
            'elec_pair_pval_nrec': elec_pair_pval_nrec,
            'elec_pair_z_nrec': elec_pair_z_nrec,
            'elec_pair_rvl_rec': elec_pair_rvl_rec,
            'elec_pair_rvl_nrec': elec_pair_rvl_nrec}