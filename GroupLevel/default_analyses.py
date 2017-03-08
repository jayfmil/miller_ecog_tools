"""
Contains default settings for various analyses to make it easier to run.
"""
import ram_data_helpers
import numpy as np
from SubjectLevel.Analyses import *


def get_default_analysis_params(analysis='classify_enc', subject_settings='default'):
    """
    Returns a dictionary of parameters for the desired combination of analysis type and subject settings.
    """

    params = {}
    if (analysis == 'classify_enc') or (analysis == 'default'):
        params['ana_class'] = subject_classifier.SubjectClassifier
        params['train_phase'] = ['enc']
        params['test_phase'] = ['enc']
        params['norm'] = 'l2'
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'classify_enc_spectral_shift':
        params['ana_class'] = subject_classifier_spectral_shift.SubjectClassifier
        params['train_phase'] = ['enc']
        params['test_phase'] = ['enc']
        params['norm'] = 'l2'
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'classify_rec':
        params['ana_class'] = subject_classifier.SubjectClassifier
        params['train_phase'] = ['rec']
        params['test_phase'] = ['rec']
        params['norm'] = 'l2'
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'classify_both':
        params['ana_class'] = subject_classifier.SubjectClassifier
        params['train_phase'] = ['enc', 'rec']
        params['test_phase'] = ['enc', 'rec']
        params['scale_enc'] = 2.0
        params['norm'] = 'l2'
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'sme_enc':
        params['ana_class'] = subject_SME.SubjectSME
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'sme_enc_timebins':
        params['ana_class'] = subject_SME_timebins.SubjectSMETime
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'sme_rec':
        params['ana_class'] = subject_SME.SubjectSME
        params['task_phase_to_use'] = ['rec']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'sme_both':
        params['ana_class'] = subject_SME.SubjectSME
        params['task_phase_to_use'] = ['enc', 'rec']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'spectral_shift_enc':
        params['ana_class'] = subject_spectral_shift.SubjectSME
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'find_peaks_enc':
        params['ana_class'] = subject_find_spectral_peaks.SubjectPeaks
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'classify_enc_top_elecs':
        params['ana_class'] = subject_classifier_using_top_features.SubjectClassifier
        params['train_phase'] = ['enc']
        params['test_phase'] = ['enc']
        params['norm'] = 'l2'
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True
        params['do_top_elecs'] = True

    else:
        print('Invalid analysis: %s' % analysis)
        return {}

    if subject_settings == 'default':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.2]
        params['end_time'] = [0.5]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    elif subject_settings == 'default_enc_rec':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc', 'rec_circle']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.2, -2.9]
        params['end_time'] = [0.5, -0.2]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    elif subject_settings == 'default_50_freqs':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.2]
        params['end_time'] = [0.5]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

    elif subject_settings == 'default_50_freqs_timebins':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.5]
        params['end_time'] = [2.0]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

        starts = np.arange(-1.5, 2.0 - 0.5 + 0.1, 0.1)
        ends = starts + 0.5
        params['time_bins'] = np.stack([starts, ends], axis=-1)

    elif subject_settings == 'TH1_full_item':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-0.75]
        params['end_time'] = [2.25]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

    elif subject_settings == 'test':
        task = 'RAM_TH1'
        params['task'] = task
        # params['subjs'] = ['R1076D', 'R1241J']
        params['subjs'] = ['R1076D']
        params['feat_phase'] = ['enc', 'rec_circle']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.2, -2.9]
        params['end_time'] = [0.5, -0.2]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    elif subject_settings == 'test_FR1':
        task = 'RAM_FR1'
        params['task'] = task
        # params['subjs'] = ['R1076D', 'R1241J']
        params['subjs'] = ['R1076D']
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [0.0]
        params['end_time'] = [1.6]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    elif subject_settings == 'default_FR1':
        task = 'RAM_FR1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        # params['subjs'] = ['R1076D']
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [0.0]
        params['end_time'] = [1.6]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

    elif subject_settings == 'default_FR1_50_freqs_timebins':
        task = 'RAM_FR1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.5]
        params['end_time'] = [2.0]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

        starts = np.arange(-1.5, 2.0 - 0.5 + 0.1, 0.1)
        ends = starts + 0.5
        params['time_bins'] = np.stack([starts, ends], axis=-1)

    elif subject_settings == 'default_YC1':
        task = 'RAM_YC1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        # params['subjs'] = ['R1076D']
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [3.5]
        params['end_time'] = [5.5]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    else:
        print('Invalid subject settings: %s' % subject_settings)
        return {}

    return params
