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

    elif analysis == 'classify_enc_triangle':
        params['ana_class'] = subject_classifier_timebins.SubjectClassifier
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

    elif analysis == 'move_still':
        params['ana_class'] = subject_move_vs_still.SubjectMoveStill
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = None
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    elif analysis == 'traveling':
        params['ana_class'] = subject_oscillation_cluster.SubjectElecCluster
        params['task_phase_to_use'] = ['enc']
        params['recall_filter_func'] = ram_data_helpers.filter_events_to_recalled
        params['load_res_if_file_exists'] = False
        params['save_res'] = True

    else:
        print('Invalid analysis: %s' % analysis)
        return {}

    if subject_settings == 'default':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [0.0]
        params['end_time'] = [1.5]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

    elif subject_settings == 'triangle':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-2.0]
        params['end_time'] = [2.0]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 8)

        start_time = params['start_time'][0]
        end_time = params['end_time'][0]
        step_size = 0.1
        window_size = np.arange(0.1, np.ptp([start_time, end_time]) + .1, .1)

        # compute all time bins based on start_time, end_time, step_size, and window_size
        edges = np.arange(start_time * 1000, end_time * 1000, step_size * 1000)
        if int(np.ptp([start_time, end_time]) * 1000) % int(step_size * 1000) == 0:
            edges = np.append(edges, end_time * 1000)
        bin_centers = np.mean(np.array(zip(edges, edges[1:])), axis=1)
        time_bins = np.array([[x - window / 2, x + window / 2] for x in bin_centers for window in
                              (window_size * 1000).astype(int) if ((x + window / 2 <= end_time * 1000) &
                                                                   (x - window / 2 >= start_time * 1000))]) / 1000

        params['time_bins'] = time_bins

    elif subject_settings == 'THR':
        task = 'RAM_THR'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [0.0]
        params['end_time'] = [1.5]
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

    elif subject_settings == 'default_move_still':
        task = 'RAM_TH1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['move']
        params['feat_type'] = 'power'
        params['start_time'] = use_duration_field
        params['end_time'] = use_duration_field
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

        starts = np.arange(-1.5, 2.0 - 0.1 + 0.05, 0.05)
        ends = starts + 0.1
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

        starts = np.arange(-1.5, 2.0 - 0.1 + 0.05, 0.05)
        ends = starts + 0.1
        params['time_bins'] = np.stack([starts, ends], axis=-1)

    elif subject_settings == 'default_PAL1_50_freqs_timebins':
        task = 'RAM_PAL1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [-1.5]
        params['end_time'] = [2.0]
        params['bipolar'] = True
        params['freqs'] = np.logspace(np.log10(1), np.log10(200), 50)

        starts = np.arange(-1.5, 2.0 - 0.1 + 0.05, 0.05)
        ends = starts + 0.1
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

    elif subject_settings == 'traveling_FR1':
        task = 'RAM_FR1'
        params['task'] = task
        params['subjs'] = ram_data_helpers.get_subjs_and_montages(task)
        params['feat_phase'] = ['enc']
        params['feat_type'] = 'power'
        params['start_time'] = [0.0]
        params['end_time'] = [1.6]
        params['bipolar'] = False
        params['do_compute_sme'] = True
        params['freqs'] = np.logspace(np.log10(2), np.log10(32), 129)

    else:
        print('Invalid subject settings: %s' % subject_settings)
        return {}

    return params

def use_duration_field(events):
    # import pdb
    start_times = np.array([0.0] * len(events))
    end_times = events.duration/1000.
    end_times[end_times > 10.] = 10.
    bad = end_times < .5
    events = events[~bad]
    start_times = start_times[~bad]
    end_times = end_times[~bad]
    return events, start_times, end_times