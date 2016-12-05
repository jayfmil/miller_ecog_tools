"""
Contains default settings for various classifications analyses.
"""
import ram_data_helpers
import numpy as np


def get_default_analysis_params(analysis='enc'):

    defaults = {}
    if analysis == 'all_events_class_enc':
        defaults = {
                'subjs': ram_data_helpers.get_subjs('RAM_TH1'),
                'task': 'RAM_TH1',
                'train_phase': ['enc'],
                'test_phase': ['enc'],
                'feat_phase': ['enc', 'rec_circle'],
                'start_time': [-1.2, -2.9],
                'end_time': [0.5, -0.2],
                'bipolar': True,
                'freqs': np.logspace(np.log10(1), np.log10(200), 8),
                'feat_type': 'power',
                'recall_filter_func': ram_data_helpers.filter_events_to_recalled,
                'exclude_by_rec_time': False,
                'force_reclass': False,
                'save_class': True,
            }

    elif analysis == 'test':
        defaults = {
            'subjs': ['R1076D', 'R1236J'],
            'task': 'RAM_TH1',
            'train_phase': ['enc'],
            'test_phase': ['enc'],
            'feat_phase': ['enc', 'rec_circle'],
            'start_time': [-1.2, -2.9],
            'end_time': [0.5, -0.2],
            'bipolar': True,
            'freqs': np.logspace(np.log10(1), np.log10(200), 8),
            'feat_type': 'power',
            'recall_filter_func': ram_data_helpers.filter_events_to_recalled,
            'exclude_by_rec_time': False,
            'force_reclass': False,
            'save_class': True,
        }

    return defaults
