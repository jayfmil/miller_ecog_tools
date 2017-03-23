"""
Functions for filtering a subject's data to exclude sessions that do not have enough trials (some sessions are cut
short).
"""
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp
from statsmodels.stats.proportion import proportions_chisquare
import pdb

# need to add in support for more tasks
criteria = {'RAM_TH1': {'n_lists': 25, 'ev_string': 'trial', 'perf_string': 'norm_err', 'chance': .5},
            'RAM_FR1': {'n_lists': 15, 'ev_string': 'list', 'perf_string': 'recalled'},
            'RAM_THR': {'n_lists': 15, 'ev_string': 'trial', 'perf_string': 'recalled'},
            'RAM_YC1': {'n_lists': 30, 'ev_string': 'blocknum', 'perf_string': 'norm_err'}}


def remove_first_session_if_worse(subj_obj):
    """
    Compare performance in the first session to performance in all later sessions. If the performance for the subject is
    significantly worse in the first session, remove it.
    """

    sessions = subj_obj.subject_data.events.data['session']
    if len(np.unique(sessions)) > 1:

        # first session indices
        inds0 = subj_obj.subject_data.events.data['session'] == subj_obj.subject_data.events.data['session'][0]

        # all later sessions
        inds1 = subj_obj.subject_data.events.data['session'] != subj_obj.subject_data.events.data['session'][0]

        # comparing the two distributions
        # chisquare test for the free recall task
        if 'FR' in subj_obj.task:
            corr0 = np.sum(subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds0])
            corr1 = np.sum(subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds1])
            total0 = len(subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds0])
            total1 = len(subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds1])
            chisqr, p, _ = proportions_chisquare([corr0, corr1], [total0, total1])

        # ttest for treasure hunt
        elif 'TH' in subj_obj.task:
            t, p = ttest_ind(subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds0],
                             subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']][inds1])
        else:
            print('Task not recognized in subject_exclusions.py')
            p = 1

        # if sig
        if p < .05:
            print('%s: removing first session based on performance difference.' % subj_obj.subj)
            subj_obj.subject_data = subj_obj.subject_data[inds1]
            subj_obj.task_phase = subj_obj.task_phase[inds1]
    return subj_obj


def remove_abridged_sessions(subj_obj):
    """
    Loop over each session in the subj_obj.subject_data.events and see if it meets the minumum number of trials. If not,
    exclude it.
    """

    sessions = subj_obj.subject_data.events.data['session']
    bad_evs = np.ones(sessions.shape, dtype=bool)
    uniq_sessions = np.unique(sessions)
    bad_sessions = []
    for session in uniq_sessions:

        # number of unique lists in the session
        sess_inds = sessions == session
        sess_lists = subj_obj.subject_data.events[sess_inds].data[criteria[subj_obj.task]['ev_string']]
        n = np.unique(sess_lists).shape[0]

        # mark as good or bad
        is_bad = n <= criteria[subj_obj.task]['n_lists']
        bad_evs[sess_inds] = is_bad
        bad_sessions.append(is_bad)
    bad_sessions = np.array(bad_sessions)
    print '%s: Removing sessions ' %subj_obj.subj + ', '.join([str(x) for x in uniq_sessions[bad_sessions]]) + \
          ' (%d of %d)' % (np.sum(bad_sessions), len(bad_sessions))

    # remove bad sessions
    if np.all(bad_sessions):
        subj_obj.subject_data = None
        subj_obj.task_phase = None
    else:
        subj_obj.subject_data = subj_obj.subject_data[~bad_evs]
        subj_obj.task_phase = subj_obj.task_phase[~bad_evs]
    return subj_obj


def remove_subj_if_at_chance(subj_obj):
    """
    Make sure performace is not at chance. This makes sense for TH1/2, but not really other taskss
    """

    perf_data = subj_obj.subject_data.events.data[criteria[subj_obj.task]['perf_string']]
    t, p = ttest_1samp(perf_data, criteria[subj_obj.task]['chance'])
    if p > .05:
        print('%s: %.3f normalized error, at chance. Removing subject.' % (subj_obj.subj, np.mean(perf_data)))
        subj_obj.subject_data = None
        subj_obj.task_phase = None
    else:
        print('%s: %.3f normalized error, better than chance.' % (subj_obj.subj, np.mean(perf_data)))

    return subj_obj
