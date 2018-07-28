import numpy as np
import ram_data_helpers
import matplotlib
import matplotlib.pyplot as plt
import random

def plot_correct_by_trial(subj):
    events = ram_data_helpers.load_subj_events('RAM_TH1', subj)
    correct = ram_data_helpers.filter_events_to_recalled('RAM_TH1', events)
    trials = events.trial
    blocks = events.block

    uniq_session = np.unique(events.session)
    fig, ax = plt.subplots(len(uniq_session), 1, squeeze=False)
    for s_count, sess in enumerate(uniq_session):
        sess_bool = events.session == sess
        trials_sess = trials[sess_bool]

        blocks_sess = blocks[sess_bool]
        corr_sess = correct[sess_bool]

        # ax[s_count].scatter(trials_sess[corr_sess], np.zeros(np.sum(corr_sess)), color='r')
        # ax[s_count].scatter(trials_sess[~corr_sess], np.zeros(np.sum(~corr_sess)), color='k')


        ax[s_count, 0].scatter(np.where(corr_sess)[0]+1, np.zeros(np.sum(corr_sess)), color='r')
        ax[s_count, 0].scatter(np.where(~corr_sess)[0]+1, np.zeros(np.sum(~corr_sess)), color='k')
        ax[s_count, 0].set_ylim(-0.01, .01)
        ax[s_count, 0].set_xlim(0, 101)
        ax[s_count, 0].set_yticks([])
        if s_count+1 < len(uniq_session):
            ax[s_count, 0].set_xticks([])

        trial_lines = np.where(np.diff(trials_sess))[0] + 1.5
        for trial_line in trial_lines:
            ax[s_count, 0].plot([trial_line]*2, [-0.01, 0.01], '--k')

    height = 1 if s_count == 0 else s_count
    fig.set_size_inches(18, height)

    return trials_sess


def pcorr_by_class(subj):
    events = ram_data_helpers.load_subj_events('RAM_TH1', subj)
    correct = ram_data_helpers.filter_events_to_recalled('RAM_TH1', events)
    # print np.mean(correct)


    uniq_session = np.unique(events.session)
    # fig, ax = plt.subplots(len(uniq_session), 1, squeeze=False)
    iters = 1000
    pcorr_after_good = np.zeros(len(uniq_session))
    pcorr_after_good_shuff = np.zeros((iters, len(uniq_session)))
    pval = np.zeros(len(uniq_session))
    for s_count, sess in enumerate(uniq_session):
        sess_bool = events.session == sess
        corr_sess = correct[sess_bool]
        print np.mean(corr_sess)

        good_resp = np.where(corr_sess[:-1])[0]
        good_next_resp = corr_sess[good_resp + 1]
        pcorr_after_good[s_count] = np.mean(good_next_resp)

        for i in range(iters):
            random.shuffle(corr_sess)
            good_resp = np.where(corr_sess[:-1])[0]
            good_next_resp = corr_sess[good_resp + 1]
            pcorr_after_good_shuff[i, s_count] = np.mean(good_next_resp)

        pval[s_count] = np.mean(pcorr_after_good_shuff[:, s_count] > pcorr_after_good[s_count])
    return pcorr_after_good, pcorr_after_good_shuff, pval

        # bad_resp = np.where(~corr_sess[:-1])[0]
        # bad_next_resp = corr_sess[bad_resp + 1]
