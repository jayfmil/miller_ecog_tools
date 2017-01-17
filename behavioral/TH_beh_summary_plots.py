import numpy as np
import ram_data_helpers
import matplotlib


def accuracy_by_conf_norm_err(subject_objs):

    subjs = [subj.subj for subj in subject_objs]
    acc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    acc_by_conf_all[:] = np.nan
    perc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    perc_by_conf_all[:] = np.nan

    hist_by_conf_all = np.zeros((20, 3, len(subjs)), dtype=float)
    hist_by_conf_all[:] = np.nan

    for i, subj in enumerate(subjs):
        print subj
        events = ram_data_helpers.load_subj_events('RAM_TH1', subj)

        # JFM: LOSO ONLY HERE
        # if len(np.unique(events.session)) == 1:
        #     print('%s: skipping.' % subj)
            # continue

        for conf in xrange(3):
            acc_by_conf_all[i, conf] = 1 - events.norm_err[events.confidence == conf].mean()
            perc_by_conf_all[i, conf] = (events.confidence == conf).mean()
            [hist, b] = np.histogram(events.norm_err[events.confidence == conf], bins=20, range=(0, 1))
            hist = hist / float(np.sum(hist))
            hist_by_conf_all[:, conf, i] = hist

    return acc_by_conf_all, perc_by_conf_all, hist_by_conf_all

def accuracy_by_conf_dist_err(subject_objs):

    subjs = [subj.subj for subj in subject_objs]
    acc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    acc_by_conf_all[:] = np.nan
    perc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    perc_by_conf_all[:] = np.nan

    hist_by_conf_all = np.zeros((20, 3, len(subjs)), dtype=float)
    hist_by_conf_all[:] = np.nan

    for i, subj in enumerate(subjs):
        print subj
        events = ram_data_helpers.load_subj_events('RAM_TH1', subj)

        for conf in xrange(3):
            acc_by_conf_all[i, conf] = events.distErr[events.confidence == conf].mean()
            perc_by_conf_all[i, conf] = (events.confidence == conf).mean()
            [hist, b] = np.histogram(events.distErr[events.confidence == conf], bins=20, range=(0, 90))
            hist = hist / float(np.sum(hist))
            hist_by_conf_all[:, conf, i] = hist

    return acc_by_conf_all, perc_by_conf_all, hist_by_conf_all


def median_dist_err_by_subj(subject_objs):

    subjs = [subj.subj for subj in subject_objs]
    med_errs = np.zeros((len(subjs)))

    for i, subj in enumerate(subjs):
        print(subj)
        events = ram_data_helpers.load_subj_events('RAM_TH1', subj)
        med_errs[i] = np.median(events.distErr)
    return med_errs


def p_corr_by_subj(subject_objs):

    subjs = [subj.subj for subj in subject_objs]
    p_corr = np.zeros((len(subjs)))

    for i, subj in enumerate(subjs):
        print(subj)
        events = ram_data_helpers.load_subj_events('RAM_TH1', subj)
        p_corr[i] = np.mean(ram_data_helpers.filter_events_to_recalled('RAM_TH1', events))
    return p_corr
