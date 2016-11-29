import numpy as np
import ram_data_helpers
import pdb

def retrieval_timing_subj(subj):
    events = ram_data_helpers.load_subj_events('RAM_TH1', subj, task_phase='rec')
    move_choice_loc = np.stack([events.move_rt, events.choice_rt, events.reactionTime], axis=1).mean(axis=0)
    move_choice_loc = np.expand_dims(move_choice_loc, axis=1).T
    return move_choice_loc, np.mean(events.distErr)

def retrieval_timing_all(subjs=None):
    if subjs is None:
        subjs = ram_data_helpers.get_subjs('RAM_TH1')

    move_choice_loc_all = None
    dist_errs_all = None
    for subj in subjs:
        print subj
        move_choice_loc_subj, dist_err_subj = retrieval_timing_subj(subj)

        # pdb.set_trace()
        move_choice_loc_all = np.concatenate([move_choice_loc_all, move_choice_loc_subj], axis=0) \
            if move_choice_loc_all is not None else move_choice_loc_subj

        dist_errs_all = np.concatenate([dist_errs_all, [dist_err_subj]], axis=0) \
            if dist_errs_all is not None else np.array([dist_err_subj])

    return np.array(subjs), move_choice_loc_all, dist_errs_all