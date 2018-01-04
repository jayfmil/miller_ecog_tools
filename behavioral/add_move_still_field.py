import numpy as np
import os
from numpy.lib.recfunctions import append_fields, merge_arrays
# import ram_data_helpers
from scipy.stats import linregress
from scipy.ndimage import find_objects, label
import pdb

base_path = '/data/eeg/'
import platform
if platform.system() == 'Darwin':
    base_path = '/Volumes/rhino/data/eeg/'


def process_event_file(events, use_json=True):

    subj = events.subject[0]
    sessions = events.session
    uniq_sessions = np.unique(sessions)
    new_ev = []

    for session in uniq_sessions:
        sess_ev = events[events.session==session]

        # make sure the log file exists
        if use_json:
            log_file = sess_ev[0].eegfile.split('ephys')[0] + 'behavioral/current_source/logs/playerPaths.par'
            if not os.path.exists(log_file):
                log_dir = os.path.join(base_path, subj, 'behavioral/TH1/session_' + str(session))
                log_file = os.path.join(log_dir, 'playerPaths.par')
        else:
            log_dir = os.path.join(base_path, subj, 'behavioral/TH1/session_' + str(session))
            log_file = os.path.join(log_dir, 'playerPaths.par')
        if not os.path.exists(log_file):
            print(log_file + ' not found.')
        else:
            print log_file
            # open log and loop over all lines
            log = open(log_file, 'r')

            # load info from path file
            mstime = []
            trial = []
            chest_num = []
            x = []
            y = []
            heading = []

            for line in log.readlines():
                line = line.replace('\r','')
                tokens = line[:-1].split('\t')
                if len(tokens) > 1:
                    mstime.append(float(tokens[0]))
                    trial.append(float(tokens[3]))
                    chest_num.append(float(tokens[4])+1)
                    x.append(float(tokens[5]))
                    y.append(float(tokens[6]))
                    heading.append(float(tokens[7]))
            log.close()

            mstime = np.array(mstime)
            trial = np.array(trial)
            chest_num = np.array(chest_num)
            x = np.array(x)
            y = np.array(y)
            heading = np.array(heading)

            all_move_times = []
            all_still_times = []
            all_move_exc_rotation_times = []
            orig_ev_num = []

            for ev, trial_chest in enumerate(zip(sess_ev.trial, sess_ev.chestNum, sess_ev.type, sess_ev.mstime)):
                if trial_chest[2] == 'CHEST':
                    orig_ev_num.append(ev)
                    # pdb.set_trace()

                    path_inds = (trial == trial_chest[0]) & (chest_num == trial_chest[1])

                    # do a little smoothing on the locations to help avoid single frames of stillness
                    path_x = np.convolve(x[path_inds], np.ones((3,)) / 3, 'same')
                    path_x[0] = x[path_inds][0]
                    path_x[-1] = path_x[-2]
                    path_y = np.convolve(y[path_inds], np.ones((3,)) / 3, 'same')
                    path_y[0] = y[path_inds][0]
                    path_y[-1] = path_y[-2]

                    # path_data = np.stack([, y[path_inds], heading[path_inds]], -1)
                    path_data = np.stack([path_x, path_y, heading[path_inds]], -1)
                    at_chest = (path_data[:, 0] == path_data[-1, 0]) & (path_data[:, 1] == path_data[-1, 1])
                    path_data = path_data[~at_chest]
                    mstimes_trial_chest = mstime[path_inds][~at_chest]

                    # sometimes there are times that are 1 ms after a previous one. Remove.
                    times_to_keep = np.concatenate([[1000], np.diff(mstimes_trial_chest)]) > 5
                    path_data = path_data[times_to_keep]
                    mstimes_trial_chest = mstimes_trial_chest[times_to_keep]

                    still = np.all(np.diff(path_data, axis=0) == 0, axis=1)
                    still = np.concatenate([[True], still])
                    still_slices = find_objects(label(still)[0])
                    move_slices = find_objects(label(~still)[0])

                    still_exc_rotation = np.all(np.diff(path_data[:, :2], axis=0) == 0, axis=1)
                    still_exc_rotation = np.concatenate([[True], still_exc_rotation])
                    move_exc_rotation_slices = find_objects(label(~still_exc_rotation)[0])
                    # pdb.set_trace()

                    # this is a little weird, but I think it makes sense. If

                    still_ms_times = []
                    for this_slice in still_slices:
                        still_slice_times = mstimes_trial_chest[this_slice]
                        if len(still_slice_times) > 1:
                            still_ms_times.append([still_slice_times[0], still_slice_times[-1]])
                    if len(still_ms_times) > 0:
                        still_ms_times = np.stack(still_ms_times, 0) - trial_chest[3]
                        still_ms_times = still_ms_times[(still_ms_times[:, 1] - still_ms_times[:, 0]) > 100.]
                        all_still_times.append(still_ms_times)
                    else:
                        all_still_times.append(np.array([]))

                    # move_starts = [move_slices[0][0].start]
                    # move_stops = []
                    # for i, this_slice in enumerate(move_slices):

                    move_ms_times = []
                    # if trial_chest[0] == 2:
                    #     pdb.set_trace()
                    for this_slice in move_slices:
                        move_slice_times = mstimes_trial_chest[this_slice]
                        if len(move_slice_times) > 1:
                            move_ms_times.append([move_slice_times[0], move_slice_times[-1]])
                    move_ms_times = np.stack(move_ms_times, 0) - trial_chest[3]
                    move_ms_times = move_ms_times[(move_ms_times[:, 1] - move_ms_times[:, 0]) > 100.]
                    all_move_times.append(move_ms_times)

                    move_exc_rotation_ms_times = []
                    for this_slice in move_exc_rotation_slices:
                        move_exc_rotation_slice_times = mstimes_trial_chest[this_slice]
                        if len(move_exc_rotation_slice_times) > 1:
                            move_exc_rotation_ms_times.append([move_exc_rotation_slice_times[0], move_exc_rotation_slice_times[-1]])
                    move_exc_rotation_ms_times = np.stack(move_exc_rotation_ms_times, 0) - trial_chest[3]
                    move_exc_rotation_ms_times = move_exc_rotation_ms_times[(move_exc_rotation_ms_times[:, 1] - move_exc_rotation_ms_times[:, 0]) > 100.]
                    all_move_exc_rotation_times.append(move_exc_rotation_ms_times)

            # max_moves = np.max([x.shape[0] for x in all_move_times])
            # max_stills = np.max([x.shape[0] for x in all_still_times])
            max_moves = 50
            max_stills = 50

            # max_moves = np.max([x.shape[0] for x in all_move_times])
            # max_stills = np.max([x.shape[0] for x in all_still_times])
            new_array = np.recarray(len(sess_ev, ), dtype=[('move_starts', 'float', max_moves),
                                                           ('move_ends', 'float', max_moves),
                                                           ('still_starts', 'float', max_stills),
                                                           ('still_ends', 'float', max_stills),
                                                           ('move_exc_rotation_starts', 'float', max_moves),
                                                           ('move_exc_rotation_ends', 'float', max_moves)
                                                           ])

            for ev_num, this_ev in enumerate(all_move_times):
                tmp_move_start_ev = np.zeros(max_moves)
                tmp_move_start_ev[:] = np.nan
                tmp_move_start_ev[:this_ev.shape[0]] = this_ev[:, 0]
                new_array[orig_ev_num[ev_num]]['move_starts'] = tmp_move_start_ev

                tmp_move_stop_ev = np.zeros(max_moves)
                tmp_move_stop_ev[:] = np.nan
                tmp_move_stop_ev[:this_ev.shape[0]] = this_ev[:, 1]
                new_array[orig_ev_num[ev_num]]['move_ends'] = tmp_move_stop_ev
                # pdb.set_trace()

            for ev_num, this_ev in enumerate(all_move_exc_rotation_times):
                tmp_move_exc_rotation_start_ev = np.zeros(max_moves)
                tmp_move_exc_rotation_start_ev[:] = np.nan
                tmp_move_exc_rotation_start_ev[:this_ev.shape[0]] = this_ev[:, 0]
                new_array[orig_ev_num[ev_num]]['move_exc_rotation_starts'] = tmp_move_exc_rotation_start_ev

                tmp_move_exc_rotation_stop_ev = np.zeros(max_moves)
                tmp_move_exc_rotation_stop_ev[:] = np.nan
                tmp_move_exc_rotation_stop_ev[:this_ev.shape[0]] = this_ev[:, 1]
                new_array[orig_ev_num[ev_num]]['move_exc_rotation_ends'] = tmp_move_exc_rotation_stop_ev
                # pdb.set_trace()

            for ev_num, this_ev in enumerate(all_still_times):
                tmp_still_start_ev = np.zeros(max_stills)
                tmp_still_start_ev[:] = np.nan
                if len(this_ev) > 0:
                    tmp_still_start_ev[:this_ev.shape[0]] = this_ev[:, 0]
                new_array[orig_ev_num[ev_num]]['still_starts'] = tmp_still_start_ev

                tmp_still_stop_ev = np.zeros(max_stills)
                tmp_still_stop_ev[:] = np.nan
                if len(this_ev) > 0:
                    tmp_still_stop_ev[:this_ev.shape[0]] = this_ev[:, 1]
                new_array[orig_ev_num[ev_num]]['still_ends'] = tmp_still_stop_ev

            new_ev.append(merge_arrays([sess_ev, new_array], flatten=True, asrecarray=True))

    if len(new_ev) == 1:
        new_ev = new_ev[0]
    else:
        # pdb.set_trace()
        new_ev = np.concatenate(new_ev)
    new_ev = new_ev.view(np.recarray)
    ev_order = np.argsort(new_ev, order=('mstime'))
    new_ev = new_ev[ev_order]
    return new_ev
