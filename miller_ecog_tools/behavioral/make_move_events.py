import numpy as np
import os
from numpy.lib.recfunctions import append_fields, merge_arrays
# import ram_data_helpers
from scipy.stats import linregress
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
            log_file = sess_ev[0].eegfile.split('ephys')[0] + 'behavioral/current_source/logs/session_log.txt'
        else:
            log_dir = os.path.join(base_path, subj, 'behavioral/TH1/session_' + str(session))
            log_file = os.path.join(log_dir, subj+'Log.txt')
        if not os.path.exists(log_file):
            print(log_file + ' not found.')
        else:
            # print log_file
            # open log and loop over all lines
            log = open(log_file, 'r')

            instruct_times = []
            nav_start_times = []
            treasure_times = []
            treasure_numbers = []

            for line in log.readlines():
                line = line.replace('\r','')
                tokens = line[:-1].split('\t')
                if len(tokens) > 1:
                    if tokens[3] == 'SHOWING_INSTRUCTIONS':
                        instruct_times.append(float(tokens[0]))
                    elif tokens[3] == 'TRIAL_NAVIGATION_STARTED':
                        nav_start_times.append(float(tokens[0]))
                    elif tokens[3] == 'TREASURE_OPEN':
                        treasure_times.append(float(tokens[0]))
                        treasure_numbers.append(tokens[2])
            log.close()

            # for each nav start, find the previous instruct closest in time
            nav_start_times = np.array(nav_start_times)
            instruct_times = np.array(instruct_times)
            still_times = []
            for nav_time in nav_start_times:
                still_times.append(instruct_times[np.where(nav_time - instruct_times > 0)[0][-1]])
            still_times = np.array(still_times)
            still_durs = nav_start_times - still_times

            # for each treasure open, find the previous nav start closest in time
            # treasure_times = np.array(treasure_times)
            # treasure_numbers = np.array(treasure_numbers)
            # nav_start_durs = []
            # for treasure_time, treasure_num in zip(treasure_times, treasure_numbers):
            #     if treasure_num == 'TreasureChest000':
            #         diffs = treasure_time - nav_start_times
            #         nav_start_durs.append(diffs[np.where(diffs > 0)[0][-1]])
            # nav_start_durs = np.array(nav_start_durs)

            new_nav_starts = []
            nav_start_durs = []
            treasure_times = np.array(treasure_times)
            treasure_numbers = np.array(treasure_numbers)
            possible_first_chests = treasure_numbers == 'TreasureChest000'
            for nav_start_time in nav_start_times:
                diffs = treasure_times[possible_first_chests] - nav_start_time
                if np.any(diffs > 0):
                    nav_start_durs.append(diffs[diffs > 0].min())
                    new_nav_starts.append(nav_start_time)
                else:
                    print('Could not find matching first chest.')
            nav_start_durs = np.array(nav_start_durs)
            nav_start_times = np.array(new_nav_starts)

            treasure_move_start_times = []
            treasure_move_durs = []
            trials = np.unique(sess_ev.trial)
            for trial in trials:
                trial_ev = sess_ev[sess_ev.trial == trial]
                for i, ev in enumerate(trial_ev):
                    if (ev.chestNum < 4) & (i+1 != len(trial_ev)):
                        treasure_move_start_times.append(ev.mstime+1500)
                        treasure_move_durs.append(trial_ev[i+1].mstime - (ev.mstime+1500))
            treasure_move_start_times = np.array(treasure_move_start_times)
            treasure_move_durs = np.array(treasure_move_durs)

            beh_times = sess_ev.mstime
            eeg_samps = sess_ev.eegoffset
            reg_info = linregress(beh_times, eeg_samps)

            all_times = np.concatenate([still_times, nav_start_times, treasure_move_start_times])
            all_durs = np.concatenate([still_durs, nav_start_durs, treasure_move_durs]).astype(int)
            all_samps = np.round(all_times * reg_info[0] + reg_info[1]).astype(int)
            all_types = np.array(['still'] * still_times.shape[0] + ['move'] * (nav_start_times.shape[0]+treasure_move_start_times.shape[0]))

            new_sess_ev = np.recarray(len(all_times,), dtype=[('mstime', list),
                                                            ('type', 'U256'),
                                                            ('eegfile', 'U256'),
                                                            ('subject', 'U256'),
                                                            ('eegoffset', list),
                                                            ('duration', list),
                                                            ('session', list),
                                                            ('trial', list),
                                                            ])
            for i, this_ev in enumerate(zip(all_times, all_durs, all_samps, all_types)):
                new_sess_ev[i]['mstime'] = this_ev[0]
                new_sess_ev[i]['duration'] = this_ev[1]
                new_sess_ev[i]['eegoffset'] = this_ev[2]
                new_sess_ev[i]['type'] = this_ev[3]
                new_sess_ev[i]['eegfile'] = sess_ev[0].eegfile
                new_sess_ev[i]['session'] = sess_ev[0].session
                new_sess_ev[i]['subject'] = sess_ev[0].subject

            ev_order = np.argsort(new_sess_ev, order=('mstime'))
            new_sess_ev = new_sess_ev[ev_order]
            trial_count = -1
            for i in range(new_sess_ev.shape[0]):
                if new_sess_ev[i]['type'] == 'still':
                    trial_count += 1
                new_sess_ev[i]['trial'] = trial_count

            new_ev.append(new_sess_ev)
    if len(new_ev) == 1:
        new_ev = new_ev[0]
    else:
        new_ev = np.concatenate(new_ev)
        new_ev = new_ev.view(np.recarray)
    ev_order = np.argsort(new_ev, order=('mstime'))
    new_ev = new_ev[ev_order]
    return new_ev
