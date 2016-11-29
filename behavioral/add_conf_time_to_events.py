import numpy as np
import os
from numpy.lib.recfunctions import append_fields
# import ram_data_helpers
import pdb

base_path = '/data/eeg/'

def process_event_file(events):

    # arrays that will be added to events
    move_array_all = np.zeros(len(events))
    move_array_all[:] = np.nan
    choice_array_all = np.zeros(len(events))
    choice_array_all[:] = np.nan

    # For each session, parse the original log file
    subj = events.subject[0]
    sessions = events.session
    uniq_sessions = np.unique(sessions)

    for session in uniq_sessions:

        # make sure the log file exists
        log_dir = os.path.join(base_path, subj, 'behavioral/TH1/session_' + str(session))
        log_file = os.path.join(log_dir, subj+'Log.txt')
        if not os.path.exists(log_file):
            print(log_file + ' not found.')
        else:

            # open log and loop over all lines
            log = open(log_file, 'r')

            # keep track of three things: rec item, time of first movement, and choice time
            rec_items = []
            rec_time = []
            rec_move = []
            rec_choice = []
            move_found = False
            choice_found = False

            for line in log.readlines():
                line = line.replace('\r','')
                tokens = line[:-1].split('\t')
                if len(tokens) > 1:

                    if tokens[2] == 'Trial Event' and tokens[3] == 'RECALL_SPECIAL':

                        pres_time = float(tokens[0])
                        rec_time.append(pres_time)
                        rec_items.append(tokens[4])
                        move_found = False
                        choice_found = False

                    if tokens[3] == 'REMEMBER_ANSWER_MOVEMENT' and not move_found:

                        # ignore really entries really soon after the item presentation, as
                        # those do not correspond to user input
                        t = float(tokens[0])
                        if (t - pres_time) > 25:
                            rec_move.append(t)
                            move_found = True

                    if tokens[3] == 'REMEMBER_RESPONSE' and not choice_found:
                        # print tokens
                        # print float(t) - float(tokens[0])
                        rec_choice.append(float(tokens[0]))
                        choice_found = True

                        # account for when they don't move the selector from the starting position
                        if move_found is False:
                            rec_move.append(float(tokens[0]))
                            move_found = True
            log.close()

            lists = [rec_items, rec_move, rec_choice]
            n = len(lists[0])
            if not all(len(l) == n for l in lists):
                print 'Size mismatch in recall arrays, not adding fields to events for %s session %d' % (subj, session)
            else:

                rec_items = np.array(rec_items)
                rec_time = np.array(rec_time)
                rec_move = np.array(rec_move)
                rec_choice = np.array(rec_choice)

                sess_events = events[sessions == session]
                move_array = np.zeros(len(sess_events))
                move_array[:] = np.nan
                choice_array = np.zeros(len(sess_events))
                choice_array[:] = np.nan
                item_names = sess_events.item_name
                item_names[np.array([np.size(x) == 0 for x in sess_events.item_name])] = u''
                for rec_item, this_time, this_move, this_choice in zip(rec_items, rec_time, rec_move, rec_choice):
                    inds = np.where(item_names == rec_item)[0]
                    move_array[inds] = this_move - this_time
                    choice_array[inds] = this_choice - this_time
                choice_array_all[sessions == session] = choice_array
                move_array_all[sessions == session] = move_array

    events = append_fields(events, 'choice_rt', choice_array_all, dtypes=float, usemask=False, asrecarray=True)
    events = append_fields(events, 'move_rt', move_array_all, dtypes=float, usemask=False, asrecarray=True)
    return events