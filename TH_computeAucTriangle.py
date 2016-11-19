import numpy as np
from copy import deepcopy
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from random import shuffle
import re
import os
from glob import glob
import cPickle as pickle
import cluster_helper.cluster

power_dir = '/scratch/jfm2/python_power/TH/8_freqs/'
def create_auc_triangle_for_subj(subj):
    """
    fffffffff
    """

    print subj
    # get list of directories with containing downsampled power data
    # sort in numerical order by window size
    data_dirs = glob(os.path.join(power_dir, 'window_*'))
    data_dirs.sort(key=natural_keys)

    # loop over each directory
    for i, this_dir in enumerate(data_dirs):
        print this_dir

        # if subject has data file, load
        data_file = os.path.join(this_dir, subj + '.p')
        if not os.path.exists(data_file):
            return
        with open(data_file, 'rb') as f:
            subj_data = pickle.load(f)

        # get boolean of recalled and not recalled items, also get sessions and cross-val labels
        recalls = filter_events_to_recalled(subj_data)
        sessions = subj_data.events.data['session']
        if len(np.unique(subj_data.events.data['session'])) > 1:
            cv_sel = sessions
        else:
            cv_sel = subj_data.events.data['trial']

        # transpose to have events as first dimensions
        pow_data = np.transpose(subj_data.data, (2, 0, 1, 3))

        # array to hold auc values for each time bin
        if i == 0:
            full_time_axis = np.round(subj_data['time'].data / .05) * .05
            aucs = np.empty((len(data_dirs), pow_data.shape[3]))
            aucs[:] = np.nan

        # loop over each time window
        for t in range(pow_data.shape[3]):
            # print subj_data['time'].data[t]
            pow_data_time = pow_data[:, :, :, t].reshape(pow_data.shape[0], -1)
            bin_ind = full_time_axis == round(subj_data['time'].data[t] / .05) * .05
            try:
                aucs[i, bin_ind] = compute_classifier(recalls, sessions, cv_sel, pow_data_time)
            except ValueError:
                print('Error processing %s' % subj)
    return aucs

def compute_classifier(recalls, sessions, cv_sel, pow_mat):

    # normalize data by session
    uniq_sessions = np.unique(sessions)
    for sess in uniq_sessions:
        sess_event_mask = (sessions == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)

    # initialize classifier
    lr_classifier = LogisticRegression(C=7.2e-4, penalty='l2', class_weight='auto', solver='liblinear')

    # hold class probability
    probs = np.empty_like(recalls, dtype=np.float)

    uniq_cv = np.unique(cv_sel)
    for cv in uniq_cv:

        # fit classifier to train data
        mask_train = (cv_sel != cv)
        pow_train = pow_mat[mask_train]
        rec_train = recalls[mask_train]
        lr_classifier.fit(pow_train, rec_train)

        # now estimate classes of train data
        pow_test = pow_mat[~mask_train]
        rec_train = recalls[~mask_train]
        test_probs = lr_classifier.predict_proba(pow_test)[:, 1]
        probs[~mask_train] = test_probs

    auc = roc_auc_score(recalls, probs)
    return auc

def process_subjs(subjs, do_par):

    if not do_par:
        auc_triangle_list = map(create_auc_triangle_for_subj, subjs)
    else:
        with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=50,
                                                 cores_per_job=1, extra_params={'mem': 10}) as view:
            auc_triangle_list = view.map(create_auc_triangle_for_subj, subjs)
    return auc_triangle_list

def filter_events_to_recalled(subj_data):
    not_low_conf = subj_data.events.data['confidence'] > 0
    not_far_dist = subj_data.events.data['distErr'] < np.max([np.median(subj_data.events.data['distErr']),
                                                                        subj_data.events.data['radius_size'][0]])
    recalled = not_low_conf & not_far_dist
    return recalled


def get_th_subjs():
    """Returns list of subjects who performed TH1."""
    subjs = glob('/data/events/RAM_TH1/R*_events.mat')
    subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
    subjs.sort()
    return subjs

##################
## http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
###################

if __name__ == '__main__':
    subjs = get_th_subjs()
    process_subjs(subjs, False)