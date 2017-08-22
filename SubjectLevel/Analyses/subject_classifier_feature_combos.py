import pdb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# from scipy.stats import ttest_ind
from scipy.misc import comb
from itertools import combinations, islice, chain
from SubjectLevel.Analyses.subject_classifier import SubjectClassifier as SC
from tqdm import tqdm
from joblib import Parallel, delayed

# from copy import deepcopy
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ttest_1samp, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

class SubjectClassifier(SC):
    """
    Version of SubjectClassifier that classifies the using only the best univariate electrode of frequency, then the
    best two, then the best three, then four, ....
    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectClassifier, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)

        # string to use when saving results files
        self.res_str = 'classify_best_combo_feats.p'

        # number of features to make combinations from
        self.num_feats = 2
        self.n_cores = 4

    def analysis(self, permute=False):
        """

        """

        # make a copy of the data so we can restore to original after we are done
        subject_data = deepcopy(self.subject_data)

        loso = True
        Cs = self.C
        if len(np.unique(self.subject_data.events.data['session'])) == 1:
            Cs = SubjectClassifier.default_C
            loso = False

        cross_val_dict = self.cross_val_dict

        # X = self.subject_data[:, feat_columns, :]
        X = self.subject_data.data.reshape(self.subject_data.shape[0], -1)
        X = self.normalize_power(X)
        X = X.reshape(subject_data.shape)
        Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)


        X = X.reshape(X.shape[0], -1)

        # now loop over all the cross validation folds
        for cv in cross_val_dict.keys():
            # print(cv_num)

            # Training data for fold
            x_train = X[cross_val_dict[cv]['train_bool']]
            task_train = cross_val_dict[cv]['train_phase']
            y_train = Y[cross_val_dict[cv]['train_bool']]

            # Test data forfold
            x_test = X[cross_val_dict[cv]['test_bool']]
            task_test = cross_val_dict[cv]['test_phase']
            y_test = Y[cross_val_dict[cv]['test_bool']]

            # normalize train test and scale test data. This is ugly because it has to account for a few
            # different contingencies. If the test data phase is also a train data phase, then scale test data for
            # that phase based on the training data for that phase. Otherwise, just zscore the test data.
            for phase in self.train_phase:
                x_train[task_train == phase] = zscore(x_train[task_train == phase], axis=0)

            for phase in self.test_phase:
                if phase in self.train_phase:
                    x_test[task_test == phase] = zmap(x_test[task_test == phase],
                                                      x_train[task_train == phase], axis=0)
                else:
                    x_test[task_test == phase] = zscore(x_test[task_test == phase], axis=0)
            cross_val_dict[cv]['y_train'] = y_train
            cross_val_dict[cv]['y_test'] = y_test
            cross_val_dict[cv]['x_train'] = x_train.reshape((x_train.shape[0], len(self.freqs), -1))
            cross_val_dict[cv]['x_test'] = x_test.reshape((x_test.shape[0], len(self.freqs), -1))
        # pdb.set_trace()

        # compute feature combinations
        combos = combinations(range(self.subject_data.shape[1]), self.num_feats)

        # store auc from each combo
        n_combos = int(comb(self.subject_data.shape[1], self.num_feats))
        aucs_by_combo = []

        # store the actual combination
        combo_features = []

        chunk_size = 200 if n_combos >= 200 else n_combos
        n_chunks = int(np.ceil(n_combos / float(chunk_size)))
        chunked_combo = chunks(combos, size=chunk_size)
        # pdb.set_trace()

        if self.pool:
            for this_chunk in tqdm(range(n_chunks)):

                combos_chunk = list(chunked_combo.next())
                data_as_list = zip([Y]*chunk_size, [Cs]*chunk_size,
                                   [loso]*chunk_size, [cross_val_dict]*chunk_size,
                                   [self.norm]*chunk_size, [self.train_phase]*chunk_size,
                                   [self.test_phase] * chunk_size, [self.scale_enc] * chunk_size,
                                   combos_chunk)
                res_as_list = self.pool.map(classify, data_as_list)
                # res_as_list = map(classify, data_as_list)
                aucs_by_combo.extend(res_as_list)
                combo_features.extend([np.array(c) for c in combos_chunk])
        else:
            with Parallel(n_jobs=self.n_cores, backend='threading') as parallel:

                for this_chunk in tqdm(range(n_chunks)):

                    combos_chunk = list(chunked_combo.next())
                    data_as_list = zip([Y]*chunk_size, [Cs]*chunk_size,
                                       [loso]*chunk_size, [cross_val_dict]*chunk_size,
                                       [self.norm]*chunk_size, [self.train_phase]*chunk_size,
                                       [self.test_phase] * chunk_size, [self.scale_enc] * chunk_size,
                                       combos_chunk)
                    res_as_list = parallel(delayed(classify)(x) for x in data_as_list)
                    aucs_by_combo.extend(res_as_list)
                    combo_features.extend([np.array(c) for c in combos_chunk])


        # pdb.set_trace()
        aucs_by_combo = np.array(aucs_by_combo)
        combo_features = np.stack(combo_features, 0)
        # pdb.set_trace()
        #
        # for i, combo in tqdm(enumerate(combos)):
        #     self.subject_data = subject_data[:, combo, :]
        #
        #     # classify using the base classifier code
        #     super(SubjectClassifier, self).analysis()
        #     # pdb.set_trace()
        #     # store auc for this permation in an easier to access array
        #     aucs_by_combo[i] = self.res['auc']
        #     combo_features[i] = np.array(combo)

        # make subject data be the full dataset again
        self.subject_data = subject_data

        # finally, store results
        self.res['aucs_by_combo'] = aucs_by_combo
        self.res['combo_features'] = combo_features.astype(int)

    def plot_combos(self):

        feat_array = np.zeros((len(self.freqs), len(self.res['aucs_by_combo'])))
        for i, combo in enumerate(self.res['combo_features']):
            feat_array[combo, i] = 1

        aucs_best_to_worst = np.argsort(self.res['aucs_by_combo'])[::-1]
        feat_array = feat_array[:, aucs_best_to_worst]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(self.res['aucs_by_combo'][aucs_best_to_worst])
        ax2.imshow(feat_array, interpolation='nearest', cmap='gray_r', aspect='auto')
        ax2.invert_yaxis()

def chunks(iterator, size=10):
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def classify(input_list):
    # Get class labels
    # Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

    Y, Cs, loso, cross_val_dict, norm, train_phase, test_phase, scale_enc, feat_columns = input_list
    # X = X[:, feat_columns, :]
    # X = X.reshape(X.shape[0], -1)


    # if leave-one-session-out (loso) cross validation, this will hold area under the curve for each hold out
    fold_aucs = np.empty(shape=(len(cross_val_dict.keys()), len(Cs)), dtype=np.float)

    # will hold the predicted class probability for all the test data
    probs = np.empty(shape=(Y.shape[0], len(Cs)), dtype=np.float)

    # This outer loop is for all the different penalty (C) values given. If more than one, the optimal value will
    # be chosen using bias correction based on Tibshirani and Tibshirani 2009, Annals of Applied Statistics. This
    # only works for multi-session (loso) data, otherwise we don't have enough data to compute help out AUC.
    for c_num, c in enumerate(Cs):

        lr_classifier = LogisticRegression(C=c, penalty=norm, solver='liblinear')

        # now loop over all the cross validation folds
        for cv_num, cv in enumerate(cross_val_dict.keys()):
            # print(cv_num)

            # Training data for fold

            x_train = cross_val_dict[cv]['x_train'][:, feat_columns, :].reshape(cross_val_dict[cv]['x_train'].shape[0], -1)
            task_train = cross_val_dict[cv]['train_phase']
            y_train = cross_val_dict[cv]['y_train']

            # Test data forfold
            x_test = cross_val_dict[cv]['x_test'][:, feat_columns, :].reshape(cross_val_dict[cv]['x_test'].shape[0], -1)
            # task_test = cross_val_dict[cv]['test_phase']
            y_test = cross_val_dict[cv]['y_test']

            # # normalize train test and scale test data. This is ugly because it has to account for a few
            # # different contingencies. If the test data phase is also a train data phase, then scale test data for
            # # that phase based on the training data for that phase. Otherwise, just zscore the test data.
            # for phase in train_phase:
            #     x_train[task_train == phase] = zscore(x_train[task_train == phase], axis=0)
            #
            # for phase in test_phase:
            #     if phase in train_phase:
            #         x_test[task_test == phase] = zmap(x_test[task_test == phase],
            #                                           x_train[task_train == phase], axis=0)
            #     else:
            #         x_test[task_test == phase] = zscore(x_test[task_test == phase], axis=0)

            # weight observations by number of positive and negative class
            y_ind = y_train.astype(int)

            # if we are training on both encoding and retrieval and we are scaling the encoding weights,
            # seperate the ecnoding and retrieval positive and negative classes so we can scale them later
            if len(train_phase) > 1:
                y_ind[task_train == 'rec'] += 2

            # compute the weight vector as the reciprocal of the number of items in each class, divided by the mean
            # class frequency
            recip_freq = 1. / np.bincount(y_ind)
            recip_freq /= np.mean(recip_freq)

            # scale the encoding classes. Sorry for the identical if statements
            if len(train_phase) > 1:
                recip_freq[:2] *= scale_enc
                recip_freq /= np.mean(recip_freq)
            weights = recip_freq[y_ind]


            lr_classifier.fit(x_train, y_train, sample_weight=weights)


            test_probs = lr_classifier.predict_proba(x_test)[:, 1]

            probs[cross_val_dict[cv]['test_bool'], c_num] = test_probs
            if loso:
                fold_aucs[cv_num, c_num] = roc_auc_score(y_test, test_probs)

        # If multi sessions, compute bias corrected AUC (if only one C values given, this has no effect) because
        # the bias will be 0. AUC is the average AUC across folds for the the bees value of C minus the bias term.
        train_bool = np.any(np.stack([cross_val_dict[x]['train_bool'] for x in cross_val_dict]), axis=0)
        test_bool = np.any(np.stack([cross_val_dict[x]['test_bool'] for x in cross_val_dict]), axis=0)

        if loso and len(cross_val_dict.keys()) > 1:
            aucs = fold_aucs.mean(axis=0)
            bias = np.mean(fold_aucs.max(axis=1) - fold_aucs[:, np.argmax(aucs)])
            auc = aucs.max() - bias
            C = Cs[np.argmax(aucs)]
            # auc = fold_aucs[-1][0]
        else:
            # is not multi session, AUC is just computed by aggregating all the hold out probabilities
            auc = roc_auc_score(Y[test_bool], probs[test_bool])
            C = Cs[0]
    return auc

def classify2(X, Y, Cs, loso, cross_val_dict, norm, train_phase, test_phase, scale_enc, feat_columns):
    # Get class labels
    # Y = self.recall_filter_func(self.task, self.subject_data.events.data, self.rec_thresh)

    X = X[:, feat_columns, :]
    X = X.reshape(X.shape[0], -1)


    # if leave-one-session-out (loso) cross validation, this will hold area under the curve for each hold out
    fold_aucs = np.empty(shape=(len(cross_val_dict.keys()), len(Cs)), dtype=np.float)

    # will hold the predicted class probability for all the test data
    probs = np.empty(shape=(Y.shape[0], len(Cs)), dtype=np.float)

    # This outer loop is for all the different penalty (C) values given. If more than one, the optimal value will
    # be chosen using bias correction based on Tibshirani and Tibshirani 2009, Annals of Applied Statistics. This
    # only works for multi-session (loso) data, otherwise we don't have enough data to compute help out AUC.
    for c_num, c in enumerate(Cs):

        lr_classifier = LogisticRegression(C=c, penalty=norm, solver='liblinear')

        # now loop over all the cross validation folds
        for cv_num, cv in enumerate(cross_val_dict.keys()):
            # print(cv_num)

            # Training data for fold
            x_train = X[cross_val_dict[cv]['train_bool']]
            task_train = cross_val_dict[cv]['train_phase']
            y_train = Y[cross_val_dict[cv]['train_bool']]

            # Test data forfold
            x_test = X[cross_val_dict[cv]['test_bool']]
            task_test = cross_val_dict[cv]['test_phase']
            y_test = Y[cross_val_dict[cv]['test_bool']]

            # normalize train test and scale test data. This is ugly because it has to account for a few
            # different contingencies. If the test data phase is also a train data phase, then scale test data for
            # that phase based on the training data for that phase. Otherwise, just zscore the test data.
            for phase in train_phase:
                x_train[task_train == phase] = zscore(x_train[task_train == phase], axis=0)

            for phase in test_phase:
                if phase in train_phase:
                    x_test[task_test == phase] = zmap(x_test[task_test == phase],
                                                      x_train[task_train == phase], axis=0)
                else:
                    x_test[task_test == phase] = zscore(x_test[task_test == phase], axis=0)

            # weight observations by number of positive and negative class
            y_ind = y_train.astype(int)

            # if we are training on both encoding and retrieval and we are scaling the encoding weights,
            # seperate the ecnoding and retrieval positive and negative classes so we can scale them later
            if len(train_phase) > 1:
                y_ind[task_train == 'rec'] += 2

            # compute the weight vector as the reciprocal of the number of items in each class, divided by the mean
            # class frequency
            recip_freq = 1. / np.bincount(y_ind)
            recip_freq /= np.mean(recip_freq)

            # scale the encoding classes. Sorry for the identical if statements
            if len(train_phase) > 1:
                recip_freq[:2] *= scale_enc
                recip_freq /= np.mean(recip_freq)
            weights = recip_freq[y_ind]


            lr_classifier.fit(x_train, y_train, sample_weight=weights)


            test_probs = lr_classifier.predict_proba(x_test)[:, 1]

            probs[cross_val_dict[cv]['test_bool'], c_num] = test_probs
            if loso:
                fold_aucs[cv_num, c_num] = roc_auc_score(y_test, test_probs)

        # If multi sessions, compute bias corrected AUC (if only one C values given, this has no effect) because
        # the bias will be 0. AUC is the average AUC across folds for the the bees value of C minus the bias term.
        train_bool = np.any(np.stack([cross_val_dict[x]['train_bool'] for x in cross_val_dict]), axis=0)
        test_bool = np.any(np.stack([cross_val_dict[x]['test_bool'] for x in cross_val_dict]), axis=0)

        if loso and len(cross_val_dict.keys()) > 1:
            aucs = fold_aucs.mean(axis=0)
            bias = np.mean(fold_aucs.max(axis=1) - fold_aucs[:, np.argmax(aucs)])
            auc = aucs.max() - bias
            C = Cs[np.argmax(aucs)]
            # auc = fold_aucs[-1][0]
        else:
            # is not multi session, AUC is just computed by aggregating all the hold out probabilities
            auc = roc_auc_score(Y[test_bool], probs[test_bool])
            C = Cs[0]
    return auc
