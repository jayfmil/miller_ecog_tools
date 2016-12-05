from TH_Classify_with_stPPC_support import ClassifyTH
import os
import cPickle as pickle
import pdb
import ram_data_helpers
import numpy as np
from scipy.stats.mstats import zscore, zmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

class CombineFeatures(ClassifyTH):

    def __init__(self, feature_paths, feat_types, subjs=None, C=[7.2e-4], pool=None):
        ClassifyTH.__init__(self, subjs=subjs, C=C, pool=pool)
        self.feature_paths = feature_paths
        self.feat_types = feat_types

    def run(self):
        class_data_all_subjs = []
        for subj in self.subjs:
            try:
                subj_class_res = self.loadAndCombine(subj)
                if subj_class_res is not None:
                    class_data_all_subjs.append(subj_class_res)
            except:
                pass
        self.res = class_data_all_subjs

    def loadAndCombine(self, subj):
        file1 = os.path.join(self.feature_paths[0], subj, self.feat_types[0], subj+'_features.p')
        file2 = os.path.join(self.feature_paths[1], subj, self.feat_types[1], subj+'_features.p')

        if (not os.path.exists(file1)) or (not os.path.exists(file2)):
            print 'missing features for %s' % subj
            return None

        with open(file1, 'rb') as f:
            feat_data1 = pickle.load(f)

        with open(file2, 'rb') as f:
            feat_data2 = pickle.load(f)



        recalls = ram_data_helpers.filter_events_to_recalled(self.task, feat_data1.events.data)

        # create cross validation labels
        sessions = feat_data1.events.data['session']
        if len(np.unique(feat_data1.events.data['session'])) > 1:
            cv_sel = sessions
        else:
            cv_sel = feat_data1.events.data['trial']


        if feat_data1.ndim > 3:
            subj_data1 = np.transpose(feat_data1.data, (2, 1, 0, 3))
        else:
            subj_data1 = np.transpose(feat_data1.data, (2, 1, 0))
        subj_data1 = subj_data1.reshape((subj_data1.shape[0], -1))

        if feat_data2.ndim > 3:
            subj_data2 = np.transpose(feat_data2.data, (2, 1, 0, 3))
        elif feat_data2.ndim > 2:
            subj_data2 = np.transpose(feat_data2.data, (2, 1, 0))
        else:
            subj_data2 = np.transpose(feat_data2.data, (1, 0))
        subj_data2 = subj_data2.reshape((subj_data2.shape[0], -1))
        subj_data = np.hstack([subj_data1, subj_data2])

        # run classify
        subj_res = self.compute_classifier(subj, recalls, sessions, cv_sel, subj_data)
        subj_res['subj'] = subj
        subj_res['events'] = feat_data1['events']
        if self.feat_type == 'power':
            subj_res['loc_tag'] = feat_data1.attrs['loc_tag']
            subj_res['anat_region'] = feat_data1.attrs['anat_region']
        return subj_res

    def compute_classifier(self, subj, recalls, sessions, cv_sel, feat_mat):

        # normalize data by session (this only makes sense for power?)
        # if self.feat_type == 'power':
        uniq_sessions = np.unique(sessions)
        for sess in uniq_sessions:
            sess_event_mask = (sessions == sess)
            feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

        # initialize classifier
        # lr_classifier = LogisticRegression(C=self.C, penalty=self.norm, class_weight='auto', solver='liblinear')

        # hold class probability
        probs = np.empty_like(recalls, dtype=np.float)

        # hold class probability
        probs = np.empty(shape=(recalls.shape[0], len(self.C)), dtype=np.float)
        # loop over each train set and classify test

        uniq_cv = np.unique(cv_sel)
        fold_aucs = np.empty(shape=(len(uniq_cv), len(self.C)), dtype=np.float)
        aucs = np.empty(shape=(len(self.C)), dtype=np.float)
        for c_num, c in enumerate(self.C):
            lr_classifier = LogisticRegression(C=c, penalty=self.norm, class_weight='balanced', solver='liblinear')

            # loop over each train set and classify test
            # uniq_cv = np.unique(cv_sel)
            for cv_num, cv in enumerate(uniq_cv):

                # create mask of just training data
                mask_train = (cv_sel != cv)
                feats_to_use_bool = np.ones((feat_mat.shape[1]), dtype=bool)
                feats_train = feat_mat[mask_train]
                feats_test = feat_mat[~mask_train]
                rec_train = recalls[mask_train]
                x_train = zscore(feats_train[:, feats_to_use_bool], axis=0)
                x_test = zmap(feats_test[:, feats_to_use_bool], feats_train[:, feats_to_use_bool], axis=0)
                lr_classifier.fit(x_train, rec_train)
                test_probs = lr_classifier.predict_proba(x_test)[:, 1]
                probs[~mask_train, c_num] = test_probs
                fold_aucs[cv_num, c_num] = roc_auc_score(recalls[~mask_train], test_probs)

            # compute AUC
            # auc = roc_auc_score(recalls, probs)
        aucs = fold_aucs.mean(axis=0)
        bias = np.mean(fold_aucs.max(axis=1) - fold_aucs[:, np.argmax(aucs)])
        auc =  aucs.max() - bias

        # output results, including AUC, lr object (fit to all data), prob estimates, and class labels
        subj_res = {}
        subj_res['auc'] = auc
        subj_res['subj'] = subj
        print auc

        # NEED TO ALSO MAKE PPC WITH ALL SESS
        subj_res['lr_classifier'] = lr_classifier.fit(feat_mat, recalls)
        subj_res['probs'] = probs[:, np.argmax(aucs)]
        subj_res['classes'] = recalls
        subj_res['tercile'] = self.compute_terciles(subj_res)
        subj_res['sessions'] = sessions
        subj_res['cv_sel'] = cv_sel
        subj_res['cv_type'] = 'loso' if len(np.unique(sessions)) > 1 else 'lolo'
        return subj_res











