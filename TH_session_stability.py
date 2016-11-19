import numpy as np
import re
import os
from glob import glob
import pdb
from scipy.stats.mstats import zscore, zmap
from scipy.stats import binned_statistic, sem, ranksums, ttest_1samp, ttest_ind
from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.TalReader import TalReader
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import roc_auc_score, roc_curve
import cPickle as pickle
import cluster_helper.cluster
import ram_data_helpers
from TH_load_features import load_features, compute_subj_dir_path
import create_stPPC_network
import matplotlib
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.distance import squareform
from plotly.offline import plot, init_notebook_mode, iplot, iplot_mpl
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode()

class ClassifyTH:
    # default paths
    event_path = '/data/events/RAM_TH1/'
    data_path = '/data/eeg'
    base_dir = '/scratch/jfm2/python/'

    def __init__(self, subjs=None, task='RAM_TH1', task_phase='enc', bipolar=True, freqs=None, start_time=-1.2,
                 end_time=0.5, norm='l2', feat_type='power', stPPC_filtering=False,
                 reduce_features=False, do_rand=False, C=7.2e-4, pool=None):

        # if subjects not given, get list from /data/events/TH1 directory
        if subjs is None:
            subjs = ram_data_helpers.get_subjs(task)
        self.subjs = subjs
        print self.subjs

        # I usually work with RAM_TH, but this code should be mostly agnostic to the actual task run
        self.task = task

        # classifying encoding or retrieval ('enc' | 'rec')
        self.task_phase = task_phase

        # this is stupid but don't let it precess R1132C - this subject didn't use the confident judgements so we can't
        #  work with the data. Also R1208C has an issue where an event occured close to the end of the eeg, and it
        #  crashes
        if task == 'RAM_TH1':
            self.subjs = [subj for subj in self.subjs if subj != 'R1132C']
            # self.subjs = [subj for subj in self.subjs if subj != 'R1208C']

        # monopolar (average re-reference) or bipolar
        self.bipolar = bipolar

        # these are the frequencies where we will compute power
        self.freqs = freqs

        # time bin to use
        self.start_time = start_time
        self.end_time = end_time

        # type of regularization, penalty parameter
        self.norm = norm
        self.C = C

        # features to use for regression
        self.feat_type = feat_type
        self.do_rand = do_rand

        # if doing parallel jobs, pool with be a cluster_helper object, otherwise None
        self.pool = pool

        # where to save data
        self.base_dir = os.path.join(ClassifyTH.base_dir, task)

        # stores results for all subjects
        self.res = None

    def run_classify_for_all_subjs(self):
        class_data_all_subjs = []
        for subj in self.subjs:

            # load events and loop over each session
            events = ram_data_helpers.load_subj_events(self.task, subj)
            thresh = np.max([np.median(events['distErr']), events['radius_size'][0]])
            sessions = np.unique(events.session)

            # only run if multiple sessions
            if len(sessions) <= 1:
                print 'Only 1 session for %s, skipping.' % subj
                continue

            for session in sessions:
                print 'Processing %s session %d.' % (subj, session)
                sess_ev = events[events.session == session]

                # define base directory for subject
                f1 = self.freqs[0]
                f2 = self.freqs[-1]
                subj_base_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f_session_stability' % (len(self.freqs), f1, f2),
                                             '%s_start_%.1f_stop_%.1f' % (self.task_phase, self.start_time, self.end_time),
                                             '%s_session_%d' % (subj, session))

                # sub directories hold electrode data, feature data, and classifier output
                subj_elec_dir = os.path.join(subj_base_dir, 'elec_data')
                subj_feature_dir = os.path.join(subj_base_dir, '%s' % self.feat_type)
                subj_class_dir = os.path.join(subj_base_dir, 'C_%.4f_norm_%s' % (self.C, self.norm))

                # this holds the classifier results
                save_file = os.path.join(subj_class_dir, subj + '_' + self.feat_type + '.p')

                # features used for classification
                feat_file = os.path.join(subj_feature_dir, subj + '_features.p')

                overwrite_features = False
                if os.path.exists(save_file):
                    with open(save_file, 'rb') as f:
                        subj_data = pickle.load(f)
                    if len(sess_ev) == len(subj_data['events']):
                        print 'Classifier exists for %s. Skipping.' % subj
                        class_data_all_subjs.append(subj_data)
                        self.res = class_data_all_subjs
                        continue
                    else:
                        overwrite_features = True
                if os.path.exists(feat_file):
                    with open(feat_file, 'rb') as f:
                        feat_data = pickle.load(f)
                    if len(sess_ev) != len(feat_data['events']):
                        overwrite_features = True

                # make directory if missing
                if not os.path.exists(subj_feature_dir):
                    try:
                        os.makedirs(subj_class_dir)
                    except OSError:
                        pass
                    try:
                        os.makedirs(subj_feature_dir)
                    except OSError:
                        pass
                    try:
                        os.makedirs(subj_elec_dir)
                    except OSError:
                        pass

                # run classifier for subject
                try:
                    subj_classify = self.run_classify_for_single_subj(subj, session, feat_file, subj_elec_dir,
                                                                      overwrite_features, thresh)
                    class_data_all_subjs.append(subj_classify)
                    with open(save_file, 'wb') as f:
                        pickle.dump(subj_classify, f, protocol=-1)
                except ValueError:
                    print 'Error processing %s.' % subj
                self.res = class_data_all_subjs

    def run_classify_for_single_subj(self, subj, session, feat_file, subj_elec_dir, overwrite_features, thresh):
        """
        Runs logistic regression classifier to predict recalled/not-recalled items. Logic is as follows:
        1) Create classifier features, if they do not exist.
        2) Classify using leave one session out cross validation. If only one session, use leave one list out
        """

        # check if features exist. If not, create
        if not os.path.exists(feat_file) or overwrite_features or self.feat_type[-2:] == '_r':
            print '%s features do not exist or overwriting for %s. Creating.' % (self.feat_type, subj)

            # feat_func = self.get_feature_func(self.feat_type)
            # subj_features = feat_func(subj, self.start_time, self.end_time, self.freqs, self.bipolar, self.pool)
            subj_features = load_features(subj, self.task, self.task_phase, self.start_time, self.end_time,
                                          self.freqs, self.bipolar, self.feat_type, False, subj_elec_dir, self.pool,
                                          session=[session])
            with open(feat_file, 'wb') as f:
                pickle.dump(subj_features, f, protocol=-1)

        # if exist, load from disk
        else:
            with open(feat_file, 'rb') as f:
                subj_features = pickle.load(f)

        recalls = ram_data_helpers.filter_events_to_recalled(self.task, subj_features.events.data, thresh)

        # create cross validation labels
        sessions = subj_features.events.data['session']
        if len(np.unique(subj_features.events.data['session'])) > 1:
            cv_sel = sessions
        else:
            cv_sel = subj_features.events.data['trial']

        # transpose to have events as first dimensions
        # reshape to events by obs
        if self.feat_type != 'ppc':
            subj_features = subj_features.transpose()
            subj_data = subj_features.data.reshape(subj_features.data.shape[0], -1)
        else:
            subj_data = np.transpose(subj_features.data, (2, 1, 0, 3))
            subj_data = subj_data.reshape((subj_data.shape[0], subj_data.shape[1]*subj_data.shape[2], -1))

        # run classify
        subj_res = self.compute_classifier(subj, recalls, sessions, cv_sel, subj_data)
        subj_res['subj'] = subj
        subj_res['events'] = subj_features['events']
        if self.feat_type == 'power':
            subj_res['loc_tag'] = subj_features.attrs['loc_tag']
            subj_res['anat_region'] = subj_features.attrs['anat_region']

        return subj_res

    def compute_classifier(self, subj, recalls, sessions, cv_sel, feat_mat):
        if self.do_rand:
            feat_mat = np.random.rand(*feat_mat.shape)
            #### FIGURE OUT HOW LOLO CAN DO SO WELL WITH RANDOM DATA. IS IT THE ZSCORING ALL TOGETHER? HAS TO BE

        # normalize data by session (this only makes sense for power?)
        if self.feat_type == 'power':
            uniq_sessions = np.unique(sessions)
            for sess in uniq_sessions:
                sess_event_mask = (sessions == sess)
                feat_mat[sess_event_mask] = zscore(feat_mat[sess_event_mask], axis=0)

        # initialize classifier
        lr_classifier = LogisticRegression(C=self.C, penalty=self.norm, class_weight='auto', solver='liblinear')

        # hold class probability
        probs = np.empty_like(recalls, dtype=np.float)

        # loop over each train set and classify test
        uniq_cv = np.unique(cv_sel)
        for cv_num, cv in enumerate(uniq_cv):

            # create mask of just training data
            mask_train = (cv_sel != cv)
            feats_to_use_bool = np.ones((feat_mat.shape[1]), dtype=bool)

            if self.feat_type != 'ppc':
                feats_train = feat_mat[mask_train]
                feats_test = feat_mat[~mask_train]
            else:
                feats_train = feat_mat[mask_train, :, cv_num]
                feats_test = feat_mat[~mask_train, :, cv_num]
            rec_train = recalls[mask_train]

            # feats_to_use = np.ones((feat_mat.shape[1]), dtype=bool)
            x_train = zscore(feats_train[:, feats_to_use_bool], axis=0)
            x_test = zmap(feats_test[:, feats_to_use_bool], feats_train[:, feats_to_use_bool], axis=0)

            # feat_mat_reduced = feat_mat[:, feats_to_use]
            # pow_train = zscore(feat_mat_reduced[mask_train], axis=0)
            # lr_classifier.fit(pow_train, rec_train)
            lr_classifier.fit(x_train, rec_train)

            # now estimate classes of train data
            # pow_test = zmap(feat_mat_reduced[~mask_train], feat_mat_reduced[mask_train], axis=0)
            # rec_train = recalls[~mask_train]
            # test_probs = lr_classifier.predict_proba(pow_test)[:, 1]
            test_probs = lr_classifier.predict_proba(x_test)[:, 1]
            probs[~mask_train] = test_probs

        # compute AUC
        auc = roc_auc_score(recalls, probs)

        # output results, including AUC, lr object (fit to all data), prob estimates, and class labels
        subj_res = {}
        subj_res['auc'] = auc
        subj_res['subj'] = subj
        print auc

        subj_res['lr_classifier'] = lr_classifier.fit(feat_mat, recalls)
        subj_res['probs'] = probs
        subj_res['classes'] = recalls
        subj_res['tercile'] = self.compute_terciles(subj_res)
        subj_res['sessions'] = sessions
        subj_res['cv_sel'] = cv_sel
        subj_res['cv_type'] = 'loso' if len(np.unique(sessions)) > 1 else 'lolo'

        # also compute forward model feature importance for each electrode if we are using power
        if self.feat_type == 'power':
            probs_log = np.log(probs / (1 - probs))
            covx = np.cov(feat_mat.T)
            covs = np.cov(probs_log)
            W = subj_res['lr_classifier'].coef_
            A = np.dot(covx, W.T) / covs
            subj_res['forward_model'] = np.reshape(A, (-1, len(self.freqs)))
            subj_res['tstat'] = ttest_ind(feat_mat[recalls],feat_mat[~recalls])[0]
        return subj_res

    def plot_terciles(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

         # get terciles from specified subjects with specifed cv_type
        terciles = np.stack([x['tercile'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])

        yerr = sem(terciles, axis=0) * 1.96
        plt.bar(range(3), np.nanmean(terciles, axis=0), align='center', color=[.5, .5, .5], linewidth=2, yerr=yerr,
                error_kw={'ecolor': 'k', 'linewidth': 2, 'capsize': 5, 'capthick': 2})

    def plot_auc_hist(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get aucs from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])

        hist = np.histogram(aucs, np.linspace(0.025, .975, 20))
        plt.bar(np.linspace(0.05, .95, 19), hist[0].astype(np.float) / np.sum(hist[0]), .05, align='center',
                color=[.5, .5, .5], linewidth=2)

    def aucs(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get terciles from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])
        return aucs

    def mean_auc(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        # get terciles from specified subjects with specifed cv_type
        aucs = np.stack([x['auc'] for x in self.res if x['cv_type'] in cv_type and x['subj'] in subjs])
        return aucs.mean()

    def print_res_table(self, subjs=None, cv_type=('loso', 'lolo')):
        if subjs is None:
            subjs = self.subjs

        table = np.stack([[x['subj'], x['auc'], x['cv_type'], self.feat_type] for x in self.res if
                          x['cv_type'] in cv_type and x['subj'] in self.subjs])

        print 'Subject\tAUC\tCV\tfeatures'
        for row in table:
            print '%s\t%.3f\t%s\t%s' % (row[0][1:], float(row[1]), row[2], row[3])

    def compute_session_corr(self, subjs=None, weights_or_power='weights', plot_all_or_avg='all',
                                regions=('IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC')):
        if subjs is None:
            subjs = np.unique(np.array([res['subj'] for res in self.res]))

        freq_x_regions_x_subjs = []
        region_counts = []
        for subj in subjs:
            subj_res = [res for res in self.res if res['subj'] == subj]
            if weights_or_power == 'weights':
                weights = np.stack([np.reshape(w['lr_classifier'].coef_, (-1, len(self.freqs))) for w in subj_res], 1)
            elif weights_or_power == 'power':
                weights = np.stack([np.reshape(w['tstat'], (-1, len(self.freqs))) for w in subj_res], 1)
            loc_dict = ram_data_helpers.bin_elec_locs(subj_res[0]['loc_tag'], subj_res[0]['anat_region'])
            freq_x_regions = np.empty(weights[0].shape[1] * len(regions)).reshape(weights[0].shape[1], len(regions))
            freq_x_regions[:] = np.nan
            region_counts_subj = np.zeros(len(regions))
            for f, weight_freq in enumerate(weights.T):
                for i, r in enumerate(regions):

                    if np.sum(loc_dict[r]) > 0:
                        if f == 0:
                            # print 'subject %s, region %s, num elecs %d' % (subj, r, np.sum(loc_dict[r]))
                            region_counts_subj[i] = np.sum(loc_dict[r])
                        # corr_mat = np.corrcoef(weight_freq[:, loc_dict[r]])
                        # mean_weight = weight_freq[:, loc_dict[r]].mean(axis=0)
                        # std_weight = weight_freq[:, loc_dict[r]].std(axis=0)
                        # freq_x_regions[f, i] = np.mean(mean_weight / std_weight)
                        # triu_inds = np.triu_indices_from(corr_mat, k=1)
                        # freq_x_regions[f, i] = corr_mat[triu_inds[0], triu_inds[1]].mean()
                        if subj == 'R1167M' and (r=='MTL'):
                            pdb.set_trace()

                        [ts, ps] = ttest_1samp(weight_freq[:, loc_dict[r]], 0, axis=0)
                        freq_x_regions[f, i] = np.mean(np.abs(ts))
            freq_x_regions_x_subjs.append(freq_x_regions)
            region_counts.append(region_counts_subj)
        region_counts = np.stack(region_counts, -1)
        corr_heatmap = np.stack(freq_x_regions_x_subjs, -1)

        # plot heatmap. If just one subject, plot actual forward model weights. Otherwise, plot t-stat against zero
        if corr_heatmap.shape[2] > 1:
            plot_data, p = ttest_1samp(corr_heatmap, 0, axis=2, nan_policy='omit')
            colorbar_str = 't-stat'
        else:
            plot_data = np.nanmean(corr_heatmap, axis=2)
            # colorbar_str = 'Corr Coef'
            colorbar_str = 'Mean T-stat'

        if weights_or_power == 'weights':
            title_str = 'classifier weights'
        else:
            title_str = 'SME t-stat'

        cmap = self.matplotlib_to_plotly('RdBu_r')
        cmap = self.matplotlib_to_plotly('YlOrRd')
        x_pos = [[.1, .4], [.6, .9]]
        x_pos_bar = [[.41, .43], [.91, .93]]
        save_dir = os.path.join(self.base_dir, '%d_freqs_%.1f_%.1f_session_stability' % (len(self.freqs),
                                                                                         self.freqs[0], self.freqs[-1]),
                                '%s_start_%.1f_stop_%.1f' % (self.task_phase, self.start_time, self.end_time))
        if plot_all_or_avg == 'all':
            fname = os.path.join(save_dir, 'stabilty_map_all_subjs_%s.html' % title_str.replace(' ', '_'))
            rows = int(np.ceil(float(corr_heatmap.shape[2])/2))
            subplot_titles = [j for i in zip(subjs,['']*len(subjs)) for j in i]
            fig = tools.make_subplots(rows=rows, cols=4)#, subplot_titles=subplot_titles)
            ax_count = 1
            for i in range(corr_heatmap.shape[2]):
                row = int(np.ceil(float(i+1)/2))
                col = i % 2 + 1
                real_col = 1 if col == 1 else 3
                data = corr_heatmap[:, :, i]
                region_labels = ['%s (%d)' % (x[0], x[1]) for x in zip(regions, region_counts[:, i])]
                # clim = 1
                # clim = np.max(np.abs([np.nanmin(data), np.nanmax(data)]))
                clim = np.nanpercentile(data.flatten(), 90)
                # if clim > 5:
                #     clim = 5
                fig.append_trace(go.Heatmap(z=data, y=np.round(self.freqs*10)/10, x=region_labels, zmin=0, zmax=clim,
                                            colorscale=cmap, showscale=False), row, real_col)
                fig.append_trace(go.Heatmap(z=[[x] for x in np.linspace(0, clim, 101)],
                                            y=[np.round([x*10])/10 for x in np.linspace(0, clim, 101)],
                                            zmin=0, zmax=clim, showscale=False, colorscale=cmap), row, real_col+1)

                fig['layout']['yaxis' + str(int(ax_count))].update(type='category')
                fig['layout']['xaxis' + str(int(ax_count))].update(domain=x_pos[col-1],title=subjs[i])
                fig['layout']['xaxis' + str(int(ax_count + 1))].update(domain=x_pos_bar[col-1],ticks='',showticklabels=False)
                fig['layout']['yaxis' + str(int(ax_count+1))].update(ticks='', showticklabels=True, side='right',
                                                                     dtick=10, type='category')
                ax_count += 2
            fig['layout'].update(height=450*rows, width=800, title='Mean abs(t-stat) Across Sessions (%s)' % title_str)
            iplot(fig)
            plot(fig, include_plotlyjs=True, show_link=False, filename=fname)

        else:
            n_subjs = corr_heatmap.shape[2]
            fname = os.path.join(save_dir, 'stabilty_map_avg_subjs_%s.html' % title_str.replace(' ', '_'))
            plot_data1 = np.nanmean(corr_heatmap, axis=2)
            plot_data2, p = ttest_1samp(corr_heatmap, 0, axis=2, nan_policy='omit')
            # clim = np.max(np.abs([np.nanmin(plot_data2), np.nanmax(plot_data2)]))
            clim = np.nanpercentile(plot_data2.flatten(), 90)
            # if clim > 7:
            #     clim = 7
            # clim2 = np.max(np.abs([np.nanmin(plot_data1), np.nanmax(plot_data1)]))
            clim2 = np.nanpercentile(plot_data1.flatten(), 90)
            # if clim2 > 6:
            #     clim2 = 6
            fig = tools.make_subplots(rows=1, cols=4)

            # left plot: mean corr coeff across subjs
            region_labels = ['%s (%d)' % (x[0], x[1]) for x in zip(regions, np.sum(region_counts, axis=1))]
            fig.append_trace(go.Heatmap(z=plot_data1, y=np.round(self.freqs*10)/10, x=region_labels, zmin=0,
                                        zmax=clim2, showscale=False, colorscale=cmap), row=1, col=1)
            fig.append_trace(go.Heatmap(z=[[x] for x in np.linspace(0, clim2, 101)],
                                        y=[np.round([x * 10]) / 10 for x in np.linspace(0, clim2, 101)],
                                        zmin=0, zmax=clim2, showscale=False, colorscale=cmap), 1, 2)

            # right plot: tstat against zero
            fig.append_trace(go.Heatmap(z=plot_data2, y=np.round(self.freqs * 10) / 10, x=region_labels, zmin=0,
                                        zmax=clim, showscale=False, colorscale=cmap), row=1, col=3)
            fig.append_trace(go.Heatmap(z=[[x] for x in np.linspace(0, clim, 101)],
                                    y=[np.round([x * 10]) / 10 for x in np.linspace(0, clim, 101)],
                                    zmin=0, zmax=clim, showscale=False, colorscale=cmap), 1, 4)
            fig['layout']['yaxis1'].update(type='category')
            fig['layout']['xaxis1'].update(domain=x_pos[0], tickangle=45, title='mean(t-stat)')
            fig['layout']['xaxis2'].update(domain=x_pos_bar[0], ticks='', showticklabels=False)
            fig['layout']['yaxis2'].update(ticks='', showticklabels=True, side='right', dtick=10, type='category')
            fig['layout']['yaxis3'].update(type='category')
            fig['layout']['xaxis3'].update(domain=x_pos[1], tickangle=45, title='t-test vs 0')
            fig['layout']['xaxis4'].update(domain=x_pos_bar[1], ticks='', showticklabels=False)
            fig['layout']['yaxis4'].update(ticks='', showticklabels=True, side='right', dtick=10, type='category')
            fig['layout'].update(title='Mean abs(t-stat) Across Sessions (%s, %d subjects)' % (title_str, n_subjs))
            iplot(fig)
            plot(fig, include_plotlyjs=True, show_link=False, filename=fname)
            # plot(fig, filename='/home1/jfm2/python/TH/testplot2', image='svg')
        #
        # clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        # fig, ax = plt.subplots(1, 1)
        # plt.imshow(plot_data[::-1], interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
        # cb = plt.colorbar()
        # cb.set_label(label=colorbar_str, size=16)  # ,rotation=90)
        # cb.ax.tick_params(labelsize=12)
        # region_total = np.sum(region_counts, axis=1)
        # region_labels = ['%s (%d)' %(x[0], x[1]) for x in zip(regions, region_total)]
        # plt.xticks(range(len(regions)), region_labels, fontsize=16, rotation=-45)
        # plt.yticks(range(0, len(self.freqs), 7)[::-1], np.round(self.freqs[range(0, len(self.freqs), 7)] * 10) / 10,
        #            fontsize=16)
        # plt.ylabel('Frequency', fontsize=16)
        # plt.tight_layout()
        return corr_heatmap

    def compute_feature_heatmap(self, subjs=None, cv_type=('loso', 'lolo'),
                                regions=('IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC')):
        if subjs is None:
            subjs = self.subjs

        # concatenate all foward model features into one 3d array (freqs x regions x subjs)
        freq_x_regions_x_subjs = []
        for subj_res in self.res:
            if (subj_res['subj'] in subjs) and (subj_res['cv_type'] in cv_type):
                loc_dict = ram_data_helpers.bin_elec_locs(subj_res['loc_tag'], subj_res['anat_region'])
                A = subj_res['forward_model']
                freq_x_regions = np.empty(A.shape[1] * len(regions)).reshape(A.shape[1], len(regions))
                freq_x_regions[:] = np.nan
                for i, r in enumerate(regions):
                    freq_x_regions[:, i] = A[loc_dict[r], :].mean(axis=0)
                freq_x_regions_x_subjs.append(freq_x_regions)
        feature_heatmap = np.stack(freq_x_regions_x_subjs, -1)

        # plot heatmap. If just one subject, plot actual forward model weights. Otherwise, plot t-stat against zero
        if feature_heatmap.shape[2] > 1:
            plot_data, p = ttest_1samp(feature_heatmap, 0, axis=2, nan_policy='omit')
            colorbar_str = 't-stat'
        else:
            plot_data = np.nanmean(feature_heatmap, axis=2)
            colorbar_str = 'Feature Importance'

        clim = np.max(np.abs([np.nanmin(plot_data), np.nanmax(plot_data)]))
        fig, ax = plt.subplots(1, 1)
        plt.imshow(plot_data[::-1], interpolation='nearest', cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
        cb = plt.colorbar()
        cb.set_label(label=colorbar_str, size=16)  # ,rotation=90)
        cb.ax.tick_params(labelsize=12)
        plt.xticks(range(len(regions)), regions, fontsize=16, rotation=-45)
        plt.yticks(range(0, len(self.freqs), 7)[::-1], np.round(self.freqs[range(0, len(self.freqs), 7)] * 10) / 10,
                   fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tight_layout()
        return feature_heatmap


    @classmethod
    def compute_terciles(cls, subj_res):
        binned_data = binned_statistic(subj_res['probs'], subj_res['classes'], statistic='mean',
                                       bins=np.percentile(subj_res['probs'], [0, 33, 67, 100]))
        tercile_delta_rec = (binned_data[0] - np.mean(subj_res['classes'])) / np.mean(subj_res['classes']) * 100
        return tercile_delta_rec

    @staticmethod
    def matplotlib_to_plotly(cmap_name):
        cmap = matplotlib.cm.get_cmap(cmap_name)
        h = 1.0 / (255 - 1)
        pl_colorscale = []

        for k in range(255):
            C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def get_num_elecs(self, subj):
        tal_path = os.path.join('/data/eeg', subj, 'tal', subj + '_talLocs_database_bipol.mat')
        tal_reader = TalReader(filename=tal_path)
        monopolar_channels = tal_reader.get_monopolar_channels()
        bipolar_pairs = tal_reader.get_bipolar_pairs()
        return len(bipolar_pairs) if self.bipolar else len(monopolar_channels)