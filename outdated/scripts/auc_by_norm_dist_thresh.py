import sys
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
import cluster_helper.cluster
import ram_data_helpers
import TH_Classify
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, iplot_mpl
from plotly.graph_objs import *
import plotly.tools as tls


if __name__ == '__main__':

    freqs = np.logspace(np.log10(1), np.log10(200), 8)
    # with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=100,
    #                                          cores_per_job=1,
    #                                          extra_params={"resources": "h_vmem=10G"}) as pool:
    enc_objs = []
    threshes = np.arange(.05, .96, .05)
    for thresh in threshes:
        enc = TH_Classify.ClassifyTH(force_reclass=True,
                                     save_class=False, freqs=freqs, compute_pval=False,
                                     recall_filter_func=ram_data_helpers.filter_events_to_recalled_norm_thresh,
                                     rec_thresh=thresh, pool=None)
        enc.run_classify_for_all_subjs()
        enc_objs.append(enc)

    # get list of all subjects (they may differ between thresholds if some thresholds had errors
    all_subjs = []
    [all_subjs.extend(x.subjs) for x in enc_objs]
    all_subjs = np.unique(np.array(all_subjs))

    f1 = enc_objs[0].freqs[0]
    f2 = enc_objs[0].freqs[-1]
    bipol_str = 'bipol' if enc_objs[0].bipolar else 'mono'
    tbin_str = '1_bin' if enc_objs[0].time_bins is None else str(enc_objs[0].time_bins.shape[0]) + '_bins'
    save_dir = os.path.join(enc_objs[0].base_dir, '%d_freqs_%.1f_%.1f_%s' % (len(enc_objs[0].freqs), f1, f2, bipol_str),
                            '%s_start_%.1f_stop_%.1f' % (enc_objs[0].train_phase, enc_objs[0].start_time,
                                                         enc_objs[0].end_time), tbin_str, 'figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot each subject
    for subj in all_subjs:
        save_file = os.path.join(save_dir, subj + '_auc_by_norm_err.pdf')
        thresh_subj = []
        pos_class_perc = []
        aucs = np.empty(shape=threshes.shape)
        aucs[:] = np.nan
        labels = np.copy(np.copy(threshes)).astype(str)
        for i, enc in enumerate(enc_objs):
            for curr_subj in enc.res:
                if curr_subj['subj'] == subj:
                    #                 aucs.append(curr_subj['auc'])
                    aucs[i] = curr_subj['auc']
                    # pos_class_perc.append(np.mean(curr_subj['classes']))
                    labels[i] = str(int(threshes[i] * 100) / 100.) + ' (%.2f)' % np.mean(curr_subj['classes'])
        fig = plt.gcf()
        plt.clf()
        #     _=plt.plot(thresh_subj, aucs)
        plt.plot(threshes, aucs, 'k', linewidth=3)
        plt.ylabel('AUC', fontsize=16)
        xlabel = plt.xlabel('Normalized Dist. Err.', fontsize=16)
        plt.title(subj + ' ' + curr_subj['cv_type'], fontsize=14)
        _ = plt.xticks(threshes, labels, fontsize=14, rotation=-45)
        _ = plt.yticks(fontsize=16)
        plt.xlim(0, 1)
        plt.grid()
        fig.set_size_inches(12, 6)
        plt.savefig(save_file, bbox_extra_artists = [xlabel], bbox_inches = 'tight')
