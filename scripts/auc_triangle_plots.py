import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import cluster_helper.cluster
import TH_ClassifyTriangle
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, iplot_mpl
from plotly.graph_objs import *
import plotly.tools as tls


if __name__ == '__main__':

    freqs = np.logspace(np.log10(1), np.log10(200), 8)
    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=100,
                                             cores_per_job=1,
                                             extra_params={"resources": "h_vmem=12G"}) as pool:
        rec = TH_ClassifyTriangle.ClassifyTH(freqs=freqs, start_time=-2.0,
                                             end_time=2.0, pool=pool)
        rec.run_classify_for_all_subjs()

    f1 = rec.freqs[0]
    f2 = rec.freqs[-1]
    bipol_str = 'bipol' if rec.bipolar else 'mono'
    tbin_str = '1_bin' if rec.time_bins is None else str(rec.time_bins.shape[0]) + '_bins'
    save_dir = os.path.join(rec.base_dir, '%d_freqs_%.1f_%.1f_%s' % (len(rec.freqs), f1, f2, bipol_str),
                            'triangle_%s_start_%.1f_stop_%.1f' % (rec.train_phase, rec.start_time,
                                                         rec.end_time), tbin_str, 'figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for subj in rec.subjs:
        try:
            save_file = os.path.join(save_dir, subj + '_auc_triangle.pdf')
            fig = plt.gcf()
            fig.clf()
            plot_data, best_time, best_window = rec.plot_triangle(subjs=[subj])
            cv_type = [x['cv_type'] for x in rec.res if x['subj'] == subj][0]
            max_auc = np.nanmax(plot_data)
            _ = plt.title(subj + ': max %.2f, %.2f s center, %.2f s size' % (max_auc, best_time, best_window), fontsize=14)
            clim = np.abs(np.array([np.nanmin(plot_data), np.nanmax(plot_data)]) - .5).max()
            plt.clim(.5 - clim, .5 + clim)
            plt.tight_layout()
            plt.savefig(save_file)
        except ValueError:
            print 'Error with %s' % subj

    # lolo
    fig = plt.gcf()
    fig.clf()
    plot_data, best_time, best_window = rec.plot_triangle(cv_type=['lolo'])
    max_auc = np.nanmax(plot_data)
    _ = plt.title('LOLO: max %.2f, %.2f s center, %.2f s size' % (max_auc, best_time, best_window), fontsize=14)
    clim = np.abs(np.array([np.nanmin(plot_data), np.nanmax(plot_data)]) - .5).max()
    plt.clim(.5 - clim, .5 + clim)
    plt.tight_layout()
    save_file = os.path.join(save_dir, 'LOLO_auc_triangle.pdf')
    plt.savefig(save_file)

    # loso
    fig = plt.gcf()
    fig.clf()
    plot_data, best_time, best_window = rec.plot_triangle(cv_type=['loso'])
    max_auc = np.nanmax(plot_data)
    _ = plt.title('LOSO: max %.2f, %.2f s center, %.2f s size' % (max_auc, best_time, best_window), fontsize=14)
    clim = np.abs(np.array([np.nanmin(plot_data), np.nanmax(plot_data)]) - .5).max()
    plt.clim(.5 - clim, .5 + clim)
    plt.tight_layout()
    save_file = os.path.join(save_dir, 'LOSO_auc_triangle.pdf')
    plt.savefig(save_file)