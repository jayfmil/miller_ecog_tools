import ram_data_helpers
import numpy as np
from GroupLevel.Analyses import group_SME, group_spectral_shift
from GroupLevel import group_brain_viz
from surfer import Surface, Brain
from mayavi import mlab
from scipy.stats import ttest_1samp

import pdb
import os
os.environ['SUBJECTS_DIR'] = '/Users/jmiller/data/eeg/freesurfer/subjects/'


TASK = 'RAM_TH1'


def load_sme(regress_broadband=False):

    # custom list of subjects, excludes R1219C
    subjs = np.array(ram_data_helpers.get_subjs('RAM_TH1'))
    # subjs = np.array(ram_data_helpers.get_subjs('RAM_FR1'))
    # subjs = np.array(ram_data_helpers.get_subjs('RAM_YC1'))
    # subjs = subjs[subjs != 'R1065J']
    subjs = subjs[subjs != 'R1219C']

    # Use 50 freqs
    freqs = np.logspace(np.log10(1), np.log10(200), 50)

    # load group data
    if not regress_broadband:
        sme = group_SME.GroupSME(freqs=freqs, load_res_if_file_exists=True, open_pool=False,
                                 task=TASK, subjs=subjs, subject_settings='default_50_freqs',
                                 base_dir='/Users/jmiller/data/python')
    else:
        sme = group_spectral_shift.GroupSpectralShift(freqs=freqs, load_res_if_file_exists=True, open_pool=False,
                                                      task=TASK, subjs=subjs, subject_settings='default_50_freqs',
                                                      base_dir='/Users/jmiller/data/python')
    sme.process()
    return sme


def sme_brain(sme, res_inds, res_key='ts', n_perms=100, file_ext='lfa'):

    # res_inds = np.where(sme.subject_objs[0].freqs <= 10)[0]
    l_ts_by_subj, r_ts_by_subj = group_brain_viz.get_elec_ts_verts(sme.subject_objs, res_inds, res_key=res_key)

    l_ts, l_ps = ttest_1samp(l_ts_by_subj, 0, axis=1, nan_policy='omit')
    r_ts, r_ps = ttest_1samp(r_ts_by_subj, 0, axis=1, nan_policy='omit')

    l_ts_perm = np.zeros((n_perms, l_ts.shape[0]))
    r_ts_perm = np.zeros((n_perms, r_ts.shape[0]))
    for i in range(n_perms):
        print('Perm %d of %d' % (i+1, n_perms))

        # left
        flipped_l_subjs = np.random.rand(l_ts_by_subj.shape[1]) < .5
        tmp_l = l_ts_by_subj.copy()
        tmp_l[:, flipped_l_subjs] = -tmp_l[:, flipped_l_subjs]

        # ttest this set of vertices with half subjects flipped against 0
        l_ts_perm_tmp, l_ps_perm_tmp = ttest_1samp(tmp_l, 0, axis=1, nan_policy='omit')

        # I'm not sure why the masked data still have values. Make them nans
        l_ts_perm_tmp_data = l_ts_perm_tmp.data
        l_ts_perm_tmp_data[l_ts.mask] = np.nan

        # store this permuation
        l_ts_perm[i] = l_ts_perm_tmp_data

        # right
        flipped_r_subjs = np.random.rand(r_ts_by_subj.shape[1]) < .5
        tmp_r = r_ts_by_subj.copy()
        tmp_r[:, flipped_r_subjs] = -tmp_r[:, flipped_r_subjs]

        # ttest this set of vertices with half subjects flipped against 0
        r_ts_perm_tmp, r_ps_perm_tmp = ttest_1samp(tmp_r, 0, axis=1, nan_policy='omit')

        # I'm not sure why the masked data still have values. Make them nans
        r_ts_perm_tmp_data = r_ts_perm_tmp.data
        r_ts_perm_tmp_data[r_ts.mask] = np.nan

        # store this permuation
        r_ts_perm[i] = r_ts_perm_tmp_data

    l_keep_verts = np.sum(~np.isnan(l_ts_by_subj), axis=1) >= 5
    l_ts_perm[:, ~l_keep_verts] = np.nan
    l_ts.data[~l_keep_verts] = np.nan

    r_keep_verts = np.sum(~np.isnan(r_ts_by_subj), axis=1) >= 5
    r_ts_perm[:, ~r_keep_verts] = np.nan
    r_ts.data[~r_keep_verts] = np.nan

    brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast',
                  background='white', offscreen=False)

    l_thresh = np.nanpercentile(l_ts_perm, [2.5, 97.5])
    r_thresh = np.nanpercentile(r_ts_perm, [2.5, 97.5])
    thresh = np.nanpercentile(np.concatenate([l_ts_perm, r_ts_perm]), [2.5, 97.5])
    print(l_thresh)
    print(r_thresh)
    print(thresh)

    # thresh = [-.5,.5]
    brain.add_data(l_ts, -5, 5, colormap='RdBu_r', hemi='lh', thresh=l_thresh[1], remove_existing=True, colorbar=False)
    brain.add_data(-l_ts, -5, 5, colormap='RdBu', hemi='lh', thresh=-l_thresh[0], remove_existing=False, colorbar=False)
    brain.add_data(~l_keep_verts, 0, 6, colormap='gray', hemi='lh', alpha=1, thresh=1, remove_existing=False,
                   colorbar=False)

    brain.add_data(r_ts, -5, 5, colormap='RdBu_r', hemi='rh', thresh=r_thresh[1], remove_existing=True, colorbar=False)
    brain.add_data(-r_ts, -5, 5, colormap='RdBu', hemi='rh', thresh=-r_thresh[0], remove_existing=False, colorbar=False)
    brain.add_data(~r_keep_verts, 0, 6, colormap='gray', hemi='rh', alpha=1, thresh=1, remove_existing=False,
                   colorbar=False)

    # make figs and save
    base_dir = os.path.split(os.path.abspath(__file__))[0]

    # left
    mlab.view(azimuth=180, distance=500)
    brain.save_image(os.path.join(base_dir, 'figs', '%s_left_%s.png' % (TASK, file_ext)))

    # right
    mlab.view(azimuth=0, distance=500)
    brain.save_image(os.path.join(base_dir, 'figs', '%s_right_%s.png' % (TASK, file_ext)))


def coverage_brain(sme, file_ext='coverage'):

    # res_inds = np.where(sme.subject_objs[0].freqs <= 10)[0]
    l_by_subj, r_by_subj = group_brain_viz.get_elec_coverage_verts(sme.subject_objs)
    max_count = np.max([l_by_subj.sum(axis=1).max(), r_by_subj.sum(axis=1).max()])

    l_count = l_by_subj.sum(axis=1)
    r_count = r_by_subj.sum(axis=1)

    brain = Brain('average', 'both', 'pial', views='lateral', cortex='low_contrast',
                  background='white', offscreen=False)

    brain.add_data(l_count, 4, max_count, colormap='hot', hemi='lh', remove_existing=True, colorbar=False)
    brain.add_data(r_count, 4, max_count, colormap='hot', hemi='rh', remove_existing=True, colorbar=False)

    base_dir = os.path.split(os.path.abspath(__file__))[0]
    mlab.view(azimuth=180, distance=500)
    brain.save_image(os.path.join(base_dir, 'figs', '%s_left_%s.png' % (TASK, file_ext)))

    # right
    mlab.view(azimuth=0, distance=500)
    brain.save_image(os.path.join(base_dir, 'figs', '%s_right_%s.png' % (TASK, file_ext)))
    print('Max subject count: %d' % int(max_count))

if __name__ == '__main__':
    pass
