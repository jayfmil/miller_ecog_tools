# Behavioral figures
# - Performance as a function of confidence
#   - Bar? Histogram? What performance measure?
#
# Neural figures
# - Average SME by region
# - Individual example subjects
#   - plot brains, electrodes
# - don't forget about left vs right

# from guidata import qthelpers # needed tp fix: ValueError: API 'QString' has already been set to version 1
# from mayavi import mlab
# from surfer import Surface, Brain
import numpy as np
# import nibabel as nib
import ram_data_helpers
import matplotlib
import os
# import RAM_plotBrain
import pdb
# from ptsa.data.readers.TalReader import TalReader
os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'

def accuracy_by_conf():
    subjs = ram_data_helpers.get_subjs('RAM_TH1')
    
    acc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    acc_by_conf_all[:] = np.nan
    perc_by_conf_all = np.zeros((len(subjs), 3), dtype=float)
    perc_by_conf_all[:] = np.nan

    hist_by_conf_all = np.zeros((20, 3, len(subjs)), dtype=float)
    hist_by_conf_all[:] = np.nan


    for i, subj in enumerate(subjs):
        print subj
        events = ram_data_helpers.load_subj_events('RAM_TH1', subj)

        # JFM: LOSO ONLY HERE
        if len(np.unique(events.session)) == 1:
            print('%s: skipping.' % subj)
            continue

        # acc_by_conf = np.zeros(3, dtype=float)
        # dist_hist = np.zeros((3, 20), dtype=float)
        # percent_conf = np.zeros(3, dtype=float)
        for conf in xrange(3):
            acc_by_conf_all[i, conf] = 1 - events.norm_err[events.confidence == conf].mean()
            perc_by_conf_all[i, conf] = (events.confidence == conf).mean()
            [hist, b] = np.histogram(events.norm_err[events.confidence == conf], bins=20, range=(0, 1))
            hist = hist/float(np.sum(hist))
            hist_by_conf_all[:, conf, i] = hist

    return acc_by_conf_all, perc_by_conf_all, hist_by_conf_all


def plot_indiv_brain_SME(classify_obj, subj):
    subj_res = [x for x in classify_obj.res if x['subj']==subj][0]
    ram_data_helpers.load_subj_elecs(subj)
    colors = matplotlib.cm.ScalarMappable(cmap='RdBu_r').to_rgba(subj_res['univar_ts'])
    return subj_res

def plot_all_elecs(bipolar=True):
    subjs = ram_data_helpers.get_subjs('RAM_TH1')

    xs = []
    ys = []
    zs = []
    lobe = []
    for subj in subjs:
        print(subj)
        try:
            tal_ext = '_bipol.mat' if bipolar else '_monopol.mat'
            tal_path = os.path.join('/data/eeg', subj, 'tal', subj + '_talLocs_database' + tal_ext)
            tal_reader = TalReader(filename=tal_path)
            tal_struct = tal_reader.read()


            xs.append(tal_struct['avgSurf']['x'])
            subj_y = tal_struct['avgSurf']['y']
            ys.append(subj_y)
            zs.append(tal_struct['avgSurf']['z'])

            subj_lobe = tal_struct.Loc2
            subj_lobe[(subj_lobe == 'Parietal Lobe') & (subj_y > 0)] = 'Frontal Lobe'
            lobe.append(subj_lobe)
        except:
            print('error with %s' % subj)

    # hemi = 'both', surface = 'pial', views = 'lateral',
    # cortex = 'classic', background = 'dimgray

    x = np.concatenate(xs, -1)
    y = np.concatenate(ys, -1)
    z = np.concatenate(zs, -1)
    lobe = np.concatenate(lobe, -1)

    N = len(z)
    scalars = np.arange(N)
    elec_colors = (np.zeros((N, 4)) * 255).astype(np.uint8)
    elec_colors[:, -1] = 255

    colors = np.array([[141, 211, 199],[255, 255, 179],[190, 186, 218],[251, 128, 114],[128, 177, 211],[253, 180, 98]])
    uniq_regions = np.unique(lobe)
    for i, region in enumerate(uniq_regions):
        inds = lobe == region
        elec_colors[inds, :-1] = colors[i]

    # colors[[2, 3, 4], :-1] = [1, 2, 3]
    brain = Brain('average', 'both', 'pial', views='lateral', cortex='classic',
                  background='white', offscreen=False, size=(800,800))
    v = mlab.view()
    brain.pts = mlab.points3d(x, y, z, scalars, scale_factor=(10. * .4), opacity=1,
                                   scale_mode='none')
    brain.pts.glyph.color_mode = 'color_by_scalar'
    brain.pts.module_manager.scalar_lut_manager.lut.table = elec_colors
    mlab.view(*v)

    return xs, ys, zs, lobe, brain



def plot_class_res_on_brain(classify_obj, freq_inds=None):

    if freq_inds is None:
        freq_inds = np.ones(classify_obj.freqs.shape, dtype=bool)
    labels_lh, ctab_lh, all_names_lh = nib.freesurfer.read_annot('/home1/jfm2/fs/average/label/lh.aparc.annot')
    all_names_lh = np.array(all_names_lh)
    lh_roi_data = np.zeros(shape=(len(all_names_lh)))

    labels_rh, ctab_rh, all_names_rh = nib.freesurfer.read_annot('/home1/jfm2/fs/average/label/rh.aparc.annot')
    all_names_rh = np.array(all_names_rh)
    rh_roi_data = np.zeros(shape=(len(all_names_rh)))

    lh_subjs_by_region = np.zeros((len(all_names_lh), len(classify_obj.res)))
    lh_subjs_by_region[:] = np.nan
    rh_subjs_by_region = np.zeros((len(all_names_lh), len(classify_obj.res)))
    rh_subjs_by_region[:] = np.nan
    for s_count, subj_res in enumerate(classify_obj.res):
        print(s_count)
        data = np.mean(subj_res['univar_ts'][:, freq_inds], axis=1)
        loc_dict = ram_data_helpers.bin_elec_locs(subj_res['loc_tag'], subj_res['anat_region'],
                                                  subj_res['chan_tags'])
        is_right = loc_dict['is_right']
        for r_count, region in enumerate(all_names_lh):

            lh_inds = ~is_right & (subj_res['anat_region'] == region)
            if np.any(lh_inds):
                lh_subjs_by_region[r_count, s_count] = np.mean(data[lh_inds])

            rh_inds = is_right & (subj_res['anat_region'] == region)
            if np.any(rh_inds):
                rh_subjs_by_region[r_count, s_count] = np.mean(data[rh_inds])

    l_mean = np.nanmean(lh_subjs_by_region, axis=1)
    r_mean = np.nanmean(rh_subjs_by_region, axis=1)
    vtx_data_lh = l_mean[labels_lh]
    vtx_data_rh = r_mean[labels_rh]
    # pdb.set_trace()
    return vtx_data_lh, vtx_data_rh

    # lim = np.max(np.abs([np.min(l_mean), np.max(l_mean)]))
    # brain = Brain('average', 'both', 'pial', views='lateral', cortex='classic',
    #               background='white', offscreen=False, size=(800,800))
    # v = mlab.view()
    # brain.add_data(-vtx_data_lh, -lim, lim, colormap="RdBu", alpha=.8, remove_existing=True, hemi='lh',
    #                      colorbar=False)



# lh_roi_data = np.zeros(shape=(len(all_names_lh)))
# # lh_roi_data[:] = np.nan
# rh_roi_data = np.zeros(shape=(len(all_names_rh)))
# # rh_roi_data[:] = np.nan
#
# for r, region in enumerate(regions):
#     names = lookup_label(region)
#     for name in names:
#         lh_roi_data[all_names_lh == name] = data[0, r] if n[0, r] > 5 else 0.0
#         rh_roi_data[all_names_rh == name] = data[1, r] if n[1, r] > 5 else 0.0
#
# vtx_data_lh = lh_roi_data[labels_lh]
# vtx_data_rh = rh_roi_data[labels_rh]
# return vtx_data_lh, vtx_data_rh
#
#



