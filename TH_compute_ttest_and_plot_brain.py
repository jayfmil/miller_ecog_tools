import TH_compute_ttest
import RAM_plotBrain
from ptsa.data.readers.TalReader import TalReader
import os
import numpy as np
from glob import glob
import re
import nibabel as nib
from mayavi import mlab
# import matplotlib.pyplot as plt
# plt.ion()

from tvtk.api import tvtk
from tvtk.common import configure_input_data

# from pyface.api import GUI

regions = ['IFG', 'MFG', 'SFG', 'TC', 'STG', 'IPC', 'SPC', 'OC', 'PRECENTRAL', 'POSTCENTRAL']
cmap = np.ones(shape=(63,3))
cmap[0:32,0] = np.linspace(0,1,32)
cmap[0:32,1] = np.linspace(0,1,32)
cmap[31:,1] = np.linspace(1,0,32)
cmap[31:,2] = np.linspace(1,0,32)

def process_all_subjs(freq_bins):
    power_dir = '/scratch/jfm2/python_power/TH/50_freqs/window_size_1000_step_size_10'
    subjs = glob(os.path.join(power_dir, 'R*.p'))
    subjs = [re.search(r'R\d\d\d\d[A-Z](_\d+)?', f).group() for f in subjs]
    subjs.sort()

    # t stat by region by time
    t_region_all = []
    for subj in subjs:
        print 'Processing %s' % subj
        t_region_subj = process_subj(subj, freq_bins)
        t_region_all.append(t_region_subj)

    t_region_all = np.stack(t_region_all, -1)

    # remove regions with less than 5 subjs
    n = np.sum(~np.isnan(t_region_all[:, :, 0]), axis=2)

    t_region_mean = np.nanmean(t_region_all, axis=3)
    lim = np.max(np.abs([np.min(t_region_mean), np.max(t_region_mean)]))
    bins = np.linspace(-lim, lim, 63)

    # create initial brain
    brain = RAM_plotBrain.BrainPlot(hemi='split', cortex=('Greys', -1, 2, True),views='lateral',avg_surf=True, offscreen=False)
    brain.show_brain()

    # iterate over time bins. I know this is not the right python way to do it
    frame = 0
    degs = np.linspace(180.,180.*4,np.shape(t_region_mean)[2])
    timesMS = np.arange(-1500,1500,10)
    # for i in np.arange(0,np.shape(t_region_mean)[2]):
    for i in np.arange(np.shape(t_region_mean)[2]):
    # for i in np.arange(1):
        data = t_region_mean[:,:,i]

        vtx_data_lh, vtx_data_rh = create_data_overlay(data, regions, n)
        brain.brain.add_data(-vtx_data_lh, -lim, lim, colormap="RdBu", alpha=.8, remove_existing=True, hemi='lh', colorbar=False)
        mlab.text(.4, .9, str(timesMS[frame]) + ' MS', line_width=1, width=.2)
        brain.brain.add_data(-vtx_data_rh, -lim, lim, colormap="RdBu", alpha=.8, remove_existing=True, hemi='rh', colorbar=False)
        # mlab.view(azimuth=degs[frame])
        # mlab.text(.4,.9,str(timesMS[frame])+' MS',line_width=1,width=.2)
        # mlab.savefig('/home1/jfm2/brain_movie_new/frame_' + "%05d" % (frame,) + '.png')
        brain.brain.save_image('/home1/jfm2/brain_movie_new_hfa/frame_' + "%05d" % (frame,) + '.png')

        # this is needed for the figures to save correctly I have no idea why it's ridiculous
        # fig = mlab.gcf()
        # rw = tvtk.RenderWindow(size=fig.scene._renwin.size, off_screen_rendering=1)
        # rw.add_renderer(fig.scene._renderer)
        #
        # w2if = tvtk.WindowToImageFilter()
        # w2if.magnification = fig.scene.magnification
        # w2if.input = rw
        # ex = tvtk.PNGWriter()
        # ex.file_name = '/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png'
        # configure_input_data(ex, w2if.output)
        # w2if.update()
        # ex.write()




        # mlab.savefig(figure=fig, filename='/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')

        # mlab.savefig('/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')
        # mlab.show()
        # _gui = GUI()
        # orig_val = _gui.busy
        # _gui.set_busy(busy=True)
        # _gui.set_busy(busy=orig_val)
        # _gui.process_events()
        # imgmap = mlab.screenshot(brain.brain_surf[0]._f, mode='rgba', antialiased=True)
        # plt.imshow(imgmap)
        # imgmap = mlab.screenshot(mode='rgba', antialiased=True)
        # plt.imshow(imgmap)
        # plt.imsave(arr=imgmap, fname='/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')
        # brain.brain.save_single_image('/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')
        # brain.brain.save_image('/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')

        frame += 1
        # for f in range(2):
        #     mlab.view(azimuth=180 + frame)
        #     brain.brain.save_single_image('/home1/jfm2/brain_movie/frame_' + "%05d" % (frame,) + '.png')


    # return t_region_mean, brain, regions, bins

def create_data_overlay(data, regions, n):
    labels_lh, ctab_lh, all_names_lh = nib.freesurfer.read_annot('/home1/jfm2/fs/average/label/lh.aparc.annot')
    all_names_lh = np.array(all_names_lh)
    labels_rh, ctab_rh, all_names_rh = nib.freesurfer.read_annot('/home1/jfm2/fs/average/label/rh.aparc.annot')
    all_names_rh = np.array(all_names_rh)
    lh_roi_data = np.zeros(shape=(len(all_names_lh)))
    # lh_roi_data[:] = np.nan
    rh_roi_data = np.zeros(shape=(len(all_names_rh)))
    # rh_roi_data[:] = np.nan

    for r, region in enumerate(regions):
        names = lookup_label(region)
        for name in names:
            lh_roi_data[all_names_lh == name] = data[0, r] if n[0, r] > 5 else 0.0
            rh_roi_data[all_names_rh == name] = data[1, r] if n[1, r] > 5 else 0.0

    vtx_data_lh = lh_roi_data[labels_lh]
    vtx_data_rh = rh_roi_data[labels_rh]
    return vtx_data_lh, vtx_data_rh


def add_color_to_brain(brain, data, regions, bins):
    for h, hemi in enumerate(['lh', 'rh']):
        for r, region in enumerate(regions):
            labels = lookup_label(region)
            c_ind = np.where(np.min(abs(bins - data[h, r])) == abs(bins - data[h, r]))
            print c_ind
            for label in labels:
                print label
                brain.brain.add_label(os.path.join('/home1/jfm2/fs/jfm_labels/',hemi+'.'+label+'.label'),
                                      hemi=hemi)#, color=cmap[c_ind,:][0][0])

def lookup_label(region):
    if region == 'IFG':
        labels = ['parsopercularis', 'parsorbitalis', 'parstriangularis']
    elif region == 'MFG':
        labels = ['caudalmiddlefrontal', 'rostralmiddlefrontal']
    elif region == 'SFG':
        labels = ['superiorfrontal']
    elif region == 'TC':
        labels = ['middletemporal', 'inferiortemporal']
    elif region == 'STG':
        labels = ['superiortemporal']
    elif region == 'IPC':
        labels = ['inferiorparietal', 'supramarginal']
    elif region == 'SPC':
        labels = ['superiorparietal', 'precuneus']
    elif region == 'OC':
        labels = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']
    elif region == 'PRECENTRAL':
        labels = ['precentral']
    elif region == 'POSTCENTRAL':
        labels = ['postcentral']
    return labels

def process_subj(subj, freq_bins):

    # compute ttest at each elec and freq
    ttest = TH_compute_ttest.ComputeTTest(subj)
    t, p, freqs = ttest.compute_ttest_at_each_time(subj)
    t = np.mean(t[(freqs >= freq_bins[0]) & (freqs <= freq_bins[1]), :], axis=0)

    tal = load_tal(subj)
    rois = bin_elecs_by_region(tal)

    t_by_region = np.empty(shape=(2, len(regions), np.shape(t)[1]))
    t_by_region[:] = np.nan

    for h, hemi in enumerate(['left', 'right']):
        for r, region in enumerate(regions):
            roi_inds = rois[hemi] & rois[region]
            if roi_inds.any():
                t_by_region[h, r, :] = np.mean(t[roi_inds,:],axis=0)

    return t_by_region


def load_tal(subj):
    tal_ext = '_bipol.mat' #if self.bipolar else '_monopol.mat'
    tal_path = os.path.join('/data/eeg', subj, 'tal', subj + '_talLocs_database' + tal_ext)
    tal_reader = TalReader(filename=tal_path)
    tal_struct = tal_reader.read()
    return tal_struct


def bin_elecs_by_region(tal):
    regions = tal['avgSurf']['anatRegion']

    rois = {}
    rois['left'] = tal['avgSurf']['x'] < 0
    rois['right'] = tal['avgSurf']['x'] > 0
    rois['IFG'] = (regions == 'parsopercularis') | (regions == 'parsorbitalis') | (regions == 'parstriangularis')
    rois['MFG'] = (regions == 'caudalmiddlefrontal') | (regions == 'rostralmiddlefrontal')
    rois['SFG'] = regions == 'superiorfrontal'
    rois['TC'] = (regions == 'middletemporal') | (regions == 'inferiortemporal')
    rois['STG'] = regions == 'superiortemporal'
    rois['IPC'] = (regions == 'inferiorparietal') | (regions == 'supramarginal')
    rois['SPC'] = (regions == 'superiorparietal') | (regions == 'precuneus')
    rois['OC'] = (regions == 'lateraloccipital') | (regions == 'lingual') | (regions == 'cuneus') | (regions == 'pericalcarine')
    rois['PRECENTRAL'] = regions == 'precentral'
    rois['POSTCENTRAL'] = regions == 'postcentral'
    return rois

if __name__ == '__main__':
    process_all_subjs([1, 12])