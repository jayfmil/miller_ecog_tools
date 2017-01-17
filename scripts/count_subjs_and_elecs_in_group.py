import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def count_subjs_and_elecs(subject_objs):
    regions = np.array(['IFG', 'MFG', 'SFG', 'MTL', 'Hipp', 'TC', 'IPC', 'SPC', 'OC'])
    elec_count = np.zeros((len(subject_objs), len(regions)), dtype=int)

    for i, subj in enumerate(subject_objs):
        elec_count[i] = [np.sum(subj.elec_locs[x]) for x in regions]
    return elec_count, regions

def plot_counts(elec_count, regions):

    with plt.style.context('myplotstyle.mplstyle'):
        f, ax = plt.subplots(2, sharex=True)
        _ = ax[0].bar(np.arange(regions.shape[0]), elec_count.sum(axis=0), align='center', zorder=5,
                      color='#1F77B4', lw=1.5)
        ax[0].set_ylabel('Electrode Count', fontsize=14)
        ax[0].set_yticklabels(np.array(ax[0].get_yticks().astype(int)), fontsize=12)
        ax[0].get_yaxis().set_label_coords(-0.13, 0.5)

        _ = ax[1].bar(np.arange(regions.shape[0]), np.sum(elec_count > 0, axis=0), align='center', zorder=5,
                      color='#1F77B4', lw=1.5)
        ax[1].set_ylabel('Subject Count', fontsize=14)
        ax[1].set_xlabel('Region', fontsize=14)
        ax[1].set_yticklabels(np.array(ax[1].get_yticks().astype(int)), fontsize=12)
        _ = plt.xticks(np.arange(regions.shape[0]), regions, fontsize=12)
        ax[1].get_yaxis().set_label_coords(-0.13, 0.5)
        f.set_size_inches(6, 4)
