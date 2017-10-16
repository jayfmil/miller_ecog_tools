import numpy as np
import pandas as pd


def filter_out_bad_elecs(subj, bad_elec_table):

    bad_ictal = bad_elec_table[bad_elec_table.index == [int(subj.subj[2:5])]]['IctalOnset']
    if not isinstance(bad_ictal.values[0], str):
        bad_ictal = ''
    else:
        bad_ictal = bad_ictal.values[0]

    bad_spiking = bad_elec_table[bad_elec_table.index == [int(subj.subj[2:5])]]['IctalSpiking']
    if not isinstance(bad_spiking.values[0], str):
        bad_spiking = ''
    else:
        bad_spiking = bad_spiking.values[0]

    bad_elecs = bad_spiking + ' ' + bad_ictal
    bad_elecs = np.array(bad_elecs.split())

    if len(bad_elecs) == 0:
        print('%s: no bad elecs.' % subj.subj)
        return subj

    bad_bools = np.zeros(len(subj.subject_data.attrs['chan_tags'])).astype(bool)
    for bad_elec in bad_elecs:
        is_bad = np.any(np.array([pair.split('-') for pair in subj.subject_data.attrs['chan_tags']]) == bad_elec, axis=1)
        bad_bools[is_bad] = True

    subj.res['ts'] = subj.res['ts'][:, ~bad_bools]
    subj.res['ps'] = subj.res['ps'][:, ~bad_bools]
    subj.res['zs'] = subj.res['zs'][:, ~bad_bools]
    subj.elec_xyz_avg = subj.elec_xyz_avg[~bad_bools]
    subj.elec_xyz_indiv = subj.elec_xyz_indiv[~bad_bools]
    for key in subj.elec_locs.keys():
        subj.elec_locs[key] = subj.elec_locs[key][~bad_bools]
    return subj