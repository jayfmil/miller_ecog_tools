"""
Code for loading Neurtex Brain Research Institute data. Basically, loads custom neuralynx-like files. Only for use
with this project, as there a lot of custom file types and directory paths/filenames.

Built for use with data from the continuous recognition (CRM) paradigm.
"""

import numpy as np
import pandas as pd
import re

# file constants
HEADER_SIZE = 16 * 1024
BLOCK_SIZE = 512

# behavioral data is all in one big table. Defined at module scope
my_globals = {'master_table_path': '/scratch/josh/BniData/BniData/Analysis/CRM/masterPseudoZ_r5.txt',
              'master_table_data': None}


def set_master_table(filepath):
    """
    Set the master_table_path variable. Also unloads the cached master data if loaded.

    Parameters
    ----------
    filepath: str
        The path to the file.

    """
    my_globals['master_table_path'] = filepath
    my_globals['master_table_data'] = None


def load_master_table():
    """
    Loads master data table and stores it in module variable my_globals['master_table_data']. We are caching it
    because it'll be faster than having to load this over and over.
    """

    # read table
    my_globals['master_table_data'] = pd.read_table(my_globals['master_table_path'])

    # also make a new column with a unique subject identifier
    my_globals['master_table_data']['subject'] = my_globals['master_table_data'].expID.apply(lambda x: x.split('e')[0])


def get_subjs(task='crm'):
    """
    Parameters
    ----------
    task: str
        Task to use. CURRENTLY ONLY 'crm' IS SUPPORTED

    Returns
    -------
    numpy.ndarray
        array of subject names. Technically these are sessions, not subjects.

    """
    if task != 'crm':
        print('CURRENTLY ONLY crm TASK IS SUPPORTED.')
        return

    # load master data if not cached already
    if my_globals['master_table_data'] is None:
        load_master_table()

    # return unique subjects
    return my_globals['master_table_data'].subject.unique()


def load_subj_events(task='crm', subject=''):
    """
    Parameters
    ----------
    task: str
        Task to use. CURRENTLY ONLY 'crm' IS SUPPORTED
    subject: str
        Subject to use

    Returns
    -------
    pandas.DataFrame
        DataFrame of events for the given subject.
    """

    if task != 'crm':
        print('CURRENTLY ONLY crm TASK IS SUPPORTED.')
        return

    # load master data if not cached already
    if my_globals['master_table_data'] is None:
        load_master_table()

    # filter to just this subject
    df_subj = my_globals['master_table_data'][my_globals['master_table_data'].subject == subject]

    # reduce to only columns with relavent behavioral data and unique rows
    df_subj = df_subj[['expID', 'rep', 'name', 'stTime', 'endTime', 'firstResp', 'keyEarly', 'oldKey', 'otherKey',
                       'multiPress', 'delay', 'isPaired', 'pairedWithDup', 'isFirst', 'lag']].drop_duplicates()

    return df_subj


def stat_ncs(channel_file):
    header_keys = [('NLX_Base_Class_Type', None),
                   ('AmpHiCut', float),
                   ('ADChannel', None),
                   ('ADGain', float),
                   ('AmpGain', float),
                   ('SubSamplingInterleave', int),
                   ('ADMaxValue', int),
                   ('ADBitVolts', float),
                   ('SamplingFrequency', float),
                   ('AmpLowCut', float),
                   ('HardwareSubSystemName', None),
                   ('HardwareSubSystemType', None)]

    # load header from file
    with open(channel_file, 'rb') as f:
        txt_header = f.read(HEADER_SIZE)
    txt_header = txt_header.strip(b'\x00').decode('latin-1')

    # find values and make dict
    info = {}
    for k, type_ in header_keys:
        pattern = '-(?P<name>' + k + ')\t(?P<value>[\S ]*)'
        matches = re.findall(pattern, txt_header)
        for match in matches:
            name = match[0]
            val = match[1].rstrip(' ')
            if type_ is not None:
                val = type_(val)
            info[name] = val
    return info


def load_ncs(channel_file):
    # load header info
    info = stat_ncs(channel_file)

    # define datatype for memmap
    ncs_dtype = [('timestamp', 'uint64'), ('channel', 'uint32'), ('sample_rate', 'uint32'),
                 ('nb_valid', 'uint32'), ('samples', 'int16', (BLOCK_SIZE,))]

    # load it all at once... sorry.
    data = np.memmap(channel_file, dtype=ncs_dtype, mode='r', offset=HEADER_SIZE)

    # loop over each block and create timestamps for each sample
    signals = []
    timestamps = []
    for this_block in data:
        # extend our list of the data
        signals.extend(this_block[4])

        # create timestamps for each sample in this block
        timestamps_block = np.linspace(this_block[0], this_block[0] + (1e6 / this_block[2] * (BLOCK_SIZE - 1)),
                                       BLOCK_SIZE)
        timestamps.append(timestamps_block)

    # get our final arrays
    timestamps = np.concatenate(timestamps)
    signals = np.array(signals)
    signals = signals * info['ADBitVolts']

    return signals, timestamps, info['SamplingFrequency']

def load_nse(channel_file, return_waveforms=False):

    # nse dtype
    dtype = [('timestamp', 'uint64'), ('channel', 'uint32'), ('unit_id', 'uint32')]
    dtype += [('params', 'uint32', (8,))]
    dtype += [('samples', 'uint16', (32,))]

    # load spiking data
    data = np.memmap(channel_file, dtype=dtype, mode='r', offset=HEADER_SIZE)

    # get timestamps
    timestamps = np.array([x[0] for x in data])

    # get spike waveforms
    if return_waveforms:
        spike_waveforms = np.array([x[4] for x in data])
        return timestamps, spike_waveforms
    else:
        return timestamps

def segment_ncs_into_epochs(events, rel_start_ms, rel_stop_ms, buf_ms=0, channel_numbers=None, noise_freq=[58., 62.],
             resample_freq=None, pass_band=None, use_mirror_buf=False, demean=False):
    return


