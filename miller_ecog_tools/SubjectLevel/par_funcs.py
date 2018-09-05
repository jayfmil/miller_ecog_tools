"""
Needs some cleanup. Intended to be functions that may need to be performed many times and therefore suitable for
parallelization. Functions should accept only one input.
"""
import numpy as np
import statsmodels.api as sm
import pdb
import numexpr
from scipy.signal import argrelmax
from scipy.stats import ttest_ind
from xarray import concat
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from tqdm import tqdm


def par_find_peaks_by_chan(info):
    """
    Parameters
    ----------
    p_spect_array: numpy.ndarray
        An array with dimensions frequencies x channels
    frequencies: numpy.ndarray
        An array of the frequencies used
    std_thresh: float
        Threshold in number of standard deviations above the corrected power spectra to be counted as a peak

    Returns
    -------
    peaks_all_chans: numpy.ndarray with type bool
        An array of booleans the same shape as p_spect_array, specifying if there is a peak at a given frequency
        and electrode
    """

    p_spect_array = info[0]
    frequencies = info[1]
    std_thresh = info[2]

    peaks_all_chans = np.zeros(p_spect_array.shape).astype(bool)
    for i, chan_data in enumerate(p_spect_array.T):
        x = sm.tools.tools.add_constant(np.log10(frequencies))
        model_res = sm.RLM(chan_data, x).fit()
        peak_inds = argrelmax(model_res.resid)
        peaks = np.zeros(x.shape[0], dtype=bool)
        peaks[peak_inds] = True
        above_thresh = model_res.resid > (np.std(model_res.resid) * std_thresh)
        peaks_all_chans[:,i] = peaks & above_thresh
    return peaks_all_chans


def par_robust_reg(info):
    """
    Parallelizable robust regression function

    info: two element list. first element, power spectra: # freqs x # elecs. Second element: log transformed freqs

    returns intercepts, slopes, resids
    """

    p_spects = info[0]
    x = sm.tools.tools.add_constant(info[1])

    # holds slope of fit line
    slopes = np.empty((p_spects.shape[1]))
    slopes[:] = np.nan

    # holds residuals
    resids = np.empty((p_spects.shape[0], p_spects.shape[1]))
    resids[:] = np.nan

    # holds intercepts
    intercepts = np.empty((p_spects.shape[1]))
    intercepts[:] = np.nan

    # holds mean height of fit line
    bband_power = np.empty((p_spects.shape[1]))
    bband_power[:] = np.nan

    # loop over every electrode
    for i, y in enumerate(p_spects.T):
        model_res = sm.RLM(y, x).fit()
        intercepts[i] = model_res.params[0]
        slopes[i] = model_res.params[1]
        bband_power[i] = model_res.fittedvalues.mean()
        resids[:, i] = model_res.resid

    return intercepts, slopes, resids, bband_power


def par_robust_reg_no_low_freqs(info):
    """
    Parallelizable robust regression function

    info: two element list. first element, power spectra: # freqs x # elecs. Second element: log transformed freqs

    returns intercepts, slopes, resids
    """

    p_spects = info[0]
    x = sm.tools.tools.add_constant(info[1])
    freq_inds = info[2]

    # holds slope of fit line
    slopes = np.empty((p_spects.shape[1]))
    slopes[:] = np.nan

    # holds residuals
    resids = np.empty((p_spects.shape[0], p_spects.shape[1]))
    resids[:] = np.nan

    # holds intercepts
    intercepts = np.empty((p_spects.shape[1]))
    intercepts[:] = np.nan

    # holds mean height of fit line
    bband_power = np.empty((p_spects.shape[1]))
    bband_power[:] = np.nan

    # loop over every electrode
    for i, y in enumerate(p_spects.T):
        model_res = sm.RLM(y[freq_inds], x[freq_inds]).fit()
        intercepts[i] = model_res.params[0]
        slopes[i] = model_res.params[1]
        bband_power[i] = model_res.fittedvalues.mean()
        resids[:, i] = y - ((x[:, 1]*model_res.params[1]) + model_res.params[0])

    return intercepts, slopes, resids, bband_power


def my_local_max(arr):
    """
    Returns indices of local maxima in a 1D array. Unlike scipy.signal.argrelmax, this does not ignore consecutive
    values that are peaks. It finds the last repetition.

    """
    b1 = arr[:-1] <= arr[1:]
    b2 = arr[:-1] > arr[1:]
    k = np.where(b1[:-1] & b2[1:])[0] + 1
    if arr[0] > arr[1]:
        k = np.append(k, 0)
    if arr[-1] > arr[-2]:
        k = np.append(k, len(arr) - 1)
    return k


def par_find_peaks(info):
    """
    Parallelizable peak picking function, uses robust reg but returns

    """

    # def moving_average(a, n=3):
    #     ret = np.cumsum(a, dtype=float)
    #     ret[n:] = ret[n:] - ret[:-n]
    #     return ret[n - 1:] / n

    p_spect = info[0]
    x = sm.tools.tools.add_constant(info[1])
    model_res = sm.RLM(p_spect, x).fit()
    peak_inds = my_local_max(model_res.resid)
    peaks = np.zeros(x.shape[0], dtype=bool)
    peaks[peak_inds] = True
    above_thresh = model_res.resid > np.std(model_res.resid)
    peaks = peaks & above_thresh
    return peaks




# def par_compute_power_chunk(info):
#
#     eeg = info[0]
#     freqs = info[1]
#     buf_dur = info[2]
#     time_bins = info[3]
#
#     chunk_pow_mat, _ = MorletWaveletFilterCpp(time_series=eeg, freqs=freqs,
#                                               output='power', width=5, cpus=25).filter()
#     dims = chunk_pow_mat.dims
#     # remove buffer and log transform
#     chunk_pow_mat = chunk_pow_mat.remove_buffer(buf_dur)
#     data = chunk_pow_mat.data
#     chunk_pow_mat.data = numexpr.evaluate('log10(data)')
#     dim_str = chunk_pow_mat.dims[1]
#     coord = chunk_pow_mat.coords[dim_str]
#     ev = chunk_pow_mat.events
#     freqs = chunk_pow_mat.frequency
#     sr = chunk_pow_mat.samplerate
#     # np.log10(chunk_pow_mat.data, out=chunk_pow_mat.data)
#
#     # mean power over time or time bins
#     if time_bins is None:
#         chunk_pow_mat = chunk_pow_mat.mean(axis=3)
#     else:
#         pow_list = []
#         # pow_mat = np.empty((chunk_pow_mat.shape[0], chunk_pow_mat.shape[1], chunk_pow_mat.shape[2], len(time_bins)))
#         # pdb.set_trace()
#
#
#         for tbin in tqdm(time_bins):
#             # print(t)
#             # tmp2 = [np.mean(chunk_pow_mat.data[:, :, :, inds], axis=3) for inds in tmp]
#             # tmp = [(chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1]) for tbin in self.time_bins]
#             # tmp = [np.where((chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1]))[0] for tbin in self.time_bins]
#             # tmp2 = np.expand_dims(np.stack(tmp, 0), 0)
#             # chunk_pow_mat.data.T[tmp3].mean(axis=1).T
#             # now = time.time()
#             inds = (chunk_pow_mat.time.data >= tbin[0]) & (chunk_pow_mat.time.data < tbin[1])
#             # print('Finding inds')
#             # print(time.time()-now)
#
#             # now = time.time()
#             # pow_mat[:, :, :, t] = np.mean(chunk_pow_mat.data[:, :, :, inds], axis=3)
#             # print('mean and append new')
#             # print(time.time() - now)
#             #
#             # now = time.time()
#             pow_list.append(np.mean(chunk_pow_mat.data[:, :, :, inds], axis=3))
#             # pdb.set_trace()
#             # print('mean and append')
#             # print(time.time() - now)
#         chunk_pow_mat = np.stack(pow_list, axis=3)
#         chunk_pow_mat = TimeSeriesX(data=chunk_pow_mat,
#                                     dims=['frequency', dim_str, 'events', 'time'],
#                                     coords={'frequency': freqs,
#                                             dim_str: coord,
#                                             'events': ev,
#                                             'time': time_bins.mean(axis=1),
#                                             'samplerate': sr})
#
#     return chunk_pow_mat
#
#
#
#
#
#





