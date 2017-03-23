import numpy as np
import statsmodels.api as sm
import pdb
from scipy.signal import argrelmax
from scipy.stats import ttest_ind

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
    peak_inds = argrelmax(model_res.resid)
    peaks = np.zeros(x.shape[0], dtype=bool)
    peaks[peak_inds] = True
    above_thresh = model_res.resid > np.std(model_res.resid)
    peaks = peaks & above_thresh
    return peaks
