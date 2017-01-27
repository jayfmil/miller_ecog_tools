import numpy as np
import statsmodels.api as sm
import pdb

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
        pdb.set_trace()

    return intercepts, slopes, resids, bband_power
