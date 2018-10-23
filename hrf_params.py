#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:05:57 2018

@author: Marian
"""
from pyprf_feature.analysis.utils_hrf import spm_hrf_compat
import numpy as np
import matplotlib.pyplot as plt
import itertools


def create_hrf_params(varPkDelMin, varPkDelMax, varUndDelMin, varUndDelMax,
                      varDelStp, varPkDspMin, varPkDspMax, varUndDspMin,
                      varUndDspMax, varDspStp, varPkUndRatMin, varPkUndRatMax,
                      varPkUndRatStp):
    """Create combinations of hrf parameters.

    Parameters
    ----------
    varPkDelMin : float, positive
        Minimum delay for the peak of the hrf function.
    varPkDelMax : float, positive
        Maximum delay for the peak of the hrf function.
    varUndDelMin : float, positive
        Minimum delay for the undershoot of the hrf function.
    varUndDelMax : float, positive
        Maximum delay for the undershoot of the hrf function.
    varDelStp : float, positive
        Step size for peak and undershoot delay of the hrf function.
    varPkDspMin : float, positive
        Minimum dispersion for the peak of the hrf function.
    varPkDspMax : float, positive
        Maximum dispersion for the peak of the hrf function.
    varUndDspMin : float, positive
        Minimum dispersion for the undershoot of the hrf function.
    varUndDspMax : float, positive
        Maximum dispersion for the undershoot of the hrf function.
    varDspStp : float, positive
        Step size for peak and undershoot dispersion of the hrf function.         
    varPkUndRatMin : float, positive
        Minimum peak to undershoot ratio.
    varPkUndRatMax : float, positive
        Maximum peak to undershoot ratio.
    varPkUndRatStp : float, positive
        Step size for teh peak to undershoot ratio.

    Returns
    -------
    aryPrm : numpy array
        All possible hrf parameters combinations.

    """

    # Define vector for delay to peak, default: 6
    vecPkDel = np.arange(varPkDelMin, varPkDelMax+varDelStp, varDelStp)

    # Define vector for delay to undershoot, default: 16
    vecUndDel = np.arange(varUndDelMin, varUndDelMax+varDelStp, varDelStp)

    # Define vector for peak dispersion, default: 1
    vecPkDsp = np.arange(varPkDspMin, varPkDspMax+varDspStp, varDspStp)

    # Define vector for undershoot dispersion, default: 1
    vecUndDsp = np.arange(varUndDspMin, varUndDspMax+varDspStp, varDspStp)

    # Define vector for weighting of undershoot relative to peak,
    # e.g. 6 means 1:6 weighting
    vecPkUndRat = np.arange(varPkUndRatMin, varPkUndRatMax+varPkUndRatStp,
                            varPkUndRatStp)

    # Find combinations of all parameters
    # Exclude combinations where undershoot delay less than 
    iterables = [vecPkDel, vecUndDel]
    vecDelCmb = list(itertools.product(*iterables))
    vecDelCmb = np.asarray(vecDelCmb)
    # Pass only the combinations where a1 < a2
    lgcDelCmb = np.less(vecDelCmb[:, 0], vecDelCmb[:, 1])
    vecDelCmb = vecDelCmb[lgcDelCmb]

    iterables = [vecDelCmb, vecPkDsp, vecUndDsp, vecPkUndRat]
    aryPrm = list(itertools.product(*iterables))
    for ind, item in enumerate(aryPrm):
        aryPrm[ind] = list(np.hstack(item))
    aryPrm = np.array(aryPrm)
    
    return aryPrm.astype(np.float32)


def wrap_hrf_params_dct(aryPrm, indPrm):
    """Wrap hrf parameters into dictionary.

    Parameters
    ----------
    aryPrm : numpy array
        Array with all combinations of hrf parameters.
    indPrm : integer, positive
        Index for hrf parameter array.

    Returns
    -------
    dctPrms : dictionary
        Dictionary with hrf parameters.

    """
    
    # Terieve vector with hrf parameters
    vecPrm = aryPrm[indPrm]
    
    # Create dictionary for hrf function
    dctPrms = {}
    dctPrms['peak_delay'] = float(vecPrm[0])
    dctPrms['under_delay'] = float(vecPrm[1])
    dctPrms['peak_disp'] = float(vecPrm[2])
    dctPrms['under_disp'] = float(vecPrm[3])
    dctPrms['p_u_ratio'] = float(vecPrm[4])
    
    return dctPrms



dictTest = wrap_hrf_params_dct(aryPrm, 0)

t = np.arange(30)

testBase = spm_hrf_compat(t, **dctPrms)

dctPrms['p_u_ratio'] = 15
testChange = spm_hrf_compat(t, **dctPrms)
plt.plot(t, testBase)
plt.plot(t, testChange)