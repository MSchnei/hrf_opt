# -*- coding: utf-8 -*-
"""Cythonised least squares GLM model fitting with 1 predictor."""

# Part of pyprf_feature library
# Copyright (C) 2018  Omer Faruk Gulban & Ingo Marquardt & Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# *****************************************************************************
# *** Import modules & adjust cython settings for speedup

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# *****************************************************************************


# *****************************************************************************
# *** Main function least squares solution, no cross-validation, 1 predictor

cpdef tuple cy_lst_sq_one(
    np.ndarray[np.float32_t, ndim=2] aryMdlRsp,
    np.ndarray[np.float32_t, ndim=1] vecFunc):
    """
    Cythonised least squares GLM model fitting.

    Parameters
    ----------
    aryMdlRsp : np.array
        2D numpy array, at float32 precision, containing mutliple hrf model
        time courses. Dimensionality: aryMdlRsp[time, nr of models].
    vecFunc : np.array
        1D numpy array, at float32 precision, containing a single voxel time
        course. Dimensionality: vecFunc[time].

    Returns
    -------
    vecRes : np.array
        1D numpy array with model residuals for all models.
        Dimensionality: vecRes[nr of models]
    vecPe : np.array
        1D numpy array with parameter estimates for all models.
        Dimensionality: vecPe[nr of models]


    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses. Assumes removal of the mean from
    the functional data and the model. Needs to be compiled before execution
    (see `cython_leastsquares_setup.py`).
    """

    cdef:
        float[:, :] aryMdlRsp_view = aryMdlRsp
        float[:] vecFunc_view = vecFunc
        unsigned long varNumMdls
        unsigned int varNumVols

    # Number of models in the model array:
    varNumMdls = int(aryMdlRsp.shape[1])

    # Number of volumes in the model/voxel time course:
    varNumVols = int(aryMdlRsp.shape[0])

    # Define 1D array for results (i.e. for residuals of least squares
    # solution):
    cdef np.ndarray[np.float32_t, ndim=1] vecRes = np.zeros(varNumMdls,
                                                            dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] vecPe = np.zeros(varNumMdls,
                                                           dtype=np.float32)
    # Memory view on array for results:
    cdef float[:] vecRes_view = vecRes

    # Memory view on array for parameter estimates:
    cdef float[:] vecPe_view = vecPe

    # Call optimised cdef function for calculation of residuals:
    vecRes_view, vecPe_view = func_cy_res(aryMdlRsp_view,
                                          vecFunc_view,
                                          vecRes_view,
                                          vecPe_view,
                                          varNumMdls,
                                          varNumVols)

    # Convert memory view to numpy array before returning it:
    vecRes = np.asarray(vecRes_view)
    vecPe = np.asarray(vecPe_view).T

    return vecPe, vecRes

# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, no cross-validation, 1 predictor

cdef (float[:], float[:]) func_cy_res(float[:, :] aryMdlRsp_view,
                                      float[:] vecFunc_view,
                                      float[:] vecRes_view,
                                      float[:] vecPe_view,
                                      unsigned long varNumMdls,
                                      unsigned int varNumVols):

    cdef:
        float varVarY, varCovXy, varRes, varSlope, varXhat
        unsigned int idxVol
        unsigned long idxMdl

    # Loop through models:
    for idxMdl in range(varNumMdls):

        # Covariance and residuals of current voxel:
        varVarY = 0
        varCovXy = 0
        varRes = 0

        # Loop through volumes and calculate variance of the model time course
        # and covariance between the model and the current voxel:
        for idxVol in range(varNumVols):
            varVarY += aryMdlRsp_view[idxVol, idxMdl] ** 2
            varCovXy += (vecFunc_view[idxVol]
                         * aryMdlRsp_view[idxVol, idxMdl])
        # Obtain the slope of the regression of the model on the data:
        varSlope = varCovXy / varVarY

        # Loop through volumes again in order to calculate the error in the
        # prediction:
        for idxVol in range(varNumVols):
            # The predicted voxel time course value:
            varXhat = aryMdlRsp_view[idxVol, idxMdl] * varSlope
            # Mismatch between prediction and actual voxel value (variance):
            varRes += (vecFunc_view[idxVol] - varXhat) ** 2

        vecRes_view[idxMdl] = varRes
        vecPe_view[idxMdl] = varSlope

    # Return memory view:
    return vecRes_view, vecPe_view


# *****************************************************************************

# *****************************************************************************
# *** Main function least squares solution, with cross-validation, 1 predictor

cpdef np.ndarray[np.float32_t, ndim=2] cy_lst_sq_xval_one(
    np.ndarray[np.float32_t, ndim=2] aryMdlRsp,
    np.ndarray[np.float32_t, ndim=1] vecFunc,
    np.ndarray[np.int32_t, ndim=2] aryIdxTrn,
    np.ndarray[np.int32_t, ndim=2] aryIdxTst
    ):
    """
    Cythonised least squares GLM model fitting with cross validation.

    Parameters
    ----------
    aryMdlRsp : np.array
        2D numpy array, at float32 precision, containing mutliple hrf model
        time courses. Dimensionality: aryMdlRsp[time, nr of models].
    vecFunc : np.array
        1D numpy array, at float32 precision, containing a single voxel time
        course. Dimensionality: vecFunc[time].
    aryIdxTrn : np.array
        2D numpy array, at int32 precision, containing training indices for
        cross-validation.
    aryIdxTst : np.array
        2D numpy array, at int32 precision, containing test indices for
        cross-validation.

    Returns
    -------
    aryResXval : np.array
        2D numpy array with cross validation error for all models and all
        cross-validation folds.
        Dimensionality: aryResXval[model, varNumXval]

    Notes
    -----
    Computes the least-squares solution for the model fit between the pRF time
    course model, and all voxel time courses with k-fold cross validation.
    Assumes removal of the mean from the functional data and the model.
    Needs to be compiled before execution (see `cython_leastsquares_setup.py`).
    """
    cdef:
        float[:, :] aryMdlRsp_view = aryMdlRsp
        float[:] vecFunc_view = vecFunc
        int [:, :] aryIdxTrn_view = aryIdxTrn
        int [:, :] aryIdxTst_view = aryIdxTst
        unsigned long varNumMdls, idxMdl
        unsigned int idxVol, idxXval, varNumXval, varNumVolTrn, varNumVolTst
        int[:] vecIdxTrn

    # Number of models in the model array:
    varNumMdls = int(aryMdlRsp.shape[1])
    # Number of cross-validations:
    varNumXval = int(aryIdxTrn.shape[1])
    # Number of training volumes
    varNumVolTrn = int(aryIdxTrn.shape[0])
    # Number of testing volumes
    varNumVolTst = int(aryIdxTst.shape[0])
    

    # Define 2D array for residuals (here crossvalidation error) of least
    # squares solution), initialized with all zeros here:
    cdef np.ndarray[np.float32_t, ndim=2] aryResXval = np.zeros((varNumMdls,
                                                                 varNumXval),
                                                                dtype=np.float32)

    # Memory view on array for residuals (here crossvalidation error)
    cdef float[:, :] aryResXval_view = aryResXval

    # Call optimised cdef function for calculation of residuals:
    aryResXval_view = func_cy_res_xval(aryMdlRsp_view,
                                       vecFunc_view,
                                       aryIdxTrn_view,
                                       aryIdxTst_view,
                                       aryResXval_view,
                                       varNumXval,
                                       varNumMdls,
                                       varNumVolTrn,
                                       varNumVolTst)

    # Convert memory view to numpy array before returning it:
    aryResXval = np.asarray(aryResXval_view)

    return aryResXval

# *****************************************************************************

# *****************************************************************************
# *** Function fast calculation residuals, with cross-validation, 1 predictor

cdef float[:, :] func_cy_res_xval(float[:, :] aryMdlRsp_view,
                                  float[:] vecFunc_view,
                                  int[:, :] aryIdxTrn_view,
                                  int[:, :] aryIdxTst_view,
                                  float[:, :] aryResXval_view,
                                  unsigned int varNumXval,
                                  unsigned long varNumMdls,
                                  unsigned int varNumVolTrn,
                                  unsigned int varNumVolTst):

    cdef:
        float varVarY, varCovXy, varRes, varSlope, varXhat
        unsigned int idxVol, idxXval, idxItr
        unsigned long idxMdl

    # Loop through cross-validations
    for idxXval in range(varNumXval):

        # Loop through models:
        for idxMdl in range(varNumMdls):

            # Covariance and residuals of current voxel:
            varVarY = 0
            varCovXy = 0
            varRes = 0

            # Loop through trainings volumes and calculate covariance between
            # the training model and the current voxel:
            for idxItr in range(varNumVolTrn):
                # get the training volume
                idxVol = aryIdxTrn_view[idxItr, idxXval]
                # Calculate the variance of the training model time courses
                # for this fold
                varVarY += aryMdlRsp_view[idxVol, idxMdl] ** 2
                # calculate covariance between model and voxel time course
                varCovXy += (vecFunc_view[idxVol]
                             * aryMdlRsp_view[idxVol, idxMdl])

            # Obtain the slope of the regression of the model on the data:
            varSlope = varCovXy / varVarY

            # Loop through test volumes and calculate the predicted time course
            # value and the mismatch between prediction and actual voxel value
            for idxItr in range(varNumVolTst):
                # get the test volume
                idxVol = aryIdxTst_view[idxItr, idxXval]
                # The predicted voxel time course value:
                varXhat = aryMdlRsp_view[idxVol, idxMdl] * varSlope
                # Mismatch between prediction and actual voxel value (variance):
                varRes += (vecFunc_view[idxVol] - varXhat) ** 2

            aryResXval_view[idxMdl, idxXval] = varRes

    # Return memory view
    return aryResXval_view

# *****************************************************************************
