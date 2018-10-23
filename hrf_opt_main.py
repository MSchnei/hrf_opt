# -*- coding: utf-8 -*-
"""Optimize the hrf parameters, given estimated pRF parameters."""

# Part of hrf_opt library
# Copyright (C) 2018  Marian Schneider
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


import time
import numpy as np
import multiprocessing as mp

#from hrf_opt.load_config import load_config
#from hrf_opt.hrf_opt_utils import cls_set_config
from load_config import load_config
from hrf_opt_utils import cls_set_config

###### DEBUGGING ###############
#strCsvCnfg = "/home/marian/Documents/Git/hrf_opt/config_test.csv"
#lgcTest = False
################################


def pyprf(strCsvCnfg, lgcTest=False):
    """
    Main function for hrf_opt.

    Parameters
    ----------
    strCsvCnfg : str
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of pyprf libary
        will be prepended to config file paths.

    """

    # %% Preparations

    # Check time
    print('---pRF analysis')
    varTme01 = time.time()

    # Load config parameters from csv file into dictionary:
    dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest)

    # Load config parameters from dictionary into namespace:
    cfg = cls_set_config(dicCnfg)

    # If suppressive surround flag is on, make sure to retrieve results from
    # that fitting
    if cfg.lgcSupsur is not None:
        cfg.strPathOut = cfg.strPathOut + '_supsur'

    # Convert preprocessing parameters (for temporal smoothing)
    # from SI units (i.e. [s]) into units of data array (volumes):
    cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)

    # %% Load previous pRF fitting results

    # Derive paths to the x, y, sigma winner parameters from pyprf_feature
    lstWnrPrm = [cfg.strPathOut + '_x_pos.nii.gz',
                 cfg.strPathOut + '_y_pos.nii.gz',
                 cfg.strPathOut + '_SD.nii.gz',
                 cfg.strPathOut + '_R2.nii.gz']

    # Check if fitting has been performed, i.e. whether parameter files exist
    # Throw error message if they do not exist.
    errorMsg = 'Files that should have resulted from fitting do not exist. \
                \nPlease perform pRF fitting first, calling  e.g.: \
                \npyprf_feature -config /path/to/my_config_file.csv'
    assert os.path.isfile(lstWnrPrm[0]), errorMsg
    assert os.path.isfile(lstWnrPrm[1]), errorMsg
    assert os.path.isfile(lstWnrPrm[2]), errorMsg
    assert os.path.isfile(lstWnrPrm[3]), errorMsg

    # Load the x, y, sigma winner parameters from pyprf_feature
    aryIntGssPrm = load_res_prm(lstWnrPrm,
                                lstFlsMsk=[cfg.strPathNiiMask])[0][0]

    # Get corresponding model parameters
    aryMdlParams = np.load(cfg.strPathMdlRsp + '_params.npy')
    # Get corresponding model responses
    aryMdlRsp = np.load(cfg.strPathMdlRsp + '_mdlRsp.npy')

    
    

    # %% Load empirical as well as fitted, unconvolved time courses

    # Load empirical, voxel time courses:
    cfg.strPathFuncIn

    # Load unconvolved model time courses:
    cfg.strPathMdlIn

    # Load R2 values of pRF fitting
    cfg.strPathR2, cfg.varThrR2



