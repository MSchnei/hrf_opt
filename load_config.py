# -*- coding: utf-8 -*-
"""Load py_pRF_mapping config file."""

import os
import csv
import ast

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def load_config(strCsvCnfg, lgcTest=False, lgcPrint=True):
    """
    Load py_pRF_mapping config file.

    Parameters
    ----------
    strCsvCnfg : string
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of this function
        will be prepended to config file paths.
    lgcPrint : Boolean
        Print config parameters?

    Returns
    -------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.

    """

    # Dictionary with config information:
    dicCnfg = {}

    # Open file with parameter configuration:
    # fleConfig = open(strCsvCnfg, 'r')
    with open(strCsvCnfg, 'r') as fleConfig:

        # Read file  with ROI information:
        csvIn = csv.reader(fleConfig,
                           delimiter='\n',
                           skipinitialspace=True)

        # Loop through csv object to fill list with csv data:
        for lstTmp in csvIn:

            # Skip comments (i.e. lines starting with '#') and empty lines.
            # Note: Indexing the list (i.e. lstTmp[0][0]) does not work for
            # empty lines. However, if the first condition is no fullfilled
            # (i.e. line is empty and 'if lstTmp' evaluates to false), the
            # second logical test (after the 'and') is not actually carried
            # out.
            if lstTmp and not (lstTmp[0][0] == '#'):

                # Name of current parameter (e.g. 'varTr'):
                strParamKey = lstTmp[0].split(' = ')[0]
                # print(strParamKey)

                # Current parameter value (e.g. '2.94'):
                strParamVlu = lstTmp[0].split(' = ')[1]
                # print(strParamVlu)

                # Put paramter name (key) and value (item) into dictionary:
                dicCnfg[strParamKey] = strParamVlu

    # Minimum delay for the peak of the hrf function:
    dicCnfg['varPkDelMin'] = float(dicCnfg['varPkDelMin'])
    if lgcPrint:
        print('---Minimum delay for the peak of the hrf function: '
              + str(dicCnfg['varPkDelMin']))

    # Maximum delay for the peak of the hrf function:
    dicCnfg['varPkDelMax'] = float(dicCnfg['varPkDelMax'])
    if lgcPrint:
        print('---Maximum delay for the peak of the hrf function: '
              + str(dicCnfg['varPkDelMax']))

    # Minimum delay for the undershoot of the hrf function:
    dicCnfg['varUndDelMin'] = float(dicCnfg['varUndDelMin'])
    if lgcPrint:
        print('---Minimum delay for the undershoot of the hrf function: '
              + str(dicCnfg['varUndDelMin']))

    # Maximum delay for the undershoot of the hrf function:
    dicCnfg['varUndDelMax'] = float(dicCnfg['varUndDelMax'])
    if lgcPrint:
        print('---Maximum delay for the undershoot of the hrf function: '
              + str(dicCnfg['varUndDelMax']))

    # Step size for peak and undershoot delay of the hrf function:
    dicCnfg['varDelStp'] = float(dicCnfg['varDelStp'])
    if lgcPrint:
        print('---Step size for peak & undershoot delay of the hrf function: '
              + str(dicCnfg['varDelStp']))

    # Minimum dispersion for the peak of the hrf function:
    dicCnfg['varPkDspMin'] = float(dicCnfg['varPkDspMin'])
    if lgcPrint:
        print('---Minimum dispersion for the peak of the hrf function: '
              + str(dicCnfg['varPkDspMin']))

    # Maximum dispersion for the peak of the hrf function:
    dicCnfg['varPkDspMax'] = float(dicCnfg['varPkDspMax'])
    if lgcPrint:
        print('---Maximum dispersion for the peak of the hrf function: '
              + str(dicCnfg['varPkDspMax']))

    # Minimum dispersion for the undershoot of the hrf function:
    dicCnfg['varUndDspMin'] = float(dicCnfg['varUndDspMin'])
    if lgcPrint:
        print('---Minimum dispersion for the undershoot of the hrf function: '
              + str(dicCnfg['varUndDspMin']))

    # Maximum dispersion for the undershoot of the hrf function:
    dicCnfg['varUndDspMax'] = float(dicCnfg['varUndDspMax'])
    if lgcPrint:
        print('---Maximum dispersion for the undershoot of the hrf function: '
              + str(dicCnfg['varUndDspMax']))

    # Step size for peak and undershoot dispersion of the hrf function:
    dicCnfg['varDspStp'] = float(dicCnfg['varDspStp'])
    if lgcPrint:
        print('---Step size for dispersion of the hrf function: '
              + str(dicCnfg['varDspStp']))

    # Minimum peak to undershoot ratio:
    dicCnfg['varPkUndRatMin'] = float(dicCnfg['varPkUndRatMin'])
    if lgcPrint:
        print('---Minimum peak to undershoot ratio: '
              + str(dicCnfg['varPkUndRatMin']))

    # Maximum peak to undershoot ratio:
    dicCnfg['varPkUndRatMax'] = float(dicCnfg['varPkUndRatMax'])
    if lgcPrint:
        print('---Maximum peak to undershoot ratio: '
              + str(dicCnfg['varPkUndRatMax']))

    # Step size for the peak to undershoot ratio
    dicCnfg['varPkUndRatStp'] = float(dicCnfg['varPkUndRatStp'])
    if lgcPrint:
        print('---Step size for the peak to undershoot ratio: '
              + str(dicCnfg['varPkUndRatStp']))

    # Factor by which time courses and HRF will be upsampled for the
    # convolutions
    dicCnfg['varTmpOvsmpl'] = ast.literal_eval(dicCnfg['varTmpOvsmpl'])
    if lgcPrint:
        print('---Factor by which time courses and HRF will be upsampled: '
              + str(dicCnfg['varTmpOvsmpl']))

    # Extent of temporal smoothing for fMRI data and pRF time course models
    # [standard deviation of the Gaussian kernel, in seconds]:
    # same temporal smoothing will be applied to pRF model time courses
    dicCnfg['varSdSmthTmp'] = float(dicCnfg['varSdSmthTmp'])
    if lgcPrint:
        print('---Extent of temporal smoothing (Gaussian SD in [s]): '
              + str(dicCnfg['varSdSmthTmp']))

    # Volume TR of input data [s]:
    dicCnfg['varTr'] = float(dicCnfg['varTr'])
    if lgcPrint:
        print('---Volume TR of input data [s]: ' + str(dicCnfg['varTr']))

    # Number of fMRI volumes and png files to load:
    dicCnfg['varNumVol'] = int(dicCnfg['varNumVol'])
    if lgcPrint:
        print('---Total number of fMRI volumes and png files: '
              + str(dicCnfg['varNumVol']))

    # Path to nifti file voxel time courses were saved to:
    dicCnfg['strPathFuncIn'] = ast.literal_eval(dicCnfg['strPathFuncIn'])
    if lgcPrint:
        print('---Path to nifti file voxel time courses were saved to: ')
        print('   ' + str(dicCnfg['strPathFuncIn']))

    # Path to nifti file unconvolved model time courses were saved to:
    dicCnfg['strPathMdlIn'] = ast.literal_eval(dicCnfg['strPathMdlIn'])
    if lgcPrint:
        print('---Path to file unconvolved model time courses were saved to: ')
        print('   ' + str(dicCnfg['strPathMdlIn']))

    # Were the model time courses created with suppressive surround?
    dicCnfg['lgcSupsur'] = (dicCnfg['lgcSupsur'] == 'True')
    if lgcPrint:
        print('---Model time courses were created with suppressive surround: '
              + str(dicCnfg['lgcSupsur']))

    # Path to nifti file with R2 values of pRF fitting:
    dicCnfg['strPathR2'] = ast.literal_eval(dicCnfg['strPathR2'])
    if lgcPrint:
        print('---Path to nifti file with R2 values of pRF fitting: ')
        print('   ' + str(dicCnfg['strPathR2']))

    # # R2 threshold that will be applied for voxel inclusion:
    dicCnfg['varThrR2'] = int(dicCnfg['varThrR2'])
    if lgcPrint:
        print('---R2 threshold that will applied be for voxel inclusion: '
              + str(dicCnfg['varThrR2']))

    # Number of processes to run in parallel:
    dicCnfg['varPar'] = int(dicCnfg['varPar'])
    if lgcPrint:
        print('---Number of processes to run in parallel: '
              + str(dicCnfg['varPar']))

    # should model fitting be based on k-fold cross-validation?
    # if not, set to 1
    dicCnfg['varNumXval'] = ast.literal_eval(dicCnfg['varNumXval'])
    if lgcPrint:
        print('---Model fitting will have this number of folds for xval: '
              + str(dicCnfg['varNumXval']))

    # Output basename:
    dicCnfg['strPathOut'] = ast.literal_eval(dicCnfg['strPathOut'])
    if lgcPrint:
        print('---Output basename:')
        print('   ' + str(dicCnfg['strPathOut']))

    # Is this a test?
    if lgcTest:

        # Prepend absolute path of this file to config file paths:
        dicCnfg['strPathFuncIn'] = (strDir + dicCnfg['strPathFuncIn'])
        dicCnfg['strPathOut'] = (strDir + dicCnfg['strPathOut'])
        dicCnfg['strPathMdlIn'] = (strDir + dicCnfg['strPathMdlIn'])
        dicCnfg['strPathR2'] = (strDir + dicCnfg['strPathR2'])

    return dicCnfg