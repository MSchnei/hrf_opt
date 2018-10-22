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





varTimeStepSize = 0.2  # 0.2
varDispStepSize = 0.2  # 0.2
varScaleStepSize = 0.1

# define vector for  time to peak, default: 6
varA1min = 2  # 2
varA1max = 11  # 10
vecA1 = np.arange(varA1min, varA1max+varTimeStepSize, varTimeStepSize)
varA1num = len(vecA1)

# define vector fortime to undershoot, default: 16
varA2min = 11  # 6
varA2max = 19  # 18
vecA2 = np.arange(varA2min, varA2max+varTimeStepSize, varTimeStepSize)
varA2num = len(vecA2)

# define vector for peak dispersion, default: 1
varB1min = 0.1  # 0.1
varB1max = 2.5  # 2.5
vecB1 = np.arange(varB1min, varB1max+varDispStepSize, varDispStepSize)
varB1num = len(vecB1)

# define vector for undershoot dispersion, default: 1
varB2min = 0.1  # 0.1
varB2max = 2.5  # 2.5
vecB2 = np.arange(varB2min, varB2max+varDispStepSize, varDispStepSize)
varB2num = len(vecB2)

# define vector for weighting of undershoot relative to peak,
# e.g. 6 means 1:6 weighting
varCmin = 0.000001  # 1
vecCmax = 2  # 10
# divide by 10 here to invert effect (go from 1/10, 2/10, ..., to 10/10)
vecC = np.arange(varCmin, vecCmax+varScaleStepSize, varScaleStepSize)
varCnum = len(vecC)

# find combinations of all parameters
# exclude combinations where a2 < a1
iterables = [vecA1, vecA2]
vecA12 = list(itertools.product(*iterables))
vecA12 = np.asarray(vecA12)
# pass only the combinations where a1 < a2
lgcA12combi = np.less(vecA12[:, 0], vecA12[:, 1])
vecA12 = vecA12[lgcA12combi]
# define number of combinations for a1 and a2 positions
varA12num = len(vecA12)

# %% create hrf model parameters
# get number of all possible parameter combinations
varAllCombis = varA12num * varB1num * varB1num * varCnum

iterables = [vecA12, vecB1, vecB2, vecC]
aryHrfParams = list(itertools.product(*iterables))
for ind, item in enumerate(aryHrfParams):
    aryHrfParams[ind] = list(np.hstack(item))
aryHrfParams = np.array(aryHrfParams)