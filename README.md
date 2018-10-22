# hrf_opt
Optimize hemodynamic response function parameters.

A free & open source package for finding best-fitting hemodynamic response function (HRF) parameters for fMRI data.
HRF parameters are optimized. Optimization takes place within the framework of population receptive field (pRF) parameters.

The fitting process requires, for every voxel of fMRI data, optimized pRF parameters. These ca be obtained using [pyprf_feature](https://github.com/MSchnei/pyprf_feature).

## Installation

For installation, follow these steps:

0. (Optional) Create conda environment
```bash
conda create -n env_hrf_opt python=2.7
source activate env_hrf_opt
conda install pip
```

1. Clone repository
```bash
git clone https://github.com/MSchnei/hrf_opt.git
```

2. Install numpy, e.g. by running:
```bash
pip install numpy
```

3. Install pyprf_feature with pip
```bash
pip install /path/to/cloned/hrf_opt
```
