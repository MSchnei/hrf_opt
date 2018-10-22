# hrf_opt
Optimize hemodynamic response function parameters.

A free & open source package for finding best-fitting hemodynamic response function (HRF) parameters for fMRI data.
Optimization takes place within the framework of population receptive field (pRF) parameters.

The fitting process requires, for every voxel of fMRI data, optimized pRF parameters.
These can be obtained using [pyprf_feature](https://github.com/MSchnei/pyprf_feature).

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

## References
This application is based on the following work:

* Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. NeuroImage, 39(2), 647–660. https://doi.org/10.1016/j.neuroimage.2007.09.034

* Harvey, B. M., & Dumoulin, S. O. (2011). The Relationship between Cortical Magnification Factor and Population Receptive Field Size in Human Visual Cortex: Constancies in Cortical Architecture. Journal of Neuroscience, 31(38), 13604–13612. https://doi.org/10.1523/JNEUROSCI.2572-11.2011

## License
The project is licensed under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).
