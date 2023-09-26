# SPECT PSF-Fitting
This repository is essentially used to generate one thing: a continuous function `f(x,y,d)` corresponding to the point spread function of a given SPECT system. For low/medium energy photons, `f` is usually an isotropic Gaussian function with width linearly dependent on `d`. For higher energy photons, `f` is no longer isotropic, so modeling becomes more difficult. This repository borrows theory developed in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3619230/) to obtain the function `f(x,y,d)` given a sequence of SIMIND point simulations at various radial distances. For more information, see `analysis.ipynb`. 

* Note: this is *not* my paper, nor my method. All credit goes to Se Young Chun, Jeffrey A. Fessler, and Yuni K. Dewaraja for developing and publishing this technique. Also note: I do not exactly follow their implementation. 

