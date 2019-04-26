# DEFT: data equipped with Fourier transforms

**DEFT** is a C++ library providing Fourier-transform-based tools for data on three-dimensional grids, including

  * sums of functions duplicated over a lattice;
  * interpolation and filtering with Fourier transforms;
  * various derivatives (gradients, the Laplacian, etc.) computed with Fourier transforms.

**DEFT** includes a python wrapper. For example applications, see

  * example: [sums of functions duplicated over a lattice](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/python/sum-over-lattice.ipynb);
  * example: [interpolation with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/python/interpolate.ipynb);
  * example: [computing derivatives with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/python/derivatives.ipynb).
  
**DEFT** (currently) has several dependencies:

  * [Armadillo](http://arma.sourceforge.net/), a C++ library for linear algebra & scientific computing;
  * [FFTW](http://www.fftw.org/), the "Fastest Fourier Transform in the West".

**DEFT** is work in progress.
