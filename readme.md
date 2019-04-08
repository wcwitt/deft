# DEFT: data equipped with Fourier transforms

**DEFT** is a C++ library providing Fourier-transform-based tools for data on three-dimensional grids, including

  * various derivatives (gradients, the Laplacian, etc.) computed with Fourier transforms;
  * interpolation and filtering with Fourier transforms;
  * sums of functions duplicated over a lattice.

**DEFT** includes a python wrapper. For example applications, see

  * example: [sums of functions duplicated over a lattice](/python/sum-over-lattice.ipynb);
  * example: [interpolation with Fourier transforms](/python/interpolate.ipynb).
  
**DEFT** (currently) has several dependencies:

  * [Armadillo](http://arma.sourceforge.net/), a C++ library for linear algebra & scientific computing;
  * [FFTW](http://www.fftw.org/), the "Fastest Fourier Transform in the West".

**DEFT** is work in progress.
