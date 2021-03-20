![CI](https://github.com/wcwitt/deft/workflows/CI/badge.svg)
# DEFT: data equipped with Fourier transforms

**DEFT** is a C++ library providing Fourier-transform-based tools for data on three-dimensional grids, including

  * sums of functions duplicated over a lattice;
  * interpolation and filtering with Fourier transforms;
  * various derivatives (gradients, the Laplacian, etc.) computed with Fourier transforms.

**DEFT** includes a python wrapper. For example applications, see

  * example: [sums of functions duplicated over a lattice](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/examples/sum-over-lattice.ipynb);
  * example: [interpolation with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/examples/interpolate.ipynb);
  * example: [computing derivatives with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/examples/derivatives.ipynb).
  
**DEFT** (currently) has several dependencies:

  * [FFTW](http://www.fftw.org/), the "Fastest Fourier Transform in the West".

**DEFT** is work in progress.

### installation notes

compiling:
```
cmake -H. -Bbuild
cd build
make
cd ../test
python -m unittest
```

#### for intel-mkl

prepare mkl environment:
```
conda create -n deft_mkl python=3 numpy cmake
conda activate deft_mkl
conda install -c intel mkl-devel
```
then set include and lib paths to
<path/to/anaconda>/envs/deft_mkl/include
<path/to/anaconda>/envs/deft_mkl/lib
