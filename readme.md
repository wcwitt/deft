![CI](https://github.com/wcwitt/deft/workflows/CI/badge.svg)
# DEFT: data equipped with Fourier transforms

**DEFT** is a C++ library providing Fourier-transform-based tools for data on three-dimensional grids, including

  * sums of functions duplicated over a lattice;
  * interpolation and filtering with Fourier transforms;
  * various derivatives (gradients, the Laplacian, etc.) computed with Fourier transforms.

**DEFT** has three FFT modes

  * [PocketFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/tree/cpp) (installs automatically), used by numpy;
  * [FFTW](http://www.fftw.org/) (requires separate installation), the "Fastest Fourier Transform in the West";
  * [Intel-MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) (requires separate installation), can be the fastest in practice.

**DEFT** includes python bindings constructed with pybind11. For example applications, see

  * example: [sums of functions duplicated over a lattice](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/example/sum-over-lattice.ipynb);
  * example: [interpolation with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/example/interpolate.ipynb);
  * example: [computing derivatives with Fourier transforms](https://nbviewer.jupyter.org/github/wcwitt/deft/blob/master/example/derivatives.ipynb).
  
### installation notes

##### basic build

The default installation uses PocketFFT.
 
```
mkdir build
cd build
cmake ..
make
cd ../test
python -m unittest
```

##### build with FFTW

To use FFTW, first install the library. The following uses conda for this purpose.

```
conda create --name deft-fftw python=3 numpy cmake
conda activate deft-fftw
conda install -c conda-forge fftw
mkdir build
cd build
cmake -DDEFT_FFT_TYPE=FFTW \
      -DFFTW_INCLUDE_DIR=<conda-base>/envs/deft-fftw/include \
      -DFFTW_LIBRARY_DIR=<conda-base>/envs/deft-fftw/lib \
      ..
make
cd ../test
python -m unittest
```

##### build with intel-mkl

To use MKL, first install the library. The following uses conda for this purpose.

```
conda create --name deft-mkl python=3 numpy cmake
conda activate deft-mkl
conda install -c intel mkl-devel
mkdir build
cd build
cmake -DDEFT_FFT_TYPE=MKL \
      -DMKL_INCLUDE_DIR=<conda-base>/envs/deft-mkl/include \
      -DMKL_LIBRARY_DIR=<conda-base>/envs/deft-mkl/lib \
      ..
make
export LD_LIBRARY_PATH=<conda-base>/envs/deft-mkl/lib:$LD_LIBRARY_PATH
cd ../test
python -m unittest
```
