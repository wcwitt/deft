# dependencies
armadillo_include=/home/EAC/armadillo/include/
armadillo_lib=/home/EAC/armadillo/lib/
fftw_lib=/usr/local/fftw/gcc/3.3.4/lib64/

# delete object files and library files
if [ -d obj ]; then rm -r obj; fi
mkdir obj
if [ -d lib ]; then rm -r lib; fi
mkdir lib

# compile deft library
g++ -O3 -g -Werror -Wall -fopenmp -c -I$armadillo_include -I./include -fPIC source/deft.cpp -o obj/deft.o 
g++ -O3 -g -Werror -Wall -fopenmp -shared -Wl,-soname,libdeft.so -o lib/libdeft.so obj/deft.o -L$fftw_lib -lfftw3 -lm -L$armadillo_lib -larmadillo
