#pragma once

#include "array.hpp"
#include "box.hpp"

namespace deft {

using Double3D = Array<double>;
using Complex3D = Array<std::complex<double>>;

Complex3D fourier_transform(Double3D data);
Double3D inverse_fourier_transform(Complex3D ft, std::array<size_t,3> shape);

Double3D wave_vectors_x(std::array<size_t,3> ft_shape, Box box);
Double3D wave_vectors_y(std::array<size_t,3> ft_shape, Box box);
Double3D wave_vectors_z(std::array<size_t,3> ft_shape, Box box);
Double3D wave_numbers(std::array<size_t,3> ft_shape, Box box);

Double3D gradient_x(Double3D data, Box box);
Complex3D gradient_x(Complex3D ft, Box box);

Double3D gradient_y(Double3D data, Box box);
Complex3D gradient_y(Complex3D ft, Box box);

Double3D gradient_z(Double3D data, Box box);
Complex3D gradient_z(Complex3D data, Box box);

Double3D grad_dot_grad(Double3D data, Box box);

Double3D laplacian(Double3D data, Box box);
Complex3D laplacian(Complex3D ft, Box box);

double integrate(Double3D data, Box box);

Double3D fourier_interpolate(Double3D data, std::array<size_t,3> shape);

}

#ifdef DEFT_HEADER_ONLY
#include "../source/fourier.cpp"
#endif
