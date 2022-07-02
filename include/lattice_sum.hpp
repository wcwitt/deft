#pragma once

#include "array.hpp"
#include "box.hpp"
#include "fourier.hpp"

namespace deft {

using Double3D = Array<double>;
using Complex3D = Array<std::complex<double>>;

Complex3D structure_factor(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords);

std::vector<double> cardinal_b_spline_values(double x, int order);
std::vector<double> cardinal_b_spline_derivatives(double x, int order);
std::complex<double> exponential_spline_b(int m, int N, int order);
Complex3D structure_factor_spline(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords,
        int order);

template<typename F>
Double3D array_from_lattice_sum(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords,
        F function_ft,
        int spline_order=-1);
}

#ifdef DEFT_HEADER_ONLY
#include "../source/lattice_sum.cpp"
#endif

namespace deft {

template<typename F>
Double3D array_from_lattice_sum(
    std::array<size_t,3> shape,
    Box box,
    std::vector<std::array<double,3>> xyz_coords,
    F function_ft,
    int spline_order)
{
    Complex3D ft({shape[0], shape[1], shape[2]/2+1});
    Complex3D str_fac(ft.shape());
    if (spline_order<0) {
        str_fac = structure_factor(shape,box,xyz_coords);
    } else {
        str_fac = structure_factor_spline(shape,box,xyz_coords,spline_order);
    }
    Double3D kx = wave_vectors_x(ft.shape(), box);
    Double3D ky = wave_vectors_y(ft.shape(), box);
    Double3D kz = wave_vectors_z(ft.shape(), box);
    ft.set_elements(
        [&str_fac, &kx, &ky, &kz, &xyz_coords, &function_ft](size_t i) {
            return str_fac(i) * function_ft(kx(i),ky(i),kz(i));
        });
    return inverse_fourier_transform(ft, shape) / box.volume();
}

}
