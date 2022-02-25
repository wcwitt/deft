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
        F function_ft);
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
    F function_ft)
{
    Complex3D ft({shape[0], shape[1], shape[2]/2+1});
    Double3D kx = wave_vectors_x(ft.shape(), box);
    Double3D ky = wave_vectors_y(ft.shape(), box);
    Double3D kz = wave_vectors_z(ft.shape(), box);
    ft.set_elements(
        [&kx, &ky, &kz, &xyz_coords, &function_ft](size_t i) {
            std::complex<double> struct_fact{0.0,0.0};
            for (size_t a=0; a<xyz_coords.size(); ++a) {
                double k_dot_r =   kx(i)*xyz_coords[a][0]
                                 + ky(i)*xyz_coords[a][1]
                                 + kz(i)*xyz_coords[a][2];
                struct_fact += exp(-std::complex<double>{0.0,1.0} * k_dot_r);
            }
            return struct_fact * function_ft(kx(i),ky(i),kz(i));
        });
    return inverse_fourier_transform(ft, shape) / box.volume();
}

}
