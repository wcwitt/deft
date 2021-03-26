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

template<typename F>
Double3D array_from_lattice_sum(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords,
        F function_ft);

double integrate(Double3D data, Box box);

}


// ---------- begin function definitions ---------- //


#ifdef DEFT_HEADER_ONLY
#include "../source/fourier.cpp"
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
    ft.set_elements(
        [&ft, &box, &xyz_coords, &function_ft](size_t i) {
            // get wave vector indices, shifted negative if appropriate
            int u = i / ft.strides()[0];
            int v = (i % ft.strides()[0]) / ft.strides()[1];
            int w = (i % ft.strides()[0]) % ft.strides()[1];
            u = u - (u > ft.shape()[0]/2) * ft.shape()[0]; 
            v = v - (v > ft.shape()[1]/2) * ft.shape()[1]; 
            // get wave vector and wave number
            const double kx = box.wave_vectors_x(u,v,w);
            const double ky = box.wave_vectors_y(u,v,w);
            const double kz = box.wave_vectors_z(u,v,w);
            const double k = box.wave_numbers(u,v,w);
            // compute structure factor for this k-vector
            std::complex<double> str_fact{0.0, 0.0};
            for (size_t a=0; a<xyz_coords.size(); ++a) {
                double k_dot_r = kx*xyz_coords[a][0]
                        + ky*xyz_coords[a][1] + kz*xyz_coords[a][2];
                str_fact += exp(-std::complex<double>{0.0,1.0} * k_dot_r);
            }
            return str_fact * function_ft(k);
        });
    return inverse_fourier_transform(ft, shape) / box.volume();
}

}
