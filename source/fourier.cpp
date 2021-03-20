#include "fourier.hpp"

#ifdef DEFT_POCKETFFT
#include "pocketfft_hdronly.h"
#else // fftw or mkl
#include "fftw3.h"
#endif

#ifdef DEFT_HEADER_ONLY
#define DEFT_INLINE inline
#else
#define DEFT_INLINE
#endif

namespace deft {

DEFT_INLINE
Complex3D fourier_transform(Double3D data)
{
    Complex3D ft({data.shape()[0], data.shape()[1], data.shape()[2]/2+1});
#ifdef DEFT_POCKETFFT
    // set shape, stride_r, stride_c, axes
    pocketfft::detail::shape_t shape_r =
            {data.shape()[0], data.shape()[1], data.shape()[2]};
    pocketfft::detail::stride_t stride_r(3);
    pocketfft::detail::stride_t stride_c(3);
    for (size_t i=0; i<3; ++i) {
        stride_r[i] = data.strides()[i] * sizeof(double);
        stride_c[i] = ft.strides()[i] * sizeof(std::complex<double>);
    }
    pocketfft::detail::shape_t axes{0, 1, 2};    
    // r2c transform
    pocketfft::detail::r2c(
        shape_r, stride_r, stride_c, axes, true,
        data.data(), ft.data(), 1.0);
#else // fftw or mkl
    // note: re-planning for same array shape should be fast
    // (see fftw manual section 4.2), 
    fftw_plan plan_r2c = fftw_plan_dft_r2c_3d(
        data.shape()[0], data.shape()[1], data.shape()[2], 
        data.data(), reinterpret_cast<fftw_complex*>(ft.data()),
        FFTW_MEASURE | FFTW_DESTROY_INPUT);
    fftw_execute(plan_r2c);
    fftw_destroy_plan(plan_r2c);
#endif
    ft /= data.size(); // normalization
    return ft;
}

DEFT_INLINE
Double3D inverse_fourier_transform(Complex3D ft, std::array<size_t,3> shape)
{
    Double3D data(shape);
#if DEFT_POCKETFFT
    // set shape, stride_r, stride_c, axes
    pocketfft::detail::shape_t shape_r =
            {data.shape()[0], data.shape()[1], data.shape()[2]};
    pocketfft::detail::stride_t stride_r(3);
    pocketfft::detail::stride_t stride_c(3);
    for (size_t i=0; i<3; ++i) {
        stride_r[i] = data.strides()[i] * sizeof(double);
        stride_c[i] = ft.strides()[i] * sizeof(std::complex<double>);
    }
    pocketfft::detail::shape_t axes{0, 1, 2};    
    // r2c transform
    pocketfft::detail::c2r(
        shape_r, stride_c, stride_r, axes, false,
        ft.data(), data.data(), 1.0, 1);
#else // fftw or mkl
    // note: re-planning for same array shape should be fast
    // (see fftw manual section 4.2), 
    fftw_plan plan_c2r = fftw_plan_dft_c2r_3d(
        data.shape()[0], data.shape()[1], data.shape()[2],
        reinterpret_cast<fftw_complex*>(ft.data()), data.data(),
        FFTW_MEASURE | FFTW_DESTROY_INPUT);
    fftw_execute(plan_c2r);
    fftw_destroy_plan(plan_c2r);
#endif
    return data;
}

DEFT_INLINE
Double3D wave_vectors_x(std::array<size_t,3> ft_shape, Box box)
{
    Double3D wave_vectors_x(ft_shape);
    wave_vectors_x.set_elements(
        [&wave_vectors_x, &ft_shape, &box](size_t i) {
            auto uvw = wave_vectors_x.unravel_index(i);
            return box.wave_vectors_x(
                    uvw[0] - (uvw[0] > ft_shape[0]/2) * ft_shape[0],
                    uvw[1] - (uvw[1] > ft_shape[1]/2) * ft_shape[1],
                    uvw[2]);
        });
    return wave_vectors_x;
}

DEFT_INLINE
Double3D wave_vectors_y(std::array<size_t,3> ft_shape, Box box)
{
    Double3D wave_vectors_y(ft_shape);
    wave_vectors_y.set_elements(
        [&wave_vectors_y, &ft_shape, &box](size_t i) {
            auto uvw = wave_vectors_y.unravel_index(i);
            return box.wave_vectors_y(
                    uvw[0] - (uvw[0] > ft_shape[0]/2) * ft_shape[0],
                    uvw[1] - (uvw[1] > ft_shape[1]/2) * ft_shape[1],
                    uvw[2]);
        });
    return wave_vectors_y;
}

DEFT_INLINE
Double3D wave_vectors_z(std::array<size_t,3> ft_shape, Box box)
{
    Double3D wave_vectors_z(ft_shape);
    wave_vectors_z.set_elements(
        [&wave_vectors_z, &ft_shape, &box](size_t i) {
            auto uvw = wave_vectors_z.unravel_index(i);
            return box.wave_vectors_z(
                    uvw[0] - (uvw[0] > ft_shape[0]/2) * ft_shape[0],
                    uvw[1] - (uvw[1] > ft_shape[1]/2) * ft_shape[1],
                    uvw[2]);
        });
    return wave_vectors_z;
}

DEFT_INLINE
Double3D wave_numbers(std::array<size_t,3> ft_shape, Box box)
{
    Double3D wave_numbers(ft_shape);
    wave_numbers.set_elements(
        [&wave_numbers, &ft_shape, &box](size_t i) {
            auto uvw = wave_numbers.unravel_index(i);
            return box.wave_numbers(
                    uvw[0] - (uvw[0] > ft_shape[0]/2) * ft_shape[0],
                    uvw[1] - (uvw[1] > ft_shape[1]/2) * ft_shape[1],
                    uvw[2]);
        });
    return wave_numbers;
}

DEFT_INLINE
Double3D gradient_x(Double3D data, Box box)
{
    return inverse_fourier_transform(
            gradient_x(fourier_transform(data),box), data.shape());
}

DEFT_INLINE
Complex3D gradient_x(Complex3D ft, Box box)
{
    Double3D wvx = wave_vectors_x(ft.shape(), box);
    return ft.set_elements(
        [&ft, &wvx](size_t i) {
            return std::complex<double>{0.0,1.0} * wvx(i) * ft(i);
        });
}

DEFT_INLINE
Double3D gradient_y(Double3D data, Box box)
{
    return inverse_fourier_transform(
            gradient_y(fourier_transform(data),box), data.shape());
}

DEFT_INLINE
Complex3D gradient_y(Complex3D ft, Box box)
{
    Double3D wvy = wave_vectors_y(ft.shape(), box);
    return ft.set_elements(
        [&ft, &wvy](size_t i) {
            return std::complex<double>{0.0,1.0} * wvy(i) * ft(i);
        });
}

DEFT_INLINE
Double3D gradient_z(Double3D data, Box box)
{
    return inverse_fourier_transform(
            gradient_z(fourier_transform(data),box), data.shape());
}

DEFT_INLINE
Complex3D gradient_z(Complex3D ft, Box box)
{
    Double3D wvz = wave_vectors_z(ft.shape(), box);
    return ft.set_elements(
        [&ft, &wvz](size_t i) {
            return std::complex<double>{0.0,1.0} * wvz(i) * ft(i);
        });
}

DEFT_INLINE
Double3D grad_dot_grad(Double3D data, Box box)
{
    Complex3D ft = fourier_transform(data);
    // compute g_x and store g_x*g_x in grad_dot_grad
    Double3D grad = inverse_fourier_transform(
            gradient_x(ft,box), data.shape());
    Double3D grad_dot_grad = grad*grad;
    // compute g_y and add g_y*g_y to grad_dot_grad
    grad = inverse_fourier_transform(gradient_y(ft,box), data.shape());
    grad_dot_grad += grad*grad;
    // compute g_z and add g_z*g_z to grad_dot_grad
    grad = inverse_fourier_transform(gradient_z(ft,box), data.shape());
    return grad_dot_grad += grad*grad;
}

DEFT_INLINE
Double3D laplacian(Double3D data, Box box)
{
    return inverse_fourier_transform(
            laplacian(fourier_transform(data),box), data.shape());
}

DEFT_INLINE
Complex3D laplacian(Complex3D ft, Box box)
{
    Double3D wn = wave_numbers(ft.shape(), box);
    ft.set_elements(
        [&ft, &wn](size_t i) {
            return ft(i) * -wn(i)*wn(i);
        });
    return ft;
}

DEFT_INLINE
double integrate(Double3D data, Box box)
{
    return data.sum() * box.volume() / data.size();
}

}
