#include "lattice_sum.hpp"

#ifdef DEFT_HEADER_ONLY
#define DEFT_INLINE inline
#else
#define DEFT_INLINE
#endif

namespace deft {

DEFT_INLINE
Complex3D structure_factor(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords)
{
    std::array<size_t,3> ft_shape = {shape[0], shape[1], shape[2]/2+1};
    Complex3D structure_factor(ft_shape);
    Double3D kx = wave_vectors_x(ft_shape, box);
    Double3D ky = wave_vectors_y(ft_shape, box);
    Double3D kz = wave_vectors_z(ft_shape, box);
    structure_factor.set_elements(
        [&kx, &ky, &kz, &xyz_coords](size_t i) {
            std::complex<double> str_fac{0.0,0.0};
            for (size_t a=0; a<xyz_coords.size(); ++a) {
                double k_dot_r =   kx(i)*xyz_coords[a][0]
                                 + ky(i)*xyz_coords[a][1]
                                 + kz(i)*xyz_coords[a][2];
                str_fac += exp(-std::complex<double>{0.0,1.0} * k_dot_r);
            }
            return str_fac;
        });
    return structure_factor;
}

DEFT_INLINE
std::vector<double> cardinal_b_spline_values(double x, int order) {
/*
    With n=order, returns [M_n(x+i) for i=0,1,...,n-1]
    Requires x=[0,1) and order>1
    By definition, M_n is zero for input<0 or input>n
    The basic formula is
        M_n[i] = (x+i)/(n-1)*M_{n-1}[i] + (n-x-i)/(n-1)*M_{n-1}[i-1]
*/
    if (x<0.0 or x>=1.0)
        throw std::runtime_error("cardinal_b_spline_values: invalid x");
    if (order<2)
        throw std::runtime_error("cardinal_b_spline_values: invalid order");
    auto M = std::vector<double>(order, 0.0);
    M[0] = x;      // M2(x)   = x
    M[1] = 1.0-x;  // M2(x+1) = 2-(x+1)
    for (int n=3; n<=order; ++n) {
        for (int i=n-1; i>0; --i) {
            M[i] = ((x+i)*M[i] + (n-x-i)*M[i-1]) / (n-1);
        }
        M[0] = x/(n-1)*M[0];
    }
    return M;
}

DEFT_INLINE
std::vector<double> cardinal_b_spline_derivatives(double x, int order) {
/*
    With n=order, returns [M_n'(x+i) for i=0,1,...,n-1]
    Requires x=[0,1) and order>2
    By definition, M_n' is zero for input<0 or input>n
    The basic formula is
        M_n'[i] = M_{n-1}[i] - M_{n-1}[i-1]
*/
    if (x<0.0 or x>=1.0)
        throw std::runtime_error("cardinal_b_spline_derivatives: invalid x");
    if (order<3)
        throw std::runtime_error("cardinal_b_spline_derivatives: invalid order");
    auto Mp = std::vector<double>(order, 0.0);
    auto M = cardinal_b_spline_values(x, order-1);
    Mp[0] = M[0];
    for (int i=1; i<order-1; i++) {
        Mp[i] = M[i] - M[i-1];
    }
    Mp[order-1] = -M[order-2];
    return Mp;
}

DEFT_INLINE
std::complex<double> exponential_spline_b(int m, int N, int order) {
    auto M = cardinal_b_spline_values(0,order);
    auto b = std::complex<double>(0,0);
    for (int i=0; i<order; ++i) {
        b += M[i]*exp(std::complex<double>(0,1)*(2*M_PI*m*double(i-1)/N));
    }
    return exp(std::complex<double>(0,1)*(2*M_PI*m*double(order-1)/N)) / b;
}

DEFT_INLINE
Complex3D structure_factor_spline(
        std::array<size_t,3> shape,
        Box box,
        std::vector<std::array<double,3>> xyz_coords,
        int order)
{
/*
    # Assembles the structure factor approximately using B-splines

    In one dimension for simplicity, the basic construction is
    $$
    \begin{align}
    \sum_{a=1}^{N_{atoms}} & \exp(i k_m x_a) = \sum_{a=1}^{N_{atoms}} \exp(i 2\pi (m/N) u_a) \\
        &= N b(m) \underbrace{\frac{1}{N} \sum_{l=0}^{N-1} \exp(-i 2\pi m l/N)}_{\text{DFT}}
                \underbrace{\sum_{a=1}^{N_{atoms}} \sum_c M_n(u_a + l + cN)}_{Q(l)}
    \end{align}
    $$
    The nonzero spline values are provided in an array
    $$
    M_i = M_n(x+i)  \qquad 0\le x \lt 1 \qquad i= 0, 1, ..., n-1
    $$
    which means the relevant combinations are
    $$
    M_i \qquad \text{with} \qquad l = \mathrm{wrap} \left[ i - \mathrm{floor}(u_a) \right].
    $$
    When the structure factor is instead defined with a negative exponent, one should take complex conjugate at the very end.

    See the following references for full details:

    * Essmann et al., J. Chem. Phys. 103, 8577 (1995)
    * Choly and Kaxiras, Phys. Rev. B 67, 155101 (2003)
    * Hung and Carter, Chem. Phys. Lett. 475, 163 (2009)
*/
    // ----- move this eventually -----
    // get inverse of box vector matrix
    std::array<std::array<double,3>,3> ainv;
    auto b = box.recip_vectors();
    for (size_t i=0; i<3; ++i) {
        for (size_t j=0; j<3; ++j) {
            // no transpose because b = [b1 b2 b3]^T
            ainv[i][j] = b[i][j] / (2.0*M_PI);
        }
    }
    // create function for matrix multiplication
    auto m_dot_v = [](std::array<std::array<double,3>,3> m,
                      std::array<double,3> v) {
        std::array<double,3> r;
        r[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2];
        r[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2];
        r[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2];
        return r;
    };
    // compute fractional coordinates
    std::vector<std::array<double,3>> frac_coords(xyz_coords.size());
    for (size_t i=0; i<frac_coords.size(); ++i) {
        frac_coords[i] = m_dot_v(ainv, xyz_coords[i]);
        //for (size_t j=0; j<3; ++j)
        //    frac_coords[i][j] -= std::floor(frac_coords[i][j]);
    }

    int N0 = shape[0];
    int N1 = shape[1];
    int N2 = shape[2];

    Double3D Q(shape);
    Q.set_elements(0.0);
    for (size_t i=0; i<frac_coords.size(); ++i) {
        double u0 = frac_coords[i][0]*N0;
        double u1 = frac_coords[i][1]*N1;
        double u2 = frac_coords[i][2]*N2;
        int floor0 = static_cast<int>(u0);
        int floor1 = static_cast<int>(u1);
        int floor2 = static_cast<int>(u2);
        auto M0 = cardinal_b_spline_values(u0-floor0, order);
        auto M1 = cardinal_b_spline_values(u1-floor1, order);
        auto M2 = cardinal_b_spline_values(u2-floor2, order);
        for (int i0=0; i0<order; ++i0) {
            int l0 = (i0-floor0)%N0 + (i0<floor0)*N0;
            for (int i1=0; i1<order; ++i1) {
                int l1 = (i1-floor1)%N1 + (i1<floor1)*N1;
                for (int i2=0; i2<order; ++i2) {
                    int l2 = (i2-floor2)%N2 + (i2<floor2)*N2;
                    Q(l0,l1,l2) += M0[i0]*M1[i1]*M2[i2];
                }
            }
        }
    }

    Complex3D S = fourier_transform(Q);
    for (int n0=0; n0<S.shape()[0]; ++n0) {
        std::complex<double> b0 = exponential_spline_b(n0,N0,order);
        for (int n1=0; n1<S.shape()[1]; ++n1) {
            std::complex<double> b1 = exponential_spline_b(n1,N1,order);
            for (int n2=0; n2<S.shape()[2]; ++n2) {
                std::complex<double> b2 = exponential_spline_b(n2,N2,order);
                S(n0,n1,n2) *= (N0*N1*static_cast<double>(N2)*b0*b1*b2);
                S(n0,n1,n2) = std::conj(S(n0,n1,n2));
            }
        }
    }

    return S;
}

}
