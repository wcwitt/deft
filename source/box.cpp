#include "box.hpp"

#ifdef DEFT_HEADER_ONLY
#define DEFT_INLINE inline
#else
#define DEFT_INLINE
#endif

namespace deft {

DEFT_INLINE
Box::Box(std::array<std::array<double,3>,3> vectors)
    : _vectors(vectors)
{
    set(vectors);
}

DEFT_INLINE
std::array<std::array<double,3>,3> Box::vectors() const
{
    return _vectors;
}

DEFT_INLINE
std::array<double,3> Box::lengths() const
{
    return _lengths;
}

DEFT_INLINE std::array<double,3> Box::angles() const
{
    return _angles;
}

DEFT_INLINE
double Box::volume() const
{
    return _volume;
}

DEFT_INLINE
std::array<std::array<double,3>,3> Box::recip_vectors() const
{
    return _recip_vectors;
}

DEFT_INLINE
std::array<double,3> Box::recip_lengths() const
{
    return _recip_lengths;
}

DEFT_INLINE
double Box::wave_vectors_x(int i, int j, int k) const
{
    return    i * _recip_vectors[0][0]
            + j * _recip_vectors[1][0]
            + k * _recip_vectors[2][0];
}

DEFT_INLINE
double Box::wave_vectors_y(int i, int j, int k) const
{
    return    i * _recip_vectors[0][1]
            + j * _recip_vectors[1][1]
            + k * _recip_vectors[2][1];
}

DEFT_INLINE
double Box::wave_vectors_z(int i, int j, int k) const
{
    return    i * _recip_vectors[0][2]
            + j * _recip_vectors[1][2]
            + k * _recip_vectors[2][2];
}

DEFT_INLINE
double Box::wave_numbers(int i, int j, int k) const
{
    const double kx = wave_vectors_x(i,j,k);
    const double ky = wave_vectors_y(i,j,k);
    const double kz = wave_vectors_z(i,j,k);
    return std::sqrt(kx*kx + ky*ky + kz*kz);
}

DEFT_INLINE
Box& Box::set(std::array<std::array<double,3>,3> vectors)
{
    _vectors = vectors;
    // define 'a' as matrix with box vectors in rows
    std::array<std::array<double,3>,3>& a = vectors;
    // set box lengths, angles, and volume
    for (size_t i=0; i<3; ++i) {
        _lengths[i] = std::sqrt(
                a[i][0]*a[i][0] + a[i][1]*a[i][1] + a[i][2]*a[i][2]);
    }
    _angles[0] = std::acos(1.0 / (_lengths[1] * _lengths[2])
            * (a[1][0]*a[2][0] + a[1][1]*a[2][1] + a[1][2]*a[2][2]));
    _angles[1] = std::acos(1.0 / (_lengths[0] * _lengths[2])
            * (a[0][0]*a[2][0] + a[0][1]*a[2][1] + a[0][2]*a[2][2]));
    _angles[2] = std::acos(1.0 / (_lengths[0] * _lengths[1])
            * (a[0][0]*a[1][0] + a[0][1]*a[1][1] + a[0][2]*a[1][2]));
    _volume = _determinant_3_by_3(a);
    // set reciprocal lattice vectors and lengths
    //
    //     [--b0--]                                [--a0--] 
    //     [--b1--] = 2 * pi * transpose( inverse( [--a1--] ) )
    //     [--b2--]                                [--a2--]
    //
    std::array<std::array<double,3>,3> ai = _invert_3_by_3(a);
    for (size_t i=0; i<3; ++i) {
        double bi_dot_bi = 0.0;
        for (size_t j=0; j<3; ++j) {
            const double bij = 2.0 * M_PI * ai[j][i];
            _recip_vectors[i][j] = bij;
            bi_dot_bi += bij * bij;
        }
        _recip_lengths[i] = std::sqrt(bi_dot_bi);
    }
    return *this;
}

DEFT_INLINE
double Box::_determinant_3_by_3(std::array<std::array<double,3>,3> a) const
{
    return   a[0][0] * (a[1][1]*a[2][2] - a[2][1]*a[1][2])
           - a[0][1] * (a[1][0]*a[2][2] - a[2][0]*a[1][2])
           + a[0][2] * (a[1][0]*a[2][1] - a[2][0]*a[1][1]);
}

DEFT_INLINE
std::array<std::array<double,3>,3> Box::_invert_3_by_3(
        std::array<std::array<double,3>,3> a) const
{
    std::array<std::array<double,3>,3> b;
    // compute cofactor matrix
    b[0][0] = a[1][1]*a[2][2] - a[2][1]*a[1][2];
    b[1][0] = a[1][2]*a[2][0] - a[2][2]*a[1][0];
    b[2][0] = a[1][0]*a[2][1] - a[2][0]*a[1][1];
    b[0][1] = a[0][2]*a[2][1] - a[2][2]*a[0][1];
    b[1][1] = a[0][0]*a[2][2] - a[2][0]*a[0][2];
    b[2][1] = a[0][1]*a[2][0] - a[2][1]*a[0][0];
    b[0][2] = a[0][1]*a[1][2] - a[1][1]*a[0][2];
    b[1][2] = a[0][2]*a[1][0] - a[1][2]*a[0][0];
    b[2][2] = a[0][0]*a[1][1] - a[1][0]*a[0][1];
    // compute determinant and divide
    const double det = _determinant_3_by_3(a);
    for (size_t i=0; i<3; ++i) {
        for (size_t j=0; j<3; ++j)
            b[i][j] = b[i][j] / det;
    }
    return b;
}

}
