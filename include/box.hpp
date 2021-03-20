#pragma once

#include <array>
#include <cmath>

namespace deft {

class Box {

public:

    Box(std::array<std::array<double,3>,3> vectors =
            {{{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}}});

    std::array<std::array<double,3>,3> vectors() const;
    std::array<double,3> lengths() const;
    std::array<double,3> angles() const;
    double volume() const;
    std::array<std::array<double,3>,3> recip_vectors() const;
    std::array<double,3> recip_lengths() const;

    double wave_numbers(int i, int j, int k) const;
    double wave_vectors_x(int i, int j, int k) const;
    double wave_vectors_y(int i, int j, int k) const;
    double wave_vectors_z(int i, int j, int k) const;

    Box& set(std::array<std::array<double,3>,3> vectors);

private:

    std::array<std::array<double,3>,3> _vectors;
    std::array<double,3> _lengths;
    std::array<double,3> _angles;
    double _volume;
    std::array<std::array<double,3>,3> _recip_vectors;
    std::array<double,3> _recip_lengths;

    // helper functions for 3x3 matrices
    double _determinant_3_by_3(std::array<std::array<double,3>,3> a) const;
    std::array<std::array<double,3>,3> _invert_3_by_3(
            std::array<std::array<double,3>,3> a) const;
};

}
