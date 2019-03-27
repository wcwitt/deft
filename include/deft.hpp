#include <complex>
#include <memory>
#include <fftw3.h>
#include <armadillo>
using namespace std;
using namespace arma;

#ifndef __DEFT_HPP__
#define __DEFT_HPP__

class deft{

public:

    // constructors
    deft(const size_t, const size_t, const size_t, const double*, const double*, const double*);
    deft(const deft&);

    // read data
    double at(const size_t, const size_t, const size_t) const;
    double operator()(const size_t, const size_t, const size_t) const;

    // copy data
    void copyDataFrom(const double*);

    // assignment
    void equals(const double);
    void operator=(const double);
    void equals(const deft);
    void operator=(const deft);

    // addition
    void addEquals(const double);
    void operator+=(const double);
    void addEquals(const deft);
    void operator+=(const deft);

    // subtraction
    void subtractEquals(const double);
    void operator-=(const double);
    void subtractEquals(const deft);
    void operator-=(const deft);
    // multiplication
    void multiplyEquals(const double);
    void operator*=(const double);
    void multiplyEquals(const deft);
    void operator*=(const deft);

    // division
    void divideEquals(const double);
    void operator/=(const double);
    void divideEquals(const deft);
    void operator/=(const deft);

    // elementwise math
    void pow(const double);

    // fourier transform operations
    void computeFT();
    void computeIFT();

    // derivatives (using fourier transforms)
    void computeGradientX();
    void computeGradientY();
    void computeGradientZ();
    void computeGradientSquared();
    void computeLaplacian();

    // integrate
    double integrate() const;

    // update cell geometry and reciprocal lattice vectors
    void updateGeometry(const double*, const double*, const double*);

    // read cell information
    double cellVecX(const size_t) const;
    double cellVecY(const size_t) const;
    double cellVecZ(const size_t) const;
    double cellLenX() const;
    double cellLenY() const;
    double cellLenZ() const;
    double vol() const;
    double dv() const;
    double kVecX(const size_t) const;
    double kVecX(const size_t, const size_t, const size_t) const;
    double kVecY(const size_t) const;
    double kVecY(const size_t, const size_t, const size_t) const;
    double kVecZ(const size_t) const;
    double kVecZ(const size_t, const size_t, const size_t) const;
    double kVecLen(const size_t) const;
    double kVecLen(const size_t, const size_t, const size_t) const;

    // interpolate
    deft* interpolate(const size_t new_x, const size_t new_y, const size_t new_z);

    // compute periodic superposition
    void compute_periodic_superposition(mat loc, double (*func)(double));

private:
public:

    // real-space dimensions and data
    const size_t _xDim;
    const size_t _yDim;
    const size_t _zDim;
    const size_t _dimXYZ;
    // TODO: decide whether this should remain a pointer
    cube* _data;

private:

    // cell geometry
    shared_ptr<vec> _cellVecX;
    shared_ptr<vec> _cellVecY;
    shared_ptr<vec> _cellVecZ;
    shared_ptr<double> _cellLenX;
    shared_ptr<double> _cellLenY;
    shared_ptr<double> _cellLenZ;
    shared_ptr<double> _vol;
    shared_ptr<double> _dv;

public:

    // fourier-space dimensions and data
    const size_t _xDimFT;
    const size_t _yDimFT;
    const size_t _zDimFT;
    // TODO: add ft_numXYZ
    cx_cube* _dataFT;

private:

    // reciprocal lattice vectors
    shared_ptr<cube> _kVecX;
    shared_ptr<cube> _kVecY;
    shared_ptr<cube> _kVecZ;
    shared_ptr<cube> _kVecLen;

    // fftw info
    fftw_plan _planR2C;
    fftw_plan _planC2R;

};



#endif  //  __DEFT_HPP__

