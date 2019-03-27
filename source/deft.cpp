#include "deft.hpp"

deft::deft(const size_t numX, const size_t numY, const size_t numZ, 
            const double* vecX, const double* vecY, const double* vecZ):
    _xDim(numX), _yDim(numY), _zDim(numZ),
    _dimXYZ(numX * numY * numZ),
    _xDimFT(numX/2+1),  _yDimFT(numY),  _zDimFT(numZ)
{
    // create shared pointers for cell geometry
    _cellVecX = make_shared<vec>(3);
    _cellVecY = make_shared<vec>(3);
    _cellVecZ = make_shared<vec>(3);
    _cellLenX = make_shared<double>();
    _cellLenY = make_shared<double>();
    _cellLenZ = make_shared<double>();
    _vol = make_shared<double>();
    _dv = make_shared<double>();

    // create shared pointers for reciprocal lattice vectors
    _kVecX = make_shared<cube>(_xDimFT, _yDimFT, _zDimFT);
    _kVecY = make_shared<cube>(_xDimFT, _yDimFT, _zDimFT);
    _kVecZ = make_shared<cube>(_xDimFT, _yDimFT, _zDimFT);
    _kVecLen = make_shared<cube>(_xDimFT, _yDimFT, _zDimFT);

    // initialize cell geometry and reciprocal lattice
    updateGeometry(vecX, vecY, vecZ);

    // create real space grid
    double* ptr = fftw_alloc_real(_xDim * _yDim * _zDim);
    _data = new cube(ptr, _xDim, _yDim, _zDim, false, true);

    // create fourier transform grid
    fftw_complex* ptrFT = fftw_alloc_complex(_xDimFT * _yDimFT * _zDimFT);
    _dataFT = new cx_cube(reinterpret_cast<complex<double>*>(ptrFT), _xDimFT, _yDimFT, _zDimFT, false, true);

    // initialize ffts
    // note that the array dimensions are reversed b/c armadillo uses column-major order (like fortran)
    // see: http://www.fftw.org/fftw3_doc/Reversing-array-dimensions.html#Reversing-array-dimensions
    // could try FFTW_PATIENT in place of FFTW_MEASURE?
    _planR2C = fftw_plan_dft_r2c_3d(_zDim, _yDim, _xDim, ptr, ptrFT, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    _planC2R = fftw_plan_dft_c2r_3d(_zDim, _yDim, _xDim, ptrFT, ptr, FFTW_MEASURE | FFTW_DESTROY_INPUT);
}

deft::deft(const deft& grd):
    _xDim(grd._xDim), _yDim(grd._yDim), _zDim(grd._zDim),
    _dimXYZ(grd._dimXYZ),
    _xDimFT(grd._xDimFT),  _yDimFT(grd._yDimFT),  _zDimFT(grd._zDimFT)
{

    // set cell properties
    _cellVecX = grd._cellVecX;
    _cellVecY = grd._cellVecY;
    _cellVecZ = grd._cellVecZ;
    _cellLenX = grd._cellLenX;
    _cellLenY = grd._cellLenY;
    _cellLenZ = grd._cellLenZ;
    _vol = grd._vol;
    _dv = grd._dv;

    // set k-vector properties
    _kVecX = grd._kVecX;
    _kVecY = grd._kVecY;
    _kVecZ = grd._kVecZ;
    _kVecLen = grd._kVecLen;

    // create real space grid
    double* ptr = fftw_alloc_real(_dimXYZ);
    _data = new cube(ptr, _xDim, _yDim, _zDim, false, true);

    // create fourier transform grid
    fftw_complex* ptrFT = fftw_alloc_complex(_dimXYZ);
    _dataFT = new cx_cube(reinterpret_cast<complex<double>*>(ptrFT), _xDimFT, _yDimFT, _zDimFT, false, true);

    // copy data
    // TODO ideally, this should copy _data or _ft_data, depending on state
    *_data = *grd._data;
    *_dataFT = *grd._dataFT;

    // initialize ffts
    // fftw saves accumulated wisdom (see fftw manual section 4.2), so re-planning for same array shape should be fast
    // TODO: test this assertion
    _planR2C = fftw_plan_dft_r2c_3d(_zDim, _yDim, _xDim, ptr, ptrFT, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    _planC2R = fftw_plan_dft_c2r_3d(_zDim, _yDim, _xDim, ptrFT, ptr, FFTW_MEASURE | FFTW_DESTROY_INPUT);

}

// copy data
void deft::copy_data_from(const double* rawData){
    double* dataPtr = _data->memptr();
    memcpy(dataPtr, rawData, _dimXYZ*sizeof(double));
}

// read data
double deft::at(const size_t i, const size_t j, const size_t k) const{
    return _data->at(i, j, k);
}
double deft::operator()(const size_t i, const size_t j, const size_t k) const{
    return at(i,j,k);
}

// assignment
void deft::equals(const double val){
    _data->fill(val);
}
void deft::operator=(const double val){ deft::equals(val); }

void deft::equals(const deft grd){
    *_data = *(grd._data);
}
void deft::operator=(const deft grd){ deft::equals(grd); }

// addition
void deft::addEquals(const double val){
    *_data = *_data + val;
}
void deft::addEquals(const deft grd){
    *_data = *_data + *(grd._data);
}

// subtraction
void deft::subtractEquals(const double val){
    *_data = *_data - val;
}
void deft::subtractEquals(const deft grd){
    *_data = *_data - *(grd._data);
}

// multiplication
void deft::multiplyEquals(const double val){
    *_data = *_data * val;
}
void deft::multiplyEquals(const deft grd){
    *_data = *_data % *(grd._data);
}

// division
void deft::divideEquals(const double val){
    *_data = *_data / val;
}
void deft::divideEquals(const deft grd){
    *_data = *_data / *(grd._data);
}

// elementwise math
void deft::pow(const double val){
    *_data = arma::pow(*_data, val);
}

// fourier transformation
void deft::computeFT(){
    fftw_execute(_planR2C);
    *_dataFT = *_dataFT / _dimXYZ;
}
void deft::computeIFT(){
    fftw_execute(_planC2R);
}

// derivatives (based on fourier transforms)
void deft::computeGradientX(){
    const complex<double> i(0.0,1.0);
    computeFT();
    for(size_t k=0; k<_dataFT->n_elem; ++k){
        _dataFT->at(k) *= i * _kVecX->at(k);
    }
    computeIFT();
}
void deft::computeGradientY(){
    const complex<double> i(0.0,1.0);
    computeFT();
    for(size_t k=0; k<_dataFT->n_elem; ++k){
        _dataFT->at(k) *= i * _kVecY->at(k);
    }
    computeIFT();
}
void deft::computeGradientZ(){
    const complex<double> i(0.0,1.0);
    computeFT();
    for(size_t k=0; k<_dataFT->n_elem; ++k){
        _dataFT->at(k) *= i * _kVecZ->at(k);
    }
    computeIFT();
}
void deft::computeGradientSquared(){

    // create two copies
    deft orig(*this);
    deft tmp(*this);
    
    // x direction
    tmp.computeGradientX();
    tmp.multiplyEquals(tmp);
    this->equals(tmp);

    // y direction
    tmp.equals(orig);
    tmp.computeGradientY();
    tmp.multiplyEquals(tmp);
    this->addEquals(tmp);

    // z direction
    tmp.equals(orig);
    tmp.computeGradientZ();
    tmp.multiplyEquals(tmp);
    this->addEquals(tmp);
}

void deft::computeLaplacian(){
    computeFT();
    for(size_t k=0; k<_dataFT->n_elem; ++k){
        _dataFT->at(k) *= - _kVecLen->at(k) * _kVecLen->at(k);
    }
    computeIFT();
}

// integrate
double deft::integrate() const{
    return accu(*_data) * *_dv;
}

// update geometry
void deft::updateGeometry(const double* vecX, const double* vecY, const double* vecZ){

    // set cell vectore
    _cellVecX->at(0)=vecX[0];  _cellVecX->at(1)=vecX[1]; _cellVecX->at(2)=vecX[2];
    _cellVecY->at(0)=vecY[0];  _cellVecY->at(1)=vecY[1]; _cellVecY->at(2)=vecY[2];
    _cellVecZ->at(0)=vecZ[0];  _cellVecZ->at(1)=vecZ[1]; _cellVecZ->at(2)=vecZ[2];
    *_cellLenX = norm(*_cellVecX);
    *_cellLenY = norm(*_cellVecY);
    *_cellLenZ = norm(*_cellVecZ);

    // compute volume and dv
    mat matrix(3,3);
    matrix.col(0) = *_cellVecX;
    matrix.col(1) = *_cellVecY;
    matrix.col(2) = *_cellVecZ;
    *_vol = det(matrix);
    *_dv = *_vol / _dimXYZ;

    // compute reciprocal lattice vectors
    mat recipLat = 2.0 * M_PI * trans(inv(matrix));
    vec kVec(3);
    for(uword nz=0; nz<_zDimFT; ++nz){
    for(uword ny=0; ny<_yDimFT; ++ny){
    for(uword nx=0; nx<_xDimFT; ++nx){

        // TODO: write explanation
        int nytmp = ny - static_cast<int>(ny > _yDim/2) * _yDim;
        int nztmp = nz - static_cast<int>(nz > _zDim/2) * _zDim;

        // get k vector in cartesian coordinates and store
        kVec(0) = nx;  kVec(1) = nytmp;  kVec(2) = nztmp;
        kVec = recipLat * kVec;
        _kVecX->at(nx,ny,nz) = kVec(0);
        _kVecY->at(nx,ny,nz) = kVec(1);
        _kVecZ->at(nx,ny,nz) = kVec(2);
        _kVecLen->at(nx,ny,nz) = norm(kVec);

    }}}
}

// read cell information
double deft::cellVecX(const size_t i) const{ return _cellVecX->at(i); }
double deft::cellVecY(const size_t i) const{ return _cellVecY->at(i); }
double deft::cellVecZ(const size_t i) const{ return _cellVecZ->at(i); }
double deft::cellLenX() const{ return *_cellLenX; }
double deft::cellLenY() const{ return *_cellLenY; }
double deft::cellLenZ() const{ return *_cellLenZ; }
double deft::vol() const{ return *_vol; }
double deft::dv() const{ return *_dv; }
double deft::kVecX(const size_t i) const{ return _kVecX->at(i); }
double deft::kVecX(const size_t i, const size_t j, const size_t k) const{ return _kVecX->at(i,j,k); }
double deft::kVecY(const size_t i) const{ return _kVecY->at(i); }
double deft::kVecY(const size_t i, const size_t j, const size_t k) const{ return _kVecY->at(i,j,k); }
double deft::kVecZ(const size_t i) const{ return _kVecZ->at(i); }
double deft::kVecZ(const size_t i, const size_t j, const size_t k) const{ return _kVecZ->at(i,j,k); }
double deft::kVecLen(const size_t i) const{ return _kVecLen->at(i); }
double deft::kVecLen(const size_t i, const size_t j, const size_t k) const{ return _kVecLen->at(i,j,k); }


// interpolate
deft* deft::interpolate(const size_t new_x, const size_t new_y, const size_t new_z){


    // create new grid
    deft* grd = new deft(new_x, new_y, new_z, this->_cellVecX->memptr(), this->_cellVecY->memptr(), this->_cellVecZ->memptr());

    // fourier transforms
    this->computeFT();
    grd->computeFT();
    grd->_dataFT->fill(0.0);

    // set to zero initially
    grd->_dataFT->fill(0.0);

    // extract grid dimensions
    size_t n2 = this->_yDim;  size_t n3 = this->_zDim;
    size_t dn2 = grd->_yDim;  size_t dn3 = grd->_zDim;
    // todo: double check that new grid is denser than old grid

    // compute nominal halfway point
    size_t h1 = this->_xDim/2;  size_t h2 = this->_yDim/2;  size_t h3 = this->_zDim/2;

    // ----- first part of y, first part of z -----
    (*grd->_dataFT)(span(0,h1), span(0,h2), span(0,h3)) = (*this->_dataFT)(span(0,h1), span(0,h2), span(0,h3));
    
    // ----- second part of y, first part of z -----
    size_t j = h2 + 1;
    (*grd->_dataFT)(span(0,h1), span(dn2-n2+j,dn2-1), span(0,h3)) = (*this->_dataFT)(span(0,h1), span(j,n2-1), span(0,h3));

    // ----- first part of y, second part of z -----
    size_t k = h3 + 1;
    (*grd->_dataFT)(span(0,h1), span(0,h2), span(dn3-n3+k,dn3-1)) = (*this->_dataFT)(span(0,h1), span(0,h2), span(k,n3-1));

    // ----- second part of y, second part of z -----
    j = h2 + 1;
    k = h3 + 1;
    (*grd->_dataFT)(span(0,h1), span(dn2-n2+j,dn2-1), span(dn3-n3+k,dn3-1)) = (*this->_dataFT)(span(0,h1), span(j,n2-1), span(k,n3-1));

    // return to real space
    this->computeIFT();
    grd->computeIFT();

    return grd;

}



// other fourier transform-enabled functionality
void deft::compute_periodic_superposition(mat loc, double (*func)(double)){

    // compute fourier transform
    this->computeFT();

    const complex<double> i = {0.0, 1.0};
    // loop over k-vectors
    for(size_t k=0; k<this->_dataFT->n_elem; ++k){

        vec kvec(3);
        kvec(0) = this->kVecX(k);
        kvec(1) = this->kVecY(k);
        kvec(2) = this->kVecZ(k);

        // compute structure factor for this k-vector
        complex<double> str_fact = {0.0, 0.0};
        for(size_t a=0; a<loc.n_rows; ++a){
            double k_dot_r = dot(kvec, loc.row(a));
            str_fact += exp(-i*k_dot_r);
        }

        // evaluate function
        this->_dataFT->at(k) = str_fact * func(this->kVecLen(k));
    }

    // compute inverse fourier transform and divide by volume
    this->computeIFT();
    this->divideEquals(this->vol());
}


extern "C"{

    deft* deft_c(const size_t numX, const size_t numY, const size_t numZ, 
                    const double* vecX, const double* vecY, const double* vecZ){
        return new deft(numX, numY, numZ, vecX, vecY, vecZ);
    }

    void equals_c(deft* grd, const double val){
        grd->equals(val);
    }

    double at_c(const deft* grd, const size_t i, const size_t j, const size_t k){
        return grd->at(i, j, k);
    }

    void copy_data_from_c(deft* grd, const double* rawData){
        grd->copy_data_from(rawData);
    }

    double integrate_c(const deft* grd){
        return grd->integrate();
    }

    deft* interpolate_c(deft* grd, const size_t new_x, const size_t new_y, const size_t new_z){
        return grd->interpolate(new_x, new_y, new_z);
    }

    void compute_periodic_superposition_c(deft* grd, size_t num, double* loc, double (*func)(double)){
        mat loc_mat(loc, num, 3);
        return grd->compute_periodic_superposition(loc_mat, func);
    }
}
