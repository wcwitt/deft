#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <vector>

#ifdef DEFT_OPENMP
#include "omp.h"
#endif

namespace deft {

template<typename T> class Array {

public:

    // constructor
    Array(std::array<size_t,3> shape);

    // rule of five
    virtual ~Array() = default;
    Array(const Array&) = default;
    Array& operator=(const Array&);
    Array(Array&&) = default;
    Array& operator=(Array&&);

    // shape, size, strides, unravel_index
    std::array<size_t,3> shape() const;
    size_t size() const;
    std::array<size_t,3> strides() const;
    std::array<size_t,3> unravel_index(size_t i) const;

    // getters and setters for array elements
    T operator()(size_t) const;
    T& operator()(size_t);
    T operator()(size_t, size_t, size_t) const;
    T& operator()(size_t, size_t, size_t);

    // arithmetic assignment
    Array& operator+=(const Array&);
    Array& operator+=(T);
    Array& operator-=(const Array&);
    Array& operator-=(T);
    Array& operator*=(const Array&);
    Array& operator*=(T);
    Array& operator/=(const Array&);
    Array& operator/=(T);

    // set elementwise
    Array& set_elements(T value);
    template<typename F>
    Array& set_elements(F function);
    template<typename F>
    Array& set_elements(F function, size_t start, size_t end);

    // elementwise math
    Array& compute_sqrt();
    Array& compute_pow(double);

    // other functions
    Array& fill(T value);
    T sum();

    // raw pointer to data
    const T* data() const;
    T* data();

private:

    const std::array<size_t,3> _shape;
    const size_t _size;
    const std::array<size_t,3> _strides;
    std::vector<T> _data;

};

// negation, addition, subtraction, multiplication, division
template<typename T> Array<T> operator-(Array<T> a);
template<typename T> Array<T> operator+(Array<T> a, const Array<T>& b);
template<typename T> Array<T> operator+(Array<T> a, T b);
template<typename T> Array<T> operator+(T a, Array<T> b);
template<typename T> Array<T> operator-(Array<T> a, const Array<T>& b);
template<typename T> Array<T> operator-(Array<T> a, T b);
template<typename T> Array<T> operator-(T a, Array<T> b);
template<typename T> Array<T> operator*(Array<T> a, const Array<T>& b);
template<typename T> Array<T> operator*(Array<T> a, T b);
template<typename T> Array<T> operator*(T a, Array<T> b);
template<typename T> Array<T> operator/(Array<T> a, const Array<T>& b);
template<typename T> Array<T> operator/(Array<T> a, T b);
template<typename T> Array<T> operator/(T a, Array<T> b);

}

namespace deft {

template<typename T>
Array<T>::Array(std::array<size_t,3> shape)
    : _shape(shape),
      _size(shape[0]*shape[1]*shape[2]),
      _strides({shape[1]*shape[2], shape[2], 1}),
      _data(_size)
{
}

template<typename T>
Array<T>& Array<T>::operator=(const Array<T>& other)
{
    set_elements([&other](size_t i) { return other._data[i]; });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator=(Array<T>&& other)
{
    _data = std::move(other._data);
    return *this;
}

template<typename T>
std::array<size_t,3> Array<T>::shape() const
{
    return _shape;
}

template<typename T>
size_t Array<T>::size() const
{
    return _size;
}

template<typename T>
std::array<size_t,3> Array<T>::strides() const
{
    return _strides;
}

template<typename T>
std::array<size_t,3> Array<T>::unravel_index(size_t i) const
{
    return { i / _strides[0],
            (i % _strides[0]) / _strides[1],
            (i % _strides[0]) % _strides[1] };
}

template<typename T>
T Array<T>::operator()(const size_t i) const
{
    return _data[i];
}

template<typename T>
T& Array<T>::operator()(const size_t i)
{
    return _data[i];
}

template<typename T>
T Array<T>::operator()(const size_t i, const size_t j, const size_t k) const
{
    return _data[i*_strides[0] + j*_strides[1] + k];
}

template<typename T>
T& Array<T>::operator()(const size_t i, const size_t j, const size_t k)
{
    return _data[i*_strides[0] + j*_strides[1] + k];
}

template<typename T>
Array<T>& Array<T>::operator+=(const Array<T>& other)
{
    set_elements(
        [this,&other](size_t i) {
            return _data[i] + other._data[i];
        });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator+=(T value)
{
    set_elements([this,value](size_t i) { return _data[i] + value; });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator-=(const Array<T>& other)
{
    set_elements([this,&other](size_t i) {
        return _data[i] - other._data[i];
    });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator-=(T value)
{
    set_elements([this,value](size_t i) { return _data[i] - value; });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator*=(const Array<T>& other)
{
    set_elements([this,&other](size_t i) {
        return _data[i] * other._data[i];
    });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator*=(T value)
{
    set_elements([this,value](size_t i) { return _data[i] * value; });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator/=(const Array<T>& other)
{
    set_elements([this,&other](size_t i) {
        return _data[i] / other._data[i];
    });
    return *this;
}

template<typename T>
Array<T>& Array<T>::operator/=(T value)
{
    set_elements([this,value](size_t i) { return _data[i] / value; });
    return *this;
}

template<typename T>
Array<T>& Array<T>::set_elements(T value)
{
#ifdef DEFT_OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i=0; i<_size; ++i) {
        _data[i] = value;
    }
    return *this;
}

template<typename T> template<typename F>
Array<T>& Array<T>::set_elements(F function)
{
    set_elements(function, 0, _size);
    return *this;
}

template<typename T> template<typename F>
Array<T>& Array<T>::set_elements(F function, size_t start, size_t end)
{
#ifdef DEFT_OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i=start; i<end; ++i) {
        _data[i] = function(i);
    }
    return *this;
}

template<typename T>
Array<T>& Array<T>::compute_sqrt()
{
    set_elements([this](size_t i) { return std::sqrt(_data[i]); });
    return *this;
}

template<typename T>
Array<T>& Array<T>::compute_pow(double power)
{
    set_elements([this,power](size_t i) {
        return std::pow(_data[i], power);
    });
    return *this;
}

template<typename T>
Array<T>& Array<T>::fill(T value)
{
    set_elements(value);
    return *this;
}

template<typename T>
T Array<T>::sum()
{
    T sum{};
#if !defined DEFT_OPENMP
    for (size_t i=0; i<_size; ++i) {
        sum += _data[i];
    }
#elif defined DEFT_OPENMP
    #pragma omp parallel
    {
        T local_sum{};
        #pragma omp for nowait
        for (size_t i=0; i<_size; ++i) {
            local_sum += _data[i];
        }
        #pragma omp critical
        sum += local_sum;
    }
#endif
    return sum;
}

template<typename T>
const T* Array<T>::data() const
{
    return _data.data();
}

template<typename T>
T* Array<T>::data()
{
    return _data.data();
}

template<typename T> Array<T> operator-(Array<T> a)
{
    a.set_elements([&a](size_t i) { return -a(i); });
    return a;
}

template<typename T>
Array<T> operator+(Array<T> a, const Array<T>& b)
{
    a.set_elements([&a,&b](size_t i) { return a(i) + b(i); });
    return a;
}

template<typename T>
Array<T> operator+(Array<T> a, T b)
{
    a.set_elements([&a,b](size_t i) { return a(i) + b; });
    return a;
}

template<typename T>
Array<T> operator+(T a, Array<T> b)
{
    b.set_elements([a,&b](size_t i) { return a + b(i); });
    return b;
}

template<typename T>
Array<T> operator-(Array<T> a, const Array<T>& b)
{
    a.set_elements([&a,&b](size_t i) { return a(i) - b(i); });
    return a;
}

template<typename T>
Array<T> operator-(Array<T> a, T b)
{
    a.set_elements([&a,b](size_t i) { return a(i) - b; });
    return a;
}

template<typename T>
Array<T> operator-(T a, Array<T> b)
{
    b.set_elements([a,&b](size_t i) { return a - b(i); });
    return b;
}

template<typename T>
Array<T> operator*(Array<T> a, const Array<T>& b)
{
    a.set_elements([&a,&b](size_t i) { return a(i) * b(i); });
    return a;
}

template<typename T>
Array<T> operator*(Array<T> a, T b)
{
    a.set_elements([&a,b](size_t i) { return a(i) * b; });
    return a;
}

template<typename T>
Array<T> operator*(T a, Array<T> b)
{
    b.set_elements([a,&b](size_t i) { return a * b(i); });
    return b;
}

template<typename T>
Array<T> operator/(Array<T> a, const Array<T>& b)
{
    a.set_elements([&a,&b](size_t i) { return a(i) / b(i); });
    return a;
}

template<typename T>
Array<T> operator/(Array<T> a, T b)
{
    a.set_elements([&a,b](size_t i) { return a(i) / b; });
    return a;
}

template<typename T>
Array<T> operator/(T a, Array<T> b)
{
    b.set_elements([a,&b](size_t i) { return a / b(i); });
    return b;
}

}
