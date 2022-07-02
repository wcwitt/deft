#include "array.hpp"
#include "box.hpp"
#include "fourier.hpp"
#include "lattice_sum.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace deft
{

namespace py = pybind11;

// helper function for instantiating an Array class of type T
template<typename T>
void define_array(py::module &m, std::string name)
{
    py::class_<Array<T>>(m, name.c_str(), py::buffer_protocol())

        .def(py::init<std::array<size_t,3>>())

        // shape, size, strides, unravel_index
        .def("shape", &Array<T>::shape)
        .def("size", &Array<T>::size)
        .def("strides", &Array<T>::strides)
        .def("unravel_index", &Array<T>::unravel_index)

        // getters and setters for array elements
        .def("__getitem__",
             // get with deft::Array<T>.operator=(i)
             [](const Array<T>& arr, size_t i) {
                return arr(i);
             })
        .def("__setitem__",
             // set with deft::Array<T>.operator=(i)
             [](Array<T>& arr, size_t i, T value) {
                arr(i) = value;
             })
        .def("__getitem__",
             // get with deft::Array<T>.operator=(i,j,k)
             [](const Array<T>& arr, std::array<size_t,3> ijk) {
                return arr(ijk[0], ijk[1], ijk[2]);
             })
        .def("__setitem__",
             // set with deft::Array<T>.operator=(i,j,k)
             [](Array<T>& arr, std::array<size_t,3> ijk, T value) {
                arr(ijk[0], ijk[1], ijk[2]) = value;
             })
        .def("__getitem__",
             // get by casting as numpy array and forwarding the key
             // enables slicing and use of the ellipsis
             [](const Array<T>& arr, py::object key) {
                py::array_t<T> pyarr = py::cast(arr);
                return pyarr.attr("__getitem__")(key);
             })
        .def("__setitem__",
             // set by casting as numpy array and forwarding the key
             // enables slicing and manipulations like arr[...].fill(3.0)
             [](const Array<T>& arr, py::object key, py::array_t<T> value) {
                 py::array_t<T> pyarr = py::cast(arr);
                 return pyarr.attr("__setitem__")(key, value);
             })

        // arithmetic assignment
        .def(py::self += py::self)
        .def(py::self += T())
        .def(py::self -= py::self)
        .def(py::self -= T())
        .def(py::self *= py::self)
        .def(py::self *= T())
        .def(py::self /= py::self)
        .def(py::self /= T())

        // elementwise math
        .def("compute_sqrt", &Array<T>::compute_sqrt)
        .def("compute_pow", &Array<T>::compute_pow)

        // other functions
        .def("fill", &Array<T>::fill)
        .def("sum", &Array<T>::sum)

        // negation, addition, subtraction, multiplication, division
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self + T())
        .def(T() + py::self)
        .def(py::self - py::self)
        .def(py::self - T())
        .def(T() - py::self)
        .def(py::self * py::self)
        .def(py::self * T())
        .def(T() * py::self)
        .def(py::self / py::self)
        .def(py::self / T())
        .def(T() / py::self)

        .def_buffer(
                [](Array<T>& arr) -> py::buffer_info {
                    return py::buffer_info(
                        arr.data(),
                        sizeof(T),
                        py::format_descriptor<T>::format(),
                        3,
                        {arr.shape()[0], arr.shape()[1], arr.shape()[2]},
                        {arr.strides()[0]*sizeof(T),
                         arr.strides()[1]*sizeof(T),
                         arr.strides()[2]*sizeof(T)});
                });
}

PYBIND11_MODULE(pydeft, m) {

    // array class instantiations
    define_array<double>(m, "Double3D");
    define_array<std::complex<double>>(m, "Complex3D");

    // Box class
    py::class_<Box>(m, "Box")

        .def(py::init<std::array<std::array<double,3>,3>>())

        .def("vectors", &Box::vectors)
        .def("lengths", &Box::lengths)
        .def("angles", &Box::angles)
        .def("volume", &Box::volume)
        .def("recip_vectors", &Box::recip_vectors)
        .def("recip_lengths", &Box::recip_lengths)

        .def("wave_numbers", &Box::wave_numbers)
        .def("wave_vectors_x", &Box::wave_vectors_x)
        .def("wave_vectors_y", &Box::wave_vectors_y)
        .def("wave_vectors_z", &Box::wave_vectors_z)

        .def("set", &Box::set);

    // functions
    m.def("fourier_transform", &fourier_transform);
    m.def("inverse_fourier_transform", &inverse_fourier_transform);
    m.def("wave_vectors_x", &wave_vectors_x);
    m.def("wave_vectors_y", &wave_vectors_y);
    m.def("wave_vectors_z", &wave_vectors_z);
    m.def("wave_numbers", &wave_numbers);
    m.def("gradient_x", py::overload_cast<Double3D,Box>(&gradient_x));
    m.def("gradient_x", py::overload_cast<Complex3D,Box>(&gradient_x));
    m.def("gradient_y", py::overload_cast<Double3D,Box>(&gradient_y));
    m.def("gradient_y", py::overload_cast<Complex3D,Box>(&gradient_y));
    m.def("gradient_z", py::overload_cast<Double3D,Box>(&gradient_z));
    m.def("gradient_z", py::overload_cast<Complex3D,Box>(&gradient_z));
    m.def("grad_dot_grad", &grad_dot_grad);
    m.def("laplacian", py::overload_cast<Double3D,Box>(&laplacian));
    m.def("laplacian", py::overload_cast<Complex3D,Box>(&laplacian));
    m.def("structure_factor", &structure_factor);
    m.def("cardinal_b_spline_values", &cardinal_b_spline_values);
    m.def("cardinal_b_spline_derivatives", &cardinal_b_spline_derivatives);
    m.def("exponential_spline_b", &exponential_spline_b);
    m.def("structure_factor_spline", &structure_factor_spline);
    m.def(
        "array_from_lattice_sum",
        &array_from_lattice_sum<std::function<std::complex<double>(double,double,double)>>,
        py::arg("shape"),
        py::arg("box"),
        py::arg("xyz_coords"),
        py::arg("function_ft"),
        py::arg("spline_order")=-1);

    m.def("integrate", &integrate);
    m.def("fourier_interpolate", &fourier_interpolate);
}

}
