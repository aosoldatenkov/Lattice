#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "fp_search.hpp"

namespace py = pybind11;

// Pybind11 Module Definition
PYBIND11_MODULE(fp_search_cpp, m) {
    m.doc() = "High-performance C++ implementation of the Fincke-Pohst search.";

    py::class_<FPSearch>(m, "FPSearch")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, double, double>(),
             py::arg("A"), py::arg("b"), py::arg("lbound"), py::arg("ubound"))
        .def("batch_search", &FPSearch::batch_search, py::arg("size"))
        .def("search_all", &FPSearch::search_all)
        .def("exhausted", &FPSearch::exhausted);
}
