#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "fp_search.hpp"

namespace py = pybind11;

// fp_search_cpp: Python bindings of the Fincke-Pohst search of fp_search.hpp
// (the reference Python implementation is FPSearchPy in lattice/enumeration.py)
PYBIND11_MODULE(fp_search_cpp, m) {
    m.doc() = "High-performance C++ implementation of the Fincke-Pohst search.";

    py::class_<FPSearch>(m, "FPSearch",
        "Lists the integer vectors x with lbound < (x + b)^T A (x + b) <= ubound, for a positive definite A. "
        "Values within rounding errors of the bounds may or may not be listed: choose the bounds away from "
        "attainable values.")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, double, double>(),
             py::arg("A"), py::arg("b"), py::arg("lbound"), py::arg("ubound"),
             "Raises ValueError if A is not positive definite, OverflowError if the coordinates may exceed "
             "2^52 (the exact range of double).")
        .def("batch_search", &FPSearch::batch_search, py::arg("size"),
             "The next (at most) size vectors; the search resumes from there on the next call.")
        .def("search_all", &FPSearch::search_all, "All the remaining vectors.")
        .def("exhausted", &FPSearch::exhausted, "True when all the vectors have been listed.");
}
