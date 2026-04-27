#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

class FPSearch {
private:
    int rank;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double lbound;
    double ubound;
    Eigen::MatrixXd R;
    Eigen::VectorXi x;
    Eigen::VectorXd dist;
    Eigen::VectorXd z;
    int k;

public:
    FPSearch(Eigen::MatrixXd A_in, Eigen::VectorXd b_in, double lbound_in, double ubound_in)
        : rank(A_in.rows()), A(std::move(A_in)), b(std::move(b_in)), lbound(lbound_in), ubound(ubound_in) {
        
        // Add a small regularization term to the diagonal for stability, mirroring the Python implementation
        Eigen::MatrixXd A_reg = A + Eigen::MatrixXd::Identity(rank, rank) * 1e-10;
        
        // LLT decomposition (Cholesky). A = L * L^T = U^T * U. 
        // We extract the upper triangular matrix U to match Python's np.linalg.cholesky().T
        Eigen::LLT<Eigen::MatrixXd> llt(A_reg);
        R = llt.matrixU();

        // Initialize state vectors
        x = Eigen::VectorXi::Zero(rank);
        dist = Eigen::VectorXd::Zero(rank);
        z = Eigen::VectorXd::Zero(rank);
        k = rank - 1;

        if (rank > 0) {
            double r_kk = R(k, k);
            x(k) = static_cast<int>(std::floor(std::sqrt(ubound / (r_kk * r_kk)) - b(k)));
        }
    }

    bool exhausted() {
        return rank == 0 || k == rank;
    }

    std::vector<std::vector<int>> batch_search(int size) {
        std::vector<std::vector<int>> vecs;
        
        // Return early if the search space is exhausted or invalid
        if (exhausted()) {
            return vecs;
        }

        // Hot loop: Avoid any heap allocations inside this block
        while (vecs.size() < static_cast<size_t>(size)) {
            double r_kk = R(k, k);
            double x_plus_b_minus_z = static_cast<double>(x(k)) + b(k) - z(k);
            double d = dist(k) + r_kk * r_kk * x_plus_b_minus_z * x_plus_b_minus_z;

            if (d <= ubound) {
                if (k == 0) {
                    if (d > lbound) {
                        // We reached a valid leaf node, record the vector
                        std::vector<int> res(rank);
                        for (int i = 0; i < rank; ++i) {
                            res[i] = x(i);
                        }
                        vecs.push_back(res);
                    } else if (static_cast<double>(x(k)) + b(k) > z(k)) {
                        int candidate = static_cast<int>(std::ceil(2.0 * (z(k) - b(k)) - static_cast<double>(x(k)))) + 1;
                        x(k) = std::min(x(k), candidate);
                    }
                    x(k) -= 1;
                } else {
                    // Move down the tree
                    k -= 1;
                    double r_kp1 = R(k + 1, k + 1);
                    double x_plus_b_minus_z_kp1 = static_cast<double>(x(k + 1)) + b(k + 1) - z(k + 1);
                    dist(k) = dist(k + 1) + r_kp1 * r_kp1 * x_plus_b_minus_z_kp1 * x_plus_b_minus_z_kp1;

                    // Update the center z for the new level
                    double sum = 0.0;
                    for (int j = k + 1; j < rank; ++j) {
                        sum += R(k, j) * (static_cast<double>(x(j)) + b(j));
                    }
                    z(k) = -sum / R(k, k);

                    // Set the upper bound for this coordinate.
                    // Note: std::max(0.0, ...) handles floating-point inaccuracies that could push the value slightly < 0
                    double inner_val = std::max(0.0, (ubound - dist(k)) / (R(k, k) * R(k, k)));
                    x(k) = static_cast<int>(std::floor(std::sqrt(inner_val) + z(k) - b(k)));
                }
            } else {
                // Prune the branch and step back up
                k += 1;
                if (k == rank) {
                    break;
                }
                x(k) -= 1;
            }
        }
        return vecs;
    }
};

// Pybind11 Module Definition
PYBIND11_MODULE(fp_search_cpp, m) {
    m.doc() = "High-performance C++ implementation of the Fincke-Pohst search.";

    py::class_<FPSearch>(m, "FPSearch")
        .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, double, double>(),
             py::arg("A"), py::arg("b"), py::arg("lbound"), py::arg("ubound"))
        .def("batch_search", &FPSearch::batch_search, py::arg("size"))
        .def("exhausted", &FPSearch::exhausted);
}