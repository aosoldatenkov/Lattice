#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <algorithm>


// Fincke-Pohst search for the integer vectors x with lbound < (x + b)^T A (x + b) <= ubound,
// for a positive definite matrix A. The coordinates are fixed from the last one down; at each
// level, the admissible interval of the coordinate is scanned from its top downwards.
// The values are computed in floating point, so vectors with values within rounding errors
// of lbound or ubound may or may not be returned: choose the bounds away from attainable values.
// The search state is kept between calls of batch_search, so that the vectors can be listed in batches.
class FPSearch {
private:
    int rank;
    Eigen::MatrixXd A;    // the positive definite matrix
    Eigen::VectorXd b;    // the shift
    double lbound;
    double ubound;
    Eigen::MatrixXd R;    // the Cholesky factor: A = R^T R, R upper triangular
    Eigen::Matrix<long, Eigen::Dynamic, 1> x;  // the current vector; x(k), ..., x(rank - 1) are fixed at level k
    Eigen::VectorXd dist; // dist(k): the contribution of the coordinates k + 1, ..., rank - 1 to the value
    Eigen::VectorXd z;    // z(k) - b(k): the center of the admissible interval of x(k) at level k
    int k;                // the current level (rank when the search is exhausted)

    // Called when the point at level k is outside the ellipsoid. Above the center, this only means
    // that the top of the interval was rounded up, so the scan continues downwards; pruning there
    // would skip the whole interval. Below the center, the interval is exhausted: go one level up.
    void step_outside() {
        if (static_cast<double>(x(k)) + b(k) > z(k)) {
            x(k) -= 1;
            return;
        }
        k += 1;
        if (k < rank) x(k) -= 1;
    }

public:
    // Throws std::invalid_argument if A is not positive definite, std::overflow_error if the coordinates
    // may exceed 2^52, beyond which x + b is not exact in double precision (pybind11: ValueError, OverflowError)
    FPSearch(Eigen::MatrixXd A_in, Eigen::VectorXd b_in, double lbound_in, double ubound_in)
        : rank(A_in.rows()), A(std::move(A_in)), b(std::move(b_in)), lbound(lbound_in), ubound(ubound_in) {

        // LLT decomposition (Cholesky). A = L * L^T = U^T * U.
        // No regularization: adding a multiple of the identity to A would bias all the computed
        // values upwards, and lose the vectors close to ubound.
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        if (llt.info() != Eigen::Success) {
            throw std::invalid_argument("FPSearch: the matrix must be positive definite");
        }
        R = llt.matrixU();

        // The coordinates are stored as long, and converted to double: on the ellipsoid,
        // |x_i + b_i| <= sqrt(ubound (A^-1)_ii), which must stay within the exact range 2^52 of double
        Eigen::MatrixXd A_inv = llt.solve(Eigen::MatrixXd::Identity(rank, rank));
        for (int i = 0; i < rank; ++i) {
            double r = std::abs(b(i)) + std::sqrt(std::max(0.0, ubound) * A_inv(i, i)) + 1;
            if (!(r < 4503599627370496.0)) throw std::overflow_error("FPSearch: the coordinates may exceed 2^52");
        }

        // Initialize state vectors
        x = Eigen::Matrix<long, Eigen::Dynamic, 1>::Zero(rank);
        dist = Eigen::VectorXd::Zero(rank);
        z = Eigen::VectorXd::Zero(rank);
        k = rank - 1;

        if (rank > 0) {
            double r_kk = R(k, k);
            x(k) = static_cast<long>(std::floor(std::sqrt(std::max(0.0, ubound) / (r_kk * r_kk)) - b(k)));
        }
    }

    // True when all the vectors have been listed
    bool exhausted() const {
        return rank == 0 || k == rank;
    }

    // The next (at most) size vectors; the search resumes from there on the next call
    std::vector<std::vector<long>> batch_search(int size) {
        std::vector<std::vector<long>> vecs;

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
                        std::vector<long> res(rank);
                        for (int i = 0; i < rank; ++i) {
                            res[i] = x(i);
                        }
                        vecs.push_back(res);
                    } else if (static_cast<double>(x(k)) + b(k) > z(k)) {
                        // Inside the inner ellipsoid above the center: jump to its other side
                        long candidate = static_cast<long>(std::ceil(2.0 * (z(k) - b(k)) - static_cast<double>(x(k)))) + 1;
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
                    x(k) = static_cast<long>(std::floor(std::sqrt(inner_val) + z(k) - b(k)));
                }
            } else {
                step_outside();
                if (exhausted()) {
                    break;
                }
            }
        }
        return vecs;
    }

    // All the remaining vectors
    std::vector<std::vector<long>> search_all() {
        std::vector<std::vector<long>> vecs;
        while (!exhausted()) {
            auto batch = batch_search(1 << 16);
            vecs.insert(vecs.end(), batch.begin(), batch.end());
        }
        return vecs;
    }
};
