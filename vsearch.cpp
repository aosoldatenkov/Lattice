#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <map>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <cstdlib>

namespace py = pybind11;

using int64 = long long;
using MatrixXi64 = Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXi64 = Eigen::Matrix<int64, 1, Eigen::Dynamic>;

// --- 1. Rational Number Struct for exact sorting of distances ---
struct Rational {
    int64 p, q;
    Rational(int64 num, int64 den) : p(num), q(den) {
        if (q < 0) { p = -p; q = -q; }
    }
    bool operator<(const Rational& o) const {
        // p1/q1 < p2/q2 => p1*q2 < p2*q1 (since q > 0)
        return p * o.q < o.p * q;
    }
    bool operator==(const Rational& o) const {
        return p * o.q == o.p * q;
    }
};

// --- 2. Sequential FPSearch (Runs purely inside worker threads) ---
class FPSearchSeq {
private:
    int rank;
    Eigen::MatrixXd R;
    Eigen::VectorXd b;
    double lbound, ubound;
    Eigen::VectorXi x;
    Eigen::VectorXd dist, z;
    int k;

public:
    FPSearchSeq(Eigen::MatrixXd A, Eigen::VectorXd b_in, double lbound_in, double ubound_in)
        : rank(A.rows()), b(std::move(b_in)), lbound(lbound_in), ubound(ubound_in) {
        Eigen::MatrixXd A_reg = A + Eigen::MatrixXd::Identity(rank, rank) * 1e-10;
        Eigen::LLT<Eigen::MatrixXd> llt(A_reg);
        R = llt.matrixU();
        x = Eigen::VectorXi::Zero(rank);
        dist = Eigen::VectorXd::Zero(rank);
        z = Eigen::VectorXd::Zero(rank);
        k = rank - 1;
        if (rank > 0) {
            x(k) = static_cast<int>(std::floor(std::sqrt(ubound / (R(k, k) * R(k, k))) - b(k)));
        }
    }

    std::vector<std::vector<int>> search_all() {
        std::vector<std::vector<int>> vecs;
        if (rank == 0 || k == rank) return vecs;

        while (true) {
            double r_kk = R(k, k);
            double x_plus_b_minus_z = static_cast<double>(x(k)) + b(k) - z(k);
            double d = dist(k) + r_kk * r_kk * x_plus_b_minus_z * x_plus_b_minus_z;

            if (d <= ubound) {
                if (k == 0) {
                    if (d > lbound) {
                        std::vector<int> res(rank);
                        for (int i = 0; i < rank; ++i) res[i] = x(i);
                        vecs.push_back(res);
                    } else if (static_cast<double>(x(k)) + b(k) > z(k)) {
                        int candidate = static_cast<int>(std::ceil(2.0 * (z(k) - b(k)) - static_cast<double>(x(k)))) + 1;
                        x(k) = std::min(x(k), candidate);
                    }
                    x(k) -= 1;
                } else {
                    k -= 1;
                    double r_kp1 = R(k + 1, k + 1);
                    double x_plus_b_minus_z_kp1 = static_cast<double>(x(k + 1)) + b(k + 1) - z(k + 1);
                    dist(k) = dist(k + 1) + r_kp1 * r_kp1 * x_plus_b_minus_z_kp1 * x_plus_b_minus_z_kp1;

                    double sum = 0.0;
                    for (int j = k + 1; j < rank; ++j) {
                        sum += R(k, j) * (static_cast<double>(x(j)) + b(j));
                    }
                    z(k) = -sum / R(k, k);

                    double inner_val = std::max(0.0, (ubound - dist(k)) / (R(k, k) * R(k, k)));
                    x(k) = static_cast<int>(std::floor(std::sqrt(inner_val) + z(k) - b(k)));
                }
            } else {
                k += 1;
                if (k == rank) break;
                x(k) -= 1;
            }
        }
        return vecs;
    }
};

// --- 3. RootSys & Integer Math ---
struct ChamberHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = v.size();
        for(auto& i : v) seed ^= static_cast<size_t>(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class RootSysCpp {
private:
    MatrixXi64 A;
    MatrixXi64 eye;
    MatrixXi64 M;
    RowVectorXi64 base;
    std::vector<RowVectorXi64> pos_roots;
    std::unordered_map<std::vector<int>, MatrixXi64, ChamberHash> cache;

public:
    std::vector<RowVectorXi64> sroots; // Exposed for VSearchCpp to use as initial walls

    // Added optional base_in to fully mirror the Python constructor
    RootSysCpp(MatrixXi64 A_in, const std::vector<RowVectorXi64>& roots, RowVectorXi64 base_in = RowVectorXi64()) 
        : A(std::move(A_in)) {
        
        int rank = A.rows();
        eye = MatrixXi64::Identity(rank, rank);
        
        // 1. Establish the polarization vector (base)
        if (base_in.size() == rank) {
            base = base_in;
        } else base = RowVectorXi64::Zero(rank);
        // Ensure the base vector does not lie on any root hyperplane
        // (i.e., its inner product with every root must be non-zero)
        bool valid_base = false;
        std::srand(A(0, 0)); // Seed with a value from A for reproducibility, can be any deterministic seed
        while (!valid_base) {
            valid_base = true;
            for (const auto& r : roots) {
                if ((base * A * r.transpose())(0, 0) == 0) {
                    valid_base = false;
                    for (int i = 0; i < rank; ++i) base(0, i) = std::rand() % 100000 + 1; // Avoid zero to reduce chance of hitting a wall
                    break; // Restart the check with the new base
                }
            }
        }

        // 2. Filter positive roots based on the polarization vector
        for (const auto& r : roots) {
            int64 dot = (base * A * r.transpose())(0, 0);
            if (dot > 0) {
                pos_roots.push_back(r);
            }
        }

        // 3. Sort positive roots by height (distance to base)
        std::sort(pos_roots.begin(), pos_roots.end(), [this](const RowVectorXi64& r1, const RowVectorXi64& r2) {
            return (this->base * this->A * r1.transpose())(0, 0) < (this->base * this->A * r2.transpose())(0, 0);
        });

        // 4. Extract simple roots (sroots)
        sroots.clear();
        for (const auto& r : pos_roots) {
            bool is_simple = true;
            for (const auto& s : sroots) {
                // A positive root is NOT simple if it has a positive inner product
                // with any already identified simple root (meaning it's decomposable).
                if ((r * A * s.transpose())(0, 0) > 0) {
                    is_simple = false;
                    break;
                }
            }
            if (is_simple) {
                sroots.push_back(r);
            }
        }

        // 5. Precompute M matrix for rapid chamber checking
        if (!pos_roots.empty()) {
            MatrixXi64 pos_roots_mat(pos_roots.size(), rank);
            for (size_t i = 0; i < pos_roots.size(); ++i) {
                pos_roots_mat.row(i) = pos_roots[i];
            }
            M = A * pos_roots_mat.transpose();
        }
    }

    MatrixXi64 reflection(const RowVectorXi64& r) const {
        int64 norm = (r * A * r.transpose())(0, 0);
        return eye - ((2 * A * r.transpose() * r) / norm); 
    }

    std::vector<int> closed_chamber(const RowVectorXi64& v) const {
        RowVectorXi64 prod = v * M;
        std::vector<int> signs(prod.cols());
        for (int i = 0; i < prod.cols(); ++i) {
            signs[i] = (prod(0, i) >= 0) ? 1 : -1;
        }
        return signs;
    }

    MatrixXi64 find_reflection(RowVectorXi64 v) {
        std::vector<int> c = closed_chamber(v);
        bool all_pos = true;
        for (int x : c) if (x < 0) { all_pos = false; break; }
        if (all_pos) return eye;

        if (cache.count(c)) return cache[c];

        MatrixXi64 r_mat = eye;
        std::vector<int> c0 = c;

        while (true) {
            int height = 0;
            for (int x : c) if (x < 0) height++;
            if (height == 0) break;

            bool moved = false;
            for (size_t i = 0; i < pos_roots.size(); ++i) {
                if (c[i] < 0) {
                    MatrixXi64 ref_i = reflection(pos_roots[i]);
                    RowVectorXi64 next_v = v * ref_i;
                    std::vector<int> new_c = closed_chamber(next_v);

                    if (cache.count(new_c)) {
                        MatrixXi64 final_r = r_mat * ref_i * cache[new_c];
                        cache[c0] = final_r;
                        return final_r;
                    }

                    int new_height = 0;
                    for (int x : new_c) if (x < 0) new_height++;

                    if (new_height < height) {
                        c = new_c;
                        r_mat = r_mat * ref_i;
                        v = next_v;
                        moved = true;
                        break;
                    }
                }
            }
            if (!moved) break;
        }
        cache[c0] = r_mat;
        return r_mat;
    }
};

bool is_root_core(const MatrixXi64& A, const RowVectorXi64& r) {
    RowVectorXi64 prod = r * A;
    int64 sq = (prod * r.transpose())(0, 0);
    if (sq == 0) return false;
    
    int64 current_gcd = std::abs(prod(0, 0));
    for (int i = 1; i < prod.cols(); ++i) {
        current_gcd = std::gcd(current_gcd, std::abs(prod(0, i)));
        if (current_gcd == 1) break;
    }
    return (2 * current_gcd) % std::abs(sq) == 0;
}

// --- 4. Main VSearch Class ---
class VSearchCpp {
private:
    MatrixXi64 A;
    MatrixXi64 C_int;
    Eigen::MatrixXd C_double;
    Eigen::RowVectorXd b;
    int rank;
    int exp;
    double s;
    
    std::shared_ptr<RootSysCpp> R;
    std::map<Rational, std::vector<RowVectorXi64>> roots;
    std::map<Rational, std::vector<RowVectorXi64>> walls;
    
    std::atomic<int> h_counter{0};

public:
    VSearchCpp(MatrixXi64 A_in, int exp_in) : A(std::move(A_in)), rank(A.rows()), exp(exp_in) {
        C_int = A.block(1, 1, rank - 1, rank - 1);
        C_double = C_int.cast<double>();
        
        Eigen::RowVectorXd A_row0_double = A.block(0, 1, 1, rank - 1).cast<double>();
        b = A_row0_double * C_double.inverse();
        s = static_cast<double>(A(0, 0)) - (b * C_double * b.transpose())(0, 0);
        
        if (s <= 0) throw std::runtime_error("Error initializing basis: s is non-positive");

        // Init Chamber (height 0)
        std::vector<RowVectorXi64> initial_roots;
        FPSearchSeq fps(-C_double, Eigen::VectorXd::Zero(rank - 1), 0.0, 2.0 * exp + 0.5);
        auto vecs = fps.search_all();
        
        for (const auto& u : vecs) {
            RowVectorXi64 v = RowVectorXi64::Zero(rank);
            for (size_t i = 0; i < u.size(); ++i) v(0, i + 1) = u[i];
            if (is_root_core(A, v)) initial_roots.push_back(v);
        }
        
        R = std::make_shared<RootSysCpp>(A, initial_roots);
    }

    void run(int root_batch, bool use_reflections, int num_threads) {
        if (num_threads <= 0) num_threads = std::thread::hardware_concurrency();

        std::atomic<int> valid_roots_found{0};
        std::vector<std::future<std::vector<RowVectorXi64>>> futures;

        // Release Python GIL
        py::gil_scoped_release release;

        for (int i = 0; i < num_threads; ++i) {
            futures.push_back(std::async(std::launch::async, [this, root_batch, use_reflections, &valid_roots_found]() {
                std::vector<RowVectorXi64> local_roots;
                
                while (valid_roots_found.load() < root_batch) {
                    int current_h = h_counter.fetch_add(1) + 1;
                    
                    Eigen::VectorXd b_h = current_h * b.transpose();
                    double bound = s * current_h * current_h;
                    double lbound = 0.5 + bound;
                    double ubound = 2.0 * exp + 0.5 + bound;

                    FPSearchSeq fps(-C_double, b_h, lbound, ubound);
                    auto vecs = fps.search_all();

                    for (const auto& u : vecs) {
                        RowVectorXi64 v = RowVectorXi64::Zero(rank);
                        v(0, 0) = current_h;
                        for (size_t j = 0; j < u.size(); ++j) v(0, j + 1) = u[j];

                        int64 sq = (v * A * v.transpose())(0, 0);
                        if (sq > 0 || !is_root_core(A, v)) continue;

                        if (use_reflections) {
                            // find_reflection modifies cache, needs a lock or local isolation.
                            // To maximize speed, each thread computes reflection on the fly
                            v = v * R->find_reflection(v); 
                        }
                        
                        local_roots.push_back(v);
                        valid_roots_found.fetch_add(1);
                    }
                }
                return local_roots;
            }));
        }

        // Merge thread results
        for (auto& fut : futures) {
            auto partial_roots = fut.get();
            for (const auto& v : partial_roots) {
                int64 sq = std::abs((v * A * v.transpose())(0, 0));
                int64 p = v(0, 0) * v(0, 0);
                Rational key(p, sq);
                roots[key].push_back(v);
            }
        }

        // Reacquire GIL implicitly when scope ends
    }

    void update_walls() {
        std::map<Rational, std::vector<RowVectorXi64>> new_walls;
        std::vector<RowVectorXi64> wall_list = R->sroots;

        // roots map is implicitly ordered by Rational distance due to std::map
        for (const auto& pair : roots) {
            if (pair.first.p == 0) continue; // Skip distance 0
            
            for (const auto& r : pair.second) {
                bool active = true;
                for (const auto& w : wall_list) {
                    if ((w * A * r.transpose())(0, 0) < 0) {
                        active = false;
                        break;
                    }
                }
                if (active) {
                    new_walls[pair.first].push_back(r);
                    wall_list.push_back(r);
                }
            }
        }
        
        walls = new_walls;
        roots = new_walls;
    }

    // Expose current walls back to Python
    std::vector<std::vector<int64>> get_walls() const {
        std::vector<std::vector<int64>> out;
        for (const auto& pair : walls) {
            for (const auto& v : pair.second) {
                std::vector<int64> vec_out(rank);
                for(int i=0; i<rank; ++i) vec_out[i] = v(0, i);
                out.push_back(vec_out);
            }
        }
        return out;
    }
};

PYBIND11_MODULE(vsearch_cpp, m) {
    m.doc() = "Multithreaded C++ exact-integer backend for VSearch and RootSys";
    
    // --- Expose RootSysCpp ---
    // We use std::shared_ptr so Pybind11 can safely manage memory if VSearchCpp 
    // and Python both hold references to the same RootSysCpp instance.
    py::class_<RootSysCpp, std::shared_ptr<RootSysCpp>>(m, "RootSysCpp")
        .def(py::init<MatrixXi64, std::vector<RowVectorXi64>, RowVectorXi64>(),
             py::arg("A"), py::arg("roots"), py::arg("base") = RowVectorXi64())
        .def("find_reflection", &RootSysCpp::find_reflection, py::arg("v"))
        .def("reflection", &RootSysCpp::reflection, py::arg("r"))
        .def_readonly("sroots", &RootSysCpp::sroots, "The calculated simple roots (basis)");

    // --- Expose VSearchCpp ---
    py::class_<VSearchCpp>(m, "VSearchCpp")
        .def(py::init<MatrixXi64, int>(), py::arg("A"), py::arg("exp"))
        .def("run", &VSearchCpp::run, py::arg("root_batch") = 10000, py::arg("use_reflections") = true, py::arg("num_threads") = 0)
        .def("update_walls", &VSearchCpp::update_walls)
        .def("get_walls", &VSearchCpp::get_walls);
}
