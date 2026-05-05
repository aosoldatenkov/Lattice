#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <cmath>
#include <set>
#include <map>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <gmpxx.h>
#include <iostream>
#include "fp_search.cpp"

namespace py = pybind11;

// --- 1. Pybind11 Caster for Python int <-> mpz_class ---
// Python 'int' has arbitrary precision, numpy uses 'object' dtype for it.
namespace pybind11 { namespace detail {
    template <> struct type_caster<mpz_class> {
    public:
        PYBIND11_TYPE_CASTER(mpz_class, _("int"));

        // Python -> C++
        bool load(handle src, bool) {
            if (!src) return false;
            // Convert Python int to string, then parse into GMP
            value = mpz_class(py::str(src).cast<std::string>());
            return true;
        }

        // C++ -> Python
        static handle cast(mpz_class src, return_value_policy, handle) {
            // 1. Convert GMP integer to a base-10 std::string
            std::string s = src.get_str();
            
            // 2. Use the Python C API to safely create a Python int from the string
            PyObject* py_obj = PyLong_FromString(s.c_str(), nullptr, 10);
            
            // 3. Check for parsing errors
            if (!py_obj) {
                throw py::error_already_set();
            }
            
            // 4. Safely wrap the new PyObject in Pybind11 and release ownership to Python
            return py::reinterpret_steal<py::object>(py_obj).release();
        }
    };
}}

// --- 2. Eigen NumTraits for mpz_class ---
namespace Eigen {
    template<> struct NumTraits<mpz_class> : GenericNumTraits<mpz_class> {
        typedef mpz_class Real;
        typedef mpz_class NonInteger;
        typedef mpz_class Nested;
        enum {
            IsComplex = 0,
            IsInteger = 1,
            ReadCost = 1,
            AddCost = 1,
            MulCost = 2,
            IsSigned = 1,
            RequireInitialization = 1
        };
    };
}

// --- 3. Matrix Type Aliases ---
using int_class = long;
// using int_class = mpz_class;
using MatrixXi64 = Eigen::Matrix<int_class, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXi64 = Eigen::Matrix<int_class, 1, Eigen::Dynamic>;

int int_class_to_int(const int_class& x) {
    return static_cast<int>(x);
    // return x.get_si();
}

double int_class_to_double(const int_class& x) {
    return static_cast<double>(x);
    // return x.get_d();
}

// --- 1. Rational Number Struct for exact sorting of distances ---
struct Rational {
    int_class p, q;
    Rational(int_class num, int_class den) : p(num), q(den) {
        if (q < 0) { p = -p; q = -q; }
    }
    bool operator<(const Rational& o) const {
        // p1/q1 < p2/q2 => p1*q2 < p2*q1 (since q > 0)
        return p * o.q < o.p * q;
    }
    bool operator==(const Rational& o) const {
        return p * o.q == o.p * q;
    }
    double to_double() const {
        mpq_class fraction(p, q);
        fraction.canonicalize();
        return fraction.get_d();
    }
};

// --- 2. RootSys & Integer Math ---
struct SignedPermutation {
    std::set<std::pair<int, int>> minus;
    std::set<std::pair<int, int>> plus;
};

class RootSysCpp {
private:
    MatrixXi64 A;
    MatrixXi64 eye;
    MatrixXi64 M;
    RowVectorXi64 base;
    std::vector<RowVectorXi64> pos_roots;
    std::vector<MatrixXi64> refl;
    std::vector<SignedPermutation> perm;

    void init_reflections() {
        std::vector<size_t> active;
        for (size_t i = 0; i < pos_roots.size(); ++i) {
            MatrixXi64 r = reflection(pos_roots[i]);
            refl.push_back(r);
            active.clear();
            RowVectorXi64 prod = pos_roots[i] * A;
            for (size_t j = 0; j < pos_roots.size(); ++j) {
                if ((prod * pos_roots[j].transpose())(0, 0) != 0) active.push_back(j);
            }
            SignedPermutation p;
            for (size_t j : active) {
                RowVectorXi64 u = pos_roots[j] * r;
                bool found = false;
                for (size_t k = 0; k < pos_roots.size(); ++k) {
                    if (u == pos_roots[k]) {
                        p.plus.insert(std::pair<int, int>(j, k));
                        // std::cout << i << ": " << j << "->" << k << std::endl;
                        found = true;
                        break;
                    }
                    if (u == -pos_roots[k]) {
                        p.minus.insert(std::pair<int, int>(j, k));
                        // std::cout << i << ": " << j << "-> -" << k << std::endl;
                        found = true;
                        break;
                    }
                }
                if (!found) throw std::runtime_error("Broken root system");
            }
            perm.push_back(p);
        }
    }
    
public:
    std::vector<RowVectorXi64> sroots;
    
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
        std::srand(int_class_to_int(A(0, 0))); // Seed with a value from A for reproducibility, can be any deterministic seed
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
            int_class dot = (base * A * r.transpose())(0, 0);
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
            int_class sign = (r * A * r.transpose())(0, 0) > 0 ? 1 : -1;
            bool is_simple = true;
            for (const auto& s : sroots) {
                // A positive root is NOT simple if it has a positive inner product
                // with any already identified simple root (assuming A is positive definite).
                if ((sign * r * A * s.transpose())(0, 0) > 0) {
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
        else {
            M = MatrixXi64::Zero(rank, 1); // No positive roots, zero matrix
        }

        init_reflections();
    }

    MatrixXi64 reflection(const RowVectorXi64& r) const {
        int_class norm = (r * A * r.transpose())(0, 0);
        return eye - ((2 * A * r.transpose() * r) / norm); 
    }

    std::vector<int> closed_chamber(RowVectorXi64 v) {
        RowVectorXi64 prod = v * M;
        std::vector<int> signs(prod.cols());
        for (int i = 0; i < prod.cols(); ++i) {
            signs[i] = (prod(0, i) >= 0) ? 1 : -1;
        }
        return signs;
    }

    RowVectorXi64 reflect(RowVectorXi64 v) {
        std::vector<int> c = closed_chamber(v);
        bool all_pos = true;
        for (int x : c) if (x < 0) { all_pos = false; break; }
        RowVectorXi64 new_v = v;
        while (!all_pos) {
            std::vector<int> new_c = c;
            for (size_t i = 0; i < pos_roots.size(); ++i) {
                int count = 0;
                for (auto p : perm[i].minus) {
                    if (p.first == p.second) count += c[p.first];
                    else count += c[p.first] + c[p.second];
                }
                if (count < 0) {
                    for (auto p : perm[i].plus) {
                        new_c[p.first] = c[p.second];
                        new_c[p.second] = c[p.first];
                    }
                    for (auto p : perm[i].minus) {
                        new_c[p.first] = -c[p.second];
                        new_c[p.second] = -c[p.first];
                    }
                    new_v = new_v * refl[i];
                    break;
                }
            }
            c = new_c;
            all_pos = true;
            for (int x : c) if (x < 0) { all_pos = false; break; }
        }
        return new_v;
    }
};


int_class mpz_gcd(int_class a, int_class b) {
    a = abs(a); b = abs(b);
    while (b != 0) {
        int_class t = b;
        b = a % b;
        a = t;
    }
    return a;
}

bool is_root_core(const MatrixXi64& A, const RowVectorXi64& r) {
    RowVectorXi64 prod = r * A;
    int_class sq = (prod * r.transpose())(0, 0);
    if (sq >= 0) return false;
    
    int_class current_gcd = abs(prod(0, 0));
    for (int i = 1; i < prod.cols(); ++i) {
        current_gcd = mpz_gcd(current_gcd, abs(prod(0, i)));
        if (current_gcd == 1) break;
    }
    return (2 * current_gcd) % abs(sq) == 0;
}

// --- 3. Main VSearch Class ---
class VSearchCpp {
private:
    MatrixXi64 A;
    MatrixXi64 C_int;
    RowVectorXi64 base;
    Eigen::MatrixXd C_double;
    int rank;
    int exp;
    mpq_class s;
    int num_threads;
    
    std::shared_ptr<RootSysCpp> R;
    std::map<Rational, std::vector<RowVectorXi64>> roots;
    std::map<Rational, std::vector<RowVectorXi64>> walls;
    
    int_class h_counter;
    std::vector<std::pair<int_class, FPSearch>> fps_workers;
    std::pair<int_class, FPSearch> init_fps() {
        h_counter += 1;
        Eigen::VectorXd b_h(rank - 1);
        for (int i = 0; i < rank - 1; ++i) {
            mpq_class val = mpq_class(-base(0, i + 1) * h_counter, base(0, 0));
            val.canonicalize();
            b_h(i) = val.get_d();
        }
        mpq_class bound = s * h_counter * h_counter;
        double lbound = bound.get_d();
        double ubound = 2.1 * exp + bound.get_d();
        FPSearch fps(-C_double, b_h, lbound, ubound);
        return std::pair<int_class, FPSearch>(h_counter, std::move(fps));
    }

public:
    VSearchCpp(MatrixXi64 A_in, RowVectorXi64 base_in, int exp_in, int num_threads_in) :
                A(std::move(A_in)), base(std::move(base_in)), rank(A.rows()), exp(exp_in), num_threads(num_threads_in), h_counter(0) {
        if (num_threads <= 0) num_threads = std::thread::hardware_concurrency();

        C_int = A.block(1, 1, rank - 1, rank - 1);
        C_double = C_int.unaryExpr([](const int_class& x) { return int_class_to_double(x); });
        
        RowVectorXi64 A_row0 = A.block(0, 0, 1, rank);
        s = mpq_class((A_row0 * base.transpose())(0, 0), base(0, 0));
        s.canonicalize();

        if (s <= 0) {
            std::cout << "s = " << s.get_d() << " prod = " << (A_row0 * base.transpose())(0, 0) << " base(0, 0) = " << base(0, 0) << std::endl;
            throw std::runtime_error("Error initializing basis: s is non-positive");
        }
        
        // Init Chamber (height 0)
        std::vector<RowVectorXi64> initial_roots;
        FPSearch fps(-C_double, Eigen::VectorXd::Zero(rank - 1), 0.0, 2.1 * exp + 0.5);
        auto vecs = fps.search_all();
        
        for (const auto& u : vecs) {
            RowVectorXi64 v = RowVectorXi64::Zero(rank);
            for (size_t i = 0; i < u.size(); ++i) v(0, i + 1) = u[i];
            if (is_root_core(A, v)) initial_roots.push_back(v);
        }
        
        R = std::make_shared<RootSysCpp>(A, initial_roots);

        for (int i = 0; i < num_threads; ++i) {
            fps_workers.push_back(init_fps());
        }
    }

    void run(int root_batch, bool use_reflections) {
        std::atomic<int> valid_roots_found{0};
        std::vector<std::future<std::vector<RowVectorXi64>>> futures;

        for (int i = 0; i < num_threads; ++i) {
            if (fps_workers[i].second.exhausted()) fps_workers[i] = init_fps();
        }

        // Release Python GIL
        py::gil_scoped_release release;

        for (int i = 0; i < num_threads; ++i) {
            std::pair<int_class, FPSearch>* fps_worker = &fps_workers[i];
            futures.push_back(std::async(std::launch::async, [this, i, root_batch, use_reflections, &valid_roots_found, fps_worker]() {
                std::vector<RowVectorXi64> local_roots;
                
                while (valid_roots_found.load() < root_batch) {
                    
                    if (fps_worker->second.exhausted()) break;
                    auto vecs = fps_worker->second.batch_search(10000);

                    for (const auto& u : vecs) {
                        RowVectorXi64 v = RowVectorXi64::Zero(rank);
                        v(0, 0) = fps_worker->first;
                        for (size_t j = 0; j < u.size(); ++j) v(0, j + 1) = u[j];

                        if (!is_root_core(A, v)) continue;

                        if (use_reflections) {
                            v = R->reflect(v); 
                        } else {
                            bool all_pos = true;
                            for (const auto& w : R->sroots) {
                                if ((w * A * v.transpose())(0, 0) < 0) {
                                    all_pos = false;
                                    break;
                                }
                            }
                            if (!all_pos) continue;
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
                int_class sq = abs((v * A * v.transpose())(0, 0));
                int_class p = v(0, 0) * v(0, 0);
                Rational key(p, sq);
                roots[key].push_back(v);
            }
        }

        // Reacquire GIL implicitly when scope ends
    }

    void update_walls() {
        std::map<Rational, std::vector<RowVectorXi64>> new_walls;
        std::vector<RowVectorXi64> wall_list; // = R->sroots;

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
    std::vector<std::vector<int_class>> get_walls() const {
        std::vector<std::vector<int_class>> out;
        for (const auto& w : R->sroots) {
            std::vector<int_class> vec_out(rank);
            for(int i=0; i<rank; ++i) vec_out[i] = w(0, i);
            out.push_back(vec_out);
        }
        for (const auto& pair : walls) {
            for (const auto& v : pair.second) {
                std::vector<int_class> vec_out(rank);
                for(int i=0; i<rank; ++i) vec_out[i] = v(0, i);
                out.push_back(vec_out);
            }
        }
        return out;
    }
};

// --- Conversion Helpers ---
MatrixXi64 to_matrix(const std::vector<std::vector<int_class>>& v) {
    if (v.empty()) return MatrixXi64(0, 0);
    MatrixXi64 m(v.size(), v[0].size());
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v[0].size(); ++j)
            m(i, j) = v[i][j];
    return m;
}

RowVectorXi64 to_row_vector(const std::vector<int_class>& v) {
    if (v.empty()) return RowVectorXi64(1, 0);
    RowVectorXi64 m(1, v.size());
    for (size_t i = 0; i < v.size(); ++i) m(0, i) = v[i];
    return m;
}

std::vector<int_class> to_std_vector(const RowVectorXi64& m) {
    std::vector<int_class> v(m.cols());
    for (int i = 0; i < m.cols(); ++i) v[i] = m(0, i);
    return v;
}

std::vector<std::vector<int_class>> to_std_matrix(const MatrixXi64& m) {
    std::vector<std::vector<int_class>> v(m.rows(), std::vector<int_class>(m.cols()));
    for(int i=0; i<m.rows(); ++i)
        for(int j=0; j<m.cols(); ++j)
            v[i][j] = m(i,j);
    return v;
}


PYBIND11_MODULE(vsearch_cpp, m) {
    m.doc() = "Multithreaded C++ exact GMP backend for VSearch and RootSys";

    // --- Expose RootSysCpp ---
    py::class_<RootSysCpp, std::shared_ptr<RootSysCpp>>(m, "RootSysCpp")
        .def(py::init([](const std::vector<std::vector<int_class>>& A_in, 
                         const std::vector<std::vector<int_class>>& roots_in, 
                         const std::vector<int_class>& base_in) {
            std::vector<RowVectorXi64> eigen_roots;
            for (const auto& r : roots_in) eigen_roots.push_back(to_row_vector(r));
            return std::make_shared<RootSysCpp>(to_matrix(A_in), eigen_roots, to_row_vector(base_in));
        }), py::arg("A"), py::arg("roots"), py::arg("base") = std::vector<int_class>())
        
        .def("reflect", [](RootSysCpp& self, const std::vector<int_class>& v) {
            return to_std_vector(self.reflect(to_row_vector(v)));
        }, py::arg("v"))

        .def("closed_chamber", [](RootSysCpp& self, const std::vector<int_class>& v) {
            return self.closed_chamber(to_row_vector(v));
        }, py::arg("v"))
        
        .def("reflection", [](RootSysCpp& self, const std::vector<int_class>& r) {
            return to_std_matrix(self.reflection(to_row_vector(r)));
        }, py::arg("r"))
        
        .def_property_readonly("sroots", [](const RootSysCpp& self) {
            std::vector<std::vector<int_class>> out;
            for (const auto& r : self.sroots) out.push_back(to_std_vector(r));
            return out;
        });

    // --- Expose VSearchCpp ---
    py::class_<VSearchCpp>(m, "VSearchCpp")
        .def(py::init([](const std::vector<std::vector<int_class>>& A_in, const std::vector<int_class>& base_in, const int exp, int num_threads) {
            return std::make_unique<VSearchCpp>(to_matrix(A_in), to_row_vector(base_in), exp, num_threads);
        }), py::arg("A"), py::arg("base"), py::arg("exp"), py::arg("num_threads") = 1)
        
        .def("run", &VSearchCpp::run, py::arg("root_batch") = 10000, py::arg("use_reflections") = true)
        .def("update_walls", &VSearchCpp::update_walls)
        .def("get_walls", &VSearchCpp::get_walls);
}