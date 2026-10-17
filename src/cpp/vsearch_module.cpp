// vsearch_cpp: the root search of Vinberg's algorithm (see docs/Vinberg.md).
//
// VSearchCpp enumerates the negative roots v = (h, x) of a hyperbolic lattice by height h, in the basis
// e_0, e_1, ..., e_{n-1} built by lattice.vinberg.Vinberg (e_1, ..., e_{n-1} span the orthogonal complement
// of the base point), using the Fincke-Pohst search of fp_search.hpp on each height, and selects the walls
// of the chamber containing the base point. RootSysCpp handles the finite root system of the roots
// orthogonal to the base point (height 0). All lattice arithmetic is exact, in 64-bit integers (int_class),
// with magnitudes bounded in advance (see the overflow safeguards below).

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
#include <stdexcept>
#include <string>
#include <gmpxx.h>
#include <iostream>
#include "fp_search.hpp"

namespace py = pybind11;

// --- 1. Pybind11 caster for Python int <-> mpz_class ---
// Python 'int' has arbitrary precision, numpy uses 'object' dtype for it. Only used when
// int_class is set to mpz_class (see section 3); the current int_class is long.
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
// Lets Eigen matrices have mpz_class entries (needed only with int_class = mpz_class).
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

// --- 3. Integer type and matrix type aliases ---
// The integer type of all lattice computations: 64-bit long (fast); mpz_class (GMP) is the exact,
// slower alternative, with the commented-out lines of the conversions below.
using int_class = long;
// using int_class = mpz_class;
using MatrixXi64 = Eigen::Matrix<int_class, Eigen::Dynamic, Eigen::Dynamic>;
using RowVectorXi64 = Eigen::Matrix<int_class, 1, Eigen::Dynamic>;

// Conversion to int (truncating); only used to seed the random number generator
int int_class_to_int(const int_class& x) {
    return static_cast<int>(x);
    // return x.get_si();
}

// Conversion to double, for the floating-point Fincke-Pohst search
double int_class_to_double(const int_class& x) {
    return static_cast<double>(x);
    // return x.get_d();
}

// --- 4. Rational keys for the exact ordering of the roots by distance ---
// A root v of height h has key h^2 / |(v, v)|, which is proportional to sinh^2 of the distance from the base
// point to its mirror. The fraction is not reduced: comparisons use cross-multiplication, whose magnitude is
// covered by the overflow safeguards.
struct Rational {
    int_class p, q;
    // The fraction num / den, normalized to a positive denominator
    Rational(int_class num, int_class den) : p(num), q(den) {
        if (q < 0) { p = -p; q = -q; }
    }
    bool operator<(const Rational& o) const {
        // p1/q1 < p2/q2 => p1*q2 < p2*q1 (since q > 0)
        return p * o.q < o.p * q;
    }
    // Equality of the values (not of the representations)
    bool operator==(const Rational& o) const {
        return p * o.q == o.p * q;
    }
    // The value as a double (exact rational rounding via GMP)
    double to_double() const {
        mpq_class fraction(p, q);
        fraction.canonicalize();
        return fraction.get_d();
    }
};

// --- 5. The finite root system at the base point ---

// The action of a reflection on the positive roots: the pairs (j, k) with s(r_j) = r_k (plus)
// and s(r_j) = -r_k (minus). Positive roots orthogonal to the reflected root are fixed and not listed.
struct SignedPermutation {
    std::set<std::pair<int, int>> minus;
    std::set<std::pair<int, int>> plus;
};

// A finite root system: the roots in a definite subspace, here the roots orthogonal to the base point
// of Vinberg's algorithm (the walls through the base point). A polarization vector (base) splits the roots
// into positive and negative ones and determines the simple roots sroots, i.e. a chamber of the root system.
class RootSysCpp {
private:
    MatrixXi64 A;                          // the Gram matrix of the ambient lattice
    MatrixXi64 eye;                        // the identity matrix
    MatrixXi64 M;                          // A times the positive roots as columns: v * M lists the (v, r)
    RowVectorXi64 base;                    // the polarization vector
    std::vector<RowVectorXi64> pos_roots;  // the positive roots, sorted by their product with base
    std::vector<MatrixXi64> refl;          // the reflections in the positive roots (acting on row vectors)
    std::vector<SignedPermutation> perm;   // the action of these reflections on the positive roots

    // Computes the reflection matrices of the positive roots and their signed permutations of the
    // positive roots (throws if the image of a positive root is not a root: the list is not a root system)
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
    std::vector<RowVectorXi64> sroots;     // the simple roots of the chamber determined by base

    // Builds the root system of the given roots (all the roots of a finite root system, both signs) in the
    // lattice with Gram matrix A_in. base_in is the polarization vector; if it is missing, or orthogonal
    // to some root, it is replaced by a pseudo-random vector (deterministic seed).
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

        // 3. Sort positive roots by their product with the polarization vector
        std::sort(pos_roots.begin(), pos_roots.end(), [this](const RowVectorXi64& r1, const RowVectorXi64& r2) {
            return (this->base * this->A * r1.transpose())(0, 0) < (this->base * this->A * r2.transpose())(0, 0);
        });

        // 4. Extract simple roots (sroots)
        sroots.clear();
        for (const auto& r : pos_roots) {
            int_class sign = (r * A * r.transpose())(0, 0) > 0 ? 1 : -1;
            bool is_simple = true;
            for (const auto& s : sroots) {
                // A positive root is NOT simple if it has a positive Euclidean inner product with any already
                // identified simple root (sign makes the form positive definite: the roots here have negative norm).
                // A non-simple positive root has a simple root in its support with a positive product,
                // and that simple root comes earlier in the order.
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

    // The matrix of the reflection x -> x - 2 (x, r) / (r, r) r acting on row vectors (x -> x * matrix).
    // The division is exact for a root r.
    MatrixXi64 reflection(const RowVectorXi64& r) const {
        int_class norm = (r * A * r.transpose())(0, 0);
        return eye - ((2 * A * r.transpose() * r) / norm); 
    }

    // The signs of the products (v, r) with the positive roots r (+1 for a product >= 0, -1 otherwise);
    // v lies in the closed chamber of the simple roots iff all the signs are +1
    std::vector<int> closed_chamber(RowVectorXi64 v) {
        RowVectorXi64 prod = v * M;
        std::vector<int> signs(prod.cols());
        for (int i = 0; i < prod.cols(); ++i) {
            signs[i] = (prod(0, i) >= 0) ? 1 : -1;
        }
        return signs;
    }

    // Maps v into the closed chamber of the simple roots by a sequence of reflections in positive roots,
    // and returns the image. The signs of the products with the positive roots are not recomputed but
    // tracked through the signed permutations: a reflection is applied when, among the positive roots it
    // makes negative, the current signs sum to a negative number, so that the number of negative signs drops.
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


// --- 6. The root test ---

// The non-negative GCD of two integers (for int_class; the name dates from the GMP version)
int_class mpz_gcd(int_class a, int_class b) {
    a = abs(a); b = abs(b);
    while (b != 0) {
        int_class t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Checks if r is a negative root: a primitive vector with (r, r) < 0
// such that (r, r) divides 2 (r, x) for all x (as Lattice.is_root in Python)
bool is_root_core(const MatrixXi64& A, const RowVectorXi64& r) {
    int_class content = 0;
    for (int i = 0; i < r.cols(); ++i) {
        content = mpz_gcd(content, r(0, i));
        if (content == 1) break;
    }
    if (content != 1) return false;

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



// --- 7. Overflow safeguards ---
// int_class is a 64-bit long. Instead of checking every operation, the magnitudes are bounded
// in advance (once per height) in long double, keeping a factor of 2 of headroom.
const long double INT_LIMIT = std::ldexp(1.0L, 62);

// Throws OverflowError (in Python) if a bound on the magnitude of some integers reaches INT_LIMIT
void check_limit(long double value, const std::string& what) {
    if (!(value < INT_LIMIT)) throw std::overflow_error("vsearch_cpp: possible 64-bit overflow in " + what);
}

// A bound on |u A v^T| over all u, v with |u_i| <= U[i], |v_j| <= V[j]. If all V[j] >= 1, it also bounds
// the entries of u A and all the partial sums
long double product_bound(const MatrixXi64& A, const std::vector<long double>& U, const std::vector<long double>& V) {
    long double total = 0;
    for (int j = 0; j < A.cols(); ++j) {
        long double col = 0;
        for (int i = 0; i < A.rows(); ++i) col += U[i] * std::fabs(static_cast<long double>(A(i, j)));
        total += col * V[j];
    }
    return total;
}

// --- 8. The root search ---

// The search for the negative roots of a hyperbolic lattice of signature (1, n), by height.
// A is the Gram matrix in a basis e_0, ..., e_{n-1} such that e_1, ..., e_{n-1} span the orthogonal complement
// of the base point (base, given by its coordinates), so that C = (e_i, e_j)_{i,j >= 1} is negative definite.
// A vector v = h e_0 + x has height h, and (v, v) = s h^2 - (x + b)^T (-C) (x + b) with s = (e_0, base) / base_0
// and b = -h base_{1..} / base_0; each height is searched by Fincke-Pohst in the shell where
// 1 <= -(v, v) < bound (bound = 2 exp + 0.5 for roots). See docs/Vinberg.md, section 3.
//
// In chamber mode, the roots orthogonal to base (height 0) give a finite root system, whose chamber is chosen
// by the chamber vector; the roots of positive height are kept if they have non-negative products with its
// simple roots, and update_walls() selects the walls of the chamber of Vinberg's algorithm among them.
// In vector mode, the search just lists the vectors (roots only, if only_roots) by height, starting from 0.
class VSearchCpp {
private:
    MatrixXi64 A;                 // the Gram matrix in the basis e_0, ..., e_{n-1}
    MatrixXi64 C_int;             // its block (e_i, e_j), i, j >= 1 (negative definite)
    RowVectorXi64 base;           // the coordinates of the base point
    Eigen::MatrixXd C_double;     // C_int in floating point, for Fincke-Pohst
    int rank;
    double bound;                 // the search covers -bound <= (v, v), see init_fps
    mpq_class s;                  // (e_0, base) / base_0 = (base, base) / base_0^2 > 0
    int num_threads;              // the number of worker threads, each searching one height at a time
    bool chamber_mode;            // see the class comment
    bool only_roots;              // in vector mode: list only the roots

    std::shared_ptr<RootSysCpp> R;                          // the root system at height 0
    std::map<Rational, std::vector<RowVectorXi64>> roots;   // the roots found, by distance key
    std::map<Rational, std::vector<RowVectorXi64>> walls;   // the accepted walls, by distance key
    std::vector<RowVectorXi64> vecs;                        // vector mode: the vectors found since get_vecs()

    int_class h_counter;                                    // the last height assigned to a worker
    std::vector<std::pair<int_class, FPSearch>> fps_workers; // (height, its Fincke-Pohst search) per worker

    Eigen::VectorXd Q_inv_diag;            // the diagonal of (-C)^-1
    std::vector<long double> coord_bound;  // bounds on the coordinates of all the vectors searched so far

    // Overflow safeguard, run once per height h before its search. On the searched ellipsoid,
    // |x_i + b_i| <= sqrt(ubound (Q^-1)_ii) for Q = -C, so coord_bound bounds the coordinates of all
    // the vectors found so far, and every product computed from them is bounded in advance.
    void update_coord_bound(int_class h, const Eigen::VectorXd& b_h, double ubound) {
        coord_bound[0] = std::max(coord_bound[0], static_cast<long double>(std::abs(h)) + 1);
        for (int i = 1; i < rank; ++i) {
            long double r = std::fabs(b_h(i - 1)) + std::sqrt(std::max(0.0, ubound) * Q_inv_diag(i - 1));
            coord_bound[i] = std::max(coord_bound[i], r * (1 + 1e-6L) + 1);
        }
        std::string where = "height " + std::to_string(h);
        // Products (u, v) of found vectors (is_root_core, the chamber test, update_walls)
        // and the entries 2 (e_i, r) r_j of the reflections
        check_limit(2 * product_bound(A, coord_bound, coord_bound), "the products of vectors at " + where);
        // The distance keys h^2 / |(v, v)| of roots, |(v, v)| <= bound, are compared by cross-multiplication
        check_limit(coord_bound[0] * coord_bound[0] * (bound + 1), "the distance keys at " + where);
    }

    // Assigns the next height to a worker: sets up the Fincke-Pohst search of that height
    std::pair<int_class, FPSearch> init_fps() {
        h_counter += 1;
        Eigen::VectorXd b_h(rank - 1);
        for (int i = 0; i < rank - 1; ++i) {
            // base(0, i + 1) * h_counter is computed with GMP, so it cannot overflow
            mpq_class val(mpz_class(-base(0, i + 1)) * h_counter, mpz_class(base(0, 0)));
            val.canonicalize();
            b_h(i) = val.get_d();
        }
        // The search covers s h^2 + lbound < (x + b)^T (-C) (x + b) <= s h^2 + ubound, where
        // (x + b)^T (-C) (x + b) = s h^2 - (v, v) for v = (h, x), and (v, v) is an integer.
        // Roots have -bound < -2 exp <= (v, v) <= -1, so the bounds s h^2 + 0.5 and s h^2 + bound
        // keep a margin of 0.5 from all attainable values; for all vectors (only_roots = false),
        // the lower bound s h^2 - 0.5 admits (v, v) <= 0.
        mpq_class mpq_bound = s * h_counter * h_counter;
        double lbound = mpq_bound.get_d() + (only_roots ? 0.5 : -0.5);
        double ubound = bound + mpq_bound.get_d();
        update_coord_bound(h_counter, b_h, ubound);
        FPSearch fps(-C_double, b_h, lbound, ubound);
        return std::pair<int_class, FPSearch>(h_counter, std::move(fps));
    }

    // Chamber mode of run(): the workers search their heights in parallel (without the GIL) until output_size
    // roots with non-negative products with the simple roots at height 0 have been found, or every worker has
    // exhausted its height. The roots are added to the map roots by distance key; returns their number.
    int run_chamber_mode(int output_size, int batch_size) {
        std::atomic<int> valid_roots_found{0};
        std::vector<std::future<std::vector<RowVectorXi64>>> futures;

        for (int i = 0; i < num_threads; ++i) {
            if (fps_workers[i].second.exhausted()) fps_workers[i] = init_fps();
        }

        // Release Python GIL
        py::gil_scoped_release release;

        for (int i = 0; i < num_threads; ++i) {
            std::pair<int_class, FPSearch>* fps_worker = &fps_workers[i];
            futures.push_back(std::async(std::launch::async, [this, i, output_size, batch_size, &valid_roots_found, fps_worker]() {
                std::vector<RowVectorXi64> local_roots;
                
                while (valid_roots_found.load() < output_size) {
                    
                    if (fps_worker->second.exhausted()) break;
                    auto vecs = fps_worker->second.batch_search(batch_size);

                    for (const auto& u : vecs) {
                        RowVectorXi64 v = RowVectorXi64::Zero(rank);
                        v(0, 0) = fps_worker->first;
                        for (size_t j = 0; j < u.size(); ++j) v(0, j + 1) = u[j];

                        if (!is_root_core(A, v)) continue;

                        bool all_pos = true;
                        for (const auto& w : R->sroots) {
                            if ((w * A * v.transpose())(0, 0) < 0) {
                                all_pos = false;
                                break;
                            }
                        }
                        if (!all_pos) continue;
                    
                        local_roots.push_back(v);
                        valid_roots_found.fetch_add(1);
                    }
                }
                return local_roots;
            }));
        }

        // Merge thread results
        int count = 0;
        for (auto& fut : futures) {
            auto partial_roots = fut.get();
            for (const auto& v : partial_roots) {
                int_class sq = abs((v * A * v.transpose())(0, 0));
                int_class p = v(0, 0) * v(0, 0);
                Rational key(p, sq);
                roots[key].push_back(v);
                count++;
            }
        }

        // Reacquire GIL implicitly when scope ends
        return count;
    }

    // Vector mode of run(): as run_chamber_mode, but collects the vectors (only the roots, if only_roots)
    // in vecs, without the chamber test; returns their number
    int run_vec_mode(int output_size, int batch_size) {
        std::atomic<int> vecs_found{0};
        std::vector<std::future<std::vector<RowVectorXi64>>> futures;

        for (int i = 0; i < num_threads; ++i) {
            if (fps_workers[i].second.exhausted()) fps_workers[i] = init_fps();
        }

        // Release Python GIL
        py::gil_scoped_release release;

        for (int i = 0; i < num_threads; ++i) {
            std::pair<int_class, FPSearch>* fps_worker = &fps_workers[i];
            futures.push_back(std::async(std::launch::async, [this, i, output_size, batch_size, &vecs_found, fps_worker]() {
                std::vector<RowVectorXi64> local_vecs;
                
                while (vecs_found.load() < output_size) {
                    
                    if (fps_worker->second.exhausted()) break;
                    auto vecs = fps_worker->second.batch_search(batch_size);

                    for (const auto& u : vecs) {
                        RowVectorXi64 v = RowVectorXi64::Zero(rank);
                        v(0, 0) = fps_worker->first;
                        for (size_t j = 0; j < u.size(); ++j) v(0, j + 1) = u[j];

                        if (only_roots && !is_root_core(A, v)) continue;

                        local_vecs.push_back(v);
                        vecs_found.fetch_add(1);
                    }
                }
                return local_vecs;
            }));
        }

        // Merge thread results
        int count = 0;
        for (auto& fut : futures) {
            auto partial_vecs = fut.get();
            for (const auto& v : partial_vecs) {
                vecs.push_back(v);
                count++;
            }
        }

        // Reacquire GIL implicitly when scope ends
        return count;
    }

public:
    // A_in, base_in: the Gram matrix and the base point, see the class comment; chamber_in: the vector that
    // chooses the chamber of the root system at height 0 (replaced by a random vector if it lies on a mirror).
    // Throws ValueError if C is not negative definite or (e_0, base) <= 0, RuntimeError if s <= 0,
    // OverflowError (see section 7).
    VSearchCpp(MatrixXi64 A_in, RowVectorXi64 base_in, RowVectorXi64 chamber_in, double bound_in, int num_threads_in, bool chamber_mode_in, bool only_roots_in) :
                A(std::move(A_in)), base(std::move(base_in)), rank(A.rows()), bound(bound_in), num_threads(num_threads_in),
                chamber_mode(chamber_mode_in), only_roots(only_roots_in), h_counter(-1) {
        if (num_threads <= 0) num_threads = std::thread::hardware_concurrency();

        C_int = A.block(1, 1, rank - 1, rank - 1);
        C_double = C_int.unaryExpr([](const int_class& x) { return int_class_to_double(x); });
        Eigen::LLT<Eigen::MatrixXd> llt(-C_double);
        if (llt.info() != Eigen::Success) {
            throw std::invalid_argument("VSearchCpp: the Gram matrix of e_1, ..., e_{n-1} must be negative definite");
        }
        Q_inv_diag = llt.solve(Eigen::MatrixXd::Identity(rank - 1, rank - 1)).diagonal();

        std::vector<long double> e0(rank, 0), base_abs(rank);
        e0[0] = 1;
        for (int i = 0; i < rank; ++i) base_abs[i] = std::fabs(static_cast<long double>(base(0, i)));
        check_limit(product_bound(A, e0, base_abs), "(e_0, base)");

        RowVectorXi64 A_row0 = A.block(0, 0, 1, rank);
        // The roots of positive height must have positive products (v, base) = h (e_0, base) with the base point:
        // the chamber test against the simple roots at height 0 depends on this orientation
        if ((A_row0 * base.transpose())(0, 0) <= 0) {
            throw std::invalid_argument("VSearchCpp: (e_0, base) must be positive");
        }
        s = mpq_class((A_row0 * base.transpose())(0, 0), base(0, 0));
        s.canonicalize();

        if (s <= 0) {
            std::cout << "s = " << s.get_d() << " prod = " << (A_row0 * base.transpose())(0, 0) << " base(0, 0) = " << base(0, 0) << std::endl;
            throw std::runtime_error("Error initializing basis: s is non-positive");
        }
        
        init_chamber(chamber_in);
    }

    // (Re)starts the search with the chamber chosen by the vector base (the chamber vector; it shadows the
    // member base): resets the heights and the workers, and in chamber mode searches the roots of height 0
    // and builds their root system R; the roots and walls found so far are discarded.
    void init_chamber(RowVectorXi64 base) {
        h_counter = chamber_mode ? 0 : -1;
        coord_bound.assign(rank, 0);
        fps_workers.clear();
        for (int i = 0; i < num_threads; ++i) {
            fps_workers.push_back(init_fps());
        }
        vecs.clear();

        std::vector<RowVectorXi64> initial_roots;

        if (!chamber_mode) {
            R = std::make_shared<RootSysCpp>(A, initial_roots, base);
            return;
        }

        // Init the base chamber (height 0): roots with 1 <= -(v, v) < bound
        update_coord_bound(0, Eigen::VectorXd::Zero(rank - 1), bound);
        // RootSysCpp computes the products of the roots with the chamber vector, or with a random one
        std::vector<long double> chamber_bound(rank);
        for (int i = 0; i < rank; ++i) {
            long double c = base.size() == rank ? std::fabs(static_cast<long double>(base(0, i))) : 0;
            chamber_bound[i] = std::max(c, 100001.0L);
        }
        check_limit(product_bound(A, chamber_bound, coord_bound), "the products with the chamber vector");
        FPSearch fps(-C_double, Eigen::VectorXd::Zero(rank - 1), 0.5, bound);
        auto initial_vecs = fps.search_all();
        
        for (const auto& u : initial_vecs) {
            RowVectorXi64 v = RowVectorXi64::Zero(rank);
            for (size_t i = 0; i < u.size(); ++i) v(0, i + 1) = u[i];
            if (is_root_core(A, v)) initial_roots.push_back(v);
        }
        
        R = std::make_shared<RootSysCpp>(A, initial_roots, base);
        roots.clear();
        walls.clear();
    }

    // Continues the search; batch_size is the number of Fincke-Pohst vectors examined at a time by a worker.
    // A height can be left partially searched; the next call resumes it. Returns the number of vectors found.
    int run(int output_size, int batch_size) {
        if (chamber_mode) {
            return run_chamber_mode(output_size, batch_size);
        } else {
            return run_vec_mode(output_size, batch_size);
        }
    }

    // Vinberg's selection: goes through the roots of positive height by increasing distance key, and accepts
    // a root if its products with all the previously accepted roots are non-negative. The rejected roots are
    // discarded for good: they are never walls of the chamber, whatever the order in which roots are found
    // (docs/Vinberg.md, section 5). Acceptance is provisional until all the closer roots are known (section 6).
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

    // The largest H such that all the heights 1, ..., H have been searched completely
    // (0 before the first height is done). Heights are assigned to the workers in increasing order,
    // and a worker keeps its height until its search is exhausted.
    int_class searched_height() const {
        int_class done = h_counter;
        for (const auto& w : fps_workers) {
            if (!w.second.exhausted()) done = std::min(done, w.first - 1);
        }
        return std::max(done, static_cast<int_class>(0));
    }

    // The current walls: the simple roots at height 0 (the walls through the base point), followed by the
    // accepted roots of positive height by increasing distance
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

    // Vector mode: the vectors found since the last call (by height within each call), and clears them
    std::vector<std::vector<int_class>> get_vecs() {
        std::vector<std::vector<int_class>> out;
        for (const auto& v : vecs) {
            std::vector<int_class> vec_out(rank);
            for(int i=0; i<rank; ++i) vec_out[i] = v(0, i);
            out.push_back(vec_out);
        }
        vecs.clear();
        return out;
    }
};

// --- 9. Conversion helpers between std::vector (pybind11) and Eigen ---

// A matrix from a list of rows (0 x 0 for an empty list)
MatrixXi64 to_matrix(const std::vector<std::vector<int_class>>& v) {
    if (v.empty()) return MatrixXi64(0, 0);
    MatrixXi64 m(v.size(), v[0].size());
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j < v[0].size(); ++j)
            m(i, j) = v[i][j];
    return m;
}

// A 1 x n row vector (1 x 0 for an empty list)
RowVectorXi64 to_row_vector(const std::vector<int_class>& v) {
    if (v.empty()) return RowVectorXi64(1, 0);
    RowVectorXi64 m(1, v.size());
    for (size_t i = 0; i < v.size(); ++i) m(0, i) = v[i];
    return m;
}

// The entries of a row vector
std::vector<int_class> to_std_vector(const RowVectorXi64& m) {
    std::vector<int_class> v(m.cols());
    for (int i = 0; i < m.cols(); ++i) v[i] = m(0, i);
    return v;
}

// The rows of a matrix
std::vector<std::vector<int_class>> to_std_matrix(const MatrixXi64& m) {
    std::vector<std::vector<int_class>> v(m.rows(), std::vector<int_class>(m.cols()));
    for(int i=0; i<m.rows(); ++i)
        for(int j=0; j<m.cols(); ++j)
            v[i][j] = m(i,j);
    return v;
}


// --- 10. Python bindings ---
PYBIND11_MODULE(vsearch_cpp, m) {
    m.doc() = "Multithreaded C++ backend for VSearch and RootSys: exact 64-bit integer arithmetic, "
              "with magnitudes bounded in advance (OverflowError if 64 bits might not suffice)";

    // --- Expose RootSysCpp ---
    py::class_<RootSysCpp, std::shared_ptr<RootSysCpp>>(m, "RootSysCpp",
        "A finite root system given by all its roots in a lattice with Gram matrix A; the polarization vector "
        "base (random if missing or on a mirror) determines the positive and the simple roots (sroots).")
        .def(py::init([](const std::vector<std::vector<int_class>>& A_in, 
                         const std::vector<std::vector<int_class>>& roots_in, 
                         const std::vector<int_class>& base_in) {
            std::vector<RowVectorXi64> eigen_roots;
            for (const auto& r : roots_in) eigen_roots.push_back(to_row_vector(r));
            return std::make_shared<RootSysCpp>(to_matrix(A_in), eigen_roots, to_row_vector(base_in));
        }), py::arg("A"), py::arg("roots"), py::arg("base") = std::vector<int_class>(),
        "RootSysCpp(A, roots, base=[]): A is the Gram matrix, roots all the roots of the root system.")
        
        .def("reflect", [](RootSysCpp& self, const std::vector<int_class>& v) {
            return to_std_vector(self.reflect(to_row_vector(v)));
        }, py::arg("v"), "Maps v into the closed chamber of the simple roots by reflections; returns the image.")

        .def("closed_chamber", [](RootSysCpp& self, const std::vector<int_class>& v) {
            return self.closed_chamber(to_row_vector(v));
        }, py::arg("v"), "The signs (+1 for >= 0, -1 for < 0) of the products of v with the positive roots.")
        
        .def("reflection", [](RootSysCpp& self, const std::vector<int_class>& r) {
            return to_std_matrix(self.reflection(to_row_vector(r)));
        }, py::arg("r"), "The matrix of the reflection in the root r, acting on row vectors.")
        
        .def_property_readonly("sroots", [](const RootSysCpp& self) {
            std::vector<std::vector<int_class>> out;
            for (const auto& r : self.sroots) out.push_back(to_std_vector(r));
            return out;
        }, "The simple roots.");

    // --- Expose VSearchCpp ---
    py::class_<VSearchCpp>(m, "VSearchCpp",
        "The search for the negative roots of a hyperbolic lattice by height, and the selection of the walls "
        "of the chamber in Vinberg's algorithm (see docs/Vinberg.md).")
        .def(py::init([](const std::vector<std::vector<int_class>>& A_in,
                         const std::vector<int_class>& base_in,
                         const std::vector<int_class>& chamber_in,
                         const double bound_in,
                         const int num_threads_in,
                         const bool chamber_mode_in,
                         const bool only_roots_in) {
            return std::make_unique<VSearchCpp>(to_matrix(A_in), to_row_vector(base_in), to_row_vector(chamber_in),
                                                bound_in, num_threads_in, chamber_mode_in, only_roots_in);
        }), py::arg("A"), py::arg("base"), py::arg("chamber"), py::arg("bound"), py::arg("num_threads") = 1, py::arg("chamber_mode") = true, py::arg("only_roots") = true,
        "VSearchCpp(A, base, chamber, bound, num_threads=1, chamber_mode=True, only_roots=True): A is the Gram "
        "matrix in a basis e_0, ..., e_{n-1} with e_1, ..., e_{n-1} orthogonal to the base point (coordinates "
        "base); chamber chooses the chamber at the base point; the search covers -bound <= (v, v) <= -1 "
        "(use bound = 2 exp + 0.5 for roots).")

        .def("init_chamber", [](VSearchCpp& self, const std::vector<int_class>& base) {
            self.init_chamber(to_row_vector(base));
        }, py::arg("base"), "Restarts the search with the chamber chosen by the given chamber vector.")

        .def("run", &VSearchCpp::run, py::arg("output_size"), py::arg("batch_size"),
             "Continues the search until output_size vectors are found or every worker exhausted its height; "
             "returns the number found.")
        .def("update_walls", &VSearchCpp::update_walls,
             "Selects the walls among the roots found, by increasing distance; discards the rejected roots.")
        .def("searched_height", &VSearchCpp::searched_height,
             "The largest H such that all the heights 1, ..., H have been searched completely.")
        .def("get_walls", &VSearchCpp::get_walls,
             "The walls: the simple roots at the base point, then the accepted roots by increasing distance.")
        .def("get_vecs", &VSearchCpp::get_vecs, "Vector mode: the vectors found since the last call.");
}