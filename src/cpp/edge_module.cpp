// edge_cpp: the per-edge core of edge tracing (lattice/vinberg/trace.py, edges.py) in C++.
//
// crossing_roots(A, exp, a, b, include_end): the roots whose mirrors cross the open segment (a, b), as in
//     lattice.vinberg.edges.edge_crossing_roots (see docs/EdgeTracing.md, section 4).
// trace_edge(A, exp, a, u, walls, edge_walls): follows an edge from a in the direction u to its end, as
//     trace_ray + _next_walls in lattice.vinberg.trace (cases 1 and 2 of section 3); an irrational ideal end
//     (case 3, the walk along Conway's river) is left to Python.
//
// Integers are 64-bit (long long), with 128-bit intermediate values; every operation that might overflow is
// checked, and an overflow raises std::overflow_error (OverflowError in Python), in which case the Python
// implementation, which uses arbitrary precision, is used instead. The only floating-point steps are the
// LLL reduction of the complement of an edge (only its quality depends on it: the transformation is exact and
// unimodular) and the Fincke-Pohst searches, whose bounds keep a margin of 0.5 from the attainable values; all
// the candidates are checked exactly.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "fp_search.hpp"

namespace py = pybind11;

using i64 = long long;
using i128 = __int128;
using Vec = std::vector<i64>;
using MatI = std::vector<Vec>;  // a list of row vectors

// --- checked arithmetic ---

static inline i128 mul(i128 x, i128 y) {
    i128 r;
    if (__builtin_mul_overflow(x, y, &r)) throw std::overflow_error("edge_cpp: 128-bit overflow");
    return r;
}

static inline i128 add(i128 x, i128 y) {
    i128 r;
    if (__builtin_add_overflow(x, y, &r)) throw std::overflow_error("edge_cpp: 128-bit overflow");
    return r;
}

static inline i64 to64(i128 x) {
    if (x > (i128)LLONG_MAX || x < -(i128)LLONG_MAX) throw std::overflow_error("edge_cpp: 64-bit overflow");
    return (i64)x;
}

static i128 iabs(i128 x) { return x < 0 ? -x : x; }

static i128 gcd128(i128 x, i128 y) {
    x = iabs(x);
    y = iabs(y);
    while (y != 0) {
        i128 t = x % y;
        x = y;
        y = t;
    }
    return x;
}

// x = g = gcd(a, b) >= 0 with s a + t b = g
static void ext_gcd(i128 a, i128 b, i128& g, i128& s, i128& t) {
    i128 old_r = a, r = b, old_s = 1, s1 = 0, old_t = 0, t1 = 1;
    while (r != 0) {
        i128 q = old_r / r;
        i128 tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s1; old_s = s1; s1 = tmp;
        tmp = old_t - q * t1; old_t = t1; t1 = tmp;
    }
    if (old_r < 0) { old_r = -old_r; old_s = -old_s; old_t = -old_t; }
    g = old_r; s = old_s; t = old_t;
}

// floor(sqrt(x)) for x >= 0
static i128 isqrt128(i128 x) {
    if (x < 0) throw std::invalid_argument("edge_cpp: square root of a negative number");
    if (x < 2) return x;
    i128 r = (i128)std::sqrt((long double)x);
    while (r > 0 && mul(r, r) > x) --r;
    while (mul(r + 1, r + 1) <= x) ++r;
    return r;
}

static i128 floordiv(i128 a, i128 b) {  // b > 0
    i128 q = a / b;
    if ((a % b != 0) && (a < 0)) --q;
    return q;
}

static i128 ceildiv(i128 a, i128 b) {  // b > 0
    return -floordiv(-a, b);
}

// --- the lattice ---

struct Form {
    int n;
    MatI A;
    i64 exp;

    // x A (as 128-bit values)
    std::vector<i128> xA(const Vec& x) const {
        std::vector<i128> r(n, 0);
        for (int i = 0; i < n; ++i) {
            if (x[i] == 0) continue;
            for (int j = 0; j < n; ++j) r[j] = add(r[j], mul(x[i], A[i][j]));
        }
        return r;
    }

    i128 prod(const Vec& x, const Vec& y) const {
        std::vector<i128> xa = xA(x);
        i128 s = 0;
        for (int j = 0; j < n; ++j) s = add(s, mul(xa[j], y[j]));
        return s;
    }

    i128 prod(const std::vector<i128>& xa, const Vec& y) const {
        i128 s = 0;
        for (int j = 0; j < n; ++j) s = add(s, mul(xa[j], y[j]));
        return s;
    }

    // Lattice.is_root: primitive, (r, r) != 0, (r, r) divides 2 (r, e_i)
    bool is_root(const Vec& r) const {
        i128 g = 0;
        for (i64 x : r) g = gcd128(g, x);
        if (g != 1) return false;
        std::vector<i128> ra = xA(r);
        i128 nr = prod(ra, r);
        if (nr == 0) return false;
        for (int j = 0; j < n; ++j)
            if (mul(2, ra[j]) % nr != 0) return false;
        return true;
    }
};

static Vec primitive(const std::vector<i128>& v) {
    i128 g = 0;
    for (i128 x : v) g = gcd128(g, x);
    Vec r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = to64(g > 1 ? v[i] / g : v[i]);
    return r;
}

// a x + b y for vectors
static std::vector<i128> comb(i128 a, const Vec& x, i128 b, const Vec& y) {
    std::vector<i128> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) r[i] = add(mul(a, x[i]), mul(b, y[i]));
    return r;
}

static Vec to64v(const std::vector<i128>& v) {
    Vec r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = to64(v[i]);
    return r;
}

// A basis of {y in span_Z(B) : (y, w) = 0}, from the values g_i = (B_i, w): with a unimodular T such that
// T g = (gcd, 0, ..., 0), the rows 2, ... of T B span the kernel
static MatI kernel(const Form& F, const MatI& B, const Vec& w) {
    int m = B.size();
    std::vector<i128> g(m);
    std::vector<i128> wa = F.xA(w);
    for (int i = 0; i < m; ++i) g[i] = F.prod(wa, B[i]);
    MatI rows = B;
    for (int i = 1; i < m; ++i) {
        if (g[i] == 0) continue;
        if (g[0] == 0) {
            std::swap(rows[0], rows[i]);
            std::swap(g[0], g[i]);
            continue;
        }
        i128 d, s, t;
        ext_gcd(g[0], g[i], d, s, t);
        Vec r0 = to64v(comb(s, rows[0], t, rows[i]));
        Vec ri = to64v(comb(g[i] / d, rows[0], -(g[0] / d), rows[i]));
        rows[0] = r0;
        rows[i] = ri;
        g[0] = d;
        g[i] = 0;
    }
    if (g[0] == 0) return rows;  // w is orthogonal to the whole span
    return MatI(rows.begin() + 1, rows.end());
}

// LLL reduction (delta = 0.99) of the rows of B, spanning a negative definite sublattice, for the positive
// definite form -( , ). The integer Gram matrix is computed once and updated exactly under the row operations;
// the Gram-Schmidt coefficients are floating point, updated during size reduction and recomputed from the Gram
// matrix after a swap. The row operations are exact and unimodular, so the result is a basis of the same
// lattice whatever the rounding (only the quality of the reduction depends on it).
static MatI lll_negative(const Form& F, MatI B) {
    int m = B.size();
    if (m <= 1) return B;
    std::vector<std::vector<i128>> G(m, std::vector<i128>(m));
    {
        std::vector<std::vector<i128>> BA(m);
        for (int i = 0; i < m; ++i) BA[i] = F.xA(B[i]);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j <= i; ++j) G[i][j] = G[j][i] = -F.prod(BA[i], B[j]);
    }
    std::vector<std::vector<long double>> mu(m, std::vector<long double>(m, 0));
    std::vector<long double> Bn(m);
    auto gram_schmidt = [&](int from) {
        for (int i = from; i < m; ++i) {
            for (int j = 0; j < i; ++j) {
                long double s = (long double)G[i][j];
                for (int k = 0; k < j; ++k) s -= mu[j][k] * mu[i][k] * Bn[k];
                mu[i][j] = s / Bn[j];
            }
            long double s = (long double)G[i][i];
            for (int k = 0; k < i; ++k) s -= mu[i][k] * mu[i][k] * Bn[k];
            Bn[i] = s;
        }
    };
    gram_schmidt(0);
    int k = 1, iterations = 0;
    while (k < m) {
        if (++iterations > 100000) break;  // the reduction only affects the performance
        for (int j = k - 1; j >= 0; --j) {
            long double qd = std::round(mu[k][j]);
            if (qd == 0) continue;
            if (std::fabs(qd) > 1.0e18L) throw std::overflow_error("edge_cpp: LLL coefficient too large");
            i128 q = (i128)qd;
            // b_k -= q b_j, and the Gram matrix accordingly
            for (int c = 0; c < F.n; ++c) B[k][c] = to64(add(B[k][c], -mul(q, B[j][c])));
            i128 gkk = add(add(G[k][k], -mul(mul(2, q), G[k][j])), mul(mul(q, q), G[j][j]));
            for (int i = 0; i < m; ++i)
                if (i != k) G[k][i] = G[i][k] = add(G[k][i], -mul(q, G[j][i]));
            G[k][k] = gkk;
            for (int i = 0; i < j; ++i) mu[k][i] -= qd * mu[j][i];
            mu[k][j] -= qd;
        }
        if (Bn[k] >= (0.99L - mu[k][k - 1] * mu[k][k - 1]) * Bn[k - 1]) {
            ++k;
        } else {
            std::swap(B[k], B[k - 1]);
            std::swap(G[k], G[k - 1]);
            for (int i = 0; i < m; ++i) std::swap(G[i][k], G[i][k - 1]);
            gram_schmidt(k - 1);
            k = std::max(k - 1, 1);
        }
    }
    return B;
}

// An LLL-reduced basis of L cap <a, u>^perp: the kernel of (., a) on Z^n, then the kernel of (., u) on it
static MatI plane_complement(const Form& F, const Vec& a, const Vec& u) {
    MatI E(F.n, Vec(F.n, 0));
    for (int i = 0; i < F.n; ++i) E[i][i] = 1;
    MatI K = kernel(F, kernel(F, E, a), u);
    return lll_negative(F, K);
}

// --- the roots crossing a segment ---

struct Root {
    Vec r;
    i128 A, B;  // (r, a), (r, b)
};

struct Segment {
    const Form& F;
    Vec a, b;
    i128 p, q, c, D, bound;
    i128 b1, a1, a2;
    Vec g1, g2;
    MatI Nb;
    Eigen::MatrixXd Q;
    Eigen::LLT<Eigen::MatrixXd> llt;
    int k;

    Segment(const Form& F_in, const Vec& a_in, const Vec& b_in, const MatI& Nb_in)
        : F(F_in), a(a_in), b(b_in), Nb(Nb_in) {
        p = F.prod(a, a);
        q = F.prod(b, b);
        c = F.prod(a, b);
        if (p < 0 || q < 0 || c <= 0) throw std::invalid_argument("a and b must lie in the closed future light cone");
        D = add(mul(c, c), -mul(p, q));
        if (D <= 0) throw std::invalid_argument("a and b must not be proportional");
        bound = mul(mul(2, F.exp), D);
        lambda_basis();
        k = Nb.size();
        if (k != F.n - 2) throw std::runtime_error("The complement of the segment does not have rank n - 2");
        if (k > 0) {
            Q.resize(k, k);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j) Q(i, j) = -(double)F.prod(Nb[i], Nb[j]);
            llt.compute(Q);
            if (llt.info() != Eigen::Success) throw std::runtime_error("The complement of the segment is not definite");
        }
    }

    // Lambda = {((y, b), (y, a))}: the basis s (b1, a1) + t (0, a2) with lifts g1, g2 (2-dimensional HNF)
    void lambda_basis() {
        std::vector<i128> ab = F.xA(b), aa = F.xA(a);
        bool has1 = false, has2 = false;
        i128 B1 = 0, A1 = 0, A2 = 0;
        Vec L1(F.n, 0), L2(F.n, 0);
        auto add2 = [&](i128 alpha, const Vec& lift) {
            // combine a generator (0, alpha) with (0, A2)
            if (alpha == 0) return;
            if (!has2) { A2 = alpha; L2 = lift; has2 = true; }
            else {
                i128 d, s, t;
                ext_gcd(A2, alpha, d, s, t);
                L2 = to64v(comb(s, L2, t, lift));
                A2 = d;
            }
            if (A2 < 0) { A2 = -A2; for (auto& x : L2) x = -x; }
        };
        for (int i = 0; i < F.n; ++i) {
            Vec e(F.n, 0);
            e[i] = 1;
            i128 beta = ab[i], alpha = aa[i];
            if (beta == 0) { add2(alpha, e); continue; }
            if (!has1) { B1 = beta; A1 = alpha; L1 = e; has1 = true; continue; }
            i128 d, s, t;
            ext_gcd(B1, beta, d, s, t);
            // the remainder (beta / d) h1 - (B1 / d) g has first coordinate 0
            i128 rem_alpha = add(mul(beta / d, A1), -mul(B1 / d, alpha));
            Vec rem_lift = to64v(comb(beta / d, L1, -(B1 / d), e));
            Vec newL1 = to64v(comb(s, L1, t, e));
            A1 = add(mul(s, A1), mul(t, alpha));
            B1 = d;
            L1 = newL1;
            add2(rem_alpha, rem_lift);
        }
        if (!has1 || !has2) throw std::runtime_error("The image of the lattice in the plane is not of rank 2");
        if (B1 < 0) { B1 = -B1; A1 = -A1; for (auto& x : L1) x = -x; }
        // Reduce A1 modulo A2 (keeps the lifts small)
        i128 t = floordiv(A1, A2);
        if (t != 0) { L1 = to64v(comb(1, L1, -t, L2)); A1 = add(A1, -mul(t, A2)); }
        b1 = B1; a1 = A1; a2 = A2; g1 = L1; g2 = L2;
    }

    // The roots y + z, z in L cap N, with (y + z, y + z) >= -2 exp; x2 = (x, x) for the crossing point
    void promote(const Vec& y_in, i128 x2, i128 Aval, i128 Bval, std::vector<Root>& out) const {
        if (k == 0) {
            if (F.is_root(y_in)) out.push_back({y_in, Aval, Bval});
            return;
        }
        // The coordinates of the projection of y to N: GN coords = (y, Nb) with GN = -Q; shift y by the nearest
        // vector of L cap N, so that the center lies in [-1/2, 1/2]^k
        std::vector<i128> ya = F.xA(y_in);
        Eigen::VectorXd v(k);
        for (int i = 0; i < k; ++i) v(i) = (double)F.prod(ya, Nb[i]);
        Eigen::VectorXd coords = -llt.solve(v);
        std::vector<i128> y(y_in.begin(), y_in.end());
        Eigen::VectorXd center(k);
        for (int i = 0; i < k; ++i) {
            double nearest = std::round(coords(i));
            if (std::fabs(nearest) > 4.0e18) throw std::overflow_error("edge_cpp: the projection is too large");
            i128 ni = (i128)nearest;
            if (ni != 0)
                for (int c2 = 0; c2 < F.n; ++c2) y[c2] = add(y[c2], -mul(ni, Nb[i][c2]));
            center(i) = coords(i) - nearest;
        }
        // -(y_N + z)^2 = (r_Pi, r_Pi) - (r, r) <= 2 exp + (r_Pi, r_Pi), with (r_Pi, r_Pi) = -x2 / D; margin 0.5
        long double ub = (long double)(2 * F.exp) - (long double)x2 / (long double)D + 0.5L;
        if (ub < 0) return;
        FPSearch fps(Q, center, -0.5, (double)ub);
        for (const auto& kv : fps.search_all()) {
            std::vector<i128> r = y;
            for (int i = 0; i < k; ++i) {
                if (kv[i] == 0) continue;
                for (int c2 = 0; c2 < F.n; ++c2) r[c2] = add(r[c2], mul(kv[i], Nb[i][c2]));
            }
            Vec r64 = to64v(r);
            if (F.is_root(r64)) out.push_back({r64, Aval, Bval});
        }
    }

    // All the roots crossing the open segment (with A > 0 > B); with include_end and no crossing, the roots
    // through b with the smallest A > 0 (B = 0)
    std::vector<Root> roots(bool include_end) const {
        std::vector<Root> out;
        for (i128 s = -1;; --s) {
            i128 Bv = mul(s, b1), absB = -Bv;
            // Even A = 1 violates the bound from here on
            if (add(add(q, mul(mul(2, absB), c)), mul(mul(absB, absB), p)) > bound) break;
            i128 rest = add(bound, -mul(mul(absB, absB), p));
            i128 A_max;
            if (q > 0) A_max = floordiv(add(isqrt128(add(mul(mul(absB, absB), mul(c, c)), mul(q, rest))), -mul(absB, c)), q);
            else A_max = floordiv(rest, mul(mul(2, absB), c));
            if (A_max < 1) continue;
            i128 t_min = ceildiv(add(1, -mul(s, a1)), a2), t_max = floordiv(add(A_max, -mul(s, a1)), a2);
            for (i128 t = t_min; t <= t_max; ++t) {
                i128 Av = add(mul(s, a1), mul(t, a2));
                Vec y = to64v(comb(s, g1, t, g2));
                i128 x2 = add(add(mul(mul(Av, Av), q), mul(mul(mul(2, Av), absB), c)), mul(mul(absB, absB), p));
                promote(y, x2, Av, Bv, out);
            }
        }
        if (include_end && out.empty()) {
            if (q <= 0) throw std::invalid_argument("include_end requires a finite end");
            i128 A_max = isqrt128(bound / q);
            for (i128 t = 1; mul(t, a2) <= A_max; ++t) {
                i128 Av = mul(t, a2);
                Vec y = to64v(comb(t, g2, 0, g2));
                promote(y, mul(mul(Av, Av), q), Av, 0, out);
                if (!out.empty()) break;
            }
        }
        return out;
    }
};

// t = A / |B| for a crossing root (A > 0 > B): compare t1 < t2 by cross-multiplication
static bool before(const Root& x, const Root& y) {
    return mul(x.A, -y.B) < mul(y.A, -x.B);
}

static bool same_point(const Root& x, const Root& y) {
    return mul(x.A, -y.B) == mul(y.A, -x.B);
}

// Fraction-free solution of G x = d * 1 (G k x k, nonsingular): returns (d, x) with d = det(G) (up to sign)
static std::pair<i128, std::vector<i128>> solve_ones(std::vector<std::vector<i128>> M) {
    int k = M.size();
    for (int i = 0; i < k; ++i) M[i].push_back(1);
    i128 prev = 1;
    for (int c = 0; c < k; ++c) {
        int piv = c;
        while (piv < k && M[piv][c] == 0) ++piv;
        if (piv == k) throw std::runtime_error("The walls of an edge are not independent");
        std::swap(M[piv], M[c]);
        for (int i = c + 1; i < k; ++i) {
            for (int j = c + 1; j <= k; ++j)
                M[i][j] = add(mul(M[c][c], M[i][j]), -mul(M[i][c], M[c][j])) / prev;
            M[i][c] = 0;
        }
        prev = M[c][c];
    }
    i128 d = M[k - 1][k - 1];
    // Back substitution with x = d * G^{-1} 1 (integral by Cramer's rule)
    std::vector<i128> x(k, 0);
    for (int i = k - 1; i >= 0; --i) {
        i128 s = mul(d, M[i][k]);
        for (int j = i + 1; j < k; ++j) s = add(s, -mul(M[i][j], x[j]));
        if (s % M[i][i] != 0) throw std::runtime_error("Fraction-free solution is not integral");
        x[i] = s / M[i][i];
    }
    return {d, x};
}

// The new wall at the finite end of an edge with walls W, followed from a: the root through the end with the
// smallest (r, a) > 0, then the smallest (y, r), y in span(W) with (y, w) = 1 (lattice/vinberg/trace.py,
// _next_walls)
static Vec next_wall(const Form& F, const MatI& W, const Vec& a, const std::vector<Root>& end_roots) {
    if (end_roots.empty()) throw std::runtime_error("No root through the end of an edge");
    int k = W.size();
    // y_int = d y with d = det(G), x = d G^{-1} 1; if d < 0, the order of (y, r) is reversed
    std::vector<i128> y(F.n, 0);
    i128 sign = 1;
    if (k > 0) {
        std::vector<std::vector<i128>> G(k, std::vector<i128>(k));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) G[i][j] = F.prod(W[i], W[j]);
        auto [d, x] = solve_ones(G);
        sign = d < 0 ? -1 : 1;
        for (int i = 0; i < k; ++i)
            for (int c = 0; c < F.n; ++c) y[c] = add(y[c], mul(x[i], W[i][c]));
    }
    Vec yv = to64v(y);
    size_t best = 0;
    i128 best_a = 0, best_y = 0;
    for (size_t i = 0; i < end_roots.size(); ++i) {
        i128 ra = F.prod(end_roots[i].r, a), ry = mul(sign, F.prod(end_roots[i].r, yv));
        if (i == 0 || ra < best_a || (ra == best_a && ry < best_y)) {
            best = i; best_a = ra; best_y = ry;
        }
    }
    const Vec& alpha = end_roots[best].r;
    for (const auto& w : W)
        if (F.prod(alpha, w) < 0) throw std::runtime_error("The new wall at a vertex makes an obtuse angle with the walls of the edge");
    return alpha;
}

static Form make_form(const MatI& A, i64 exp) {
    Form F;
    F.n = A.size();
    F.A = A;
    F.exp = exp;
    return F;
}

// The first crossing point of the segment (a, b), and all the roots crossing there
static std::pair<Vec, std::vector<Root>> first_crossing(const Form& F, const Vec& a, const Vec& b, std::vector<Root> found) {
    std::vector<Root> crossing;
    for (const auto& r : found) if (r.B < 0) crossing.push_back(r);
    const Root* first = &crossing[0];
    for (const auto& r : crossing) if (before(r, *first)) first = &r;
    std::vector<Root> at;
    for (const auto& r : crossing) if (same_point(r, *first)) at.push_back(r);
    // The point -B a + A b on the mirror of the first root
    return {primitive(comb(-first->B, a, first->A, b)), at};
}

// The directions of the edges at the finite vertex v with walls S (n of them): u_j generates the kernel of
// (., v) and (., S[i]) for i != j, oriented with (u_j, S[j]) > 0 (lattice/vinberg/trace.py, finite_vertex_edges)
static std::vector<Vec> vertex_directions(const Form& F, const Vec& v, const MatI& S) {
    MatI E(F.n, Vec(F.n, 0));
    for (int i = 0; i < F.n; ++i) E[i][i] = 1;
    MatI C = lll_negative(F, kernel(F, E, v));
    std::vector<Vec> result;
    for (size_t j = 0; j < S.size(); ++j) {
        MatI K = C;
        for (size_t i = 0; i < S.size(); ++i)
            if (i != j) K = kernel(F, K, S[i]);
        if (K.size() != 1) throw std::runtime_error("The walls at a finite vertex are not independent");
        std::vector<i128> u(K[0].begin(), K[0].end());
        Vec up = primitive(u);
        if (F.prod(up, S[j]) < 0) for (auto& x : up) x = -x;
        result.push_back(up);
    }
    return result;
}

// --- Python interface ---

static py::list roots_to_list(const std::vector<Root>& roots) {
    py::list result;
    for (const auto& r : roots) result.append(py::make_tuple(r.r, (i64)to64(r.A), (i64)to64(r.B)));
    return result;
}

PYBIND11_MODULE(edge_cpp, m) {
    m.doc() = "The per-edge core of edge tracing (see docs/EdgeTracing.md): crossing_roots and trace_edge.";

    m.def("crossing_roots", [](const MatI& A, i64 exp, const Vec& a, const Vec& b, bool include_end) {
        Form F = make_form(A, exp);
        MatI Nb = plane_complement(F, a, b);
        Segment S(F, a, b, Nb);
        return roots_to_list(S.roots(include_end));
    }, py::arg("A"), py::arg("exp"), py::arg("a"), py::arg("b"), py::arg("include_end") = false,
       "The roots r with (r, a) > 0 > (r, b) (and, with include_end and no such root, those through b with the "
       "smallest (r, a) > 0), as triples (root, (r, a), (r, b)). Raises OverflowError beyond 64/128 bits.");

    m.def("vertex_directions", [](const MatI& A, const Vec& v, const MatI& S) {
        Form F = make_form(A, 1);
        return vertex_directions(F, v, S);
    }, py::arg("A"), py::arg("v"), py::arg("S"),
       "The directions u_j of the edges at the finite vertex v with walls S: (u_j, v) = 0, (u_j, S[i]) = 0 for "
       "i != j, (u_j, S[j]) > 0. Raises OverflowError beyond 64/128 bits.");

    m.def("trace_edge", [](const MatI& A, i64 exp, const Vec& a, const Vec& u, const MatI& walls, const MatI& edge_walls) -> py::tuple {
        Form F = make_form(A, exp);
        i128 p = F.prod(a, a), c = F.prod(a, u), q = F.prod(u, u);
        MatI Nb = plane_complement(F, a, u);
        // 1. The end predicted by the known walls not through a: (w, a) + t (w, u) = 0, t > 0
        bool has = false;
        i128 num = 0, den = 1;
        std::vector<i128> aa = F.xA(a), ua = F.xA(u);
        for (const auto& w : walls) {
            i128 wa = F.prod(aa, w), wu = F.prod(ua, w);
            if (wa > 0 && wu < 0) {
                // t = wa / (-wu)
                if (!has || mul(wa, den) < mul(num, -wu)) { num = wa; den = -wu; has = true; }
            }
        }
        if (has) {
            Vec bvec = primitive(comb(den, a, num, u));
            if (F.prod(bvec, bvec) > 0) {
                Segment S(F, a, bvec, Nb);
                std::vector<Root> found = S.roots(true);
                bool crossing = false;
                for (const auto& r : found) if (r.B < 0) crossing = true;
                if (crossing) {
                    auto [end, at] = first_crossing(F, a, bvec, found);
                    return py::make_tuple(0, end, next_wall(F, edge_walls, a, at));
                }
                return py::make_tuple(0, bvec, next_wall(F, edge_walls, a, found));
            }
        }
        // 2. The ideal end of the line, if rational
        if (q > 0) throw std::runtime_error("The direction of a ray must not be timelike");
        Vec z;
        if (q == 0) z = u;
        else {
            i128 disc = add(mul(c, c), -mul(p, q));
            i128 s = isqrt128(disc);
            if (mul(s, s) != disc) return py::make_tuple(2, py::none(), py::none());  // irrational: Python's river walk
            z = primitive(comb(-q, a, add(c, s), u));
        }
        Segment S(F, a, z, Nb);
        std::vector<Root> found = S.roots(false);
        if (!found.empty()) {
            auto [end, at] = first_crossing(F, a, z, found);
            return py::make_tuple(0, end, next_wall(F, edge_walls, a, at));
        }
        return py::make_tuple(1, z, py::none());
    }, py::arg("A"), py::arg("exp"), py::arg("a"), py::arg("u"), py::arg("walls"), py::arg("edge_walls"),
       "Follows the edge from a in the direction u: returns (0, end, new wall) for a finite end, (1, cusp, None) "
       "for a rational ideal end, (2, None, None) for an irrational ideal end (to be handled in Python). "
       "Raises OverflowError beyond 64/128 bits.");
}
