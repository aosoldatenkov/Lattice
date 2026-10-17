"""Brute-force checks of lattice.lattice: results are compared with direct enumeration of the
vectors in a box, and with independent formulas (determinantal divisors, the inverse Gram matrix).

Membership of a vector x in the Z-span of the rows of a matrix B is tested by comparing
the Hermite normal forms of B and of B with the row x appended."""
import math
import random
from fractions import Fraction
from itertools import combinations, product
import flint as fl
import numpy as np
import pytest
import sympy as sp
from lattice._matrix import imat, imat_zero
from lattice.lattice import Lattice
from _lattice_samples import random_grams, random_definite, random_generators, ident

BOX2 = range(-4, 5)
RANK2 = [[[a, h], [h, b]] for a in BOX2 for b in BOX2 for h in BOX2 if a * b != h * h]
RANK3 = random_grams(11, 80, [3], 5) + random_definite(12, 20, [3], 2)
SMALL = random_grams(13, 40, [2, 3, 4], 4)


def box(n, r):
    return product(range(-r, r + 1), repeat=n)


def hnf_rows(B):
    """The nonzero rows of the Hermite normal form of B: a canonical basis of its Z-span."""
    if len(B) == 0:
        return []
    H = fl.fmpz_mat([[int(x) for x in r] for r in B]).hnf()
    return H.tolist()[:H.rank()]


class Span:
    """The Z-span and the Q-span of the rows of B, with membership tests."""

    def __init__(self, B):
        self.H = hnf_rows(B)
        self.rank = len(self.H)

    def __contains__(self, x):
        x = [int(t) for t in x]
        return not any(x) or hnf_rows(self.H + [x]) == self.H

    def q_contains(self, x):
        x = [int(t) for t in x]
        return not any(x) or fl.fmpz_mat(self.H + [x]).rank() == self.rank


def minors_gcd(A, k):
    """The k-th determinantal divisor: the GCD of the k x k minors."""
    M = sp.Matrix(A)
    return math.gcd(*(int(M.extract(list(r), list(c)).det()) for r in combinations(range(M.rows), k)
                      for c in combinations(range(M.cols), k)))


def max_square(d):
    return max(k for k in range(1, math.isqrt(d) + 1) if d % (k * k) == 0)


def matrix_or_empty(rows, n):
    return imat(rows) if rows else imat_zero(0, n)


class TestInvariants:
    @pytest.mark.parametrize("A", RANK2 + RANK3, ids=ident)
    def test_dgroup_from_determinantal_divisors(self, A):
        L = Lattice(len(A), A)
        divisors = [1] + [minors_gcd(A, k) for k in range(1, len(A) + 1)]
        assert L.dgroup == [divisors[k] // divisors[k - 1] for k in range(1, len(A) + 1)]
        assert L.disc == abs(divisors[-1])

    @pytest.mark.parametrize("A", RANK2 + RANK3, ids=ident)
    def test_exponent_from_inverse(self, A):
        L = Lattice(len(A), A)
        inv = sp.Matrix(A).inv()
        assert L.exp == math.lcm(*(int(sp.fraction(x)[1]) for x in inv))

    @pytest.mark.parametrize("A", RANK2 + RANK3, ids=ident)
    def test_signature_numerically(self, A):
        eig = np.linalg.eigvalsh(np.array(A, dtype=float))
        assert Lattice(len(A), A).signature == (int((eig > 0).sum()), int((eig < 0).sum()))

    @pytest.mark.parametrize("A", RANK2 + RANK3, ids=ident)
    def test_parity(self, A):
        L = Lattice(len(A), A)
        assert L.parity == any(L.square(x) % 2 for x in box(len(A), 1))


class TestVectors:
    @pytest.mark.parametrize("A", SMALL, ids=ident)
    def test_is_root(self, A):
        n = len(A)
        L = Lattice(n, A)
        basis = np.eye(n, dtype=int).tolist()
        for u in box(n, 2):
            s = L.square(u)
            # u is primitive, and the reflection x -> x - 2 (x, u) / (u, u) u maps the basis into L
            expected = s != 0 and math.gcd(*u) == 1 and all(Fraction(2 * L.product(e, u), s).denominator == 1
                                                            for e in basis)
            assert L.is_root(imat(u)) == expected

    @pytest.mark.parametrize("A", SMALL, ids=ident)
    def test_dual_vec(self, A):
        n = len(A)
        L = Lattice(n, A)
        for u in box(n, 1):
            v, d = L.dual_vec(u)
            assert d == math.gcd(*(L.product(u, x) for x in box(n, 1)))
            assert L.product(u, v) == d

    @pytest.mark.parametrize("A", SMALL, ids=ident)
    def test_products(self, A):
        n = len(A)
        L = Lattice(n, A)
        rng = random.Random(n)
        U = imat([[rng.randint(-3, 3) for _ in range(n)] for _ in range(3)])
        V = imat([[rng.randint(-3, 3) for _ in range(n)] for _ in range(2)])
        G = L.batch_prod(U, V)
        for i, j in product(range(3), range(2)):
            expected = sum(U[i, k] * A[k][m] * V[j, m] for k in range(n) for m in range(n))
            assert G[i, j] == L.product(U[i], V[j]) == L.product(U[i:i + 1], V[j]) == expected
        assert L.square(U[0]) == L.product(U[0], U[0])


class TestSublattices:
    @pytest.mark.parametrize("A", RANK2 + RANK3[:40], ids=ident)
    def test_even_sublattice(self, A):
        n = len(A)
        L = Lattice(n, A)
        B = L.even_sublattice()
        assert B.shape == (n, n)
        assert all(L.square(b) % 2 == 0 for b in B)
        span = Span(B)
        for x in box(n, 2):
            assert (x in span) == (L.square(x) % 2 == 0)

    @pytest.mark.parametrize("seed", range(40))
    def test_complement(self, seed):
        rng = random.Random(seed)
        A = SMALL[seed]
        n = len(A)
        L = Lattice(n, A)
        gens = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(rng.randint(0, n))]
        C = L.complement(matrix_or_empty(gens, n))
        assert C.shape == (n - Span(gens).rank, n)
        assert all(L.product(c, g) == 0 for c in C for g in gens)
        span = Span(C)
        for x in box(n, 2):
            assert (x in span) == all(L.product(x, g) == 0 for g in gens)

    @pytest.mark.parametrize("gens, n", random_generators(14, 60, 3, [2, 3], 3), ids=ident)
    def test_saturate(self, gens, n):
        S = Lattice.saturate(matrix_or_empty(gens, n))
        span, gspan = Span(S), Span(gens)
        assert S.shape == (gspan.rank, n)
        for x in box(n, 2):
            assert (x in span) == gspan.q_contains(x)

    @pytest.mark.parametrize("gens, n", random_generators(15, 60, 3, [2, 3], 3), ids=ident)
    def test_image_and_index(self, gens, n):
        I = Lattice.image(matrix_or_empty(gens, n))
        assert I.shape == (Span(gens).rank, n)
        assert hnf_rows(I) == hnf_rows(gens)
        # The index of span(gens) in its saturation S is the last determinantal divisor
        # of the coordinates of gens in a basis of S
        S = Lattice.saturate(matrix_or_empty(gens, n))
        if len(S) == 0:
            assert Lattice.index(matrix_or_empty(gens, n)) == 1
            return
        coords = sp.Matrix(gens) * sp.Matrix(S.tolist()).pinv()
        assert all(t.is_integer for t in coords)
        assert Lattice.index(gens) == minors_gcd(coords.tolist(), len(S))

    @pytest.mark.parametrize("seed", range(40))
    def test_fibre_product(self, seed):
        rng = random.Random(seed)
        n1, n2, k = rng.randint(0, 2), rng.randint(1, 2), rng.randint(1, 2)
        A1 = matrix_or_empty([[rng.randint(-3, 3) for _ in range(k)] for _ in range(n1)], k)
        A2 = imat([[rng.randint(-3, 3) for _ in range(k)] for _ in range(n2)])
        B1, B2 = Lattice.fibre_product(A1, A2)
        assert B1.shape[1] == n1 and B2.shape[1] == n2 and B1.shape[0] == B2.shape[0]
        assert (B1 @ A1 == B2 @ A2).all()
        span = Span(np.concatenate([B1, B2], axis=1))
        assert span.rank == B1.shape[0]
        for x in box(n1 + n2, 2):
            x = imat(x)
            assert (x in span) == (x[:n1] @ A1 == x[n1:] @ A2).all()

    @pytest.mark.parametrize("seed", range(40))
    def test_subquotient(self, seed):
        rng = random.Random(seed)
        A = SMALL[seed]
        n = len(A)
        L = Lattice(n, A)
        gens = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(rng.randint(1, n + 1))]
        # Make the form on the span degenerate, if possible: add a vector orthogonal to all gens
        C = L.complement(gens)
        if len(C):
            gens += [[int(x) for x in C[0]], [int(x) for x in imat(gens[0]) + C[0]]]
        R = L.subquotient(gens)
        G = imat(gens)
        assert R.shape == (sp.Matrix(L.batch_prod(G, G).tolist()).rank(), n)
        span = Span(gens)
        assert all(r in span for r in R)
        if len(R) == 0:
            return
        gram = sp.Matrix(L.batch_prod(R, R).tolist())
        assert gram.det() != 0
        # Every generator is a combination of the rows of R plus a vector of the kernel
        for g in gens:
            c = sp.Matrix([list(L.batch_prod(imat([g]), R)[0])]) * gram.inv()
            assert all(t.is_integer for t in c)
            w = imat(g) - imat([int(t) for t in c]) @ R
            assert all(L.product(w, h) == 0 for h in gens)


class TestConstructions:
    @pytest.mark.parametrize("A", RANK2[::5] + RANK3[:20], ids=ident)
    def test_make_primitive(self, A):
        L = Lattice(len(A), A)
        P = L.make_primitive()
        g = math.gcd(*(x for r in A for x in r))
        assert (P.A * g == L.A).all()
        assert math.gcd(*P.A.flatten().tolist()) == 1
        assert (L(5).make_primitive().A == P.A).all()

    @pytest.mark.parametrize("A", RANK2[::10], ids=ident)
    def test_sum_scale_multiple(self, A):
        L = Lattice(2, A)
        M = Lattice(1, [[3]])
        S = L + M
        assert S.A.tolist() == [A[0] + [0], A[1] + [0], [0, 0, 3]]
        assert S.disc == 3 * L.disc
        assert tuple(map(sum, zip(L.signature, M.signature))) == S.signature
        assert (3 * L).A.tolist() == (L + L + L).A.tolist() == (L * 3).A.tolist()
        assert (L(-2).A == -2 * L.A).all()
        assert L(-1).signature == L.signature[::-1]

    @pytest.mark.parametrize("A", RANK2, ids=ident)
    def test_clear_squares(self, A):
        L = Lattice(2, A)
        M = L.clear_squares()
        s = math.prod(max_square(d) for d in L.dgroup)
        assert M.disc * s * s == L.disc
        assert M.signature == L.signature
        assert all(max_square(d) == 1 for d in M.dgroup)
        assert M.clear_squares().dgroup == M.dgroup
