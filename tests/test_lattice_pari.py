"""Comparison of lattice.lattice with PARI/GP (via cypari2).

PARI conventions: matrices act on column vectors, so a set of vectors is passed to PARI
as the matrix whose columns are the vectors; mathnf(M) is the (unique) Hermite normal form
of the column span of M; matsnf lists the elementary divisors in decreasing order;
matrixqz(M, -2) is a basis of the intersection of the Q-span of the columns with Z^n;
matkerint(M) is a basis of the integral kernel; qfisom works for definite forms only.
"""
import math
import pytest
from lattice._matrix import imat_zero
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat, I_lat, Leech_lat
from _lattice_samples import (random_grams, random_definite, random_squareful, random_generators, unimodular,
                              conjugate, ident)
import random

cypari2 = pytest.importorskip("cypari2")
pari = cypari2.Pari()

CATALOGUE = [L.A.tolist() for L in [A_lat(1), A_lat(4), D_lat(5), E_lat(6), E_lat(8), U_lat(), U_lat() + E_lat(8)(-1),
                                    I_lat(1, 4), A_lat(2)(3) + U_lat()(2), Leech_lat()]]
GRAMS = random_grams(1, 150, range(1, 9), 9) + CATALOGUE
POSITIVE = random_definite(2, 40, range(1, 7), 3)
NEGATIVE = random_definite(3, 15, range(1, 6), 3, sign=-1)
INDEFINITE = [A for A in random_grams(4, 40, range(2, 8), 6) if 0 not in Lattice(len(A), A).signature]
SQUAREFUL = random_squareful(5, 40, range(1, 5))
SQUAREFUL_DEF = random_squareful(6, 30, range(1, 5), signs=(1,))
GENERATORS = random_generators(7, 120, 5, range(1, 6), 4)


def pmat(rows, n):
    """The k x n PARI matrix with the given rows."""
    return pari.matrix(len(rows), n, [int(x) for r in rows for x in r])


def pcols(rows, n):
    """The n x k PARI matrix whose columns are the given vectors."""
    return pmat(rows, n).mattranspose()


def same_span(rows1, rows2, n):
    return pari.mathnf(pcols(rows1, n)) == pari.mathnf(pcols(rows2, n))


def ints(v):
    return [int(x) for x in v]


def isometric(L, M):
    """For definite lattices; PARI needs reduced Gram matrices to avoid overflows."""
    s = 1 if L.signature[1] == 0 else -1
    return pari.qfisom(pmat((s * L.lll()).tolist(), L.rank), pmat((s * M.lll()).tolist(), M.rank)) != 0


def max_square(n):
    return math.prod(p ** (e // 2) for p, e in (ints(f) for f in pari.factor(n).mattranspose()))


def is_ssf(L):
    """The invariant factors are square-free, and for every prime p at most rank/2 are divisible by p."""
    snf = ints(pari.matsnf(pmat(L.A.tolist(), L.rank)))
    if not all(pari.issquarefree(d) for d in snf):
        return False
    primes = ints(pari.factor(L.disc)[0]) if L.disc > 1 else []
    return all(2 * sum(1 for d in snf if d % p == 0) <= L.rank for p in primes)


class TestInvariants:
    @pytest.mark.parametrize("A", GRAMS, ids=ident)
    def test_matches_pari(self, A):
        n = len(A)
        L = Lattice(n, A)
        P = pmat(A, n)
        assert L.signature == tuple(ints(pari.qfsign(P)))
        assert L.disc == abs(int(pari.matdet(P)))
        assert L.dgroup == sorted(ints(pari.matsnf(P)))
        assert L.exp == max(L.dgroup)

    @pytest.mark.parametrize("A", GRAMS[:40], ids=ident)
    def test_sum_and_scale(self, A):
        L = Lattice(len(A), A)
        M = L + A_lat(2)(-2)
        n = len(A)
        P = pmat([r + [0, 0] for r in A] + [[0] * n + [-4, 2], [0] * n + [2, -4]], n + 2)
        assert M.dgroup == sorted(ints(pari.matsnf(P)))
        assert M.signature == tuple(ints(pari.qfsign(P)))
        assert L(-3).dgroup == sorted(ints(pari.matsnf(-3 * pmat(A, len(A)))))


class TestSublattices:
    @pytest.mark.parametrize("gens, n", GENERATORS, ids=ident)
    def test_saturate(self, gens, n):
        S = Lattice.saturate(imat_zero(0, n) if not gens else gens)
        H = pari.mathnf(pcols(gens, n))
        assert S.shape == (len(H), n)
        assert same_span(S.tolist(), ints_cols(pari.matrixqz(H, -2)), n) if len(H) else True

    @pytest.mark.parametrize("gens, n", GENERATORS, ids=ident)
    def test_image(self, gens, n):
        I = Lattice.image(imat_zero(0, n) if not gens else gens)
        assert I.shape == (int(pari.matrank(pcols(gens, n))), n)
        assert same_span(I.tolist(), gens, n)

    @pytest.mark.parametrize("gens, n", GENERATORS, ids=ident)
    def test_index(self, gens, n):
        H = pari.mathnf(pcols(gens, n))
        if len(H) == 0:
            assert Lattice.index(imat_zero(0, n) if not gens else gens) == 1
            return
        S = pari.matrixqz(H, -2)
        # index^2 = det Gram(H) / det Gram(S)
        ratio = pari.matdet(H.mattranspose() * H) / pari.matdet(S.mattranspose() * S)
        assert Lattice.index(gens) ** 2 == int(ratio)

    @pytest.mark.parametrize("seed", range(80))
    def test_complement(self, seed):
        rng = random.Random(seed)
        A = GRAMS[seed]
        n = len(A)
        L = Lattice(n, A)
        gens = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(rng.randint(0, min(n, 3)))]
        C = L.complement(imat_zero(0, n) if not gens else gens)
        K = pari.matkerint(pmat(gens, n) * pmat(A, n)) if gens else pari.matid(n)
        assert C.shape == (len(K), n)
        assert same_span(C.tolist(), ints_cols(K), n)

    @pytest.mark.parametrize("A", GRAMS[:100], ids=ident)
    def test_dual_vec(self, A):
        n = len(A)
        L = Lattice(n, A)
        rng = random.Random(n)
        for _ in range(5):
            u = [rng.randint(-4, 4) * rng.choice([1, 2, 6]) for _ in range(n)]
            v, d = L.dual_vec(u)
            assert d == int(pari.content(pmat([u], n) * pmat(A, n)))
            assert L.product(u, v) == d


def ints_cols(M):
    """The columns of a PARI matrix, as lists of ints."""
    return [ints(c) for c in M] if len(M) else []


class TestLLL:
    @pytest.mark.parametrize("A", POSITIVE + NEGATIVE + [A for A in CATALOGUE if Lattice(len(A), A).signature[0]
                                                          in (0, len(A))], ids=ident)
    def test_definite(self, A):
        n = len(A)
        L = Lattice(n, A)
        M, T = L.lll(transform=True)
        assert (T @ L.A @ T.T == M).all()
        assert abs(int(pari.matdet(pmat(T.tolist(), n)))) == 1
        assert L.lll().tolist() == M.tolist()
        s = 1 if L.signature[1] == 0 else -1
        if n <= 12:  # qfisom exhausts the default PARI stack on the Leech lattice
            assert isometric(L, Lattice(n, M))
        # LLL with delta = 0.99: |b_1|^2 <= (4 / (4 delta - 1))^(n - 1) lambda_1^2
        minimum = int(pari.qfminim(pmat((s * L.A).tolist(), n), None, 0)[1])
        assert int(s * M[0, 0]) * 296 ** (n - 1) <= 400 ** (n - 1) * minimum

    @pytest.mark.parametrize("A", INDEFINITE + [A for A in CATALOGUE if 0 not in Lattice(len(A), A).signature],
                             ids=ident)
    def test_indefinite(self, A):
        n = len(A)
        L = Lattice(n, A)
        M, T = L.lll(transform=True)
        assert (T @ L.A @ T.T == M).all()
        assert abs(int(pari.matdet(pmat(T.tolist(), n)))) == 1
        assert Lattice(n, M).signature == L.signature


class TestSquareFree:
    @pytest.mark.parametrize("A", SQUAREFUL + SQUAREFUL_DEF, ids=ident)
    def test_clear_squares(self, A):
        n = len(A)
        L = Lattice(n, A)
        M = L.clear_squares()
        s = math.prod(max_square(d) for d in ints(pari.matsnf(pmat(A, n))))
        assert M.disc * s * s == L.disc
        assert M.signature == L.signature
        assert all(pari.issquarefree(d) for d in M.dgroup)

    @pytest.mark.parametrize("A", SQUAREFUL_DEF, ids=ident)
    def test_clear_squares_is_canonical(self, A):
        n = len(A)
        L = Lattice(n, A)
        L2 = Lattice(n, conjugate(A, unimodular(random.Random(n), n)))
        assert isometric(L.clear_squares(), L2.clear_squares())

    @pytest.mark.parametrize("A", SQUAREFUL + SQUAREFUL_DEF + GRAMS[:60], ids=ident)
    def test_make_ssf(self, A):
        L = Lattice(len(A), A)
        S = L.make_ssf()
        assert is_ssf(S)
        assert S.signature == L.signature
        T = S.make_ssf()
        assert (T.dgroup, T.signature) == (S.dgroup, S.signature)

    @pytest.mark.parametrize("A", SQUAREFUL_DEF + POSITIVE[:20], ids=ident)
    @pytest.mark.parametrize("p", [2, 3, 5])
    def test_make_ssf_up_to_isometry_and_scale(self, A, p):
        n = len(A)
        L = Lattice(n, A)
        S = L.make_ssf()
        primes = ints(pari.factor(S.disc * p)[0])
        if any(2 * sum(1 for d in S.dgroup if d % q == 0) == n for q in primes):
            pytest.skip("p-rank n/2: the associated lattice is not unique")
        assert isometric(S, Lattice(n, conjugate(A, unimodular(random.Random(n), n))).make_ssf())
        assert isometric(S, L(p).make_ssf())
