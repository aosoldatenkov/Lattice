"""The Fincke-Pohst search (C++ FPSearch, FPSearchPy, fincke_pohst_search) against exact enumeration.

The bounds are chosen half-way between attainable values: for an integral A and b in (1/q) Z^n,
the values (x + b)^T A (x + b) lie in (1/q^2) Z, so bounds (k + 1/2) / q^2 are never attained and
the result does not depend on rounding."""
import functools
import math
import random
from fractions import Fraction
from itertools import product
import numpy as np
import pytest
from lattice.enumeration import FPSearchPy, fincke_pohst_search

fp_search_cpp = pytest.importorskip("lattice.fp_search_cpp")


def search_cpp(A, b, lo, hi):
    return fp_search_cpp.FPSearch(np.array(A, dtype=float), np.array(b, dtype=float), lo, hi).search_all()


def search_py(A, b, lo, hi):
    return FPSearchPy(A, b, lo, hi).search_all()


def search_gen(A, b, lo, hi):
    return [list(x) for x in fincke_pohst_search(np.array(A, dtype=float), np.array(b, dtype=float), lo, hi)]


SEARCHES = [search_cpp, search_py, search_gen]


def random_problem(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    while True:
        B = [[rng.randint(-2, 2) for _ in range(n + 1)] for _ in range(n)]
        A = [[sum(B[i][k] * B[j][k] for k in range(n + 1)) for j in range(n)] for i in range(n)]
        if abs(np.linalg.det(np.array(A, dtype=float))) > 0.5:
            break
    q = rng.randint(1, 6)
    b = [Fraction(rng.randint(-3 * q, 3 * q), q) for _ in range(n)]
    hi = rng.randint(1, 20 * q * q)
    lo = rng.randint(-q * q, hi - 1)
    return A, b, Fraction(2 * lo + 1, 2 * q * q), Fraction(2 * hi + 1, 2 * q * q)


@functools.cache
def exact(seed):
    """All x with lo < (x + b)^T A (x + b) <= hi, by enumeration of a box containing the ellipsoid,
    in integers: with b = c / q, the value is (q x + c)^T A (q x + c) / q^2."""
    A, b, lo, hi = random_problem(seed)
    n = len(A)
    q = math.lcm(*(t.denominator for t in b))
    c = [int(t * q) for t in b]
    lo2, hi2 = lo * q * q, hi * q * q
    inv = np.linalg.inv(np.array(A, dtype=float))
    radius = [math.isqrt(int(max(hi, 0) * inv[i, i]) + 1) + 2 for i in range(n)]
    out = []
    for x in product(*(range(math.floor(-b[i]) - radius[i], math.ceil(-b[i]) + radius[i] + 1) for i in range(n))):
        y = [q * x[i] + c[i] for i in range(n)]
        val = sum(y[i] * A[i][j] * y[j] for i in range(n) for j in range(n))
        if lo2 < val <= hi2:
            out.append(x)
    return sorted(out)


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("search", SEARCHES, ids=["cpp", "py", "generator"])
def test_matches_exact_enumeration(seed, search):
    A, b, lo, hi = random_problem(seed)
    found = sorted(tuple(x) for x in search(A, [float(t) for t in b], float(lo), float(hi)))
    assert found == exact(seed)


@pytest.mark.parametrize("search", SEARCHES, ids=["cpp", "py", "generator"])
def test_long_thin_ellipsoid(search):
    """Q(x) = (x_0 + 1000 x_1)^2 + x_1^2 (det 1) has solutions of Q = N with |x| ~ 1000 sqrt(N).
    A regularization A + 1e-10 I raises their computed values by about 1e-10 |x|^2 ~ 1 and loses them."""
    N = 10 ** 4
    A = [[1, 1000], [1000, 1000001]]
    found = sorted(tuple(x) for x in search(A, [0.0, 0.0], N - 0.5, N + 0.5))
    expected = sorted((a - 1000 * y, y) for y in range(-100, 101) for a in range(-100, 101) if a * a + y * y == N)
    assert found == expected and len(expected) == 20  # r_2(10^4) = 4 (4 + 1)


def test_batches_cover_search_all():
    A, b, lo, hi = random_problem(3)
    args = (np.array(A, dtype=float), np.array([float(t) for t in b]), float(lo), float(hi))
    fps = fp_search_cpp.FPSearch(*args)
    batches = []
    while not fps.exhausted():
        batches += fps.batch_search(3)
    assert sorted(map(tuple, batches)) == sorted(map(tuple, fp_search_cpp.FPSearch(*args).search_all()))


def test_not_positive_definite():
    with pytest.raises(ValueError):
        fp_search_cpp.FPSearch(np.array([[1.0, 0.0], [0.0, -1.0]]), np.zeros(2), 0, 5)
    with pytest.raises(np.linalg.LinAlgError):
        FPSearchPy([[1, 0], [0, 0]], [0, 0], 0, 5)


def test_coordinates_beyond_int():
    with pytest.raises(OverflowError):
        fp_search_cpp.FPSearch(np.array([[1.0]]), np.zeros(1), 0, 1e20)
    with pytest.raises(OverflowError):
        fp_search_cpp.FPSearch(np.array([[1.0]]), np.array([3e9]), 0, 1)


def test_empty_and_negative_bounds():
    for search in SEARCHES:
        assert search([[2, 1], [1, 2]], [0.0, 0.0], -5, -1) == []
        assert sorted(map(tuple, search([[2, 1], [1, 2]], [0.0, 0.0], -1, 0.5))) == [(0, 0)]
