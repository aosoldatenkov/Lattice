"""Tests of lattice.decomposition, against Kneser's original method (all pairs of short vectors are tested
for decomposability, and the summands are spanned by the connected components of the remaining ones)."""
import random
import numpy as np
import pytest
import flint as fl
from lattice._matrix import imat, imat_diag, imat2flz
from lattice.lattice import Lattice
from lattice.decomposition import irred_decomp
from lattice.catalogue import A_lat, D_lat, E_lat, I_lat, U_lat
from lattice import fp_search_cpp
from _lattice_samples import random_definite, unimodular, conjugate


def sublattice(rows, n):
    """A canonical form of the sublattice spanned by the given rows of Z^n: its Hermite normal form."""
    H = imat2flz(imat(rows).reshape(-1, n)).hnf()
    return tuple(tuple(int(x) for x in H.tolist()[i]) for i in range(H.nrows()) if any(H.tolist()[i]))


def kneser(L):
    """The indecomposable summands of a positive definite L by Kneser's method, as canonical sublattices."""
    B, T = L.A_fl.lll(rep='gram', transform=True)
    G = np.array(B.tolist(), dtype=float)
    bound = max(int(B[i, i]) for i in range(L.rank))
    S = [imat([int(x) for x in v]) @ imat(T.tolist())
         for v in fp_search_cpp.FPSearch(G, np.zeros(L.rank), 0.5, bound + 0.5).search_all()]
    P = L.batch_prod(imat(S), imat(S))
    k = len(S)
    # v is decomposable iff v = w + (v - w) with w != 0, v orthogonal: (v, w) = (w, w) with w != v
    indec = [i for i in range(k) if not any(j != i and P[i, j] == P[j, j] for j in range(k))]
    # Connected components of the graph of non-orthogonal indecomposable vectors
    comp = {i: i for i in indec}

    def find(i):
        while comp[i] != i:
            i = comp[i]
        return i
    for a in indec:
        for b in indec:
            if P[a, b] != 0:
                comp[find(a)] = find(b)
    groups = {}
    for i in indec:
        groups.setdefault(find(i), []).append(S[i].tolist())
    return {sublattice(g, L.rank) for g in groups.values()}


def check(L):
    """irred_decomp(L) gives pairwise orthogonal summands spanning L, equal to those of Kneser's method."""
    parts = irred_decomp(L)
    n = L.rank
    assert sum(len(b) for b in parts) == n
    allrows = [r for b in parts for r in b]
    assert abs(int(imat2flz(imat(allrows)).det())) == 1
    for i, b in enumerate(parts):
        for c in parts[i + 1:]:
            assert not L.batch_prod(imat(b), imat(c)).any()
    sign = 1 if L.signature[1] == 0 else -1
    assert {sublattice(b, n) for b in parts} == kneser(L(sign))
    return parts


@pytest.mark.parametrize("diag", [[1, 1, 3], [1, 2, 3], [2, 2, 6], [1, 1, 1, 1], [3, 5, 7, 2]])
def test_diagonal(diag):
    """Decomposable vectors such as e_1 + e_2 of norm 2 < 3 used to merge summands."""
    parts = check(Lattice(len(diag), imat_diag(diag)))
    assert len(parts) == len(diag)


def test_root_lattices():
    L = E_lat(8) + A_lat(2) + D_lat(4) + A_lat(1) + A_lat(1)
    parts = check(L)
    assert sorted((len(b), int(Lattice(len(b), L.batch_prod(imat(b), imat(b))).A_fl.det())) for b in parts) == \
        [(1, 2), (1, 2), (2, 3), (4, 4), (8, 1)]


def test_negative_definite():
    parts = check(Lattice(4, imat_diag([-1, -1, -3, -2])) + A_lat(2)(-1))
    assert sorted(len(b) for b in parts) == [1, 1, 1, 1, 2]


def test_indecomposable():
    for L in (E_lat(8), A_lat(5), D_lat(5), Lattice(1, [[7]]), Lattice(2, [[2, 1], [1, 3]])):
        assert len(check(L)) == 1


def test_indefinite_rejected():
    for L in (U_lat(), I_lat(1, 2)):
        with pytest.raises(ValueError):
            irred_decomp(L)


def _random_sums(seed, count):
    """Orthogonal sums of random definite blocks of rank 1 to 3, in a random basis."""
    rng = random.Random(seed)
    blocks = random_definite(seed, 40, range(1, 4), 2) + [A.A.tolist() for A in (A_lat(2), D_lat(4), A_lat(3))]
    result = []
    for _ in range(count):
        L = None
        for A in rng.sample(blocks, rng.randint(1, 3)):
            B = Lattice(len(A), A)
            L = B if L is None else L + B
        if L.rank > 6:
            continue
        result.append(Lattice(L.rank, conjugate(L.A, unimodular(rng, L.rank))))
    return result


@pytest.mark.parametrize("L", _random_sums(1, 40))
def test_random_sums(L):
    check(L)


@pytest.mark.parametrize("seed", range(5))
def test_base_change(seed):
    """The summands are unique as sublattices: a change of basis T maps them to the summands of T L."""
    rng = random.Random(seed)
    L = Lattice(4, imat_diag([1, 1, 2, 2])) + A_lat(2)
    T = unimodular(rng, L.rank)
    M = Lattice(L.rank, conjugate(L.A, T))
    # The coordinates x in M correspond to x T in L
    mapped = {sublattice((imat(b) @ T).tolist(), L.rank) for b in irred_decomp(M)}
    assert mapped == {sublattice(b, L.rank) for b in irred_decomp(L)}
