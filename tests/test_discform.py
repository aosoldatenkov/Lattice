"""Tests of lattice.discform, partly against the brute-force discriminant forms of
_discriminant_forms.py."""
from collections import Counter
from itertools import product
import flint as fl
import pytest
from lattice._matrix import imat_diag, imat2flz
from lattice.lattice import Lattice
from lattice.discform import DiscForm
from lattice.genus import Genus
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat
from _lattice_samples import random_grams, random_squareful
from _discriminant_forms import BruteDiscForm

CATALOGUE = [L.A.tolist() for L in [A_lat(2), A_lat(3), D_lat(4), D_lat(5), U_lat(2), U_lat(4), U_lat(3) + A_lat(2),
                                    A_lat(2)(2), D_lat(4)(2) + U_lat(), Lattice(3, [[2, 0, 0], [0, 2, 0], [0, 0, -4]])]]
# Discriminant groups with many isotropic subgroups
CATALOGUE += [imat_diag(d).tolist() for d in [(2, 2, 2, 2), (4, 4), (4, -4), (2, 2, -8), (3, 3, -3), (9, -1), (8, -8), (3, 6, -6)]]
CATALOGUE += [(U_lat(2) + U_lat(2)).A.tolist(), (U_lat(3) + A_lat(2)(-1)).A.tolist(), (U_lat(4) + Lattice(1, [[4]])).A.tolist()]
# Lattices with small discriminant groups, as the isotropic elements are found by enumeration
SAMPLES = [A for A in CATALOGUE + random_grams(31, 60, range(1, 5), 5) + random_squareful(32, 50, range(1, 4))
           if Lattice(len(A), A).disc <= 256]


def is_integral(m):
    return all(x.denom() == 1 for x in m.entries())


def element(D, u):
    """The element u of A_L as a 1 x n matrix in the coordinates of the basis v_i of L^∨."""
    return fl.fmpq_mat(1, D.rank, [0] * (D.rank - len(D.ord)) + list(u))


def group(D):
    """All elements of A_L as coordinate vectors."""
    return list(product(*[range(d) for d in D.ord]))


@pytest.mark.parametrize("n", [8, 16])
def test_d_n_overlattices(n):
    # D_n^*/D_n = (Z/2)^2. Gluing in the vector class gives Z^n (odd); for n = 0 mod 8
    # the two spinor classes give even unimodular lattices (E8 and D16+).
    D = DiscForm(D_lat(n))
    assert D.ord == [2, 2]
    assert len(D.list_iso()) == 3  # elements u with b(u, u) in Z
    overlattices = [D.overlattice(sorted(s)) for s in D.list_max_isospaces()]
    assert all(M.disc == 1 and M.rank == n for M in overlattices)
    assert sorted(bool(M.parity) for M in overlattices) == [False, False, True]


def test_unimodular():
    D = DiscForm(E_lat(8))
    assert D.ord == [] and D.list_iso() == [] and D.list_max_isospaces() == []


def test_no_isotropic_elements():
    # A2^∨/A2 = Z/3 with b(u, u) = 2/3
    D = DiscForm(A_lat(2))
    assert D.ord == [3] and D.list_iso() == [] and D.list_max_isospaces() == []


def test_large_entries():
    """The Smith normal form is exact (hsnf used int64 and silently overflowed here)."""
    L = Lattice(2, [[2 * 10 ** 9 + 2, 3 ** 19], [3 ** 19, 2 * 10 ** 9 + 4]])
    D = DiscForm(L)
    assert D.ord == [d for d in L.dgroup if d > 1]
    assert is_integral(D.A.inv()) and D.A.det() * L.A_fl.det() == 1


@pytest.mark.parametrize("A", SAMPLES)
def test_embedding(A):
    """The rows of D.dual are the v_i, D.embedding gives the e_j in the v_i, and overlattice(embedding=True) gives
    the e_j in the basis of the overlattice, which preserves the form."""
    L = Lattice(len(A), A)
    D = DiscForm(L)
    E = fl.fmpq_mat(imat2flz(D.embedding))
    assert E * D.dual == fl.fmpq_mat(L.rank, L.rank, [int(i == j) for i in range(L.rank) for j in range(L.rank)])
    assert D.dual * fl.fmpq_mat(L.A_fl) * D.dual.transpose() == D.A
    for S in D.list_max_isospaces():
        M, X = D.overlattice(S, embedding=True)
        assert (imat2flz(X) * M.A_fl * imat2flz(X).transpose()) == L.A_fl
        assert Lattice.index(X) ** 2 * M.disc == L.disc


def test_iter_max_isospaces():
    """The iterator yields the same subgroups as the list, each once, and the first ones before the search ends."""
    D = DiscForm(E_lat(8)(2))
    it = D.iter_max_isospaces()
    first = next(it)
    rest = list(it)
    assert sorted([first] + rest, key=sorted) == D.list_max_isospaces() and len(rest) + 1 == 2295
    assert list(DiscForm(A_lat(2)).iter_max_isospaces()) == []


def test_overlattice_requires_orthogonal_elements():
    # U(2)^∨/U(2) = (Z/2)^2 with b(e1, e1) = b(e2, e2) = 0 and b(e1, e2) = 1/2
    D = DiscForm(U_lat(2))
    iso = D.list_iso()
    assert len(iso) == 3
    i, j = iso.index([0, 1]), iso.index([1, 0])
    with pytest.raises(ValueError):
        D.overlattice([i, j])
    assert D.overlattice([i]).disc == 1


@pytest.mark.parametrize("A", SAMPLES)
def test_dual_basis(A):
    """D.A is the Gram matrix of a basis v_i of L^∨ such that the d_i v_i form a basis of L."""
    L = Lattice(len(A), A)
    D = DiscForm(L)
    assert D.ord == [d for d in L.dgroup if d > 1]
    assert [int(D.D[i, i]) for i in range(L.rank)] == L.dgroup
    # The Gram matrix of L^∨ is the inverse of the Gram matrix of L in the dual basis
    assert is_integral(D.A.inv())
    Dq = fl.fmpq_mat(L.rank, L.rank, [D.D[i, j] for i in range(L.rank) for j in range(L.rank)])
    G, denom = (Dq * D.A * Dq).numer_denom()
    assert denom == 1 and G.det() == L.A_fl.det()
    assert Genus(Lattice(L.rank, G.tolist())) == Genus(L)
    # Ared is the reduction of D.A mod 1 on the nontrivial generators
    k = L.rank - len(D.ord)
    for i in range(len(D.ord)):
        for j in range(len(D.ord)):
            x, y = D.Ared[i, j], D.A[k + i, k + j]
            assert 0 <= x < 1 and (x - y).denom() == 1


@pytest.mark.parametrize("A", SAMPLES)
def test_bilinear_form(A):
    """The values b(u, u) (mod 1) over A_L agree with the brute-force discriminant form."""
    D = DiscForm(Lattice(len(A), A))
    B = BruteDiscForm(A)
    values = Counter()
    for u in group(D):
        x = element(D, u)
        v = (x * D.A * x.transpose())[0, 0]
        values[fl.fmpq(v.numer() % v.denom(), v.denom())] += 1
    expected = Counter()
    for u in B.elements:
        v = B.b(u, u)
        expected[fl.fmpq(v.numerator, v.denominator)] += 1
    assert values == expected


@pytest.mark.parametrize("A", SAMPLES)
def test_list_iso(A):
    D = DiscForm(Lattice(len(A), A))
    iso = D.list_iso()
    assert D.iso is iso
    assert len(iso) == len(BruteDiscForm(A).isotropic())
    assert len(set(map(tuple, iso))) == len(iso)
    for u in iso:
        assert any(u) and all(0 <= x < d for x, d in zip(u, D.ord))
        x = element(D, u)
        assert (x * D.A * x.transpose())[0, 0].denom() == 1


def _brute_max_isospaces(D):
    """The maximal sets of pairwise orthogonal isotropic elements, by enumerating all subsets."""
    iso = D.list_iso()
    vecs = [element(D, u) for u in iso]
    d = len(iso)
    orth = [[(vecs[i] * D.A * vecs[j].transpose())[0, 0].denom() == 1 for j in range(d)] for i in range(d)]
    cliques = [S for S in range(1, 2 ** d)
               if all(orth[i][j] for i in range(d) for j in range(d) if S >> i & 1 and S >> j & 1)]
    maximal = [S for S in cliques if not any(T != S and T & S == S for T in cliques)]
    return sorted(sorted(i for i in range(d) if S >> i & 1) for S in maximal)


@pytest.mark.parametrize("A", SAMPLES)
def test_max_isospaces(A):
    L = Lattice(len(A), A)
    D = DiscForm(L)
    spaces = D.list_max_isospaces()
    if len(D.iso) <= 12:
        assert sorted(sorted(S) for S in spaces) == _brute_max_isospaces(D)
    for S in spaces:
        # S together with 0 is a subgroup
        elems = {tuple(D.iso[i]) for i in S} | {tuple([0] * len(D.ord))}
        assert all(tuple((x + y) % d for x, y, d in zip(u, v, D.ord)) in elems for u in elems for v in elems)
        # The overlattice has index |S| + 1 over L, and is maximal: its discriminant group is anisotropic
        M = D.overlattice(S)
        assert M.rank == L.rank and M.signature == L.signature
        assert M.disc * (len(S) + 1) ** 2 == L.disc
        assert DiscForm(M).list_iso() == []
