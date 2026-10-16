"""Comparison of lattice.binary with PARI/GP (via cypari2) on nondegenerate binary lattices.

The norm form of BinLattice(a, b, h) is the binary quadratic form a x^2 + 2h xy + b y^2,
i.e. Qfb(a, 2h, b) in PARI, of discriminant 4(h^2 - ab) = -4 det. Even lattices correspond
to all integral forms: (A, B, C) <-> Gram matrix [[2A, B], [B, 2C]], of discriminant
B^2 - 4AC = -det. PARI does not accept forms of square discriminant, so lattices
representing zero are excluded here.

PARI conventions: equivalence of forms is SL_2(Z)-equivalence (our oriented isomorphism);
qfauto and qfisom act on column vectors; qfbsolve(Q, n, 1) lists the primitive solutions of
Q(x, y) = n up to the proper automorphisms of Q.
"""
import math
from collections import defaultdict
from itertools import combinations
import pytest
from lattice.binary import BinLattice, list_bin_lattices

cypari2 = pytest.importorskip("cypari2")
pari = cypari2.Pari()

BOX = range(-8, 9)


def represents_zero(a, b, h):
    d = h * h - a * b
    return d >= 0 and math.isqrt(d) ** 2 == d


# Nondegenerate Gram data (a, b, h) not representing zero, with entries in BOX
FORMS = [(a, b, h) for a in BOX for b in BOX for h in BOX if a * b != h * h and not represents_zero(a, b, h)]


@pytest.fixture(scope="module")
def lattices():
    return {f: BinLattice(*f) for f in FORMS}


def pari_ints(v):
    return tuple(int(x) for x in v)


def pari_sl_class(a, b, h):
    """A key of the SL_2(Z)-class of a x^2 + 2h xy + b y^2 computed by PARI: the unique reduced
    form (definite; negated if negative definite), or the minimal form in the cycle of reduced
    forms (indefinite)."""
    if a * b - h * h > 0:
        s = 1 if a > 0 else -1
        return ("definite", s, pari_ints(pari.qfbred(pari.Qfb(s * a, 2 * s * h, s * b))[:3]))
    f0 = pari.qfbred(pari.Qfb(a, 2 * h, b))
    g, cycle = f0, []
    while True:
        cycle.append(pari_ints(g[:3]))
        g = pari.qfbred(g, 1)  # one reduction step: the next form in the cycle
        if g == f0:
            return ("indefinite", min(cycle))


def pari_gl_class(a, b, h):
    """(a, b, -h) is the form in the basis (e, -f), of the opposite orientation."""
    return min(pari_sl_class(a, b, h), pari_sl_class(a, b, -h))


def positive_gram(L):
    """The Gram matrix of L, or of L(-1) if L is negative definite, as a PARI matrix."""
    s = 1 if L.signature == (2, 0) else -1
    return s, pari.matrix(2, 2, [s * L.a, s * L.h, s * L.h, s * L.b])


def proper_automorphisms(G):
    """The proper automorphisms of a positive definite Gram matrix (acting on column vectors),
    generated from qfauto."""
    gens = [tuple(tuple(int(M[i, j]) for j in range(2)) for i in range(2)) for M in pari.qfauto(G)[1]]
    mul = lambda X, Y: tuple(tuple(sum(X[i][k] * Y[k][j] for k in range(2)) for j in range(2)) for i in range(2))
    group = {((1, 0), (0, 1))}
    frontier = list(group)
    while frontier:
        X = frontier.pop()
        for Y in gens:
            Z = mul(X, Y)
            if Z not in group:
                group.add(Z)
                frontier.append(Z)
    return [X for X in group if X[0][0] * X[1][1] - X[0][1] * X[1][0] == 1]


@pytest.mark.parametrize("oriented", [False, True], ids=["GL2", "SL2"])
def test_isomorphism_classes(lattices, oriented):
    """is_isomorphic induces the same partition of FORMS as PARI's reduction."""
    key = pari_sl_class if oriented else pari_gl_class
    classes = defaultdict(list)
    for f in FORMS:
        classes[key(*f)].append(f)
    reps = defaultdict(list)
    for fs in classes.values():
        rep = lattices[fs[0]]
        assert all(rep.is_isomorphic(lattices[f], oriented=oriented) for f in fs[1:]), fs[0]
        reps[(rep.disc, rep.signature)].append(rep)
    for group in reps.values():
        for L, M in combinations(group, 2):
            assert not L.is_isomorphic(M, oriented=oriented), ((L.a, L.b, L.h), (M.a, M.b, M.h))


def test_improper_automorphism(lattices):
    """An automorphism of determinant -1 exists iff the form is SL_2(Z)-equivalent to its
    opposite (a, b, -h)."""
    for (a, b, h), L in lattices.items():
        S = L.improper_automorphism()
        assert (S is not None) == (pari_sl_class(a, b, h) == pari_sl_class(a, b, -h)), (a, b, h)
        if S is not None:
            assert (S @ L.A @ S.T == L.A).all() and S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0] == -1


def test_qfisom_definite(lattices):
    """For definite lattices, is_isomorphic agrees with PARI's lattice isomorphism test."""
    by_det = defaultdict(list)
    for L in lattices.values():
        if L.signature != (1, 1) and L.disc <= 20:
            by_det[(L.disc, L.signature)].append(L)
    for group in by_det.values():
        for L, M in combinations(group[:12], 2):
            assert L.is_isomorphic(M) == (pari.qfisom(positive_gram(L)[1], positive_gram(M)[1]) != 0)


MAX_D = 200


def sl_class_count(lats):
    """The number of SL_2(Z)-classes in a list of GL_2(Z)-class representatives."""
    return sum(1 if L.improper_automorphism() is not None else 2 for L in lats)


def primitive_even(L):
    return math.gcd(L.a // 2, L.h, L.b // 2) == 1


def test_class_numbers_definite():
    """SL_2(Z)-classes of primitive positive definite forms of discriminant -d, i.e. of
    primitive even positive definite lattices of determinant d, number h(-d)."""
    listed = list_bin_lattices(MAX_D, signature=(2, 0), parity=0)
    for d in range(1, MAX_D + 1):
        if d % 4 not in (0, 3):
            assert not listed.get(d)
            continue
        count = sl_class_count([L for L in listed.get(d, []) if primitive_even(L)])
        assert count == int(pari.qfbclassno(-d)), d


def test_class_numbers_indefinite():
    """SL_2(Z)-classes of primitive indefinite forms of nonsquare discriminant D number the
    narrow class number h+(D) = h(D), or 2 h(D) if the fundamental unit has norm +1."""
    listed = list_bin_lattices(MAX_D, signature=(1, 1), parity=0)
    for D in range(2, MAX_D + 1):
        if D % 4 not in (0, 1) or math.isqrt(D) ** 2 == D:
            continue
        narrow = int(pari.qfbclassno(D)) * (2 if int(pari.norm(pari.quadunit(D))) == 1 else 1)
        count = sl_class_count([L for L in listed.get(D, []) if primitive_even(L)])
        assert count == narrow, D


def test_genus_counts_definite():
    """Gauss: the primitive positive definite forms of discriminant -d fall into
    |Cl(-d) / Cl(-d)^2| = 2^(number of even invariants of Cl(-d)) genera."""
    listed = list_bin_lattices(MAX_D, signature=(2, 0), parity=0)
    for d, lats in listed.items():
        reps = []
        for L in (L for L in lats if primitive_even(L)):
            if not any(L.is_same_genus(M) for M in reps):
                reps.append(L)
        cyc = pari.quadclassunit(-d)[1]
        assert len(reps) == 2 ** sum(1 for c in cyc if int(c) % 2 == 0), d


REPRESENTATION_FORMS = [
    (1, 1, 0), (2, 2, 1), (4, 6, 1), (2, 10, 0), (3, 5, 1), (-2, -4, 1), (6, 6, 3),  # definite
    (1, -3, 0), (2, -3, 1), (1, -7, 2), (2, -2, 1), (-3, 5, 2), (4, -6, 3), (3, -6, 0),  # indefinite
]
BOUND = 60


@pytest.mark.parametrize("abh", REPRESENTATION_FORMS)
def test_representations(abh):
    """Primitive representations of norms n with |n| <= BOUND: list_positive / list_negative
    against qfbsolve. Definite lattices: the full sets of vectors (up to sign). Indefinite
    lattices: the numbers of orbits of the proper automorphism group <shift, -1>."""
    a, b, h = abh
    L = BinLattice(a, b, h)
    if L.signature == (1, 1):
        pos, neg = L.list_positive(BOUND), L.list_negative(BOUND)
        for n in range(1, BOUND + 1):
            for m, ours in ((n, pos), (-n, neg)):
                sols = pari.qfbsolve(pari.Qfb(a, 2 * h, b), m, 1)
                assert len(ours.get(m, ())) == len(sols), (abh, m)
        return
    s, G = positive_gram(L)
    SO = proper_automorphisms(G)
    ours_all = L.list_positive(BOUND) if s == 1 else L.list_negative(BOUND)
    for n in range(1, BOUND + 1):
        sols = [pari_ints(v) for v in pari.qfbsolve(pari.Qfb(s * a, 2 * s * h, s * b), n, 1)]
        expanded = {L._normalize((M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y)) for M in SO for x, y in sols}
        assert set(ours_all.get(s * n, ())) == expanded, (abh, n)


@pytest.mark.parametrize("abh", [f for f in REPRESENTATION_FORMS if f[0] * f[1] - f[2] ** 2 > 0])
def test_theta_series_definite(abh):
    """The numbers of all (not only primitive) vectors of each norm, up to sign: qfrep."""
    L = BinLattice(*abh)
    s, G = positive_gram(L)
    prim = L.list_positive(BOUND) if s == 1 else L.list_negative(BOUND)
    counts = pari.qfrep(G, BOUND)
    for n in range(1, BOUND + 1):
        total = sum(len(prim.get(s * n // (k * k), ())) for k in range(1, math.isqrt(n) + 1) if n % (k * k) == 0)
        assert total == int(counts[n - 1]), (abh, n)


@pytest.mark.parametrize("abh", [f for f in REPRESENTATION_FORMS if f[0] * f[1] - f[2] ** 2 > 0])
def test_minimum_definite(abh):
    L = BinLattice(*abh)
    s, G = positive_gram(L)
    minimum = int(pari.qfminim(G)[1])
    assert (L.min_positive() if s == 1 else -L.max_negative()) == minimum
