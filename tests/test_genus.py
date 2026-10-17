"""Tests of lattice.genus: the p-adic symbols and the comparison of genera.

The comparison of 2-adic symbols is checked against an independent oracle (the comparison of
discriminant quadratic forms, see _discriminant_forms.py), on hand-picked pairs where the
Jordan decompositions differ by oddity fusion or sign walking, and on families of lattices
built from 2-adically interesting blocks and hidden by random changes of basis.
"""
import random
from collections import defaultdict
from itertools import combinations, combinations_with_replacement
import pytest
from sympy import factorint
from sympy.functions.combinatorial.numbers import legendre_symbol
from lattice._matrix import imat, imat_diag
from lattice.lattice import Lattice
from lattice.genus import Genus, ComPAdic, Com2Adic
from lattice.discform import DiscForm
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat, I_lat
from test_lattice import unimodular
from _lattice_samples import random_grams, random_squareful, conjugate
from _lattice_samples import unimodular as unimodular_rng
from _discriminant_forms import same_genus


def diag(*d):
    return Lattice(len(d), imat_diag(list(d)))


def V(k=1):
    """The even binary form [[2k, k], [k, 2k]], of type II with sign -1 at 2."""
    return A_lat(2)(k)


LATTICES = {
    "II_1,9": U_lat() + E_lat(8)(-1),
    "A2+<-6>": A_lat(2) + Lattice(1, [[-6]]),
    "diag(1,1,2,4,-8)": diag(1, 1, 2, 4, -8),
    "D4+U(2)": D_lat(4) + U_lat(2),
    "diag(3,-5,7)": diag(3, -5, 7),
    "U(2)+V(4)+<-24>": U_lat(2) + V(4) + diag(-24),
    "diag(1,-4,3)+V(2)": diag(1, -4, 3) + V(2),
    "A2(2)+<-8>+<12>": A_lat(2)(2) + diag(-8, 12),
}

# Samples for the consistency checks of the symbols
SAMPLES = ([L.A.tolist() for L in LATTICES.values()] + random_grams(21, 80, range(1, 6), 6) +
           random_squareful(22, 60, range(1, 5)))


def test_symbol_string():
    assert Genus(U_lat() + E_lat(8)(-1)).str() == "II_{1, 9}()"
    assert Genus(diag(1, -4, 3)).str() == "I_{2, 1}(1_{4}^{-2}4_{7}^{1}:1^{-2}3^{1})"
    assert Genus(diag(1, -4, 3)).str(5) == ""


class TestSymbol:
    def test_unimodular(self):
        assert Genus(E_lat(8)).sym == {}
        assert Genus(I_lat(1, 3)).sym == {}

    def test_rank_one(self):
        # -6 = 2 * (-3) with -3 = 5 mod 8; -6 = 3 * (-2) with (-2 / 3) = 1
        assert Genus(diag(-6)).sym == {2: {1: Com2Adic('I', 5, -1, 1)}, 3: {1: ComPAdic(1, 1)}}

    def test_diagonal(self):
        # 2-adically <1, 3> + <-4>: oddity 1 + 3, sign (2 / 3) = -1; then -1 = 7 mod 8.
        # 3-adically <1, -4> + <3>: sign (-4 / 3) = -1; then (1 / 3) = 1.
        assert Genus(diag(1, -4, 3)).sym == {
            2: {0: Com2Adic('I', 4, -1, 2), 2: Com2Adic('I', 7, 1, 1)},
            3: {0: ComPAdic(-1, 2), 1: ComPAdic(1, 1)}}

    def test_type_II(self):
        # A2 is 2-adically even unimodular of determinant 3 (sign -1), U(2) = 2 U with det(U) = -1 = 7 mod 8.
        # 3-adically A2 = <2> + <3 * 2> and U(2) is unimodular of determinant -4: the unimodular
        # constituent <2> + U(2) has sign (-8 / 3) = 1, and the other one (2 / 3) = -1.
        assert Genus(A_lat(2) + U_lat(2)).sym == {
            2: {0: Com2Adic('II', 0, -1, 2), 1: Com2Adic('II', 0, 1, 2)},
            3: {0: ComPAdic(1, 3), 1: ComPAdic(-1, 1)}}

    def test_exponent_too_small(self):
        with pytest.raises(ValueError):
            Genus.symbol(imat([[4]]), 2, 1)

    @pytest.mark.parametrize("A", SAMPLES)
    def test_dimensions_and_scales(self, A):
        L = Lattice(len(A), A)
        exp, disc = factorint(L.exp), factorint(L.disc)
        G = Genus(L)
        assert sorted(G.sym) == sorted(exp)
        for p, s in G.sym.items():
            assert sum(c.dim for c in s.values()) == L.rank
            assert sum(e * c.dim for e, c in s.items()) == disc[p]
            assert max(s) == exp[p] and min(s) >= 0 and all(c.dim > 0 for c in s.values())

    @pytest.mark.parametrize("A", SAMPLES)
    def test_signs_and_determinant(self, A):
        """The product of the signs is the sign of the unit part of the determinant."""
        L = Lattice(len(A), A)
        det = int(L.A_fl.det())
        for p, s in Genus(L).sym.items():
            u = det // p ** sum(e * c.dim for e, c in s.items())
            expected = legendre_symbol(u, p) if p > 2 else (1 if u % 8 in (1, 7) else -1)
            prod = 1
            for c in s.values():
                prod *= c.sign
            assert prod == expected

    @pytest.mark.parametrize("A", SAMPLES)
    def test_2adic_constituents(self, A):
        L = Lattice(len(A), A)
        s = Genus(L).sym.get(2)
        if s is None:
            return
        for c in s.values():
            if c.type == 'II':
                assert c.tr == 0 and c.dim % 2 == 0
            else:
                assert c.tr % 2 == c.dim % 2
                if c.dim == 1:
                    assert c.sign == (1 if c.tr in (1, 7) else -1)
        # L is odd iff the unimodular constituent is odd
        assert L.parity == (0 in s and s[0].type == 'I')

    @pytest.mark.parametrize("A", SAMPLES)
    def test_oddity_formula(self, A):
        """signature + sum of the p-excesses = oddity (mod 8), [CS, Ch. 15, eq. (30)]."""
        L = Lattice(len(A), A)
        sym = Genus(L).sym
        if 2 not in sym and L.parity:
            return  # the 2-adic symbol of an odd lattice with odd determinant is not stored
        total = L.signature[0] - L.signature[1]
        for p, s in sym.items():
            # The terms 4 k_q for the constituents with q = p^odd and ε_q = -1
            k = 4 * sum(1 for e, c in s.items() if e % 2 == 1 and c.sign == -1)
            if p == 2:
                total -= sum(c.tr for c in s.values()) + k
            else:
                total += sum(c.dim * (p ** e - 1) for e, c in s.items()) + k
        assert total % 8 == 0


@pytest.mark.parametrize("name", LATTICES)
@pytest.mark.parametrize("seed", [0, 1])
def test_base_change_invariance(name, seed):
    L = LATTICES[name]
    T = unimodular(L.rank, seed)
    assert Genus(L) == Genus(Lattice(L.rank, T @ L.A @ T.T))


def test_distinct_genera_same_determinant():
    # x^2 + 5y^2 and 2x^2 + 2xy + 3y^2: the two genera of discriminant -20
    assert Genus(Lattice(2, [[1, 0], [0, 5]])) != Genus(Lattice(2, [[2, 1], [1, 3]]))


def test_rescaling_changes_genus():
    L = A_lat(2)
    assert Genus(L) != Genus(L(3))


def test_parity():
    assert Genus(U_lat() + E_lat(8)(-1)) != Genus(I_lat(1, 9))


def test_not_a_genus():
    G = Genus(A_lat(2))
    assert G != A_lat(2) and not (G == None)


def test_even_unimodular_rank_16():
    D = DiscForm(D_lat(16))
    even = [M for M in (D.overlattice(sorted(s)) for s in D.list_max_isospaces()) if not M.parity]
    assert len(even) == 2  # both give D16+
    assert Genus(even[0]) == Genus(E_lat(8) + E_lat(8))


# Pairs of lattices of the same genus, whose 2-adic Jordan decompositions found by
# Genus.symbol have different signs or oddities
SAME_GENUS = [
    # Oddity fusion: 1_1 2_7 vs 1_7 2_1 (one compartment, total oddity 0)
    (diag(1, -2), diag(-1, 2)),
    (diag(1, 1, -2), diag(1, -1, 2)),
    # Sign walking between adjacent scales: 1_7^1 2_3^-1 vs 1_5^-1 2_1^1
    (diag(-1, 6), diag(2, -3)),
    (diag(2, -24), diag(8, -6)),
    # Sign walking past a zero constituent between two odd ones: 1_7^1 4_3^-1 vs 1_3^-1 4_7^1
    (diag(-1, 12), diag(-4, 3)),
    (diag(2, -3, 40), diag(8, -3, 10)),
    (diag(1, -12, 14), diag(4, -3, 14)),
    # Sign walking between a type II and a type I constituent: 1^2 2_3^-1 vs 1^-2 2_7^1
    (U_lat() + diag(6), V() + diag(-2)),
    (U_lat(4) + diag(-24), V(-4) + diag(8)),
]

# Pairs of lattices of different genera with the same dimensions, types and odd p-adic symbols,
# and the same total sign at 2; only condition (b) of Genus.__eq__ distinguishes them
DIFFERENT_GENERA = [
    # The compartments {1} and {4} are separated by the zero constituent at scale 2, so
    # their oddities cannot fuse, although the total oddity is the same: 1_1 4_7 vs 1_7 4_1
    (diag(1, -4), diag(-1, 4)),
    (diag(1, 1, -4), diag(1, -1, 4)),
    (diag(2, -8), diag(-2, 8)),
    # The signs cannot walk past the two zero constituents at scales 2 and 4 (different trains)
    (diag(-1, 3, -24), diag(8, -3, -3)),
    (U_lat() + diag(24), V() + diag(-8)),
    (diag(2, 7, 24), diag(8, 6, 7)),
    (diag(1, -24, -40), diag(-1, -24, 40)),
]


@pytest.mark.parametrize("L, M", SAME_GENUS)
def test_same_genus(L, M):
    G, H = Genus(L), Genus(M)
    assert G.sym[2] != H.sym[2]  # the example does exercise the 2-adic comparison
    assert same_genus(L, M)
    assert G == H and H == G


@pytest.mark.parametrize("L, M", DIFFERENT_GENERA)
def test_different_genera(L, M):
    G, H = Genus(L), Genus(M)
    a, b = G.sym[2], H.sym[2]
    assert (L.signature, L.parity) == (M.signature, M.parity)
    assert all(G.sym[p] == H.sym[p] for p in G.sym if p > 2) and G.sym.keys() == H.sym.keys()
    assert {e: (c.dim, c.type) for e, c in a.items()} == {e: (c.dim, c.type) for e, c in b.items()}
    assert sum(a[e].sign != b[e].sign for e in a) % 2 == 0
    assert not same_genus(L, M)
    assert G != H and H != G


@pytest.mark.parametrize("c", [-1, 2, 3, 12, 3 * 2 ** 70])
def test_rescaled_pairs(c):
    """The genus is invariant under rescaling, which moves the constituents to other scales."""
    for L, M in SAME_GENUS:
        assert Genus(L(c)) == Genus(M(c))
    for L, M in DIFFERENT_GENERA:
        assert Genus(L(c)) != Genus(M(c))


def _block_family(rank, seed, count):
    """count lattices of the given rank, orthogonal sums of 1-dimensional forms ±2^k u and
    of the even forms U(2^k), V(±2^k), hidden by a random change of basis."""
    rng = random.Random(seed)
    units = [s * v for v in (1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 7) for s in (1, -1)]
    binary = [U_lat(q) for q in (1, 2, 4)] + [V(s * q) for q in (1, 2, 4) for s in (1, -1)]
    shapes = [(k, bl, dg) for k in range(rank // 2 + 1)
              for bl in combinations_with_replacement(range(len(binary)), k)
              for dg in combinations_with_replacement(units, rank - 2 * k)]
    result = []
    for k, bl, dg in rng.sample(shapes, min(count, len(shapes))):
        L = diag(*dg) if dg else None
        for i in bl:
            L = binary[i] if L is None else L + binary[i]
        T = unimodular_rng(rng, rank)
        result.append(Lattice(rank, conjugate(L.A, T)))
    return result


@pytest.mark.parametrize("rank, seed, max_disc, max_pairs", [(2, 0, 1000, 400), (3, 1, 300, 150), (4, 2, 64, 60)])
def test_against_discriminant_forms(rank, seed, max_disc, max_pairs):
    """Genus equality agrees with the comparison of discriminant quadratic forms, for the pairs
    of lattices with the same signature, parity and discriminant group."""
    buckets = defaultdict(list)
    for L in _block_family(rank, seed, 1000):
        if L.disc <= max_disc:
            buckets[(L.signature, L.parity, tuple(L.dgroup))].append(L)
    pairs = [pair for b in buckets.values() for pair in combinations(b, 2)]
    random.Random(seed).shuffle(pairs)
    same = 0
    for L, M in pairs[:max_pairs]:
        expected = same_genus(L, M)
        same += expected
        assert (Genus(L) == Genus(M)) == expected, (L.A.tolist(), M.A.tolist())
    assert 0 < same < min(len(pairs), max_pairs)  # both outcomes occur
