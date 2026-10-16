from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import pytest
from lattice._matrix import imat
from lattice.binary import BinLattice
from lattice.lattice import Lattice
from lattice.genus import Genus

BOX = range(-4, 5)


def transform(L, g):
    G = imat(g) @ L.A @ imat(g).T
    return BinLattice(G[0, 0], G[1, 1], G[0, 1])


class TestDiscriminantForm:
    def test_a2(self):
        D = BinLattice(2, 2, -1).discriminant_form()
        assert (D.orders, D.mod) == ((1, 3), 2)
        assert D.value((0, 1)) == Fraction(2, 3)  # A2^*/A2 = Z/3 with q = 2/3 mod 2

    def test_hyperbolic_plane(self):
        D = BinLattice(0, 0, 3).discriminant_form()
        assert D.orders == (3, 3)
        assert D.value((1, 0)) == D.value((0, 1)) == 0 and D.bilinear((1, 0), (0, 1)) == Fraction(1, 3)

    def test_odd_lattice_mod_1(self):
        D = BinLattice(1, 3, 0).discriminant_form()
        assert (D.orders, D.mod, D.value((0, 1))) == ((1, 3), 1, Fraction(1, 3))

    def test_generators_in_dual(self):
        for L in (BinLattice(4, 6, 2), BinLattice(2, -6, 3), BinLattice(0, 0, 4), BinLattice(6, 10, -2)):
            D = L.discriminant_form()
            for g, d in zip(D.gens, D.orders):
                assert all(L.prod(g, e).denominator == 1 for e in ((1, 0), (0, 1)))  # g in L^*
                assert all((d * x).denominator == 1 for x in g)  # d g in L
            assert D.orders[0] * D.orders[1] == L.disc

    def test_q_distinguishes(self):
        # U(2) and <2> + <2> both have the group (Z/2)^2, with q values {0, 0, 1} and {1/2, 1/2, 1}
        assert not BinLattice(0, 0, 2).discriminant_form().is_isomorphic(BinLattice(2, 2, 0).discriminant_form())
        # q(x) = 2/3 vs 4/3 mod 2: A2 and A2(-1)
        assert not BinLattice(2, 2, -1).discriminant_form().is_isomorphic(BinLattice(-2, -2, 1).discriminant_form())

    @pytest.mark.parametrize("abh", [(2, 2, -1), (4, 6, 2), (2, -6, 3), (0, 0, 6), (1, 7, 3), (-3, 5, 1)])
    @pytest.mark.parametrize("g", [[[2, 1], [1, 1]], [[1, 3], [0, -1]], [[0, 1], [1, 0]]])
    def test_base_change_invariance(self, abh, g):
        L = BinLattice(*abh)
        assert L.discriminant_form().is_isomorphic(transform(L, g).discriminant_form())

    def test_undefined(self):
        with pytest.raises(ValueError):
            BinLattice(1, 0, 0).discriminant_form()
        with pytest.raises(ValueError):
            BinLattice(Fraction(1, 2), 1, 0).discriminant_form()


class TestGenus:
    def test_genus_with_two_classes(self):
        # Discriminant -56: x^2 + 14y^2 and 2x^2 + 7y^2 form one genus, 3x^2 +- 2xy + 5y^2 the other
        L, M, N = BinLattice(1, 14, 0), BinLattice(2, 7, 0), BinLattice(3, 5, 1)
        assert L.is_same_genus(M) and not L.is_isomorphic(M)
        assert not L.is_same_genus(N)

    def test_degenerate(self):
        assert BinLattice(2, 8, 4).is_same_genus(BinLattice(0, 2, 0))
        assert not BinLattice(2, 8, 4).is_same_genus(BinLattice(0, 3, 0))
        assert BinLattice(0, 0, 0).is_same_genus(BinLattice(0, 0, 0))

    def test_matches_conway_sloane_symbols(self):
        """is_same_genus agrees with the comparison of Conway-Sloane symbols in lattice.genus."""
        lats = [BinLattice(a, b, h) for a in BOX for b in BOX for h in BOX if a * b != h * h]
        groups = defaultdict(list)
        for L in lats:
            groups[(L.signature, L.disc, L.parity)].append((L, Genus(Lattice(2, [[L.a, L.h], [L.h, L.b]]))))
        for group in groups.values():
            for (L, GL), (M, GM) in combinations(group, 2):
                assert L.is_same_genus(M) == (GL == GM), ((L.a, L.b, L.h), (M.a, M.b, M.h))
