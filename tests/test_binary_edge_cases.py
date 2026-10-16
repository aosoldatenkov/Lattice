"""Edge cases and incorrect arguments for lattice.binary."""
import time
from fractions import Fraction
import flint
import numpy as np
import pytest
import sympy
from lattice._matrix import imat
from lattice.binary import BinLattice, BinDiscForm, list_bin_lattices


def canonical(L):
    return (L.can.a, L.can.b, L.can.h)


class TestEntryTypes:
    @pytest.mark.parametrize("bad", [1.0, 0.5, np.float64(2.0), sympy.Float(1.5), 1j, "1", None, [1], True,
                                     np.bool_(True)],
                             ids=["float", "float-frac", "np.float64", "sympy.Float", "complex", "str", "None", "list",
                                  "bool", "np.bool_"])
    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_inexact_or_non_numeric_rejected(self, bad, position):
        args = [2, 3, 1]
        args[position] = bad
        with pytest.raises(TypeError):
            BinLattice(*args)

    @pytest.mark.parametrize("make", [int, np.int64, np.int8, np.uint16, flint.fmpz, sympy.Integer,
                                      lambda x: Fraction(x, 1), lambda x: flint.fmpq(x, 1),
                                      lambda x: sympy.Rational(x, 1)],
                             ids=["int", "np.int64", "np.int8", "np.uint16", "fmpz", "sympy.Integer", "Fraction",
                                  "fmpq", "sympy.Rational"])
    def test_integer_types(self, make):
        L = BinLattice(make(4), make(6), make(1))
        assert all(type(x) is int for x in (L.a, L.b, L.h))
        assert (canonical(L), L.disc, L.exp, L.parity) == (canonical(BinLattice(4, 6, 1)), 23, 23, 0)

    @pytest.mark.parametrize("make", [Fraction, flint.fmpq, sympy.Rational], ids=["Fraction", "fmpq", "sympy.Rational"])
    def test_rational_types(self, make):
        L = BinLattice(make(4, 3), make(2, 1), make(1, 3))
        assert type(L.a) is Fraction and type(L.b) is int and type(L.h) is Fraction
        assert canonical(L) == tuple(Fraction(x, 3) for x in canonical(BinLattice(4, 6, 1)))
        assert L.exp is None and L.parity is None

    def test_numpy_int64_does_not_overflow(self):
        L = BinLattice(np.int64(2**62), np.int64(2**62), np.int64(1))
        assert L.disc == 2**124 - 1

    def test_non_integral_off_diagonal_is_not_even(self):
        assert BinLattice(0, 2, Fraction(2, 3)).parity is None


class TestSpecialForms:
    def test_zero_form(self):
        L = BinLattice(0, 0, 0)
        assert (L.signature, L.disc, L.exp, L.parity, canonical(L)) == ((0, 0), 0, 0, 0, (0, 0, 0))
        assert L.zero == [] and L.river == [] and L.shift is None
        assert L.list_positive(10) == {} and L.list_negative(10) == {} and L.list_roots() == {}
        assert L.min_positive() is None and L.max_negative() is None
        assert L.is_isomorphic(BinLattice(0, 0, 0)) and L.is_isomorphic(BinLattice(0, 0, 0), oriented=True)
        assert not L.is_isomorphic(BinLattice(1, 0, 0))
        assert L.improper_automorphism() is not None
        assert L.is_same_genus(BinLattice(0, 0, 0))
        assert L.info() == "Zero lattice"
        with pytest.raises(ValueError):
            L.discriminant_form()

    @pytest.mark.parametrize("abh, sig, m", [((2, 8, 4), (1, 0), 2), ((-3, 0, 0), (0, 1), -3), ((0, 5, 0), (1, 0), 5)])
    def test_degenerate(self, abh, sig, m):
        L = BinLattice(*abh)
        assert (L.signature, L.disc, L.exp, canonical(L)) == (sig, 0, 0, (0, m, 0))
        assert len(L.zero) == 2 and L.river == [] and L.shift is None
        assert "Represents zero" in L.info()
        assert (L.min_positive(), L.max_negative()) == ((m, None) if m > 0 else (None, m))

    @pytest.mark.parametrize("h", [1, 2, 7])
    def test_hyperbolic_plane(self, h):
        L = BinLattice(0, 0, h)
        assert canonical(L) == (0, 0, h) and len(L.zero) == 4 and L.river == []
        assert (L.min_positive(), L.max_negative()) == (2 * h, -2 * h)
        assert BinLattice(0, 0, -h).is_isomorphic(L, oriented=True)

    def test_negative_definite(self):
        L = BinLattice(-2, -2, 1)
        assert L.signature == (0, 2) and L.list_positive(100) == {} and L.min_positive() is None
        assert L.max_negative() == -2 and len(L.list_negative(2)[-2]) == 3  # the roots of A2(-1), up to sign

    def test_non_primitive(self):
        L = BinLattice(6, 6, 3)
        assert (L.disc, L.exp, canonical(L)) == (27, 9, (6, 6, -3))
        assert L.is_isomorphic(BinLattice(6, 6, -3))


class TestHugeEntries:
    @pytest.mark.parametrize("k", [10**6, 10**20, 10**40])
    def test_conjugated_forms_reduce_fast(self, k):
        """Forms conjugated by a matrix with entries ~k^2 are reduced in logarithmic time."""
        g = imat([[1, k], [0, 1]]) @ imat([[1, 0], [k + 1, 1]])
        for abh in [(1, 1, 0), (4, 6, 1), (1, -3, 0), (2, -3, 1), (0, 5, 7)]:
            A = imat([[abh[0], abh[2]], [abh[2], abh[1]]])
            G = g @ A @ g.T
            start = time.perf_counter()
            L = BinLattice(G[0, 0], G[1, 1], G[0, 1])
            assert time.perf_counter() - start < 0.5
            assert L.is_isomorphic(BinLattice(*abh)) and L.is_isomorphic(BinLattice(*abh), oriented=True)

    def test_huge_definite(self):
        L = BinLattice(10**40 + 7, 10**40 + 1, 10**40 - 3)
        M = L.descend(L.can)
        assert canonical(L) == (M.a, M.b, M.h) and L.signature == (2, 0)

    def test_long_river(self):
        L = BinLattice(1, -(100**2 + 1), 0)  # x^2 - (n^2 + 1) y^2 has a river of length 4n
        assert len(L.river) == 400 and L.list_positive(1) == {1: {(1, 0)}}


class TestBadArguments:
    def setup_method(self):
        self.L = BinLattice(2, 3, 1)

    @pytest.mark.parametrize("vec", [(1,), (1, 2, 3), ()])
    def test_wrong_vector_length(self, vec):
        with pytest.raises(ValueError):
            self.L.prod(vec, (1, 0))
        with pytest.raises(ValueError):
            self.L.is_root(vec)

    def test_zero_vector_is_not_a_root(self):
        assert not self.L.is_root((0, 0))

    def test_non_primitive_root(self):
        assert BinLattice(1, 1, 0).is_root((2, 0))

    @pytest.mark.parametrize("other", [(2, 3, 1), None, imat([[2, 1], [1, 3]])], ids=["tuple", "None", "matrix"])
    def test_comparison_with_non_lattice(self, other):
        with pytest.raises(TypeError):
            self.L.is_isomorphic(other)
        with pytest.raises(TypeError):
            self.L.is_same_genus(other)

    def test_genus_of_rational_lattice(self):
        with pytest.raises(ValueError):
            BinLattice(Fraction(1, 2), 1, 0).is_same_genus(self.L)
        with pytest.raises(ValueError):
            BinLattice(Fraction(1, 2), 1, 0).discriminant_form()

    def test_bounds(self):
        L = BinLattice(1, 1, 0)
        assert L.list_positive(0) == {} and L.list_negative(0) == {}
        assert L.list_positive(-5) == L.list_positive(5)  # |bound| is used
        assert L.list_positive(Fraction(5, 2)) == L.list_positive(2)
        with pytest.raises(TypeError):
            L.list_positive(None)

    def test_discriminant_forms_of_different_parity(self):
        assert not BinLattice(2, 2, 1).discriminant_form().is_isomorphic(BinLattice(1, 3, 0).discriminant_form())
        assert isinstance(BinLattice(1, 3, 0).discriminant_form(), BinDiscForm)


class TestListBinLatticesArguments:
    def test_empty(self):
        assert list_bin_lattices(0) == {}

    @pytest.mark.parametrize("max_d", [-1, -10])
    def test_negative(self, max_d):
        with pytest.raises(ValueError):
            list_bin_lattices(max_d)

    @pytest.mark.parametrize("max_d", [2.5, "3", None])
    def test_non_integer(self, max_d):
        with pytest.raises(TypeError):
            list_bin_lattices(max_d)

    @pytest.mark.parametrize("signature", [(3, 0), (1, 0), (2, 2), "definite"])
    def test_bad_signature(self, signature):
        with pytest.raises(ValueError):
            list_bin_lattices(5, signature=signature)

    @pytest.mark.parametrize("parity", [2, -1, "odd"])
    def test_bad_parity(self, parity):
        with pytest.raises(ValueError):
            list_bin_lattices(5, parity=parity)

    def test_numpy_integer_bound(self):
        assert sorted(list_bin_lattices(np.int64(3))) == [1, 2, 3]
