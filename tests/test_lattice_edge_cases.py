"""Edge cases and incorrect arguments for lattice.lattice."""
from fractions import Fraction
import numpy as np
import pytest
from lattice._matrix import imat, imat_zero, imat_diag
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, U_lat, I_lat


class TestConstruction:
    @pytest.mark.parametrize("bad", [[[1, 2], [3, 4]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]],
                             ids=["non-symmetric", "degenerate", "zero"])
    def test_invalid_gram(self, bad):
        with pytest.raises(ValueError):
            Lattice(2, bad)

    @pytest.mark.parametrize("rank, gram", [(2, [[1, 0, 0], [0, 1, 0]]), (3, [[1, 0], [0, 1]]),
                                            (2, [[1, 0], [0, 1], [0, 0]]), (2, [1, 2]), (1, [1, 2]),
                                            (2, imat([[[1]]])), (1, [[]])],
                             ids=["2x3", "rank-mismatch", "3x2", "flat-list", "rank1-two-entries", "3-dim", "1x0"])
    def test_wrong_shape(self, rank, gram):
        with pytest.raises(ValueError):
            Lattice(rank, gram)

    @pytest.mark.parametrize("bad", [1.0, 0.5, Fraction(1, 2), Fraction(2, 1), np.float64(1), "1", None, 1j],
                             ids=["float", "float-frac", "Fraction", "Fraction-int", "np.float64", "str", "None",
                                  "complex"])
    def test_non_integer_entries_rejected(self, bad):
        with pytest.raises(TypeError):
            Lattice(2, [[2, 1], [1, bad]])

    @pytest.mark.parametrize("dtype", [np.int64, np.int8, np.uint16, object])
    def test_numpy_dtypes(self, dtype):
        L = Lattice(2, np.array([[2, 1], [1, 2]], dtype=dtype))
        assert (L.disc, L.dgroup, L.signature) == (3, [1, 3], (2, 0))
        assert L.A.dtype == object

    def test_big_entries(self):
        N = 2 ** 100
        L = Lattice(2, [[N, 1], [1, N]])
        assert L.disc == N * N - 1 and L.signature == (2, 0)
        assert L.lll().tolist() == [[N, 1], [1, N]]
        assert Lattice(2, np.array([[2 ** 62, 1], [1, 2 ** 62]])).disc == 2 ** 124 - 1

    def test_input_is_copied(self):
        A = imat([[2, 1], [1, 2]])
        L = Lattice(2, A)
        A[0, 0] = 100
        assert L.A[0, 0] == 2 and L.A_fl[0, 0] == 2

    def test_entries_are_ints(self):
        L = Lattice(2, np.array([[2, 1], [1, 2]]))
        assert all(type(x) is int for x in L.A.flatten())
        assert all(type(x) is int for x in L.dgroup) and type(L.disc) is int and type(L.exp) is int


class TestSmallRanks:
    @pytest.mark.parametrize("gram", [[], imat_zero(0, 0)], ids=["list", "imat"])
    def test_rank_zero(self, gram):
        L = Lattice(0, gram)
        assert L.A.shape == (0, 0)
        assert (L.signature, L.disc, L.dgroup, L.exp, L.parity) == ((0, 0), 1, [], 1, False)
        assert L.info() == "Zero lattice"
        assert L.lll().shape == (0, 0)
        M, T = L.lll(transform=True)
        assert M.shape == T.shape == (0, 0)
        assert L.even_sublattice().shape == (0, 0)
        assert L.complement(imat_zero(0, 0)).shape == (0, 0)
        assert (L + A_lat(2)).A.tolist() == A_lat(2).A.tolist()
        assert L.clear_squares().rank == L.make_ssf().rank == 0

    @pytest.mark.parametrize("gram", [[[-12]], [-12], -12, imat([[-12]]), imat([-12])],
                             ids=["list", "flat-list", "scalar", "imat", "1-d imat"])
    def test_rank_one(self, gram):
        L = Lattice(1, gram)
        assert L.A.shape == (1, 1)
        assert (L.signature, L.disc, L.dgroup, L.exp, L.parity) == ((0, 1), 12, [12], 12, False)
        assert L.lll().tolist() == [[-12]]
        M, T = L.lll(transform=True)
        assert M.tolist() == [[-12]] and abs(T[0, 0]) == 1
        assert L.clear_squares().A.tolist() == [[-3]]
        assert L.make_ssf().A.tolist() == [[-1]]
        assert L.make_primitive().A.tolist() == [[-1]]
        assert Lattice.saturate([[2]]).tolist() == [[1]]
        assert L.complement([[1]]).shape == (0, 1)
        assert L.even_sublattice().tolist() == [[1]]
        assert Lattice(1, [[3]]).even_sublattice().tolist() == [[2]]

    def test_rank_one_is_root(self):
        L = Lattice(1, [[6]])
        assert L.is_root([1]) and not L.is_root([0]) and not L.is_root([-2])
        # Roots are primitive, although the reflection in [3] is -1 as well
        assert not Lattice(1, [[-2]]).is_root([3])


class TestOperations:
    @pytest.mark.parametrize("k", [0, -1, 1.5, "2", None])
    def test_bad_multiplier(self, k):
        with pytest.raises(ValueError):
            A_lat(2) * k

    def test_multiplier_one_and_rmul(self):
        L = A_lat(2)
        assert (L * 1).A.tolist() == L.A.tolist()
        assert (2 * L).A.tolist() == (L * 2).A.tolist()

    def test_scale_by_zero(self):
        with pytest.raises(ValueError):
            A_lat(2)(0)

    def test_make_primitive_returns_self(self):
        L = A_lat(2)
        assert L.make_primitive() is L

    def test_info(self):
        assert A_lat(2).info() == ("Even lattice of signature (2, 0), discriminant 3 and exponent 3\n"
                                   "Discriminant group: [1, 3]")
        assert I_lat(1, 2).info().startswith("Odd lattice of signature (1, 2), discriminant 1")


class TestVectorArguments:
    L = Lattice(3, [[2, 1, 0], [1, 2, 0], [0, 0, -4]])

    @pytest.mark.parametrize("u, v", [([1, 0, 1], [0, 1, 1]), (imat([1, 0, 1]), imat([[0, 1, 1]])),
                                      (imat([[1, 0, 1]]), imat([0, 1, 1])), (imat([[1, 0, 1]]), imat([[0, 1, 1]]))],
                             ids=["lists", "1d-2d", "2d-1d", "2d-2d"])
    def test_product_shapes(self, u, v):
        assert self.L.product(u, v) == 1 - 4
        assert type(self.L.product(u, v)) is int

    @pytest.mark.parametrize("u", [[1, 0, 0], imat([1, 0, 0]), imat([[1, 0, 0]])], ids=["list", "1d", "2d"])
    def test_is_root_shapes(self, u):
        assert self.L.is_root(u)

    def test_is_root_special(self):
        assert not self.L.is_root([0, 0, 0])
        assert not U_lat().is_root([1, 0])  # isotropic
        assert self.L.is_root([0, 0, 1])  # (u, u) = -4 divides 2 (u, x) = -8 x_3
        assert not self.L.is_root([0, 0, 2])  # (u, u) = -16 divides 2 (u, x) = -16 x_3, but not primitive
        assert not self.L.is_root(imat([[0, 0, 2]]))
        assert not self.L.is_root([2, 0, 0])
        assert self.L.is_root([1, 0, 1])  # (u, u) = -2
        assert self.L.is_root([1, 1, 0])  # (u, u) = 6, 2 (u, x) = 6 (x_1 + x_2)
        assert not self.L.is_root([1, 2, 0])  # (u, u) = 14, 2 (u, e_1) = 8

    def test_dual_vec_special(self):
        v, d = self.L.dual_vec([0, 0, 0])
        assert d == 0 and v.shape == (3,)
        v, d = self.L.dual_vec([0, 0, -3])
        assert d == 12 and self.L.product([0, 0, -3], v) == 12
        v, d = self.L.dual_vec(imat([[0, 0, 1]]))
        assert d == 4

    @pytest.mark.parametrize("rank", [16, 26])
    def test_batch_prod_flint_matches_numpy(self, rank):
        """From Lattice.FLINT_MIN_RANK on, batch_prod uses FLINT; it must agree with numpy."""
        rng = np.random.default_rng(rank)
        B = imat(rng.integers(-3, 4, size=(rank + 2, rank)).tolist())
        L = Lattice(rank, (B.T @ B + imat_diag([1] * rank)).tolist())
        assert rank >= Lattice.FLINT_MIN_RANK
        U = imat(rng.integers(-10 ** 6, 10 ** 6, size=(5, rank)).tolist()) * 10 ** 12
        V = imat(rng.integers(-9, 10, size=(3, rank)).tolist())
        for u, v in [(U, V), (U[0], V), (U, V[1]), (U[0], V[1]), (imat_zero(0, rank), V), (U, imat_zero(0, rank))]:
            expected = u @ L.A @ v.transpose()
            result = L.batch_prod(u, v)
            if np.ndim(expected) == 0:  # both arguments are vectors
                assert type(result) is int and result == expected
                continue
            assert result.shape == expected.shape and result.dtype == object
            assert (result == expected).all() if result.size else True
            assert all(type(x) is int for x in np.ravel(result))

    def test_batch_prod_empty(self):
        E = imat_zero(0, 3)
        assert self.L.batch_prod(E, imat_diag([1, 1, 1])).shape == (0, 3)
        assert self.L.batch_prod(E, E).shape == (0, 0)


class TestSublatticeArguments:
    L = Lattice(3, [[2, 1, 0], [1, 2, 0], [0, 0, -4]])

    @pytest.mark.parametrize("gens", [[[1, 0, 0]], imat([[1, 0, 0]]), imat([1, 0, 0])], ids=["list", "2d", "1d"])
    def test_single_generator(self, gens):
        assert self.L.complement(gens).shape == (2, 3)
        assert Lattice.saturate(gens).shape == (1, 3)
        assert Lattice.image(gens).shape == (1, 3)
        assert Lattice.index(gens) == 1
        assert self.L.subquotient(gens).shape == (1, 3)

    @pytest.mark.parametrize("gens", [imat_zero(0, 3), [[0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                             ids=["no-rows", "zero-row", "zero-rows"])
    def test_zero_generators(self, gens):
        assert self.L.complement(gens).shape == (3, 3)
        assert Lattice.saturate(gens).shape == (0, 3)
        assert Lattice.image(gens).shape == (0, 3)
        assert Lattice.index(gens) == 1
        assert self.L.subquotient(gens).shape == (0, 3)

    def test_full_rank_generators(self):
        gens = [[2, 0, 0], [0, 3, 0], [0, 0, 1], [0, 0, 5]]
        assert self.L.complement(gens).shape == (0, 3)
        assert Lattice.saturate(gens).shape == (3, 3)
        assert Lattice.index(gens) == 6
        assert Lattice.image(gens).shape == (3, 3)

    def test_isotropic_subquotient(self):
        U = U_lat() + A_lat(1)
        assert U.subquotient([[1, 0, 0]]).shape == (0, 3)
        assert U.subquotient([[1, 0, 0], [0, 0, 1]]).tolist() in ([[0, 0, 1]], [[0, 0, -1]])

    def test_fibre_product_sign(self):
        """Pairs (x, y) with x A1 = y A2, not x A1 = -y A2."""
        B1, B2 = Lattice.fibre_product([[1]], [[1]])
        assert B1.tolist() == B2.tolist() and B1.shape == (1, 1)

    def test_fibre_product_empty(self):
        B1, B2 = Lattice.fibre_product(imat_zero(0, 2), imat([[1, 2], [2, 4]]))
        assert B1.shape == (1, 0) and B2.shape == (1, 2)
        B1, B2 = Lattice.fibre_product([[1, 0]], [[0, 1]])
        assert B1.shape == (0, 1) and B2.shape == (0, 1)


class TestSquareFree:
    @pytest.mark.parametrize("diag, cleared, ssf", [([1, 3, 9], [1, 1, 3], [1, 1, 3]),
                                                    ([2, 2, 2], [2, 2, 2], [1, 1, 1]),
                                                    ([1, 1, 8], [1, 1, 2], [1, 1, 2]),
                                                    ([4, 9, 25], [1, 1, 1], [1, 1, 1]),
                                                    ([1, 3, 3, 1], [1, 1, 3, 3], [1, 1, 3, 3]),
                                                    ([6, 6], [6, 6], [1, 1]),
                                                    ([1, 2, 6], [1, 2, 6], [1, 1, 6])],
                             ids=["1-3-9", "2-2-2", "1-1-8", "4-9-25", "tie", "6-6", "1-2-6"])
    def test_diagonal(self, diag, cleared, ssf):
        L = Lattice(len(diag), imat_diag(diag))
        assert L.clear_squares().dgroup == cleared
        assert L.make_ssf().dgroup == ssf

    def test_big_entries(self):
        N = 10 ** 20
        L = Lattice(2, imat_diag([4 * (N + 1), 9 * N * N]))
        assert L.clear_squares().dgroup == [1, N + 1]
        assert L.make_ssf().disc == N + 1
