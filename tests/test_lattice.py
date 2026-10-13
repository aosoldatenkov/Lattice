import numpy as np
import pytest
from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.enumeration import FPSearch, FPSearchPy
from lattice.catalogue import A_lat, B_lat, C_lat, D_lat, E_lat, U_lat, I_lat, II_lat_n_1, Leech_lat, Leech_lat_alt


def unimodular(rank, seed):
    """A pseudo-random matrix in GL(rank, Z), built from elementary row operations."""
    rng = np.random.default_rng(seed)
    T = imat([[int(i == j) for j in range(rank)] for i in range(rank)])
    for _ in range(4 * rank):
        i, j = rng.choice(rank, size=2, replace=False)
        T[i, :] += int(rng.integers(-2, 3)) * T[j, :]
    return T


def is_unimodular(T):
    return abs(Lattice(T.shape[0], T @ T.T).disc) == 1 if T.shape[0] == T.shape[1] else False


class TestInvariants:
    def test_e8(self):
        L = E_lat(8)
        assert (L.rank, L.signature, L.disc, L.exp) == (8, (8, 0), 1, 1)
        assert not L.parity

    def test_a2(self):
        L = A_lat(2)
        assert L.dgroup == [1, 3]
        assert L.disc == 3

    def test_odd_lorentzian(self):
        L = I_lat(1, 2)
        assert L.signature == (1, 2)
        assert L.parity

    @pytest.mark.parametrize("bad", [[[1, 2], [3, 4]], [[1, 1], [1, 1]]])
    def test_invalid_gram(self, bad):
        with pytest.raises(ValueError):
            Lattice(2, bad)


class TestOperations:
    def test_sum_multiple_scale(self):
        A2 = A_lat(2)
        assert (A2 + A2).disc == 9
        assert (3 * A2).rank == 6
        assert A2(-1).signature == (0, 2)
        assert (A2(3).make_primitive().A == A2.A).all()

    def test_is_root(self):
        L = I_lat(1, 2)
        assert L.is_root(imat([0, 1, 0]))
        assert L.is_root(imat([1, 0, 0]))
        assert not L.is_root(imat([1, 1, 0]))  # isotropic

    def test_complement(self):
        L = I_lat(1, 3)
        C = L.complement([[1, 1, 0, 0]])
        assert C.shape == (3, 4)
        assert all(L.product([1, 1, 0, 0], c) == 0 for c in C)

    def test_saturate_and_index(self):
        assert Lattice.index([[2, 0], [0, 2]]) == 4
        assert Lattice.index(Lattice.saturate([[2, 0], [0, 2]])) == 1

    def test_dual_vec(self):
        L = A_lat(2)
        v, d = L.dual_vec([1, 0])
        assert d == 1
        assert L.product([1, 0], v) == d

    def test_even_sublattice(self):
        L = I_lat(1, 2)
        basis = L.even_sublattice()
        assert Lattice.index(basis) == 2
        assert all(L.square(b) % 2 == 0 for b in basis)

    @pytest.mark.parametrize("L", [E_lat(8), E_lat(8)(-1), U_lat() + E_lat(8)(-1)], ids=["E8", "E8(-1)", "II_1,9"])
    def test_lll_transform(self, L):
        M, T = L.lll(transform=True)
        assert (T @ L.A @ T.T == M).all()
        assert is_unimodular(T)


class TestFinckePohst:
    @pytest.mark.parametrize("L", [E_lat(8), A_lat(3) + D_lat(4)], ids=["E8", "A3+D4"])
    def test_python_matches_cpp(self, L):
        A = np.array(L.A, dtype=float)
        b = np.zeros(L.rank)
        cpp = sorted(map(tuple, FPSearch(A, b, 0, 4.5).search_all()))
        py = sorted(map(tuple, FPSearchPy(A, b, 0, 4.5).search_all()))
        assert cpp == py and len(cpp) > 0

    def test_python_batches_resume(self):
        A = np.array(E_lat(8).A, dtype=float)
        fps = FPSearchPy(A, np.zeros(8), 0, 2.5)
        batches = []
        while not fps.exhausted():
            batches += fps.batch_search(50)
        assert len(batches) == len(set(map(tuple, batches))) == 240


class TestCatalogue:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_a_n(self, n):
        assert A_lat(n).disc == n + 1

    @pytest.mark.parametrize("n", [4, 5, 8])
    def test_d_n_and_c_n(self, n):
        assert D_lat(n).disc == 4
        assert C_lat(n).disc == 4

    def test_d4(self):
        assert D_lat(4).dgroup == [1, 1, 2, 2]

    def test_b_n(self):
        L = B_lat(5)
        assert L.disc == 1 and L.parity

    @pytest.mark.parametrize("n, disc", [(6, 3), (7, 2), (8, 1)])
    def test_e_n(self, n, disc):
        L = E_lat(n)
        assert L.disc == disc and not L.parity and L.signature == (n, 0)

    def test_ii_9_1(self):
        L = II_lat_n_1(9)
        assert (L.rank, L.signature, L.disc, L.parity) == (10, (9, 1), 1, False)

    @pytest.mark.parametrize("construct", [Leech_lat, Leech_lat_alt])
    def test_leech(self, construct):
        L = construct()
        assert (L.rank, L.signature, L.disc, L.parity) == (24, (24, 0), 1, False)
        A = L.lll()
        fps = FPSearch(np.array(A, dtype=float), np.zeros(24), 0, 2.5)
        assert fps.search_all() == []  # no roots
