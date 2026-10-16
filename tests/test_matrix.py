"""Edge cases of the shape convention documented in lattice/_matrix.py:
empty matrices, single rows and 1 x 1 matrices."""
import numpy as np
import flint as fl
from lattice._matrix import imat, imat_zero, imat_diag, imat2flz, flz2imat, flq2imat, nrows, list_rows
from lattice.lattice import Lattice


def flz_shape(m):
    return (m.nrows(), m.ncols())


class TestConversions:
    def test_flz2imat_keeps_shape(self):
        for r, c in [(0, 3), (1, 3), (1, 1), (3, 1), (2, 0), (0, 0)]:
            m = fl.fmpz_mat(r, c, list(range(1, r * c + 1)))
            assert flz2imat(m).shape == (r, c)

    def test_flq2imat_keeps_shape(self):
        M, d = flq2imat(fl.fmpq_mat(1, 2, [fl.fmpq(1, 2), 1]))
        assert M.shape == (1, 2) and M.tolist() == [[1, 2]] and d == 2
        M, d = flq2imat(fl.fmpq_mat(0, 2))
        assert M.shape == (0, 2) and d == 1

    def test_imat2flz_shapes(self):
        assert flz_shape(imat2flz([])) == (0, 0)
        assert flz_shape(imat2flz(imat_zero(0, 3))) == (0, 3)
        assert flz_shape(imat2flz(imat_zero(3, 0))) == (3, 0)
        assert flz_shape(imat2flz(imat([1, 2, 3]))) == (1, 3)
        assert flz_shape(imat2flz([[1, 2, 3]])) == (1, 3)

    def test_imat2flz_int64(self):
        assert imat2flz(np.array([[1, 2], [3, 4]])) == fl.fmpz_mat([[1, 2], [3, 4]])

    def test_round_trip(self):
        for M in [imat_zero(0, 3), imat_zero(2, 0), imat([[7]]), imat([[1, 2, 3]]), imat([[1], [2]])]:
            assert flz2imat(imat2flz(M)).shape == M.shape

    def test_imat_diag_empty(self):
        assert imat_diag([]).shape == (0, 0)
        assert imat_diag([3]).shape == (1, 1)


class TestRows:
    def test_nrows(self):
        assert nrows([]) == 0
        assert nrows(imat_zero(0, 3)) == 0
        assert nrows(imat([1, 2, 3])) == 1
        assert nrows(imat([[1, 2, 3]])) == 1

    def test_list_rows(self):
        assert list_rows(imat_zero(0, 3)) == []
        assert len(list_rows(imat([1, 2]))) == 1


class TestLatticeEdgeCases:
    def test_rank_one_lll(self):
        L = Lattice(1, [[4]])
        assert L.lll().shape == (1, 1)
        M, T = L.lll(transform=True)
        assert M.shape == (1, 1) and T.shape == (1, 1)

    def test_saturate(self):
        assert Lattice.saturate([[2]]).tolist() == [[1]]
        S = Lattice.saturate([[0, 0], [0, 0], [2, 0]])
        assert S.shape == (1, 2) and abs(S[0, 0]) == 1 and S[0, 1] == 0
        assert Lattice.saturate(imat_zero(0, 2)).shape == (0, 2)

    def test_index_more_rows_than_columns(self):
        assert Lattice.index([[1, 0], [0, 1], [1, 1]]) == 1
        assert Lattice.index([[2, 0], [0, 2], [2, 2]]) == 4

    def test_complement_of_full_rank(self):
        L = Lattice(2, [[2, 1], [1, 2]])
        assert L.complement([[1, 0], [0, 1]]).shape == (0, 2)
        assert L.complement(imat_zero(0, 2)).shape == (2, 2)

    def test_image_and_subquotient_of_zero(self):
        L = Lattice(2, [[2, 1], [1, 2]])
        assert Lattice.image([[0, 0]]).shape == (0, 2)
        assert L.subquotient([[0, 0]]).shape == (0, 2)
        assert Lattice.image([[2, 4]]).shape == (1, 2)

    def test_fibre_product_empty(self):
        B1, B2 = Lattice.fibre_product(imat_zero(0, 1), imat([[2], [0]]))
        assert B1.shape == (1, 0) and B2.shape == (1, 2)
        assert (B2 @ imat([[2], [0]])).tolist() == [[0]]
        B1, B2 = Lattice.fibre_product(imat_zero(0, 1), imat_zero(0, 1))
        assert B1.shape == (0, 0) and B2.shape == (0, 0)
