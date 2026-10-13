import pytest
from lattice.lattice import Lattice
from lattice.genus import Genus
from lattice.discform import DiscForm
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat
from test_lattice import unimodular

LATTICES = {
    "II_1,9": U_lat() + E_lat(8)(-1),
    "A2+<-6>": A_lat(2) + Lattice(1, [[-6]]),
    "diag(1,1,2,4,-8)": Lattice(5, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, -8]]),
    "D4+U(2)": D_lat(4) + U_lat(2),
    "diag(3,-5,7)": Lattice(3, [[3, 0, 0], [0, -5, 0], [0, 0, 7]]),
}


def test_symbol_string():
    assert Genus(U_lat() + E_lat(8)(-1)).str() == "II_{1, 9}()"


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


def test_even_unimodular_rank_16():
    D = DiscForm(D_lat(16))
    even = [M for M in (D.overlattice(sorted(s)) for s in D.list_max_isospaces()) if not M.parity]
    assert len(even) == 2  # both give D16+
    assert Genus(even[0]) == Genus(E_lat(8) + E_lat(8))
