"""Tests of lattice.vinberg.coxeter (the Coxeter diagram of the walls and its symmetries)."""
import pytest
import flint as fl
from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat, I_lat
from lattice.roots import Coxeter_graph
from lattice.vinberg.coxeter import WallDiagram
from lattice.vinberg.trace import ChamberTracer, trace_chamber

pynauty = pytest.importorskip("pynauty")


def test_incremental_gram():
    """The Gram matrix built one wall at a time, and the labels, as computed at once."""
    L = I_lat(1, 9)
    walls = trace_chamber(L, base=[1] + [0] * 9)["walls"]
    D = WallDiagram(L, walls[:4])
    assert D.add(walls[2:]) == list(range(2, len(walls)))
    G = L.batch_prod(imat(walls), imat(walls))
    assert D.gram == [[int(G[i, j]) for j in range(len(walls))] for i in range(len(walls))]
    assert D.labels() == {p: lab for p, lab in Coxeter_graph(L, walls).items() if lab != 2}


A2, D4 = A_lat(2)(-1), D_lat(4)(-1)


@pytest.mark.parametrize("L, base, order, orbits", [
    (U_lat() + E_lat(8)(-1), None, 1, [1] * 10),
    (I_lat(1, 14), [1] + [0] * 14, 2, [2] * 8 + [1]),
    (U_lat() + A2 + A2 + A2, None, 6, [3, 3, 3, 1]),
    (U_lat() + D4 + D4 + D4, None, 6, [3, 3, 3, 3, 3, 1, 1]),
])
def test_symmetries(L, base, order, orbits):
    """The order and the orbits of walls of Sym(C); the generators are integral isometries permuting the walls."""
    res = trace_chamber(L, base=base)
    D = WallDiagram(L, res["walls"])
    S = D.symmetries(res["tracer"].base)
    assert S.order == order and sorted((len(o) for o in S.wall_orbits), reverse=True) == orbits
    A = fl.fmpz_mat(L.A.tolist())
    for s, M in S.generators:
        Mf = fl.fmpz_mat(M.tolist())
        assert Mf * A * Mf.transpose() == A
        for i, w in enumerate(D.walls):
            assert list(fl.fmpz_mat([list(w)]) * Mf) == list(D.walls[s[i]])


@pytest.mark.parametrize("gram, index", [
    ([[1, 0, 0], [0, -8, 0], [0, 0, -312]], 2),  # Allcock 41.41
    ([[1, 0, 0], [0, -32, 0], [0, 0, -64]], 4),  # Allcock 160.12
])
def test_non_integral_symmetries(gram, index):
    """Symmetries of the diagram which are not integral: Sym(C) is trivial, of index aut_order in Aut(G)."""
    L = Lattice(3, gram)
    res = trace_chamber(L)
    S = WallDiagram(L, res["walls"]).symmetries(res["tracer"].base)
    assert (S.order, S.aut_order, S.index) == (1, index, index) and S.generators == []


def test_rank_deficient():
    L = I_lat(1, 4)
    D = WallDiagram(L, [[0, 1, -1, 0, 0], [0, 0, 1, -1, 0]])
    with pytest.raises(ValueError):
        D.symmetries([1, 0, 0, 0, 0])


def test_tracer_reports_symmetries(capsys):
    L = U_lat() + A2 + A2 + A2
    tracer = ChamberTracer(L, symmetry=True)
    res = tracer.run(progress=True, progress_interval=0)
    assert res["symmetry"].order == 6
    assert len(tracer.diagram) == len(tracer.walls) and tracer.diagram.walls == [tuple(int(x) for x in w) for w in tracer.walls]
    assert "symmetries of the 10 walls: order 6, 4 orbits of walls [3 x 3, 1]" in capsys.readouterr().out
