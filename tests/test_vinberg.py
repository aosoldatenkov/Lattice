from collections import Counter
import pytest
from lattice._matrix import imat
from lattice.roots import Coxeter_graph, simple_roots, irred_decomp, get_extremal_rays
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat, I_lat
from lattice.vinberg import Vinberg


def test_simple_roots_a2():
    L = A_lat(2)
    roots = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]]
    roots = [r for r in roots if L.square(r) == 2]
    simple = simple_roots(L, roots)
    assert len(simple) == 2
    assert Coxeter_graph(L, simple) == {(0, 1): 3}


def test_irred_decomp():
    components = irred_decomp(E_lat(8) + A_lat(2) + D_lat(4))
    assert sorted(len(b) for b in components) == [2, 4, 8]


def test_extremal_rays_triangle():
    # The (2, 4, inf) triangle of I_{1,2}: two finite vertices and one cusp
    L = I_lat(1, 2)
    rays, lines = get_extremal_rays([[0, 1, -1], [0, 0, 1], [1, -1, -1]], L.A)
    assert len(lines) == 0
    squares = sorted(L.square(imat([int(x) for x in r.numer_denom()[0].tolist()[0]])) for r in rays)
    assert len(squares) == 3 and squares[0] == 0 and squares[1] > 0


@pytest.mark.parametrize("n", range(2, 8))
def test_odd_unimodular(n):
    # Vinberg: the fundamental polyhedron of I_{1,n} has n + 1 walls for 2 <= n <= 9
    L = I_lat(1, n)
    walls = Vinberg(L, base=[1] + [0] * n).run(iterations=10)
    assert len(walls) == n + 1
    assert all(L.is_root(w) and L.square(w) < 0 for w in walls)


def test_ii_1_9():
    # The Coxeter diagram of II_{1,9} is E10 = T_{2,3,7}
    L = U_lat() + E_lat(8)(-1)
    walls = Vinberg(L, base=[1, 1] + [0] * 8).run(iterations=10)
    assert len(walls) == 10
    graph = Coxeter_graph(L, walls)
    assert set(graph.values()) == {2, 3}
    edges = [e for e, m in graph.items() if m == 3]
    assert len(edges) == 9  # a tree on 10 vertices
    degree = Counter(v for e in edges for v in e)
    assert sorted(degree.values()) == [1, 1, 1] + [2] * 6 + [3]
    # Arm lengths from the branch vertex: 1, 2, 6
    branch = next(v for v, d in degree.items() if d == 3)
    adj = {v: [w for e in edges if v in e for w in e if w != v] for v in degree}
    arms = []
    for start in adj[branch]:
        prev, cur, length = branch, start, 1
        while degree[cur] == 2:
            prev, cur = cur, next(w for w in adj[cur] if w != prev)
            length += 1
        arms.append(length)
    assert sorted(arms) == [1, 2, 6]
