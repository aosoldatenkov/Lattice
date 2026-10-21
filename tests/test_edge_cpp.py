"""Tests of the C++ per-edge core (edge_cpp) against the Python implementation (the reference)."""
import itertools
import random
from fractions import Fraction
import pytest
from lattice._matrix import imat, imat_diag
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, E_lat, U_lat, I_lat
from lattice.vinberg.edges import edge_crossing_roots
from lattice.vinberg.trace import trace_chamber, trace_ray, _next_walls, finite_vertex_edges
from test_edges import LATTICES

edge_cpp = pytest.importorskip("lattice.edge_cpp")


def as_set(found):
    return {(tuple(int(x) for x in r), t) for r, t in found}


def cpp_roots(L, a, b, include_end):
    return {(tuple(r), None if B == 0 else Fraction(A, -B))
            for r, A, B in edge_cpp.crossing_roots(L.A.tolist(), L.exp, [int(x) for x in a], [int(x) for x in b],
                                                   include_end)}


@pytest.mark.parametrize("index", range(len(LATTICES)))
def test_crossing_roots_random_segments(index):
    L = LATTICES[index]
    rng = random.Random(index)
    points = [imat(list(x)) for x in itertools.product(range(-3, 4), repeat=L.rank)
              if any(x) and L.square(list(x)) >= 0]
    tested = 0
    while tested < 15:
        a, b = rng.sample(points, 2)
        p, q, c = L.square(a), L.square(b), L.product(a, b)
        if c <= 0 or c * c <= p * q:
            continue
        for include_end in ((False, True) if q > 0 else (False,)):
            assert cpp_roots(L, a, b, include_end) == as_set(edge_crossing_roots(L, a, b, include_end, use_cpp=False))
        tested += 1


@pytest.mark.parametrize("n", [5, 9])
def test_edges_of_a_chamber(n):
    """crossing_roots on the edges of the chamber of I_{1,n}, and trace_edge from both ends of each edge, with all
    the walls known (predicted ends) and with none (rational and irrational ideal ends)."""
    L = I_lat(1, n)
    tracer = trace_chamber(L, base=[1] + [0] * n, chamber=[(n + 1) ** 2 - i ** 2 for i in range(n + 1)])["tracer"]
    A = L.A.tolist()
    for (e1, e2) in tracer.edges.values():
        a, b = imat(list(e1)), imat(list(e2))
        assert cpp_roots(L, a, b, False) == as_set(edge_crossing_roots(L, a, b, use_cpp=False))
    irrational = 0
    for vert in tracer.vertices.values():
        if vert["ideal"]:
            continue
        v = vert["vector"]
        S = [tracer.walls[i] for i in sorted(vert["walls"])]
        _, edges = finite_vertex_edges(L, v, S)
        # The directions computed in C++
        dirs = edge_cpp.vertex_directions(A, [int(x) for x in v], [[int(x) for x in w] for w in S])
        assert [tuple(d) for d in dirs] == [tuple(int(x) for x in u) for _, u in edges]
        for known in (tracer.walls, []):
            for j, u in edges:
                W = [S[i] for i in range(len(S)) if i != j]
                code, end, alpha = edge_cpp.trace_edge(A, L.exp, [int(x) for x in v], [int(x) for x in u],
                                                       [[int(x) for x in w] for w in known],
                                                       [[int(x) for x in w] for w in W])
                if code == 2:
                    irrational += 1
                    continue
                pend, pideal, roots = trace_ray(L, v, u, list(known), use_cpp=False)
                assert tuple(end) == tuple(int(x) for x in pend) and (code == 1) == pideal
                if not pideal:
                    assert tuple(alpha) == tuple(int(x) for x in _next_walls(L, W, v, pend, roots)[-1])
    assert irrational > 0 or n < 9


def test_overflow_falls_back_to_python():
    """Beyond 64/128 bits, crossing_roots raises OverflowError, and edge_crossing_roots uses Python instead."""
    L = Lattice(3, imat_diag([1, -3, -1]))
    k = 2 ** 40
    a, b = imat([k, 0, 0]), imat([7 * k, 4 * k, k])
    with pytest.raises(OverflowError):
        edge_cpp.crossing_roots(L.A.tolist(), L.exp, [int(x) for x in a], [int(x) for x in b], False)
    assert as_set(edge_crossing_roots(L, a, b)) == as_set(edge_crossing_roots(L, a, b, use_cpp=False))
    assert as_set(edge_crossing_roots(L, a, b)) == as_set(edge_crossing_roots(L, imat([1, 0, 0]), imat([7, 4, 1])))


@pytest.mark.parametrize("L, base, chamber", [
    (I_lat(1, 10), [1] + [0] * 10, [121 - i ** 2 for i in range(11)]),
    (I_lat(1, 7), [9, 1, 2, 3, 0, 1, 4, 2], None),
    (U_lat() + E_lat(8)(-1), None, None),
    (Lattice(3, [[2, 1, 60], [1, -2, 0], [60, 0, -120]]), None, None),
    (U_lat(2) + A_lat(2)(-1), [2, 3, 1, 0], None),
])
def test_tracing_with_and_without_cpp(L, base, chamber):
    ref = trace_chamber(L, base=base, chamber=chamber, cpp=False)
    res = trace_chamber(L, base=base, chamber=chamber)
    assert {tuple(int(x) for x in w) for w in res["walls"]} == {tuple(int(x) for x in w) for w in ref["walls"]}
    assert (res["vertices"], res["cusps"], res["edges"]) == (ref["vertices"], ref["cusps"], ref["edges"])
    assert set(res["tracer"].vertices) == set(ref["tracer"].vertices)


def test_large_coordinates():
    """Short segments with large coordinates in U + D_20(-1): the C++ search agrees with Python or falls back with
    OverflowError (a badly conditioned complement used to raise another error)."""
    from lattice.catalogue import D_lat
    L = D_lat(20)(-1) + U_lat()
    A = L.A.tolist()
    rng = random.Random(3)
    checked = 0
    for scale in (3, 300, 30000):
        done = 0
        while done < 4:
            a = imat([rng.randint(-scale, scale) for _ in range(20)] + [rng.randint(1, 8 * scale), rng.randint(1, 8 * scale)])
            if L.square(a) <= 0:
                continue
            b = 3 * a + imat([rng.randint(-2, 2) for _ in range(22)])
            p, q, c = L.square(a), L.square(b), L.product(a, b)
            if q <= 0 or c <= 0 or c * c <= p * q:
                continue
            done += 1
            try:
                found = cpp_roots(L, a, b, False)
            except OverflowError:
                continue
            assert found == as_set(edge_crossing_roots(L, a, b, use_cpp=False))
            checked += 1
    assert checked > 0
