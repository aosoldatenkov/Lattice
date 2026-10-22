"""Tests of lattice.vinberg.trace (edge tracing), against RunTest + cdd, certify_polytope and Allcock's list."""
import io
import contextlib
import pytest
from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, E_lat, U_lat, I_lat
from lattice.roots import cone_structure
from lattice.vinberg import RunTest
from lattice.vinberg.certify import certify_polytope
from lattice.vinberg.trace import ChamberTracer, trace_chamber


def keys(walls):
    return {tuple(int(x) for x in w) for w in walls}


@pytest.mark.parametrize("n", range(2, 12))
def test_vinberg_lattices(n):
    """Same walls as RunTest, and the same numbers of vertices, cusps and edges as cdd."""
    L = I_lat(1, n)
    base, chamber = [1] + [0] * n, [(n + 1) ** 2 - i ** 2 for i in range(n + 1)]
    res = trace_chamber(L, base=base, chamber=chamber)
    with contextlib.redirect_stdout(io.StringIO()):
        walls = RunTest(L, base=base, chamber=chamber, n_iter=0)
    assert keys(res["walls"]) == keys(walls)
    cone = cone_structure(walls, L.A)
    squares = [(r * L.A_fl * r.transpose())[0, 0] for r in cone["rays"]]
    assert res["vertices"] == sum(1 for s in squares if s > 0)
    assert res["cusps"] == sum(1 for s in squares if s == 0)
    assert res["edges"] == sum(len(a) for a in cone["adjacency"]) // 2


@pytest.mark.parametrize("L, base, nwalls", [
    (I_lat(1, 3), [5, 1, 2, 3], 4),
    (I_lat(1, 7), [9, 1, 2, 3, 0, 1, 4, 2], 8),
    (U_lat() + E_lat(8)(-1), [3, 2, 1, 0, 0, 0, 0, 0, 0, 0], 10),  # the E_10 diagram
    (Lattice(3, [[1, 0, 0], [0, -3, 1], [0, 1, -3]]), [3, 1, 1], 3),
    (U_lat(2) + A_lat(2)(-1), [2, 3, 1, 0], 4),
])
def test_descent_from_interior(L, base, nwalls):
    """The base point lies on no mirror: the tracing starts by descending to a vertex."""
    res = trace_chamber(L, base=base)
    assert len(res["walls"]) == nwalls
    assert certify_polytope(L, res["walls"])["certified"] is True


# Lattices of Allcock's list (the reduced forms of experiments/allcock.py) with their numbers of walls and cusps;
# the edges of the compact polygons have irrational ideal ends
ALLCOCK = [
    ("0.1", [[0, 1, 0], [1, 0, 0], [0, 0, -4]], 3, 1),
    ("2.1", [[4, 0, 0], [0, -2, 1], [0, 1, -2]], 3, 0),
    ("3.9", [[3, 0, 0], [0, -1, 0], [0, 0, -8]], 4, 1),
    ("12.3", [[-1, 0, 0], [0, 4, -2], [0, -2, -25]], 8, 2),
    ("40.1", [[2, 2, -1], [2, -4, 2], [-1, 2, -14]], 4, 0),
    ("104.11", [[2, 1, 60], [1, -2, 0], [60, 0, -120]], 16, 0),
    ("118.40", [[1, 0, 0], [0, -1155, 0], [0, 0, -1155]], 18, 0),
    ("300.2", [[2, -2, 1], [-2, -8, 4], [1, 4, -182]], 8, 0),
]


@pytest.mark.parametrize("name, gram, nwalls, ncusps", ALLCOCK, ids=[x[0] for x in ALLCOCK])
def test_allcock(name, gram, nwalls, ncusps):
    L = Lattice(3, gram)
    res = trace_chamber(L)
    # A polygon: as many vertices (finite and ideal) as walls and edges
    assert len(res["walls"]) == nwalls == res["vertices"] + res["cusps"] == res["edges"]
    assert res["cusps"] == ncusps
    assert certify_polytope(L, res["walls"])["certified"] is True


def test_structure():
    """Each edge has two ends, whose walls contain the walls of the edge; finite vertices have n - 1 walls."""
    L = I_lat(1, 9)
    tracer = ChamberTracer(L, base=[1] + [0] * 9, chamber=[100 - i ** 2 for i in range(10)])
    tracer.run()
    for edge, ends in tracer.edges.items():
        assert len(ends) == 2 and len(edge) == L.rank - 2
        assert all(edge <= tracer.vertices[e]["walls"] for e in ends)
    for v in tracer.vertices.values():
        assert v["ideal"] == (L.square(v["vector"]) == 0)
        assert all(L.product(v["vector"], tracer.walls[i]) == 0 for i in v["walls"])
        if not v["ideal"]:
            assert len(v["walls"]) == L.rank - 1


def test_check_detects_missing_edges():
    """The final check fails if an edge at a vertex is not marked, or if the edges do not pair up."""
    tracer = ChamberTracer(I_lat(1, 5), base=[1] + [0] * 5)
    tracer.run()
    tracer._check()
    done = tracer._done[0]
    tracer._done[0] = done & (done - 1)
    with pytest.raises(RuntimeError):
        tracer._check()
    tracer._done[0] = done
    tracer._parity ^= 1
    with pytest.raises(RuntimeError):
        tracer._check()


def test_max_vertices():
    with pytest.raises(RuntimeError):
        trace_chamber(I_lat(1, 9), base=[1] + [0] * 9, max_vertices=2)


def test_progress(capsys):
    trace_chamber(I_lat(1, 4), base=[1, 0, 0, 0, 0], progress=True)
    assert "done: 5 walls, 4 vertices, 1 cusps, 10 edges" in capsys.readouterr().out


@pytest.mark.parametrize("L, base, chamber", [
    (I_lat(1, 10), [1] + [0] * 10, [121 - i ** 2 for i in range(11)]),
    (I_lat(1, 7), [9, 1, 2, 3, 0, 1, 4, 2], None),
    (Lattice(3, [[2, 1, 60], [1, -2, 0], [60, 0, -120]]), None, None),
])
def test_parallel(L, base, chamber):
    """Worker processes give the same chamber, vertices and edges as the main process alone."""
    serial = trace_chamber(L, base=base, chamber=chamber)
    parallel = trace_chamber(L, base=base, chamber=chamber, threads=3)
    assert keys(parallel["walls"]) == keys(serial["walls"])
    for key in ("vertices", "cusps", "edges"):
        assert parallel[key] == serial[key]
    s, p = serial["tracer"], parallel["tracer"]
    assert set(s.vertices) == set(p.vertices)
    assert {frozenset(e) for e in s.edges.values()} == {frozenset(e) for e in p.edges.values()}


def test_cpp_error_falls_back_to_python(monkeypatch):
    """A failed consistency check in C++ is recorded, and the edge is traced again in Python."""
    import lattice.vinberg.trace as trace
    if trace.edge_cpp is None:
        pytest.skip("no edge_cpp")
    calls = []

    class Failing:
        def __getattr__(self, name):
            return getattr(trace.edge_cpp_module, name)

        def trace_edge(self, *args):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("simulated failure")
            return trace.edge_cpp_module.trace_edge(*args)

    monkeypatch.setattr(trace, "edge_cpp_module", trace.edge_cpp, raising=False)
    monkeypatch.setattr(trace, "edge_cpp", Failing())
    L, base = I_lat(1, 7), [1] + [0] * 7
    res = trace_chamber(L, base=base)
    assert res["cpp_errors"] == 1 and res["tracer"].cpp_errors[0]["error"] == "simulated failure"
    monkeypatch.undo()
    ref = trace_chamber(L, base=base)
    assert keys(res["walls"]) == keys(ref["walls"]) and res["edges"] == ref["edges"]


def test_failed_task_input_saved(monkeypatch):
    """The input of a failed task is saved to a JSON file named in the error."""
    import json
    import os
    import re
    import lattice.vinberg.trace as trace

    def failing(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(trace, "_task_edge", failing)
    with pytest.raises(RuntimeError, match="simulated failure") as info:
        trace_chamber(I_lat(1, 4), base=[1, 0, 0, 0, 0])
    path = re.search(r"saved in (\S+\.json)", str(info.value)).group(1)
    with open(path) as f:
        saved = json.load(f)
    os.remove(path)
    assert saved["kind"] == "edge" and len(saved["args"][0]) == 5 and saved["gram"][0][0] == 1
