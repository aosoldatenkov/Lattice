"""Tests of lattice.vinberg.certify."""
import io
import contextlib
import pytest
from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.catalogue import I_lat
from lattice.vinberg import RunTest
from lattice.vinberg.certify import certify_polytope, roots_by_height

# The chamber of I_{1,2}: a triangle with angles pi/4 (between the first two walls), pi/2 and 0
CHAMBER = [[0, 1, -1], [0, 0, 1], [1, -1, -1]]
# The chamber and its image under the reflection in its third wall: a triangle with angles pi/4, pi/4, 0
# bounded by mirrors, containing two chambers; the mirror (1, -1, -1) crosses its edge through the two
# finite vertices and passes through its cusp
TWO_CHAMBERS = [[0, 1, -1], [0, 0, 1], [2, -2, -1]]


@pytest.mark.parametrize("method", ["projection", "height"])
def test_chamber(method):
    report = certify_polytope(I_lat(1, 2), CHAMBER, method=method)
    assert report["certified"] is True and report["violations"] == []
    assert (report["vertices"], report["cusps"], report["edges"]) == (2, 1, 3)


@pytest.mark.parametrize("method", ["projection", "height"])
def test_two_chambers(method):
    L = I_lat(1, 2)
    report = certify_polytope(L, TWO_CHAMBERS, method=method)
    assert report["certified"] is False
    kinds = {kind for kind, _, _ in report["violations"]}
    assert kinds == {"edge", "cusp"}
    # The violating roots are the mirror of the third wall of the chamber, with either sign
    assert {tuple(abs(x) for x in r) for _, _, r in report["violations"]} == {(1, 1, 1)}


def test_skipped_search():
    report = certify_polytope(I_lat(1, 2), TWO_CHAMBERS, max_height=0, method="height")
    # The edge search from the vertex (1, 0, 0) needs height 4; the violation at the cusp is still found
    assert report["skipped"] and report["certified"] is False


def test_infinite_volume():
    with pytest.raises(ValueError):
        certify_polytope(I_lat(1, 2), CHAMBER[:2] + [[0, 1, 0]])


def test_roots_by_height():
    # The roots of I_{1,2} orthogonal to (1, 0, 0) (height 0, both signs) and those of height 1
    L = I_lat(1, 2)
    roots = {tuple(int(x) for x in r) for r in roots_by_height(L, imat([1, 0, 0]), 1)}
    assert {r for r in roots if r[0] == 0} == {(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 1, 1),
                                                (0, -1, -1), (0, 1, -1), (0, -1, 1)}
    assert all(L.is_root(list(r)) and 0 <= r[0] <= 1 for r in roots)


def run_test_walls(n):
    L = I_lat(1, n)
    with contextlib.redirect_stdout(io.StringIO()):
        return RunTest(L, base=[1] + [0] * n, chamber=[(n + 1) ** 2 - i ** 2 for i in range(n + 1)], n_iter=0)


@pytest.mark.parametrize("n", range(3, 9))
def test_run_test_walls_certified_height(n):
    assert certify_polytope(I_lat(1, n), run_test_walls(n), method="height")["certified"] is True


@pytest.mark.parametrize("n", range(3, 12))
def test_run_test_walls_certified(n):
    assert certify_polytope(I_lat(1, n), run_test_walls(n))["certified"] is True


def test_unknown_method():
    with pytest.raises(ValueError):
        certify_polytope(I_lat(1, 2), CHAMBER, method="other")


@pytest.mark.parametrize("method", ["projection", "height"])
def test_progress(method, capsys):
    certify_polytope(I_lat(1, 2), CHAMBER, method=method, progress=True, progress_interval=0)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Polyhedron: 2 vertices, 1 cusps, 3 edges"
    assert lines[-1].endswith("vertices 2/2, cusps 1/1, edges 3/3")
