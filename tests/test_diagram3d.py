"""Tests of lattice.draw.diagram3d (3D pictures of Coxeter diagrams, WebGL pages)."""
import base64
import json
import re
import numpy as np
import pytest
from lattice.catalogue import A_lat, E_lat, U_lat, I_lat
from lattice.vinberg.coxeter import WallDiagram
from lattice.vinberg.trace import trace_chamber
from lattice.draw import CoxeterDiagram3D, layout3d


@pytest.fixture(scope="module")
def i114():
    L = I_lat(1, 14)
    return L, WallDiagram(L, trace_chamber(L, base=[1] + [0] * 14)["walls"])


@pytest.mark.parametrize("method", ["stress", "spectral", "geometric"])
def test_layouts(i114, method):
    L, D = i114
    X = layout3d(len(D), D.labels(), method=method, diagram=D, center=[1] + [0] * 14)
    assert X.shape == (len(D), 3) and np.isfinite(X).all()
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) + np.eye(len(D))
    assert d.min() > 1e-6  # no two walls at the same place


def test_symmetric_walls_separated():
    """Walls exchanged by a symmetry used to coincide in the classical scaling, and stay so in the majorization."""
    L = I_lat(1, 17)
    D = WallDiagram(L, trace_chamber(L, base=[1] + [0] * 17)["walls"])
    X = layout3d(len(D), D.labels())
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) + np.eye(len(D)) * 1e9
    assert d.min() > 0.1


def test_page(i114, tmp_path):
    """The page embeds the positions and the edges of each kind (as index pairs), and the orbits to color by."""
    L, D = i114
    S = D.symmetries([1] + [0] * 14)
    path = str(tmp_path / "d.html")
    C = CoxeterDiagram3D(L, D, method="geometric", symmetry=S, center=[1] + [0] * 14)
    C.draw(path, highlight=[1, 2])
    text = open(path).read()
    data = json.loads(re.search(r"const DATA = (.*);\n", text).group(1))
    assert data["k"] == len(D) and data["highlight"] == [1, 2]
    pos = np.frombuffer(base64.b64decode(data["pos"]), dtype=np.float32).reshape(-1, 3)
    assert np.allclose(pos, C.xyz, atol=1e-5)
    pairs = set()
    for kind, b64 in data["edges"].items():
        e = np.frombuffer(base64.b64decode(b64), dtype=np.uint16).reshape(-1, 2)
        assert len(e) == data["counts"][kind]
        pairs |= {tuple(sorted(map(int, p))) for p in e}
    assert pairs == set(D.labels())
    assert sorted(data["byOrbit"]["legend"]) == sorted(f"orbit of {len(o)}" for o in S.wall_orbits)
    with pytest.raises(ValueError):
        C.draw(str(tmp_path / "d.svg"))


def test_invariant_center():
    """The geometric layout of a symmetric chamber: the center is fixed by the symmetries, so the distances from it
    are constant on the orbits of walls."""
    from lattice.draw.diagram3d import _invariant_point
    L = U_lat() + A_lat(2)(-1) + A_lat(2)(-1) + A_lat(2)(-1)
    res = trace_chamber(L)
    D = WallDiagram(L, res["walls"])
    S = D.symmetries(res["tracer"].base)
    A = np.array(L.A.tolist(), dtype=float)
    mats = [np.array(M.tolist(), dtype=float) for _, M in S.generators]
    b = _invariant_point(A, np.array([float(x) for x in res["tracer"].base]), mats)
    assert all(np.allclose(b @ M, b) for M in mats)
    W = np.array(D.walls, dtype=float)
    heights = (W @ A @ b) / np.sqrt(np.abs(np.diag(np.array(D.gram, dtype=float))))
    for o in S.wall_orbits:
        assert np.allclose(heights[o], heights[o[0]])
    assert (heights >= -1e-9).all()
