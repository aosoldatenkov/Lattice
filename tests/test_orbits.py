"""Tests of lattice.vinberg.orbits (edge tracing up to symmetry)."""
import itertools
import math
import numpy as np
import pytest
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, D_lat, U_lat, I_lat
from lattice.vinberg.trace import ChamberTracer, trace_chamber, load_walls
from lattice.vinberg.orbits import GroupAction, OrbitTracer, trace_chamber_by_orbits, verify_certificate

pytest.importorskip("pynauty")

A2, D4 = A_lat(2)(-1), D_lat(4)(-1)


def pairs_action(m):
    """S_m acting on the pairs of m points: generators (a transposition and an m-cycle) and the pair positions."""
    pairs = list(itertools.combinations(range(m), 2))
    pos = {p: i for i, p in enumerate(pairs)}

    def induced(perm):
        return [pos[tuple(sorted((perm[a], perm[b])))] for a, b in pairs]
    t = list(range(m))
    t[0], t[1] = 1, 0
    return [induced(t), induced(list(range(1, m)) + [0])], pos


def test_group_action():
    gens, pos = pairs_action(5)
    G = GroupAction(gens, 10, order=120)
    assert G.order == 120 and (G.elements[0] == np.arange(10)).all()
    # Sets in the same orbit have the same canonical image, and the element maps the set to it
    for S in ([pos[(0, 1)], pos[(2, 3)]], [pos[(1, 4)], pos[(0, 2)]], [pos[(0, 1)], pos[(1, 2)]]):
        image, e = G.canonical(S)
        assert tuple(sorted(int(x) for x in G.elements[e][S])) == image
    assert G.canonical([pos[(0, 1)], pos[(2, 3)]])[0] == G.canonical([pos[(1, 4)], pos[(0, 2)]])[0]
    assert G.canonical([pos[(0, 1)], pos[(2, 3)]])[0] != G.canonical([pos[(0, 1)], pos[(1, 2)]])[0]
    # The stabilizer of a matching of 2 pairs in S_5: 8
    assert G.stabilizer_mask([pos[(0, 1)], pos[(2, 3)]]) == 8
    H, gens_H = np.zeros(120, dtype=bool), []
    H = G.extend(H, gens_H, G.index(gens[0]))
    assert H.sum() == 2
    H = G.extend(H, gens_H, G.index(gens[1]))
    assert H.sum() == 120
    with pytest.raises(RuntimeError):
        GroupAction(gens, 10, order=60)


@pytest.mark.parametrize("L, base", [
    (I_lat(1, 14), [1] + [0] * 14),
    (U_lat() + A2 + A2 + A2, None),
    (U_lat() + D4 + D4 + D4, None),
    (Lattice(3, [[1, 0, 0], [0, -32, 0], [0, 0, -64]]), None),  # Allcock 160.12: symmetries of the diagram not in O(L)
])
def test_same_as_full_tracing(L, base):
    """The walls and the numbers of vertices, cusps and edges, from the orbits, as from the full traversal."""
    full = trace_chamber(L, base=base)
    res = trace_chamber_by_orbits(L, full["walls"], base=base)
    assert (res["walls"], res["vertices"], res["cusps"], res["edges"]) == \
        (len(full["walls"]), full["vertices"], full["cusps"], full["edges"])
    assert {tuple(int(x) for x in w) for w in res["chamber_walls"]} == {tuple(int(x) for x in w) for w in full["walls"]}
    assert res["certified_order"] == res["group_order"] and res["restarts"] == 0
    assert res["orbits"] < full["vertices"] + full["cusps"] or res["group_order"] == 1


def test_parallel():
    L = U_lat() + D4 + D4 + D4
    walls = trace_chamber(L)["walls"]
    serial = trace_chamber_by_orbits(L, walls)
    parallel = trace_chamber_by_orbits(L, walls, threads=3)
    for key in ("walls", "vertices", "cusps", "edges", "orbits", "certified_order"):
        assert serial[key] == parallel[key]


@pytest.mark.parametrize("known", [6, 0])
def test_incomplete_walls(known):
    """A wall outside the known ones restarts the traversal with it; the result is the same."""
    L = U_lat() + A2 + A2 + A2
    full = trace_chamber(L)
    res = OrbitTracer(L, full["walls"][:known]).run()
    assert res["restarts"] == len(full["walls"]) - known
    assert (res["walls"], res["vertices"], res["cusps"], res["edges"]) == \
        (len(full["walls"]), full["vertices"], full["cusps"], full["edges"])
    assert res["certified_order"] == 6


def test_walls_of_another_chamber():
    L = U_lat() + A2 + A2 + A2
    walls = trace_chamber(L)["walls"]
    with pytest.raises(ValueError):
        OrbitTracer(L, [-w for w in walls])


def test_walls_file(tmp_path):
    """The walls saved during a run, with its base point, chamber and seed, are used to certify by orbits."""
    L = U_lat() + D4 + D4 + D4
    path = str(tmp_path / "walls.json")
    tracer = ChamberTracer(L, walls_file=path, seed=3)
    tracer.run(progress=False)
    data = load_walls(path)
    assert data["seed"] == 3 and len(data["walls"]) == len(tracer.walls) == 17
    res = trace_chamber_by_orbits(L, path)
    assert (res["walls"], res["vertices"], res["edges"]) == (17, tracer.result["vertices"], tracer.result["edges"])


@pytest.mark.parametrize("L, base", [
    (I_lat(1, 14), [1] + [0] * 14),
    (U_lat() + D4 + D4 + D4, None),
])
def test_nauty_method(L, base):
    """Orbits identified by canonical labeling (for groups too large to enumerate): the same result."""
    walls = trace_chamber(L, base=base)["walls"]
    ref = trace_chamber_by_orbits(L, walls, base=base, method="enumerate")
    res = trace_chamber_by_orbits(L, walls, base=base, method="nauty")
    for key in ("walls", "vertices", "cusps", "edges", "orbits", "certified_order"):
        assert res[key] == ref[key]
    assert res["certified_order"] == res["group_order"]


def test_nauty_method_needs_integral_symmetries():
    L = Lattice(3, [[1, 0, 0], [0, -32, 0], [0, 0, -64]])  # Allcock 160.12
    with pytest.raises(ValueError):
        OrbitTracer(L, trace_chamber(L)["walls"], method="nauty")


@pytest.mark.parametrize("L, base, method", [
    (U_lat() + D4 + D4 + D4, None, "enumerate"),
    (I_lat(1, 14), [1] + [0] * 14, "nauty"),
])
def test_certificate(L, base, method, tmp_path):
    """A certificate is verified by retracing the edges of its representatives; the counts are those of the run."""
    path = str(tmp_path / "cert.json")
    walls = trace_chamber(L, base=base)["walls"]
    res = trace_chamber_by_orbits(L, walls, base=base, method=method, certificate=path)
    ver = verify_certificate(path, method=method)
    assert ver["verified"] is True
    for key in ("walls", "vertices", "cusps", "edges", "orbits", "certified_order"):
        assert ver[key] == res[key]


def test_certificate_tampering(tmp_path):
    """Wrong counts, a missing representative, a wrong generator or a missing wall are detected."""
    import json
    L = U_lat() + A2 + A2 + A2
    path = str(tmp_path / "cert.json")
    trace_chamber_by_orbits(L, trace_chamber(L)["walls"], certificate=path)
    with open(path) as f:
        data = json.load(f)

    def check(change):
        bad = json.loads(json.dumps(data))
        change(bad)
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            json.dump(bad, f)
        with pytest.raises(RuntimeError):
            verify_certificate(bad_path)

    check(lambda d: d["counts"].update(edges=d["counts"]["edges"] + 1))
    check(lambda d: d["representatives"].pop())
    check(lambda d: d["representatives"][0].update(orbit=d["representatives"][0]["orbit"] + 1))
    check(lambda d: d["group"]["generators"][0]["matrix"][0].__setitem__(0, d["group"]["generators"][0]["matrix"][0][0] + 1))
    # A missing wall: the walls of the certificate are not those of the chamber
    missing = data["walls"][-1]

    def drop_wall(d):
        i = d["walls"].index(missing)
        d["walls"].pop(i)
        for g in d["group"]["generators"]:
            g["perm"] = [x - (x > i) for x in g["perm"] if x != i]
        for r in d["representatives"]:
            r["walls"] = [x - (x > i) for x in r["walls"] if x != i]
    check(drop_wall)
