"""Tests of lattice.vinberg.nonreflective (certificates of non-reflectivity)."""
import copy
import flint as fl
import pytest
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, I_lat
from lattice.vinberg.nonreflective import search_nonreflective, verify_certificate, has_infinite_order

pytest.importorskip("pynauty")

A3 = A_lat(3)(-1)


def test_infinite_order():
    assert not has_infinite_order(fl.fmpz_mat([[0, -1], [1, 0]]))          # order 4
    assert not has_infinite_order(fl.fmpz_mat([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))  # order 3
    assert has_infinite_order(fl.fmpz_mat([[2, 1], [1, 1]]))               # hyperbolic
    assert has_infinite_order(fl.fmpz_mat([[1, 1], [0, 1]]))               # parabolic: cyclotomic, not semisimple


@pytest.mark.parametrize("L, base, nwalls", [
    (Lattice(1, [[2]]) + A3, None, 5),
    (I_lat(1, 9), [1] + [0] * 9, 10),
    (Lattice(3, [[1, 0, 0], [0, -1155, 0], [0, 0, -1155]]), None, 18),
])
def test_reflective(L, base, nwalls):
    """On reflective lattices the traversal ends (the identifications made along the way have finite order)."""
    res = search_nonreflective(L, base=base, check_interval=0)
    assert res["reflective"] is True and res["certificate"] is None and len(res["tracer"].walls) == nwalls


@pytest.mark.parametrize("L", [
    Lattice(1, [[7]]) + A3,
    Lattice(1, [[11]]) + A3,
    Lattice(9, [[1 - 2 * int(i == j) for j in range(9)] for i in range(9)]),   # Baragar's Lambda_9
])
def test_symmetry_certificate(L, tmp_path):
    """A symmetry of infinite order of the chamber, found while tracing it, and verified independently."""
    path = str(tmp_path / "cert.json")
    res = search_nonreflective(L, check_interval=0, max_vertices=20000, certificate=path)
    assert res["reflective"] is False and res["certificate"]["type"] == "symmetry"
    v = verify_certificate(path)
    assert v["verified"] is True and has_infinite_order(v["matrix"])


def test_geodesic_certificate():
    """The descent from the base point meets no mirror over a period of the stabilizer of its plane."""
    L = Lattice(3, [[1, 0, 0], [0, -10, 0], [0, 0, -13]])
    res = search_nonreflective(L, check_interval=0, max_vertices=20000)
    assert res["reflective"] is False and res["certificate"]["type"] == "geodesic"
    assert verify_certificate(res["certificate"])["verified"] is True


def test_tampering():
    """A wrong bijection, a wrong path, or an element of finite order are rejected."""
    L = Lattice(1, [[11]]) + A3
    cert = search_nonreflective(L, check_interval=0, max_vertices=20000)["certificate"]
    verify_certificate(cert)
    bad = copy.deepcopy(cert)
    ident = bad["identifications"][0]
    ident["r"]["path"] = ident["x"]["path"]
    with pytest.raises(RuntimeError):
        verify_certificate(bad)
    bad = copy.deepcopy(cert)
    bad["identifications"][0]["x"] = bad["identifications"][0]["r"]   # the identity: finite order
    bad["identifications"][0]["sigma"] = list(range(len(bad["identifications"][0]["r"]["walls"])))
    with pytest.raises(RuntimeError):
        verify_certificate(bad)


def test_large_coordinates_fall_back_to_python():
    """Coordinates beyond 64 bits (far in an infinite chamber) are searched in Python, not passed to C++."""
    from lattice._matrix import imat
    from lattice.vinberg.edges import edge_crossing_roots
    L = Lattice(3, [[1, 0, 0], [0, -3, 0], [0, 0, -1]])
    k = 2 ** 70
    a, b = imat([k, 0, 0]), imat([7 * k, 4 * k, k])
    as_set = lambda found: {(tuple(int(x) for x in r), t) for r, t in found}
    found = as_set(edge_crossing_roots(L, a, b))
    assert found and found == as_set(edge_crossing_roots(L, imat([1, 0, 0]), imat([7, 4, 1])))


def test_weyl_vector():
    """E_10 = U + E_8(-1): the Weyl vector of its chamber, of norm 1240; no lattice Weyl vector for <7> + A_3(-1)."""
    from lattice.catalogue import E_lat, U_lat
    from lattice.vinberg.trace import trace_chamber
    from lattice.vinberg.nonreflective import weyl_vector
    L = U_lat() + E_lat(8)(-1)
    rho = weyl_vector(L, trace_chamber(L)["walls"])
    r = fl.fmpq_mat([[rho[i, 0] for i in range(10)]])
    assert (r * fl.fmpq_mat(fl.fmpz_mat(L.A.tolist())) * r.transpose())[0, 0] == 1240
    res = search_nonreflective(Lattice(1, [[7]]) + A3, check_interval=0, max_vertices=20000)
    assert res["analysis"]["weyl_vector"] is None


@pytest.mark.parametrize("k, kind", [(7, "virtually cyclic"), (14, "parabolic"), (19, "non-elementary")])
def test_symmetry_types(k, kind):
    """The structure of Sym(C) from the symmetries found: a fixed cusp, a preserved axis, or neither."""
    from lattice.vinberg.nonreflective import classify_element
    L = Lattice(1, [[k]]) + A3
    res = search_nonreflective(L, check_interval=0, max_vertices=20000, n_elements=6)
    a = res["analysis"]
    assert a["type"] == kind
    if kind == "parabolic":
        c = a["cusp"]
        assert L.square(c) == 0 and all(list((fl.fmpz_mat([c]) * M).entries()) == c for M in res["search"].elements)
        assert all(h >= 0 for _, h in a["heights"])
        assert all(classify_element(L, M)[0] in ("parabolic", "elliptic") for M in res["search"].elements)
