"""Tests of lattice.vinberg.edges against a brute-force enumeration of the roots in a box."""
import itertools
import random
from fractions import Fraction
import pytest
from lattice._matrix import imat, imat_diag
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, U_lat, I_lat
from lattice.vinberg.edges import edge_crossing_roots

LATTICES = [I_lat(1, 2), I_lat(1, 3), U_lat(4) + Lattice(1, [[-2]]), Lattice(3, imat_diag([1, -3, -1])),
            U_lat(2) + Lattice(1, [[-1]]) + Lattice(1, [[-2]]), U_lat(3) + A_lat(2)(-1),
            Lattice(3, imat_diag([2, -1, -6]))]


def box_roots(L, box):
    return [imat(list(x)) for x in itertools.product(range(-box, box + 1), repeat=L.rank)
            if any(x) and L.is_root(list(x))]


def check(L, a, b, roots, box):
    """The roots in the box crossing the segment are found, and the roots found in the box cross it."""
    found = edge_crossing_roots(L, a, b)
    keys = {tuple(int(x) for x in r) for r, _ in found}
    expected = {tuple(r.tolist()) for r in roots if L.product(r, a) > 0 > L.product(r, b)}
    assert expected <= keys
    assert {k for k in keys if max(abs(x) for x in k) <= box} == expected
    for r, t in found:
        assert L.is_root(r) and t == Fraction(L.product(r, a), -L.product(r, b))
    assert [t for _, t in found] == sorted(t for _, t in found)
    return found


@pytest.mark.parametrize("L, a, b, count", [
    (I_lat(1, 2), [1, 0, 0], [3, -2, -2], 1),         # two finite vertices
    (I_lat(1, 3), [1, -1, 0, 0], [1, 0, -1, 0], 1),   # two ideal vertices
    (U_lat(4) + Lattice(1, [[-2]]), [1, 1, 0], [1, 1, 2], 2),
    (Lattice(3, imat_diag([1, -3, -1])), [2, 1, 0], [7, 4, 1], 1),
])
def test_edges(L, a, b, count):
    a, b = imat(a), imat(b)
    box = 6 if L.rank == 3 else 4
    assert len(check(L, a, b, box_roots(L, box), box)) == count


@pytest.mark.parametrize("index", range(len(LATTICES)))
def test_random_segments(index):
    L = LATTICES[index]
    rng = random.Random(index)
    box = 6 if L.rank == 3 else 3
    roots = box_roots(L, box)
    points = [imat(list(x)) for x in itertools.product(range(-3, 4), repeat=L.rank)
              if any(x) and L.square(list(x)) >= 0]
    tested = 0
    while tested < 10:
        a, b = rng.sample(points, 2)
        p, q, c = L.square(a), L.square(b), L.product(a, b)
        if c <= 0 or c * c <= p * q:
            continue
        check(L, a, b, roots, box)
        tested += 1


def test_invalid_segments():
    L = I_lat(1, 2)
    with pytest.raises(ValueError):
        edge_crossing_roots(L, imat([1, 0, 0]), imat([0, 1, 0]))  # b outside the light cone
    with pytest.raises(ValueError):
        edge_crossing_roots(L, imat([1, 0, 0]), imat([2, 0, 0]))  # proportional
