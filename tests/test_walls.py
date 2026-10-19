"""Tests of lattice.vinberg.walls (cusp_walls, geodesic_walls, RunTest) and of the helpers in lattice.roots."""
import io
import contextlib
import itertools
import math
import random
from fractions import Fraction
import flint as fl
import pytest
from lattice._matrix import imat, imat_diag, flz2imat
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, U_lat, I_lat
from lattice.roots import Coxeter_graph, simple_roots
from lattice.vinberg import Vinberg
from lattice.vinberg.walls import cusp_walls, geodesic_walls, RunTest


def oriented(L, roots, base):
    return {tuple(int(x) for x in (r if L.product(r, base) > 0 else -r)) for r in roots if L.product(r, base) != 0}


CUSP_CASES = [
    # exp(v^perp / v) = 2 < exp(L) = 4: the roots of norm -8 used to be missed
    (U_lat(4) + Lattice(1, [[-2]]), [1, 1, 2], [1, 1, 0]),
    (U_lat(2) + Lattice(1, [[-2]]) + Lattice(1, [[-6]]), [1, 0, 0, 0], [1, 1, 0, 0]),
    (U_lat(3) + A_lat(2)(-1), [1, 0, 0, 0], [1, 1, 0, 0]),
    (U_lat() + Lattice(1, [[-4]]) + Lattice(1, [[-4]]), [1, 0, 0, 0], [1, 1, 0, 0]),
    (Lattice(3, [[2, 0, 0], [0, -2, 0], [0, 0, -4]]), [1, 1, 0], [1, 0, 0]),
    # The root e_1 + 2 e_3 has the non-primitive image 2 e_3 in v^perp / v
    (U_lat(2) + Lattice(1, [[-1]]) + Lattice(1, [[-2]]), [1, 0, 0, 0], [1, 1, 0, 0]),
]


@pytest.mark.parametrize("L, v, base", CUSP_CASES)
def test_cusp_walls(L, v, base):
    """For each root r orthogonal to v with (r, base) > 0 and r - t v (t > 0) not such a root, r is found."""
    v, base = imat(v), imat(base)
    found = oriented(L, cusp_walls(L, v, base), base)
    box = range(-5, 6)
    roots = {x for x in itertools.product(box, repeat=L.rank)
             if any(x) and L.product(list(x), v) == 0 and L.product(list(x), base) > 0 and L.is_root(list(x))}
    minimal = [r for r in roots if not any(tuple(c - t * w for c, w in zip(r, v)) in roots for t in range(1, 11))]
    assert minimal and all(r in found for r in minimal)
    assert all(L.is_root(list(r)) and L.product(list(r), v) == 0 for r in found)


def test_cusp_walls_non_primitive_image():
    L = U_lat(2) + Lattice(1, [[-1]]) + Lattice(1, [[-2]])
    found = oriented(L, cusp_walls(L, imat([1, 0, 0, 0]), imat([1, 1, 0, 0])), imat([1, 1, 0, 0]))
    assert (1, 0, 2, 0) in found or (1, 0, -2, 0) in found


def plane_roots(L, base):
    """The roots (x, y, 0) of L = diag(1, -3, -1) with (r, base) != 0, oriented by base, sorted by distance."""
    roots = []
    for y in range(-3000, 3001):
        for s in (-1, -2, -3, -6):
            x2 = 3 * y * y + s
            x = math.isqrt(x2) if x2 >= 0 else -1
            if x < 0 or x * x != x2:
                continue
            for r in {(x, y, 0), (-x, y, 0)}:
                d = L.product(list(r), base)
                if d != 0 and L.is_root(list(r)):
                    r = r if d > 0 else tuple(-c for c in r)
                    roots.append((Fraction(d * d, -s), r))
    return sorted(roots)


@pytest.mark.parametrize("base", [[2, 1, 0], [7, 4, 0], [97, 56, 0], [1351, 780, 0]])
def test_geodesic_walls_far_base(base):
    """The closest mirrors on both sides of the projection of the base point, also far along the river
    (the window of shifts used to be fixed around the given basis)."""
    L = Lattice(3, imat_diag([1, -3, -1]))
    base = imat(base)
    roots = plane_roots(L, base)
    first = roots[0][1]
    second = next(r for _, r in roots if L.product(list(r), list(first)) > 0)
    found = {tuple(int(x) for x in w) for w in geodesic_walls(L, imat([[1, 0, 0], [0, 1, 0]]), base)}
    assert found == {first, second}


def run_quietly(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return RunTest(*args, **kwargs)


@pytest.mark.parametrize("n", range(2, 10))
def test_run_test_small(n):
    # Vinberg: the chamber of I_{1,n} has n + 1 walls for n <= 9; the refinement starts from the base walls
    L = I_lat(1, n)
    walls = run_quietly(L, base=[1] + [0] * n, chamber=[(n + 1) ** 2 - i ** 2 for i in range(n + 1)], n_iter=0)
    assert len(walls) == n + 1


@pytest.mark.parametrize("n", [10, 11])
def test_run_test_matches_certified(n):
    L = I_lat(1, n)
    base, chamber = [1] + [0] * n, [(n + 1) ** 2 - i ** 2 for i in range(n + 1)]
    with contextlib.redirect_stdout(io.StringIO()):
        V = Vinberg(L, base=base, chamber=chamber, h_batch=4)
        certified = V.run(iterations=10 ** 5, certify=True)
    assert V.certified
    walls = run_quietly(L, base=base, chamber=chamber, n_iter=0)
    assert {tuple(int(x) for x in w) for w in walls} == {tuple(int(x) for x in w) for w in certified}


def test_update_rays_without_walls():
    V = Vinberg(I_lat(1, 3), base=[1, 0, 0, 0])
    V.active_walls = []
    assert V.update_rays() is False and V.rays == [] and V.lines == []


class TestRoots:
    def test_coxeter_graph_indices(self):
        # The walls of I_{1,3} in a scrambled order: the keys are the indices in the list
        L = I_lat(1, 3)
        walls = [[0, 0, 0, 1], [1, -1, -1, -1], [0, 1, -1, 0], [0, 0, 1, -1]]
        assert Coxeter_graph(L, walls) == {(0, 1): 4, (0, 2): 2, (0, 3): 4, (1, 2): 2, (1, 3): 2, (2, 3): 3}

    def test_coxeter_graph_ultraparallel(self):
        # Two roots of norm -2 with product 3: cos^2 = 9/4
        L = Lattice(2, [[-2, 3], [3, -2]])
        assert Coxeter_graph(L, [[1, 0], [0, 1]]) == {(0, 1): Fraction(9, 4)}

    def test_simple_roots_keeps_global_random_state(self):
        L = A_lat(2)
        random.seed(0)
        a = random.random()
        random.seed(0)
        simple_roots(L, [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]])
        assert random.random() == a

    def test_simple_roots_fmpz_entries(self):
        # Roots coming from flint matrices have fmpz entries
        L = A_lat(2)
        roots = list(flz2imat(fl.fmpz_mat([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]])))
        simple = simple_roots(L, roots)
        assert len(simple) == 2 and Coxeter_graph(L, simple) == {(0, 1): 3}
