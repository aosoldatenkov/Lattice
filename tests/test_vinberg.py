import math
from collections import Counter
from fractions import Fraction
import numpy as np
import pytest
from lattice._matrix import imat, imat_diag, imat2flz
from lattice.lattice import Lattice
from lattice.roots import Coxeter_graph, simple_roots, get_extremal_rays
from lattice.catalogue import A_lat, D_lat, E_lat, U_lat, I_lat
from lattice.vinberg import Vinberg


def test_simple_roots_a2():
    L = A_lat(2)
    roots = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1]]
    roots = [r for r in roots if L.square(r) == 2]
    simple = simple_roots(L, roots)
    assert len(simple) == 2
    assert Coxeter_graph(L, simple) == {(0, 1): 3}


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


@pytest.mark.parametrize("diag", [[1, -1, -1, -2], [3, -1, -1, -1], [1, -1, -1, -4]])
def test_cpp_roots_are_primitive(diag):
    # The C++ search sees vectors such as 2 e_2, whose reflection is integral; they are not roots
    from lattice import vsearch_cpp
    L = Lattice(len(diag), imat_diag(diag))
    VS = vsearch_cpp.VSearchCpp(L.A.tolist(), [1] + [0] * (len(diag) - 1), [0] * len(diag), 2 * L.exp + 0.5,
                                1, False, True)
    roots = []
    for _ in range(8):
        VS.run(10 ** 4, 10 ** 3)
        roots += VS.get_vecs()
    assert len(roots) > 0
    assert all(math.gcd(*r) == 1 and L.is_root(r) and L.square(r) < 0 for r in roots)


def search_setup(L, base):
    """The Gram matrix M in the basis used by Vinberg, and the coordinates of the base vector in it."""
    V = Vinberg(L, base=base)
    B, _ = imat2flz(V.basis).inv().numer_denom()
    return V.M, [int(x) for x in (imat2flz(V.base) * B).tolist()[0]]


def brute_roots(M, beta, h):
    """The roots v = (h, x) with (v, v) < 0, from all integer points of a box around the searched ellipsoid."""
    n = M.rank
    A = np.array(M.A.tolist(), dtype=np.int64)
    s = Fraction(int(A[0] @ np.array(beta)), beta[0])
    center = [-Fraction(beta[i + 1] * h, beta[0]) for i in range(n - 1)]
    inv = np.linalg.inv(-A[1:, 1:].astype(float))
    radius = [math.sqrt((float(s) * h * h + 2 * M.exp + 1) * inv[i, i]) for i in range(n - 1)]
    ranges = [np.arange(math.floor(-c - r) - 1, math.ceil(-c + r) + 2) for c, r in zip(center, radius)]
    grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(n - 1, -1).T
    V = np.hstack([np.full((len(grid), 1), h), grid])
    norms = np.einsum("ij,jk,ik->i", V, A, V)
    candidates = V[(norms < 0) & (norms >= -2 * M.exp)]
    return {tuple(int(t) for t in v) for v in candidates if M.is_root(v)}


@pytest.mark.parametrize("diag, base", [([3, -1, -2, -5], [2, 1, 1, 0]), ([1, -1, -1, -6], [1, 0, 0, 0]),
                                        ([1, -2, -3], [3, 1, 1])])
def test_cpp_search_is_complete(diag, base):
    """In vec mode with one thread, each call of run() searches the next height (0, 1, 2, ...)."""
    from lattice import vsearch_cpp
    M, beta = search_setup(Lattice(len(diag), imat_diag(diag)), base)
    VS = vsearch_cpp.VSearchCpp(M.A.tolist(), beta, [0] * M.rank, 2 * M.exp + 0.5, 1, False, True)
    for h in range(12):
        VS.run(10 ** 9, 10 ** 4)
        found = {tuple(v) for v in VS.get_vecs()}
        assert found == brute_roots(M, beta, h), h
        assert VS.searched_height() == h


@pytest.mark.parametrize("diag", [[1, -1, -1, -1, -1, -1], [3, -1, -2, -5], [1, -1, -1, -1, -10]])
def test_certified_run(diag):
    L = Lattice(len(diag), imat_diag(diag))
    base = [1] + [0] * (len(diag) - 1)
    walls = Vinberg(L, base=base).run(iterations=200)
    V = Vinberg(L, base=base)
    certified = V.run(iterations=200, certify=True)
    assert V.certified and V.VS.searched_height() >= V.required_height() > 0
    assert {tuple(map(int, w)) for w in certified} == {tuple(map(int, w)) for w in walls}


def test_base_point_from_complement_is_oriented():
    """list_bases takes the base point from a complement, of arbitrary sign; with (e_0, base) < 0 the chamber
    test used to reject the right roots and the search never terminated (Allcock's list, lattice 2)."""
    L = Lattice(3, [[1, 0, 0], [0, -3, 1], [0, 1, -3]])
    V = Vinberg(L, h_batch=2)
    assert L.product(V.basis[0], V.base) > 0
    walls = V.run(iterations=50)
    assert len(walls) == 3 and V.update_rays()


def test_run_with_no_walls_found():
    """An iteration in which the search returns no walls used to fail, as imat([]) has shape (0,)."""
    class NoWalls:
        def __init__(self, VS):
            self.VS = VS

        def get_walls(self):
            return []

        def __getattr__(self, name):
            return getattr(self.VS, name)

    L = I_lat(1, 3)
    V = Vinberg(L, base=[1, 0, 0, 0])
    V.VS = NoWalls(V.VS)
    walls = V.run(iterations=2)
    assert all(any(np.array_equal(w, b) for b in V.base_walls) for w in walls)


@pytest.mark.parametrize("factor", [2, 2.1, 3])
def test_bound_factor(factor):
    # A larger search window only adds non-roots, which are rejected; the walls do not change
    L = I_lat(1, 4)
    walls = Vinberg(L, base=[1, 0, 0, 0, 0]).run(iterations=10)
    V = Vinberg(L, base=[1, 0, 0, 0, 0])
    V.bound_factor = factor
    V._init_basis()
    assert {tuple(map(int, w)) for w in V.run(iterations=10, certify=True)} == {tuple(map(int, w)) for w in walls}
    assert V.certified


def test_small_bound_factor_not_certified():
    V = Vinberg(I_lat(1, 3), base=[1, 0, 0, 0])
    V.bound_factor = 1.5
    V._init_basis()
    with pytest.raises(ValueError):
        V.run(certify=True)


@pytest.mark.parametrize("gram, base", [([[1, 0, 0], [0, -3, 1], [0, 1, -3]], [3, 1, 1]),
                                        ([[516, -36, -72], [-36, 2, 5], [-72, 5, 10]], [1, 0, 0]),
                                        (imat_diag([1, -1, -1, -1, -10]).tolist(), [4, 1, 1, 1, 0])])
def test_reduced_basis(gram, base):
    L = Lattice(len(gram), gram)
    V = Vinberg(L, base=base)
    B = V.basis
    _, d = L.dual_vec(V.base)
    assert abs(int(imat2flz(B).det())) == 1
    assert L.product(B[0], V.base) == d > 0
    assert all(L.product(b, V.base) == 0 for b in B[1:])
    # The complement is LLL-reduced, and the axis has a reduced projection to it
    C = L.batch_prod(B[1:], B[1:])
    assert (Lattice(len(C), C).lll() == C).all()
    proj = np.linalg.solve(np.array(C, dtype=float), np.array(L.batch_prod(B[1:], B[:1]), dtype=float))
    assert np.all(np.abs(proj) <= 0.5 + 1e-9)


def test_base_point_candidates():
    L = Lattice(3, [[-516, 36, 72], [36, -2, -5], [72, -5, -10]])(-1)
    V = Vinberg(L, n_candidates=20)
    candidates = V.list_bases(20)
    keys = []
    for key, base, _ in candidates:
        _, d = L.dual_vec(base)
        assert L.square(base) > 0 and math.gcd(*[int(x) for x in base]) == 1
        assert key == (Fraction(L.square(base), d * d), L.square(base))
        keys.append(key)
    assert keys == sorted(keys) and len(candidates) == 20
    assert np.array_equal(V.base, candidates[0][1])


@pytest.mark.parametrize("base", [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
def test_non_positive_base_rejected(base):
    # Used to be silently replaced by a base point from list_bases
    with pytest.raises(ValueError):
        Vinberg(I_lat(1, 3), base=base)


def test_cpp_rejects_negative_orientation():
    from lattice import vsearch_cpp
    with pytest.raises(ValueError):
        vsearch_cpp.VSearchCpp([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [-1, 0, 0], [0, 0, 0], 2.5, 1, True, True)


class TestCppOverflow:
    def make(self, A, base, chamber, bound, chamber_mode=True):
        from lattice import vsearch_cpp
        return vsearch_cpp.VSearchCpp(A, base, chamber, bound, 1, chamber_mode, True)

    def test_huge_search_bound(self):
        # Coordinates up to sqrt(3e18) at height 0: products of two such vectors may overflow
        with pytest.raises(OverflowError):
            self.make([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0], [0, 0, 0], 3e18)

    def test_huge_chamber_vector(self):
        with pytest.raises(OverflowError):
            self.make([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0], [3 * 10 ** 18, 0, 0], 2.5)

    def test_huge_base_vector(self):
        with pytest.raises(OverflowError):
            self.make([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [2 ** 62, 0, 0], [0, 0, 0], 2.5)

    def test_growing_height(self):
        # Fine at heights 0 and 1; at height 2, products reach about 65 * 2^56 > 2^62
        N = 2 ** 56
        VS = self.make([[N, 0, 0], [0, -N, 0], [0, 0, -N]], [1, 0, 0], [0, 0, 0], 2 * N + 0.5, chamber_mode=False)
        with pytest.raises(OverflowError):
            for _ in range(5):
                VS.run(10 ** 6, 10 ** 3)

    def test_too_large_entries_rejected_at_the_boundary(self):
        with pytest.raises(TypeError):
            self.make([[2 ** 63, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0], [0, 0, 0], 2.5)

    def test_complement_not_negative_definite(self):
        with pytest.raises(ValueError):
            self.make([[1, 0, 0], [0, -1, 0], [0, 0, 1]], [1, 0, 0], [0, 0, 0], 2.5)


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
