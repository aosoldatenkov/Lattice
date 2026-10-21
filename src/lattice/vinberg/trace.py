"""Edge tracing: the fundamental chamber of a reflective lattice, found and certified in one pass.

Let L be a lattice of signature (1, n - 1) and C a chamber of its reflection group W. No mirror of W crosses
an edge of C in a relative interior point: near such a point, C is the wedge of the n - 2 walls through the
edge times a line along it, and a mirror transversal to the edge would cut the interior of C. So an edge of
C, followed from one of its ends, ends exactly at the first mirror crossing its line, or at the ideal point
of its line if no mirror crosses it before. If C has finite volume, its 1-skeleton (the finite and ideal
vertices and the edges) is finite and connected, so following all the edges from one vertex finds every
vertex, every edge and every wall of C (each wall contains a vertex), and the result is correct by
construction.

The walls of C at a new vertex are found locally (they need not be the mirrors found crossing the edge):
  - at a finite vertex q, the roots orthogonal to q form a finite root system R_q of rank n - 1 (height 0
    of a root search from q); the walls of C at q are its simple roots for the chamber of R_q containing
    C. If F are the walls of C along the incoming segment and u points from q back along it, this is the
    chamber containing u + eps y, where y in span(F) has (y, w) = 1 for w in F (_local_walls);
  - at an ideal vertex z (a cusp), the roots through z form an affine root system; the walls of C at z
    are the walls of its chamber containing C, found by Vinberg's acceptance criterion among the roots
    through z, closest to a point m of the incoming segment first, with the walls F through m as the
    walls at the base point (_cusp_local_walls). In each class of parallel roots (cusp_root_classes),
    only the closest one can be accepted.
A finite vertex has exactly n - 1 walls and n - 1 edges, obtained by dropping one wall. At a cusp, the walls
form an affine diagram with k components and n - 2 + k nodes, and the edges are obtained by dropping one
node from each component.

Following an edge (_trace_ray) from a point a in the direction u, i.e. along x(t) = a + t u, t > 0:
  - if a known wall crosses the line inside hyperbolic space, the first such point b is a candidate end,
    and the segment (a, b) is checked with edge_crossing_roots: the edge ends at the first root crossing
    it, or at b;
  - otherwise, if the ideal end of the line is rational (an isotropic vector z), (a, z) is checked in the
    same way, and the edge ends at the first crossing, or at the cusp z;
  - otherwise (an irrational ideal point; then a is finite, and the plane of the edge is anisotropic), walk
    along Conway's river of the plane: the positive vectors on its bank ahead of a approach the ideal point
    (one period of the river shifted by its automorphism), and the segments between them are checked one
    after the other (overlapping, so that no point is missed) until a crossing is found; for a chamber of
    finite volume this terminates. The bank vectors have small norms and consecutive ones are close, so
    every segment costs about the same (_river_points).

The starting vertex is found by descending from a base point p (with a chamber at p): from a point q in the
relative interior of a face, follow a generic direction inside the face to the first mirror, which lies on a
face of higher codimension, until a vertex is reached.
"""
from __future__ import annotations
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, Future, wait, FIRST_COMPLETED
from fractions import Fraction
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import flint as fl
from lattice._matrix import IMat, imat, imat2flz, nrows
from lattice.lattice import Lattice
from lattice.vinberg.walls import cusp_walls
from lattice.binary import BinLattice
from lattice._matrix import inv2x2
from lattice.vinberg.edges import edge_crossing_roots, lll_negative_definite
from lattice.vinberg.certify import roots_by_height


def _prim(v) -> IMat:
    v = imat([int(x) for x in v])
    g = math.gcd(*[int(x) for x in v])
    return v // g if g > 1 else v


def _key(v) -> Tuple[int, ...]:
    return tuple(int(x) for x in v)


def _rank(rows) -> int:
    return imat2flz(imat(rows)).rank() if len(rows) else 0


def _simple_roots(L: Lattice, roots: List[IMat], direction: IMat) -> List[IMat]:
    """The simple roots of the finite root system roots for the chamber containing direction, which must
    be orthogonal to none of them: the positive roots by increasing (direction, r), kept if linearly
    independent of those kept before."""
    if not roots:
        return []
    R = np.array([[int(x) for x in r] for r in roots], dtype=object)
    p = R @ (np.array(L.A.tolist(), dtype=object) @ np.array([int(x) for x in direction], dtype=object))
    if any(x == 0 for x in p):
        raise ValueError("The direction is orthogonal to a root")
    order = sorted((i for i in range(len(roots)) if p[i] > 0), key=lambda i: p[i])
    simple = []
    for i in order:
        if _rank(simple + [roots[i]]) > len(simple):
            simple.append(roots[i])
    return simple


def _dual_in_span(L: Lattice, F: List[IMat]) -> IMat:
    """An integral vector y in the span of F with (y, w) equal to the same positive integer for all w in F."""
    if not F:
        return imat([0] * L.rank)
    G = fl.fmpq_mat(imat2flz(L.batch_prod(imat(F), imat(F))))
    c = G.solve(fl.fmpq_mat(len(F), 1, [1] * len(F)))
    den = math.lcm(*[int(c[i, 0].q) for i in range(len(F))])
    coeffs = imat([int(c[i, 0] * den) for i in range(len(F))])
    return coeffs @ imat(F)


def _local_walls(L: Lattice, q: IMat, F: List[IMat], u: IMat) -> List[IMat]:
    """The walls of the chamber at the finite point q, given the walls F of the chamber along the incoming
    segment (all of them pass through q) and the direction u (orthogonal to q and F) from q back along it."""
    roots = roots_by_height(L, q, 0)
    y = _dual_in_span(L, F)
    A = np.array(L.A.tolist(), dtype=object)
    R = np.array([[int(x) for x in r] for r in roots], dtype=object)
    pu = R @ (A @ np.array([int(x) for x in u], dtype=object)) if len(roots) else []
    py = R @ (A @ np.array([int(x) for x in y], dtype=object)) if len(roots) else []
    # dir = K u + y with K so large that the sign of (dir, r) is that of (u, r) when it is nonzero; the roots
    # orthogonal to u are orthogonal to the incoming segment, hence in span(F), and (y, r) != 0 for them
    K = 1
    for a, b in zip(pu, py):
        if a != 0:
            K = max(K, abs(b) // abs(a) + 1)
        elif b == 0:
            raise ValueError("A root is orthogonal to the incoming segment but not in the span of its walls")
    walls = _simple_roots(L, roots, K * u + y)
    keys = {_key(w) for w in walls}
    if not all(_key(w) in keys for w in F):
        raise RuntimeError("The local walls do not contain the walls of the incoming segment")
    return walls


def _next_walls(L: Lattice, W: List[IMat], a: IMat, end: IMat, end_roots: List[IMat]) -> List[IMat]:
    """The walls at the finite end of an edge with walls W, followed from a: W and the one more simple root
    alpha of the root system at end. In the ordering of _local_walls, dir = K u + y with u = (end, end) a -
    (a, end) end and K large, (dir, r) = K (end, end) (r, a) + (y, r) for the roots r through end, so alpha is
    the root through end with the smallest (r, a) > 0, then the smallest (y, r): all these roots are among
    end_roots (the roots through end crossing the line of the edge, from the edge search)."""
    if not end_roots:
        raise RuntimeError("No root through the end of an edge")
    y = _dual_in_span(L, W)
    R = np.array([[int(x) for x in r] for r in end_roots], dtype=object)
    RA = R @ np.array(L.A.tolist(), dtype=object)
    pa = RA @ np.array([int(x) for x in a], dtype=object)
    py = RA @ np.array([int(x) for x in y], dtype=object)
    alpha = end_roots[min(range(len(end_roots)), key=lambda i: (pa[i], py[i]))]
    if not all(L.product(alpha, w) >= 0 for w in W):
        raise RuntimeError("The new wall at a vertex makes an obtuse angle with the walls of the edge")
    return list(W) + [alpha]


def _cusp_local_walls(L: Lattice, z: IMat, F: List[IMat], m: IMat) -> List[IMat]:
    """The walls of the chamber at the cusp z, given a finite point m on the incoming segment (from m to z)
    and the walls F of the chamber through m: Vinberg's acceptance among the roots through z, with base
    point m and base walls F."""
    A = np.array(L.A.tolist(), dtype=object)
    cands = cusp_walls(L, z, m)
    if cands:
        C = np.array([[int(x) for x in r] for r in cands], dtype=object)
        CA = C @ A
        pm = CA @ np.array([int(x) for x in m], dtype=object)
        norms = (CA * C).sum(axis=1)
        keep = [k for k in range(len(cands)) if pm[k] > 0]
        keep.sort(key=lambda k: Fraction(int(pm[k]) ** 2, -int(norms[k])))
    else:
        keep = []
    # The products of the candidates with the base walls F and with each other, computed at once
    Fm = np.array([[int(x) for x in w] for w in F], dtype=object).reshape(len(F), L.rank)
    PF = (CA[keep] @ Fm.T) if keep and len(F) else np.zeros((len(keep), 0), dtype=object)
    PC = CA[keep] @ C[keep].T if keep else np.zeros((0, 0), dtype=object)
    accepted_idx = []
    for i in range(len(keep)):
        if (PF[i] >= 0).all() and all(PC[i, j] >= 0 for j in accepted_idx):
            accepted_idx.append(i)
    accepted = list(F) + [cands[keep[i]] for i in accepted_idx]
    if _rank(accepted) != L.rank - 1:
        raise RuntimeError(f"The roots through the cusp {_key(z)} have rank {_rank(accepted)} < {L.rank - 1}: "
                           f"the chamber has infinite volume")
    return accepted


def _components(L: Lattice, walls: List[IMat]) -> List[List[int]]:
    """The connected components of the Coxeter diagram of the walls (edges: nonzero products)."""
    k = len(walls)
    G = L.batch_prod(imat(walls), imat(walls))
    seen, comps = set(), []
    for i in range(k):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            j = stack.pop()
            comp.append(j)
            for l in range(k):
                if l not in seen and G[j, l] != 0:
                    seen.add(l)
                    stack.append(l)
        comps.append(sorted(comp))
    return comps


# --- the work on edges, vertices and cusps (run in the main process or in worker processes) ---

def _crossing(L: Lattice, a: IMat, b: IMat, found) -> Tuple[IMat, bool, List[IMat]]:
    """The first crossing point of the segment (a, b) from the result of edge_crossing_roots, with all the
    roots crossing there (the result lists all the roots crossing the segment, sorted by position)."""
    r, t0 = found[0]
    A, B = L.product(r, a), L.product(r, b)
    return _prim(-B * a + A * b), False, [r for r, t in found if t == t0]


def trace_ray(L: Lattice, a: IMat, u: IMat, walls: List[IMat], complement: IMat = None,
              max_steps: int = 200) -> Tuple[IMat, bool, List[IMat]]:
    """Follows x(t) = a + t u, t > 0, from the point a (finite or ideal, in the closure of the chamber) to the
    first mirror, or to the ideal end of the line; walls are known walls, used to predict the end (the result
    does not depend on them). Returns (end, ideal, end_roots), where end_roots are all the roots through a
    finite end with (r, a) > 0 (None for an ideal end). complement is an optional LLL-reduced basis of the
    orthogonal complement of the line in L."""
    p, c, q = L.square(a), L.product(a, u), L.square(u)
    # The candidate end from the known walls not through a: (w, a) + t (w, u) = 0 with t > 0
    best = None
    if walls:
        W = np.array([[int(x) for x in w] for w in walls], dtype=object)
        WA = W @ np.array(L.A.tolist(), dtype=object)
        pa = WA @ np.array([int(x) for x in a], dtype=object)
        pu = WA @ np.array([int(x) for x in u], dtype=object)
        for wa, wu in zip(pa, pu):
            if wa > 0 and wu < 0:
                t = Fraction(int(wa), -int(wu))
                if best is None or t < best:
                    best = t
    if best is not None:
        b = _prim(best.denominator * a + best.numerator * u)
        if L.square(b) > 0:
            found = edge_crossing_roots(L, a, b, include_end=True, complement=complement)
            # The crossings come first, then the roots through b (t = None)
            if found and found[0][1] is not None:
                return _crossing(L, a, b, found)
            return b, False, [r for r, t in found if t is None]
    # The ideal end of the line: (a + t u)^2 = p + 2 t c + t^2 q = 0 for the root t_inf > 0. From a finite
    # point, u is orthogonal to a, so q < 0; from a cusp, u is the other isotropic vector of the plane
    # (q = 0), and the ideal end is u itself
    if q > 0:
        raise RuntimeError("The direction of a ray must not be timelike")
    if q == 0:
        z, disc = u, None
    else:
        disc = c * c - p * q
        s = math.isqrt(disc)
        # t_inf = (c + sqrt(disc)) / (-q)
        z = _prim((-q) * a + (c + s) * u) if s * s == disc else None
    if z is not None:
        found = edge_crossing_roots(L, a, z, complement=complement)
        if found:
            return _crossing(L, a, z, found)
        return z, True, None
    # Irrational ideal end (then a is finite): walk along Conway's river of the plane, through the positive
    # vectors on its bank ahead of a, and check overlapping segments between them
    if p <= 0:
        raise RuntimeError("Irrational ideal end from an ideal point")
    points = [a]
    for b in _river_points(L, a, u):
        points.append(b)
        if len(points) >= 3:
            # (points[-3], b) contains points[-2] in its interior; the previous segment ended at points[-2]
            start = points[-3]
            found = edge_crossing_roots(L, start, b, complement=complement)
            if found:
                return _crossing(L, start, b, found)
        if len(points) > max_steps:
            break
    raise RuntimeError(f"No mirror found along the ray from {_key(a)} after {max_steps} steps")


def _river_points(L: Lattice, a: IMat, u: IMat):
    """The positive vectors on the bank of Conway's river of the plane Pi = <a, u> (anisotropic, of signature
    (1, 1), with u orthogonal to a), in the order of their positions t along a + t u, from the first one with
    t > 0 towards the ideal end (t -> t_inf), oriented with (x, a) > 0.

    The bank vectors of one period of the river (BinLattice.river), shifted by the powers of its automorphism
    S, are all the bank vectors; their positions are monotonic along the river. Consecutive ones are close
    (one edge of the river) and have small norms (the values on the bank are bounded by the discriminant), so
    the segments between them are short: the cost of edge_crossing_roots does not grow along the walk."""
    basis = L.saturate([a, u])
    G = L.batch_prod(basis, basis)
    B = BinLattice(G[0, 0], G[1, 1], G[0, 1])
    if B.shift is None:
        raise RuntimeError("The plane of an edge with an irrational end has no periodic river")
    period = []
    for edge in B.river[:-1]:
        if not period or edge.e != period[-1]:
            period.append(edge.e)
    S = imat(B.shift)
    S_inv, _ = inv2x2(B.shift)
    p, q = L.square(a), L.square(u)

    def point(v):
        x = imat([int(c) for c in v]) @ basis
        return x if L.product(x, a) > 0 else -x

    def position(x):
        # x = alpha a + beta u with alpha = (x, a) / (a, a), beta = (x, u) / (u, u); t = beta / alpha
        return Fraction(L.product(x, u) * p, L.product(x, a) * q)

    def shifted(k):
        T = imat([[1, 0], [0, 1]])
        for _ in range(abs(k)):
            T = T @ (S if k > 0 else S_inv)
        return [point(imat(list(e)) @ T) for e in period]

    # The direction of the walk: the positions increase along the river with S, or with S^-1
    t0 = [position(x) for x in shifted(0)]
    t1 = [position(x) for x in shifted(1)]
    step = 1 if t1[0] > t0[0] else -1
    order = (lambda xs: xs) if step == 1 else (lambda xs: xs[::-1])
    # Go back to a period starting at a position <= 0, then walk forward from the first point with t > 0
    k = 0
    while position(order(shifted(k))[0]) > 0:
        k -= step
    # The last bank vector of a period can be the first one of the next period: skip repetitions, which would
    # break the overlapping of the segments in trace_ray
    last = position(a)
    while True:
        for x in order(shifted(k)):
            t = position(x)
            if t > last:
                last = t
                yield _prim(x)
        k += step


def _kernel(L: Lattice, C: IMat, u: IMat) -> IMat:
    """An LLL-reduced basis of {y in span_Z(C) : (y, u) = 0} (C a basis of a negative definite sublattice)."""
    g = imat2flz(L.batch_prod(C, imat([u.tolist()])))  # the column ((c_i, u))
    _, T = g.hnf(transform=True)
    # T g = (gcd, 0, ..., 0)^T, so the rows 2, 3, ... of T span the kernel of g
    K = imat([[int(x) for x in row] for row in T.tolist()[1:]]).reshape(-1, nrows(C)) @ C
    return lll_negative_definite(L, K)


def finite_vertex_edges(L: Lattice, v: IMat, S: List[IMat]) -> Tuple[IMat, List[Tuple[int, IMat]]]:
    """The edges at the finite vertex v with walls S: an LLL-reduced basis C of L cap v^perp (negative
    definite), and the pairs (j, u_j), u_j the direction of the edge without the wall S[j]: u_j = x C with
    x M = e_j, M = (C, S), so that (u_j, S[i]) = 0 for i != j and (u_j, S[j]) > 0."""
    C = lll_negative_definite(L, L.complement([v]))
    Minv = fl.fmpq_mat(imat2flz(L.batch_prod(C, imat([w.tolist() for w in S])))).inv()
    result = []
    for j in range(len(S)):
        row = [Minv[j, i] for i in range(len(S))]
        den = math.lcm(*[int(x.q) for x in row])
        result.append((j, _prim(imat([int(x * den) for x in row]) @ C)))
    return C, result


def cusp_edges(L: Lattice, z: IMat, S: List[IMat]) -> List[Tuple[Tuple[int, ...], IMat]]:
    """The edges at the cusp z with walls S, as pairs (dropped, u): dropped are the positions in S of one wall
    from each component of the Coxeter diagram of S, and u is the other isotropic vector of the plane of the
    edge, oriented into the chamber (the edge is the segment (z, u))."""
    result = []
    comps = _components(L, S)
    for dropped in product(*comps):
        rest = [S[i] for i in range(len(S)) if i not in dropped]
        plane = L.complement(rest)
        if nrows(plane) != 2:
            raise RuntimeError("The plane of an edge at a cusp is not of rank 2")
        # For b independent of z, (s z + t b)^2 = 0 for (s, t) = (-(b, b), 2 (z, b)); then (u, z) = 2 (z, b)^2 > 0
        b = next(imat([int(x) for x in b]) for b in plane if _rank([z, b]) == 2)
        u = _prim(-L.square(b) * z + 2 * L.product(z, b) * b)
        if not all(L.product(u, S[i]) > 0 for i in dropped):
            raise RuntimeError("Inconsistent orientation of an edge at a cusp")
        result.append((tuple(dropped), u))
    return result


def _task_edge(L, max_steps, a, u, C, edge_walls, walls):
    """Traces an edge from a in the direction u; returns (end, ideal, new wall at a finite end or None)."""
    a, u = imat(list(a)), imat(list(u))
    complement = _kernel(L, imat([list(r) for r in C]), u) if C is not None else None
    end, ideal, end_roots = trace_ray(L, a, u, [imat(list(w)) for w in walls], complement, max_steps)
    if ideal:
        return _key(end), True, None
    W = [imat(list(w)) for w in edge_walls]
    alpha = _next_walls(L, W, a, end, end_roots)[-1]
    return _key(end), False, _key(alpha)


def _task_vertex(L, v, S):
    """The data of the edges at a new finite vertex: (C, [(j, u_j)])."""
    C, edges = finite_vertex_edges(L, imat(list(v)), [imat(list(w)) for w in S])
    return [_key(r) for r in C], [(j, _key(u)) for j, u in edges]


def _task_cusp(L, z, edge_walls, m):
    """The walls at a new cusp (reached along an edge with walls edge_walls from the finite point m) and the
    data of its edges: (walls, [(dropped, u)])."""
    z, m = imat(list(z)), imat(list(m))
    S = _cusp_local_walls(L, z, [imat(list(w)) for w in edge_walls], m)
    return [_key(w) for w in S], [(d, _key(u)) for d, u in cusp_edges(L, z, S)]


_WORKER = {}


def _init_worker(gram, max_steps):
    _WORKER["L"] = Lattice(len(gram), gram)
    _WORKER["max_steps"] = max_steps


def _run_task(kind, args):
    L = _WORKER["L"]
    if kind == "edge":
        return _task_edge(L, _WORKER["max_steps"], *args)
    if kind == "vertex":
        return _task_vertex(L, *args)
    return _task_cusp(L, *args)


class _InlineExecutor:
    """A stand-in for ProcessPoolExecutor running the tasks immediately in this process."""

    def __init__(self, gram, max_steps):
        _init_worker(gram, max_steps)

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class ChamberTracer:
    """Traces the 1-skeleton of a chamber of the reflection group of L (signature (1, n - 1)); see the module
    docstring. After run(), the attributes are:
        walls: the walls (roots, oriented towards the chamber);
        vertices: {vector key: {"vector", "walls" (frozenset of indices in walls), "ideal" (bool)}};
        edges: {frozenset of wall indices: [the keys of its two ends]}.

    With threads > 1, the edges, and the new vertices and cusps, are processed by a pool of worker processes:
    the main process only keeps the registries and the queue. The order in which the edges are traced does not
    change the result (the known walls only predict the ends, which are certified by the searches)."""

    def __init__(self, L: Lattice, base: List[int] | IMat = None, chamber: List[int] | IMat = None,
                 max_vertices: int = None, max_steps: int = 200, seed: int = 0, threads: int = 1):
        if L.signature[0] != 1 or L.signature[1] == 0:
            raise ValueError("The lattice should be of signature (1, n) with n > 0")
        self.L = L
        self.n = L.rank
        self.walls: List[IMat] = []
        self.wall_index: Dict[Tuple[int, ...], int] = {}
        self.vertices: Dict[Tuple[int, ...], dict] = {}
        self.edges: Dict[frozenset, list] = {}
        self.max_vertices = max_vertices
        self.max_steps = max_steps
        self.threads = threads
        self.rng = random.Random(seed)
        if base is None:
            from lattice.vinberg.algorithm import Vinberg
            base = Vinberg(L).base
        self.base = _prim(base)
        if L.square(self.base) <= 0:
            raise ValueError("The base point must have a positive square")
        self.chamber = imat([int(x) for x in chamber]) if chamber is not None else None

    def _add_walls(self, walls) -> frozenset:
        idx = []
        for w in walls:
            k = _key(w)
            if k not in self.wall_index:
                self.wall_index[k] = len(self.walls)
                self.walls.append(imat(list(k)))
            idx.append(self.wall_index[k])
        return frozenset(idx)

    def _descend(self) -> Tuple[Tuple[int, ...], List[IMat], bool]:
        """Descends from the base point to a vertex of the chamber (see the module docstring); returns the
        vertex, its walls and whether it is ideal."""
        L = self.L
        q = self.base
        roots = roots_by_height(L, q, 0)
        if roots:
            direction = self.chamber
            if direction is None:
                while True:
                    direction = imat([self.rng.randint(1, 10 ** 6) for _ in range(self.n)])
                    if all(L.product(direction, r) != 0 for r in roots):
                        break
            F = _simple_roots(L, roots, direction)
        else:
            F = []
        while _rank(F) < self.n - 1:
            # A generic direction in the face through q: orthogonal to q and F, short (the cost of the
            # searches along the ray grows with the size of the vectors), so from an LLL-reduced basis of the
            # complement (negative definite) with small coefficients; a choice orthogonal to a root at the
            # end of the ray (not generic) is replaced by another one
            basis = lll_negative_definite(L, L.complement(F + [q]))
            for attempt in range(100):
                u = sum((self.rng.randint(-3, 3) * imat([int(x) for x in b]) for b in basis), imat([0] * self.n))
                if not any(u):
                    continue
                end, ideal, _ = trace_ray(L, q, u, F, max_steps=self.max_steps)
                try:
                    if ideal:
                        return end, _cusp_local_walls(L, end, F, q), True
                    back = L.square(end) * q - L.product(q, end) * end
                    F, q = _local_walls(L, end, F, _prim(back)), end
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("No generic direction found in the descent")
        return q, F, False

    # --- the traversal ---

    def run(self, progress: bool = False, progress_interval: float = 10) -> dict:
        """Traces the chamber; returns a summary dictionary (also stored in self.result)."""
        L = self.L
        start = last = time.perf_counter()
        gram = [[int(x) for x in row] for row in L.A.tolist()]
        if self.threads > 1:
            executor = ProcessPoolExecutor(max_workers=self.threads, initializer=_init_worker,
                                           initargs=(gram, self.max_steps))
        else:
            executor = _InlineExecutor(gram, self.max_steps)
        pending = {}                 # future -> (kind, data)
        waiting = set()              # the keys of the new vertices whose data are being computed
        queue = []                   # the edges to trace: (edge key, vertex key, u, C)
        submitted = set()            # the keys of the edges traced or being traced
        self.vertex_edges = {}       # vertex key -> the keys of its edges
        window = 4 * max(self.threads, 1)

        def submit(kind, args, data):
            pending[executor.submit(_run_task, kind, args)] = (kind, data)

        def new_vertex(key, walls, ideal, data):
            """Registers a vertex with known walls; for a finite vertex, the data of its edges are computed
            by a task, and for a cusp they come with its walls."""
            self.vertices[key] = {"vector": imat(list(key)), "walls": self._add_walls(walls), "ideal": ideal}
            if self.max_vertices is not None and len(self.vertices) > self.max_vertices:
                raise RuntimeError(f"More than {self.max_vertices} vertices")
            if ideal:
                add_edges(key, data)
            else:
                # The positions j in the result of the vertex task refer to this order of the walls
                self.order[key] = [self.wall_index[_key(w)] for w in walls]
                waiting.add(key)
                submit("vertex", (key, [_key(w) for w in walls]), key)

        def add_edges(key, edge_data, C=None):
            vert = self.vertices[key]
            S = [self.wall_index[_key(w)] for w in edge_data[0]] if vert["ideal"] else None
            keys = []
            if vert["ideal"]:
                for dropped, u in edge_data[1]:
                    rest = frozenset(S[i] for i in range(len(S)) if i not in dropped)
                    keys.append(rest)
                    queue_edge(rest, key, u, None)
            else:
                S = self.order[key]
                for j, u in edge_data:
                    rest = frozenset(S[i] for i in range(len(S)) if i != j)
                    keys.append(rest)
                    queue_edge(rest, key, u, C)
            self.vertex_edges[key] = keys

        def queue_edge(ekey, vkey, u, C):
            if ekey not in submitted:
                submitted.add(ekey)
                queue.append((ekey, vkey, u, C))

        def end_reached(ekey, vkey, end, ideal, alpha):
            self.edges[ekey] = [vkey, end]
            if end in self.vertices or end in waiting:
                return
            a = self.vertices[vkey]["vector"]
            edge_walls = [self.walls[i] for i in sorted(ekey)]
            if ideal:
                waiting.add(end)
                m = a if L.square(a) > 0 else a + imat(list(end))
                submit("cusp", (end, [_key(w) for w in edge_walls], _key(m)), end)
            else:
                new_vertex(end, edge_walls + [imat(list(alpha))], False, None)

        # The start vertex (in this process)
        self.order = {}
        v, walls, ideal = self._descend()
        first = _key(v)
        if ideal:
            self.vertices[first] = {"vector": v, "walls": self._add_walls(walls), "ideal": True}
            add_edges(first, ([_key(w) for w in walls], [(d, _key(u)) for d, u in cusp_edges(L, v, walls)]))
        else:
            new_vertex(first, walls, False, None)
        try:
            while queue or pending:
                while queue and len(pending) < window:
                    ekey, vkey, u, C = queue.pop()
                    args = (vkey, u, C, [_key(self.walls[i]) for i in sorted(ekey)], [_key(w) for w in self.walls])
                    submit("edge", args, (ekey, vkey))
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    kind, data = pending.pop(future)
                    result = future.result()
                    if kind == "edge":
                        end, ideal, alpha = result
                        end_reached(data[0], data[1], end, ideal, alpha)
                    elif kind == "vertex":
                        C, edges = result
                        waiting.discard(data)
                        add_edges(data, edges, C)
                    else:
                        walls, edges = result
                        waiting.discard(data)
                        self.vertices[data] = {"vector": imat(list(data)), "walls": self._add_walls(walls),
                                               "ideal": True}
                        add_edges(data, (walls, edges))
                now = time.perf_counter()
                if progress and now - last >= progress_interval:
                    last = now
                    print(f"[{now - start:.0f} s] {len(self.walls)} walls, {self._count(False)} vertices, "
                          f"{self._count(True)} cusps, {len(self.edges)} edges traced, "
                          f"{len(queue) + len(pending)} in the queue", flush=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        self._check()
        self.result = {"walls": self.walls, "vertices": self._count(False), "cusps": self._count(True),
                       "edges": len(self.edges), "start": first, "seconds": round(time.perf_counter() - start, 3)}
        if progress:
            print(f"[{self.result['seconds']:.0f} s] done: {len(self.walls)} walls, {self.result['vertices']} vertices, "
                  f"{self.result['cusps']} cusps, {self.result['edges']} edges", flush=True)
        return self.result

    def _check(self):
        """Consistency of the 1-skeleton: every edge listed at a vertex has been traced, its two ends contain
        its walls, and the vertex is one of its ends."""
        for vkey, ekeys in self.vertex_edges.items():
            for ekey in ekeys:
                ends = self.edges.get(ekey)
                if ends is None or vkey not in ends:
                    raise RuntimeError("An edge at a vertex does not end at it")
        for ekey, ends in self.edges.items():
            if len(set(ends)) != 2 or not all(ekey <= self.vertices[e]["walls"] for e in ends):
                raise RuntimeError("An edge does not have two ends containing its walls")

    def _count(self, ideal: bool) -> int:
        return sum(1 for v in self.vertices.values() if v["ideal"] == ideal)


def trace_chamber(L: Lattice, base: List[int] | IMat = None, chamber: List[int] | IMat = None,
                  progress: bool = False, threads: int = 1, **kwargs) -> dict:
    """Finds the walls of a chamber of the reflection group of L by edge tracing, with its vertices, cusps and
    edges (see the module docstring); the result is certified by construction. base is the starting point
    and chamber a vector choosing the chamber at it (as in Vinberg); threads > 1 traces the edges in parallel
    worker processes. Returns the summary of ChamberTracer.run(), with the tracer itself under "tracer"."""
    tracer = ChamberTracer(L, base=base, chamber=chamber, threads=threads, **kwargs)
    result = tracer.run(progress=progress)
    result["tracer"] = tracer
    return result
