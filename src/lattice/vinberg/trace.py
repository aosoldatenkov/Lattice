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
  - otherwise (an irrational ideal point; then a is finite), the segments between the points
    b_k = 2^k a + m_k u with m_k / 2^k -> t_inf from below are checked one after the other (overlapping,
    so that no point is missed) until a crossing is found; for a chamber of finite volume this terminates.

The starting vertex is found by descending from a base point p (with a chamber at p): from a point q in the
relative interior of a face, follow a generic direction inside the face to the first mirror, which lies on a
face of higher codimension, until a vertex is reached.
"""
from __future__ import annotations
import math
import random
import time
from fractions import Fraction
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import flint as fl
from lattice._matrix import IMat, imat, imat2flz, nrows
from lattice.lattice import Lattice
from lattice.vinberg.walls import cusp_walls
from lattice.vinberg.edges import edge_crossing_roots
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


def _cusp_local_walls(L: Lattice, z: IMat, F: List[IMat], m: IMat) -> List[IMat]:
    """The walls of the chamber at the cusp z, given a finite point m on the incoming segment (from m to z)
    and the walls F of the chamber through m: Vinberg's acceptance among the roots through z, with base
    point m and base walls F."""
    candidates = [r for r in cusp_walls(L, z, m) if L.product(r, m) > 0]
    candidates.sort(key=lambda r: Fraction(L.product(r, m) ** 2, -L.square(r)))
    accepted = list(F)
    for r in candidates:
        if all(L.product(r, a) >= 0 for a in accepted):
            accepted.append(r)
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


class ChamberTracer:
    """Traces the 1-skeleton of a chamber of the reflection group of L (signature (1, n - 1)); see the module
    docstring. After run(), the attributes are:
        walls: the walls (roots, oriented towards the chamber);
        vertices: {vector key: {"vector", "walls" (frozenset of indices in walls), "ideal" (bool)}};
        edges: {frozenset of wall indices: [the keys of its two ends]}."""

    def __init__(self, L: Lattice, base: List[int] | IMat = None, chamber: List[int] | IMat = None,
                 max_vertices: int = None, max_steps: int = 200, seed: int = 0):
        if L.signature[0] != 1 or L.signature[1] == 0:
            raise ValueError("The lattice should be of signature (1, n) with n > 0")
        self.L = L
        self.n = L.rank
        self.A = np.array(L.A.tolist(), dtype=object)
        self.walls: List[IMat] = []
        self.wall_index: Dict[Tuple[int, ...], int] = {}
        self.vertices: Dict[Tuple[int, ...], dict] = {}
        self.edges: Dict[frozenset, list] = {}
        self.max_vertices = max_vertices
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        if base is None:
            from lattice.vinberg.algorithm import Vinberg
            base = Vinberg(L).base
        self.base = _prim(base)
        if L.square(self.base) <= 0:
            raise ValueError("The base point must have a positive square")
        self.chamber = imat([int(x) for x in chamber]) if chamber is not None else None

    # --- walls and points ---

    def _add_walls(self, walls: List[IMat]) -> frozenset:
        idx = []
        for w in walls:
            k = _key(w)
            if k not in self.wall_index:
                self.wall_index[k] = len(self.walls)
                self.walls.append(imat(list(k)))
            idx.append(self.wall_index[k])
        return frozenset(idx)

    def _prod_walls(self, v: IMat) -> np.ndarray:
        """The products of all known walls with v."""
        if not self.walls:
            return np.zeros(0, dtype=object)
        W = np.array([[int(x) for x in w] for w in self.walls], dtype=object)
        return W @ (self.A @ np.array([int(x) for x in v], dtype=object))

    # --- following a ray ---

    def _crossing_point(self, a: IMat, b: IMat, r: IMat) -> IMat:
        """The point of the segment (a, b) on the mirror of r, with (r, a) > 0 > (r, b)."""
        A, B = self.L.product(r, a), self.L.product(r, b)
        return _prim(-B * a + A * b)

    def _trace_ray(self, a: IMat, u: IMat) -> Tuple[IMat, bool]:
        """Follows x(t) = a + t u, t > 0, from the point a (finite or ideal, in the closure of the chamber) to
        the first mirror, or to the ideal end of the line. Returns (end, ideal)."""
        L = self.L
        p, c, q = L.square(a), L.product(a, u), L.square(u)
        # The candidate end from the known walls not through a: (w, a) + t (w, u) = 0 with t > 0
        pa, pu = self._prod_walls(a), self._prod_walls(u)
        best = None
        for wa, wu in zip(pa, pu):
            if wa > 0 and wu < 0:
                t = Fraction(int(wa), -int(wu))
                if best is None or t < best:
                    best = t
        if best is not None:
            b = _prim(best.denominator * a + best.numerator * u)
            if L.square(b) > 0:
                found = edge_crossing_roots(L, a, b)
                if found:
                    return self._crossing_point(a, b, found[0][0]), False
                return b, False
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
            found = edge_crossing_roots(L, a, z)
            if found:
                return self._crossing_point(a, z, found[0][0]), False
            return z, True
        # Irrational ideal end (then a is finite and c = 0 can be assumed after projecting u): approach it
        # through b_k = 2^k a + m_k u with m_k / 2^k < t_inf, and check overlapping segments
        if p <= 0:
            raise RuntimeError("Irrational ideal end from an ideal point")
        points = [a]
        for k in range(1, self.max_steps + 1):
            # m_k = floor(2^k t_inf) = floor((2^k c + sqrt(4^k disc)) / (-q)), strictly below 2^k t_inf
            m = (math.isqrt(disc * 4 ** k) + c * 2 ** k) // (-q)
            if m <= 0:
                continue
            b = _prim(2 ** k * a + m * u)
            if _key(b) == _key(points[-1]):
                continue
            points.append(b)
            if len(points) >= 3:
                # (points[-3], b) contains points[-2] in its interior; the previous segment ended at points[-2]
                start = points[-3]
                found = edge_crossing_roots(L, start, b)
                if found:
                    return self._crossing_point(start, b, found[0][0]), False
        raise RuntimeError(f"No mirror found along the ray from {_key(a)} after {self.max_steps} steps")

    # --- vertices ---

    def _register_vertex(self, v: IMat, walls: List[IMat], ideal: bool) -> Tuple[int, ...]:
        key = _key(v)
        if key not in self.vertices:
            self.vertices[key] = {"vector": v, "walls": self._add_walls(walls), "ideal": ideal}
            self.queue.extend(self._vertex_edges(key))
        return key

    def _vertex_edges(self, key) -> List[Tuple[frozenset, Tuple[int, ...], IMat]]:
        """The edges at a vertex, as (edge key, vertex key, direction from the vertex into the edge)."""
        L, vert = self.L, self.vertices[key]
        v, S = vert["vector"], sorted(vert["walls"])
        result = []
        if not vert["ideal"]:
            for s in S:
                rest = [i for i in S if i != s]
                d = L.complement([self.walls[i] for i in rest] + [v])
                if nrows(d) != 1:
                    raise RuntimeError("The walls at a finite vertex are not independent")
                u = imat([int(x) for x in d[0]])
                if L.product(u, self.walls[s]) < 0:
                    u = -u
                result.append((frozenset(rest), key, u))
        else:
            comps = _components(L, [self.walls[i] for i in S])
            for dropped in product(*[[S[j] for j in comp] for comp in comps]):
                rest = [i for i in S if i not in dropped]
                plane = L.complement([self.walls[i] for i in rest])
                if nrows(plane) != 2:
                    raise RuntimeError("The plane of an edge at a cusp is not of rank 2")
                # The other isotropic vector of the plane: for b independent of v, (s v + t b)^2 = 0 for
                # (s, t) = (-(b, b), 2 (v, b)); then (u, v) = 2 (v, b)^2 > 0, and the edge is the segment (v, u)
                b = next(imat([int(x) for x in b]) for b in plane if _rank([v, b]) == 2)
                u = _prim(-L.square(b) * v + 2 * L.product(v, b) * b)
                if not all(L.product(u, self.walls[i]) > 0 for i in dropped):
                    raise RuntimeError("Inconsistent orientation of an edge at a cusp")
                result.append((frozenset(rest), key, u))
        return result

    def _descend(self) -> Tuple[int, ...]:
        """Descends from the base point to a vertex of the chamber (see the module docstring)."""
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
            basis = L.complement(F + [q])
            _, U = Lattice(nrows(basis), L.batch_prod(basis, basis)).lll(transform=True)
            basis = U @ basis
            for attempt in range(100):
                u = sum((self.rng.randint(-3, 3) * imat([int(x) for x in b]) for b in basis), imat([0] * self.n))
                if not any(u):
                    continue
                end, ideal = self._trace_ray(q, u)
                try:
                    if ideal:
                        return self._register_vertex(end, _cusp_local_walls(L, end, F, q), True)
                    back = L.square(end) * q - L.product(q, end) * end
                    F, q = _local_walls(L, end, F, _prim(back)), end
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("No generic direction found in the descent")
        return self._register_vertex(q, F, False)

    # --- the traversal ---

    def run(self, progress: bool = False, progress_interval: float = 10) -> dict:
        """Traces the chamber; returns a summary dictionary (also stored in self.result)."""
        L = self.L
        self.queue = []
        start = last = time.perf_counter()
        first = self._descend()
        while self.queue:
            if self.max_vertices is not None and len(self.vertices) > self.max_vertices:
                raise RuntimeError(f"More than {self.max_vertices} vertices")
            ekey, vkey, u = self.queue.pop()
            ends = self.edges.setdefault(ekey, [])
            if vkey not in ends:
                ends.append(vkey)
            if len(ends) >= 2:
                continue
            a = self.vertices[vkey]["vector"]
            end, ideal = self._trace_ray(a, u)
            ekey_walls = [self.walls[i] for i in sorted(ekey)]
            ek = _key(end)
            if ek not in self.vertices:
                if ideal:
                    m = a if L.square(a) > 0 else a + end
                    walls = _cusp_local_walls(L, end, ekey_walls, m)
                else:
                    back = L.square(end) * a - L.product(a, end) * end
                    walls = _local_walls(L, end, ekey_walls, _prim(back))
                self._register_vertex(end, walls, ideal)
            if not ekey <= self.vertices[ek]["walls"]:
                raise RuntimeError("The end of an edge does not contain the walls of the edge")
            if ek not in ends:
                ends.append(ek)
            now = time.perf_counter()
            if progress and now - last >= progress_interval:
                last = now
                done = sum(1 for e in self.edges.values() if len(e) >= 2)
                print(f"[{now - start:.0f} s] {len(self.walls)} walls, {self._count(False)} vertices, "
                      f"{self._count(True)} cusps, {done} edges traced, {len(self.queue)} in the queue", flush=True)
        bad = [k for k, e in self.edges.items() if len(e) != 2]
        if bad:
            raise RuntimeError(f"{len(bad)} edges do not have two ends")
        self.result = {"walls": self.walls, "vertices": self._count(False), "cusps": self._count(True),
                       "edges": len(self.edges), "start": first, "seconds": round(time.perf_counter() - start, 3)}
        if progress:
            print(f"[{self.result['seconds']:.0f} s] done: {len(self.walls)} walls, {self.result['vertices']} vertices, "
                  f"{self.result['cusps']} cusps, {self.result['edges']} edges", flush=True)
        return self.result

    def _count(self, ideal: bool) -> int:
        return sum(1 for v in self.vertices.values() if v["ideal"] == ideal)


def trace_chamber(L: Lattice, base: List[int] | IMat = None, chamber: List[int] | IMat = None,
                  progress: bool = False, **kwargs) -> dict:
    """Finds the walls of a chamber of the reflection group of L by edge tracing, with its vertices, cusps and
    edges (see the module docstring); the result is certified by construction. base is the starting point
    and chamber a vector choosing the chamber at it (as in Vinberg); returns the summary of
    ChamberTracer.run(), with the tracer itself under "tracer"."""
    tracer = ChamberTracer(L, base=base, chamber=chamber, **kwargs)
    result = tracer.run(progress=progress)
    result["tracer"] = tracer
    return result
