"""Non-reflectivity certificates: symmetries of infinite order of the chamber of the reflection group.

L is reflective iff Sym(C) is finite, since O^+(L) = W(L) Sym(C) for a chamber C of the reflection group W(L).
Sym(C) has finitely many orbits of vertices on C (O^+(L) has finite covolume), and if it is infinite it contains
elements of infinite order (it is finitely generated, as a quotient of O^+(L), and linear, as a subgroup of it: an
infinite such torsion group does not exist, by Schur's theorem). So exploring the 1-skeleton of C (ChamberTracer)
either ends (L is reflective, and the chamber is certified) or eventually meets two vertices exchanged by a symmetry
of infinite order, which certifies that L is not reflective.

Identification of vertices. A symmetry g of C mapping a finite vertex x to a vertex r maps the walls at x to the
walls at r: g is determined by a bijection sigma of the walls preserving their products (an isomorphism of the local
Coxeter diagrams, found by nauty) with (x, x) = (r, r), as g = B_x^-1 P_sigma B_r in the bases B_x = (walls at x, x)
and B_r = (walls at r, r). Conversely such a g is an isometry, and it is a symmetry of C iff it is integral (it maps
the local cone of C at x to that at r, so gC and C share a vertex and its local cone). In the coordinates of B_x,
L is the lattice Lambda_x = Z^n B_x^-1, and g is integral iff Lambda_x P_sigma = Lambda_r. So for each class of
vertices (same (x, x) and isomorphic local diagrams, by the canonical form of nauty) a representative r is kept with
the orbit of Lambda_r under the automorphisms of its local diagram (with a transversal; keys: Hermite normal forms):
a new vertex x is identified with r iff Lambda_x P_sigma0 is in this orbit, for one isomorphism sigma0, which gives
g. The elements found are tested for infinite order (a non-cyclotomic factor of the characteristic polynomial, or
g^m != 1 for m the lcm of the cyclotomic orders), and so are the products g h^-1 of elements of the same class.

The certificate records the start of the traversal, paths of edges from a root vertex to the vertices involved, and
the isomorphisms; verify_certificate retraces the paths (the same searches as the tracer) and checks the rest.

Infinite geodesics. A geodesic with irrational ends (in a rational anisotropic plane) crossed by no mirror lies in the
closure of one chamber (the mirrors through its points are those containing it, all along it), which then has
infinite volume. This is decided by one period of the stabilizer of the plane (see trace.trace_ray), and found when
the descent from the base point of the tracing (which starts on no mirror transversal to its ray) meets no mirror:
the certificate is then the point and the direction.
"""
from __future__ import annotations
import json
import math
import time
from typing import Dict, List, Sequence, Tuple
import numpy as np
import flint as fl
from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.vinberg.coxeter import WallDiagram
from lattice.vinberg.trace import (ChamberTracer, InfiniteGeodesic, cusp_edges, roots_by_height, trace_ray, _river,
                                   _simple_roots, _dual_in_span, _bits, _key, _task_edge, _task_cusp)

try:
    import pynauty
except ImportError:  # optional dependency (pip install lattice[symmetry])
    pynauty = None

CERTIFICATE_FORMAT = "lattice.vinberg.nonreflective certificate 1"


def has_infinite_order(M: fl.fmpz_mat) -> bool:
    """Whether the integer matrix M has infinite order: some irreducible factor of its characteristic polynomial is
    not cyclotomic (a hyperbolic element), or M^m != 1 for m the lcm of the orders of the cyclotomic factors (a
    parabolic element, not semisimple)."""
    orders = []
    for f, _ in M.charpoly().factor()[1]:
        m = f.is_cyclotomic()
        if m == 0:
            return True
        orders.append(int(m))
    m = math.lcm(*orders)
    one = fl.fmpz_mat([[int(i == j) for j in range(M.nrows())] for i in range(M.nrows())])
    return M ** m != one


def classify_element(L: Lattice, M: fl.fmpz_mat) -> Tuple[str, object]:
    """The type of an isometry M of L preserving the time orientation (row action): ("elliptic", None) for finite
    order, ("parabolic", c) with c a primitive isotropic vector it fixes (up to sign: its unique fixed point at
    infinity: the radical of the form on the fixed space of the unipotent power M^m, m the lcm of the cyclotomic
    orders), or ("hyperbolic", None) (a non-cyclotomic, Salem factor of the characteristic polynomial)."""
    n = M.nrows()
    orders = []
    for f, _ in M.charpoly().factor()[1]:
        m = f.is_cyclotomic()
        if m == 0:
            return "hyperbolic", None
        orders.append(int(m))
    one = fl.fmpz_mat([[int(i == j) for j in range(n)] for i in range(n)])
    U = M ** math.lcm(*orders)
    if U == one:
        return "elliptic", None
    K, nullity = (U - one).transpose().nullspace()            # columns: the x with x U = x
    K = fl.fmpz_mat([[int(K[i, j]) for i in range(n)] for j in range(nullity)])
    R, r = (K * fl.fmpz_mat(L.A.tolist()) * K.transpose()).nullspace()
    if r != 1:
        raise RuntimeError("The fixed space of a parabolic element does not have a radical of rank 1")
    c = [int(x) for x in (fl.fmpz_mat([[int(R[i, 0]) for i in range(R.nrows())]]) * K).entries()]
    g = math.gcd(*c)
    return "parabolic", [x // g for x in c]


def weyl_vector(L: Lattice, walls) -> fl.fmpq_mat:
    """The solution rho of (rho, r) = -(r, r)/2 for all the given walls (a rational column vector), or None if this
    system is inconsistent: then the chamber has no lattice Weyl vector, since the condition holds for all its walls
    (if consistent, rho is a candidate, checked only on these walls)."""
    if len(walls) < L.rank or fl.fmpz_mat([[int(x) for x in w] for w in walls]).rank() < L.rank:
        raise ValueError("The walls do not span the space")
    M = fl.fmpq_mat([[int(x) for x in w] for w in walls]) * fl.fmpq_mat(fl.fmpz_mat(L.A.tolist()))
    rhs = fl.fmpq_mat([[fl.fmpq(-L.square(w), 2)] for w in walls])
    rho = (M.transpose() * M).solve(M.transpose() * rhs)
    return rho if M * rho == rhs else None


def _inverse(M: fl.fmpz_mat) -> fl.fmpz_mat:
    """The inverse of a unimodular integer matrix, as an integer matrix."""
    I = fl.fmpq_mat(M).inv()
    return fl.fmpz_mat([[int(I[i, j]) for j in range(I.ncols())] for i in range(I.nrows())])


def _lattice_key(C: fl.fmpq_mat) -> Tuple:
    """A canonical form of the lattice spanned by the rows of the rational matrix C: its denominator and the Hermite
    normal form of its multiple."""
    d = math.lcm(*[int(x.q) for x in C.entries()])
    H = fl.fmpz_mat([[int(C[i, j] * d) for j in range(C.ncols())] for i in range(C.nrows())]).hnf()
    return d, tuple(int(x) for x in H.entries())


def _permute_columns(C: fl.fmpq_mat, sigma: Sequence[int]) -> fl.fmpq_mat:
    """C P_sigma: column i of C becomes column sigma(i) (the last column, for the vertex, is fixed)."""
    n = C.ncols()
    rows = [[None] * n for _ in range(C.nrows())]
    for i in range(C.nrows()):
        for j in range(n):
            rows[i][sigma[j] if j < len(sigma) else j] = C[i, j]
    return fl.fmpq_mat(rows)


class _LocalVertex:
    """The data of a finite vertex x for the identification: its walls (in a fixed order), the basis B_x, the
    lattice Lambda_x = Z^n B_x^-1, and the canonical labeling of its local diagram."""

    def __init__(self, L: Lattice, vid: int, x: Tuple[int, ...], walls: List[Tuple[int, ...]], canonical=None):
        self.vid, self.x, self.walls = vid, x, walls
        self.k = len(walls)
        self.diagram = WallDiagram(L, walls)
        if canonical is not None:                 # (key, label), computed elsewhere (in a worker)
            self.key, self.label = canonical
        else:
            g = self.diagram.nauty_graph("gram")
            self.diagram.set_marked(g, (), "gram")
            D = self.diagram
            values = tuple(sorted({D.gram[i][j] for i in range(self.k) for j in range(i + 1, self.k)}))
            self.key = (L.square(list(x)), tuple(sorted(D.norms())), values, pynauty.certificate(g))
            self.label = pynauty.canon_label(g)
        self.B = fl.fmpq_mat([list(w) for w in walls] + [list(x)])
        self.Binv = self.B.inv()

    def isomorphism_to(self, other: "_LocalVertex") -> List[int]:
        """An isomorphism sigma0 of the local diagrams (wall i of self -> wall sigma0[i] of other), from the
        canonical labelings (lab[t]: the vertex at the canonical position t; the walls are the first layer)."""
        sigma = [0] * self.k
        for a, b in zip(self.label, other.label):
            if a < self.k:
                sigma[a] = b
        return sigma

    def automorphisms(self) -> List[List[int]]:
        g = self.diagram.nauty_graph("gram")
        self.diagram.set_marked(g, (), "gram")
        return [list(p[:self.k]) for p in pynauty.autgrp(g)[0]]


class _Class:
    """A class of vertices with a representative r, and the orbit of Lambda_r under the automorphisms of its local
    diagram, with a transversal: key -> t with Lambda_r P_t = the lattice of the key."""

    def __init__(self, r: _LocalVertex, max_orbit: int):
        self.r = r
        self.orbit: Dict[Tuple, List[int]] = {}
        self.complete = True
        start = list(range(r.k))
        self.orbit[_lattice_key(r.Binv)] = start
        frontier = [(r.Binv, start)]
        gens = r.automorphisms()
        while frontier:
            new = []
            for C, t in frontier:
                for a in gens:
                    C2 = _permute_columns(C, a)
                    key = _lattice_key(C2)
                    if key not in self.orbit:
                        if len(self.orbit) >= max_orbit:
                            self.complete = False
                            return
                        self.orbit[key] = [a[t[i]] for i in range(r.k)]   # first t, then a
                        new.append((C2, self.orbit[key]))
            frontier = new
        self.members: List[Tuple[_LocalVertex, fl.fmpz_mat, List[int]]] = []

    def identify(self, x: _LocalVertex):
        """The symmetry g mapping x to the representative (with its bijection of the walls), or None."""
        sigma0 = x.isomorphism_to(self.r)
        key = _lattice_key(_permute_columns(x.Binv, sigma0))
        t = self.orbit.get(key)
        if t is None:
            return None
        t_inv = [0] * self.r.k
        for i, ti in enumerate(t):
            t_inv[ti] = i
        sigma = [t_inv[sigma0[i]] for i in range(x.k)]              # first sigma0, then t^-1
        g = x.Binv * _permute_columns(fl.fmpq_mat([[int(i == j) for j in range(x.k + 1)] for i in range(x.k + 1)]),
                                      sigma) * self.r.B
        if any(e.q != 1 for e in g.entries()):
            raise RuntimeError("The identification of two vertices did not give an integral isometry")
        return fl.fmpz_mat([[int(g[i, j]) for j in range(g.ncols())] for i in range(g.nrows())]), sigma


class SymmetrySearch:
    """Looks for symmetries of infinite order of the chamber traced by a ChamberTracer, among the finite vertices
    registered so far (see the module docstring); update() processes the vertices added since the last call, and
    can be used as the callback of ChamberTracer.run. max_orbit bounds the orbits of the lattices under the
    automorphisms of local diagrams (a class with a larger orbit is skipped); max_products bounds the number of
    earlier members of a class with which the products g h^-1 are tested."""

    def __init__(self, tracer: ChamberTracer, max_orbit: int = 20000, max_products: int = 20,
                 max_vertices: int = None, n_elements: int = 1):
        if pynauty is None:
            raise ImportError("The symmetry search needs pynauty (pip install pynauty, or lattice[symmetry])")
        self.tracer = tracer
        self.L = tracer.L
        self.max_orbit, self.max_products, self.max_vertices = max_orbit, max_products, max_vertices
        self.classes: Dict[Tuple, _Class] = {}
        self.seen = 0
        self.identified = 0
        self.finite = 0
        self.certificate = None
        self.n_elements = n_elements      # the number of elements of infinite order to collect before stopping
        self.elements: List[fl.fmpz_mat] = []   # all the symmetries found
        self.infinite: List[fl.fmpz_mat] = []   # those of infinite order

    def _found(self, M: fl.fmpz_mat, identifications, word) -> bool:
        """Records a symmetry; the first one of infinite order gives the certificate."""
        self.elements.append(M)
        if not has_infinite_order(M):
            return False
        self.infinite.append(M)
        if self.certificate is None:
            self.certificate = self._certificate(identifications, word)
        return True

    def update(self, tracer=None) -> bool:
        """Processes the new vertices; returns True if a certificate was found (or max_vertices exceeded), to stop
        the traversal."""
        t = self.tracer
        n_vertices = len(t._keys)
        for vid in range(self.seen, n_vertices):
            if len(self.infinite) >= self.n_elements:
                break
            if t._ideal[vid]:
                continue
            x = _LocalVertex(self.L, vid, t._vector(vid), [t.wall_keys[i] for i in _bits(t._walls[vid])])
            cls = self.classes.get(x.key)
            if cls is None:
                self.classes[x.key] = _Class(x, self.max_orbit)
                continue
            if not cls.complete:
                continue
            found = cls.identify(x)
            if found is None:
                continue
            g, sigma = found
            self.identified += 1
            if not self._found(g, [(x, cls.r, sigma)], [(0, 1)]):
                self.finite += 1
            # Products with earlier members of the class: g h^-1 maps x to the member y of h
            for y, h, tau in cls.members[-self.max_products:]:
                if len(self.infinite) >= self.n_elements:
                    break
                self._found(g * _inverse(h), [(x, cls.r, sigma), (y, cls.r, tau)], [(0, 1), (1, -1)])
            cls.members.append((x, g, sigma))
        self.seen = n_vertices
        return len(self.infinite) >= self.n_elements or (self.max_vertices is not None and
                                                         n_vertices > self.max_vertices)

    def analysis(self) -> dict:
        """What the symmetries found say about Sym(C) (see analyze_symmetries)."""
        return analyze_symmetries(self.L, self.elements, self.tracer.walls, self.tracer.base)

    # --- the certificate ---

    def _root(self) -> Tuple[int, List[int]]:
        """A finite root vertex for the paths: the start if finite, else its first finite neighbour (with the
        wall mask of the edge from it to the start)."""
        t = self.tracer
        start = 0
        if not t._ideal[start]:
            return start, []
        for vid in range(1, len(t._keys)):
            if t._parent[vid] == start and not t._ideal[vid]:
                vids, masks = t.path(vid)
                return vid, [masks[0]]
        raise RuntimeError("No finite vertex next to the start")

    def _path(self, vid: int) -> List[List[List[int]]]:
        """The path of edges from the root to the vertex: the walls of each edge (as vectors)."""
        t = self.tracer
        root, prefix = self._root()
        masks = t.path(vid)[1]
        if root != 0:
            vids, _ = t.path(vid)
            if len(vids) > 1 and vids[1] == root:
                masks = masks[1:]          # the path goes through the root: start from it
            else:
                masks = prefix + masks
        return [[list(t.wall_keys[i]) for i in _bits(E)] for E in masks]

    def _certificate(self, identifications, word) -> dict:
        t = self.tracer
        root, _ = self._root()
        return {
            "format": CERTIFICATE_FORMAT,
            "type": "symmetry",
            "gram": [[int(x) for x in row] for row in self.L.A.tolist()],
            "root": {"vector": list(t._vector(root)), "walls": [list(t.wall_keys[i]) for i in _bits(t._walls[root])]},
            "identifications": [{"x": {"vector": list(x.x), "walls": [list(w) for w in x.walls],
                                       "path": self._path(x.vid)},
                                 "r": {"vector": list(r.x), "walls": [list(w) for w in r.walls],
                                       "path": self._path(r.vid)},
                                 "sigma": [int(s) for s in sigma]} for x, r, sigma in identifications],
            "word": [[int(i), int(e)] for i, e in word],
        }


def analyze_symmetries(L: Lattice, elements, walls, base) -> dict:
    """The structure of a group of isometries containing the given elements (e.g. the symmetries of the chamber
    found by SymmetrySearch). A virtually abelian (elementary) infinite discrete group either fixes a cusp (all its
    elements of infinite order are parabolic, with that fixed point) or preserves a geodesic (it is virtually cyclic:
    its elements of infinite order are hyperbolic with that axis, and its commutators have finite order). Returns:
        "type": "parabolic" (all the elements fix the cusp of the parabolic ones), "virtually cyclic" (all the
            hyperbolic ones, and their conjugates by all the elements, have commutators of finite order),
            "non-elementary" (rigorous: two parabolic elements with different cusps, a parabolic and a hyperbolic
            one, a commutator of infinite order of hyperbolic ones, or an element not fixing the cusp), or "finite"
            (no element of infinite order);
        "cusp": the fixed cusp (parabolic type), with "heights": the pairs (-(r, r), (c, r)) over the walls;
        "weyl_vector": the solution of (rho, r) = -(r, r)/2 over the walls (a candidate lattice Weyl vector), or
            None if inconsistent (rigorous: no lattice Weyl vector), or "undetermined" if the walls do not span."""
    kinds = [(M, *classify_element(L, M)) for M in elements]
    infinite = [(M, k, c) for M, k, c in kinds if k != "elliptic"]
    result = {"elements": len(elements), "infinite": len(infinite),
              "parabolic": sum(1 for _, k, _ in infinite if k == "parabolic"),
              "hyperbolic": sum(1 for _, k, _ in infinite if k == "hyperbolic")}
    try:
        rho = weyl_vector(L, walls)
        result["weyl_vector"] = None if rho is None else [rho[i, 0] for i in range(L.rank)]
    except ValueError:
        result["weyl_vector"] = "undetermined"
    if not infinite:
        result["type"] = "finite"
        return result
    b = [int(x) for x in base]
    cusps = set()
    for M, k, c in infinite:
        if k == "parabolic":
            cusps.add(tuple(c if L.product(c, b) > 0 else [-x for x in c]))
    if result["parabolic"] and result["hyperbolic"] or len(cusps) > 1:
        result["type"] = "non-elementary"
        return result
    if cusps:
        c = list(next(iter(cusps)))
        fixes = all(list((fl.fmpz_mat([c]) * M).entries()) == c for M in elements)
        result["type"] = "parabolic" if fixes else "non-elementary"
        result["cusp"] = c
        result["heights"] = sorted({(-L.square(w), L.product(c, w)) for w in walls})
        return result
    # Hyperbolic elements only: commutators with each other and with their conjugates by all the elements
    hyp = [M for M, k, _ in infinite]
    g = hyp[0]
    tests = [h for h in hyp[1:]] + [M * g * _inverse(M) for M in elements]
    for h in tests:
        if has_infinite_order(g * h * _inverse(g) * _inverse(h)):
            result["type"] = "non-elementary"
            return result
    result["type"] = "virtually cyclic"
    return result


def search_nonreflective(L: Lattice, base=None, chamber=None, threads: int = 1, progress: bool = False,
                         check_interval: float = 5, max_vertices: int = None, certificate: str = None,
                         n_elements: int = 1, **kwargs) -> dict:
    """Traces the chamber of L and looks for a symmetry of infinite order of it at the same time (every
    check_interval seconds). Returns {"reflective": True (the traversal ended: the chamber is certified), False (a
    certificate was found) or None (max_vertices exceeded first), "certificate", "result" (of the tracer),
    "tracer", "search", "analysis"}; with certificate, a path, the certificate is also saved there (JSON). With
    n_elements > 1, the search goes on until that many symmetries of infinite order are found, for the analysis of
    Sym(C) (analyze_symmetries: parabolic, virtually cyclic or non-elementary; heights of the fixed cusp; lattice
    Weyl vector)."""
    tracer = ChamberTracer(L, base=base, chamber=chamber, threads=threads, **kwargs)
    search = SymmetrySearch(tracer, max_vertices=max_vertices, n_elements=n_elements)
    try:
        result = tracer.run(progress=progress, callback=search.update, callback_interval=check_interval)
    except InfiniteGeodesic as e:
        # In the descent from the base point: the chamber contains a whole geodesic with irrational ends
        search.certificate = {"format": CERTIFICATE_FORMAT, "type": "geodesic",
                              "gram": [[int(x) for x in row] for row in L.A.tolist()],
                              "point": [int(x) for x in e.a], "direction": [int(x) for x in e.u]}
        result = {"complete": False}
    if not result["complete"] and len(search.infinite) < n_elements:
        search.update()
    reflective = True if result["complete"] else (False if search.certificate is not None else None)
    if progress:
        verdict = {True: "reflective (the chamber is certified)", False: "not reflective (a symmetry of infinite "
                   "order of the chamber was found)", None: "undecided"}[reflective]
        print(f"{verdict}: {len(tracer._keys)} vertices, {len(tracer.walls)} walls, {len(search.classes)} classes "
              f"of vertices, {search.identified} identifications ({search.finite} of finite order)", flush=True)
    if certificate is not None and search.certificate is not None:
        with open(certificate, "w") as f:
            json.dump(search.certificate, f)
    return {"reflective": reflective, "certificate": search.certificate, "result": result, "tracer": tracer,
            "analysis": search.analysis() if reflective is False and search.elements else None,
            "search": search}


# --- verification ---

def _local_walls_ok(L: Lattice, v, walls) -> bool:
    """Whether the walls are the walls of a chamber of the local root system at the finite vertex v: n - 1
    independent roots orthogonal to v, the simple roots for the chamber containing the direction y in their span
    with (y, s) > 0 for all of them."""
    v = imat(list(v))
    S = [imat(list(w)) for w in walls]
    if L.square(v) <= 0 or len(S) != L.rank - 1 or fl.fmpz_mat([list(w) for w in walls]).rank() != L.rank - 1:
        return False
    if any(L.product(v, s) != 0 or not L.is_root(s) for s in S):
        return False
    y = _dual_in_span(L, S)
    simple = _simple_roots(L, roots_by_height(L, v, 0), y)
    return {_key(s) for s in simple} == {_key(s) for s in S}


def _follow(L: Lattice, vector, walls, path, cpp: bool = True):
    """Follows the path (the walls of each edge) from a vertex of the chamber with the given walls: each edge is
    traced (with the searches of trace.py), and the walls at its end are computed. Returns the last vertex and its
    walls."""
    a, S, ideal = tuple(int(x) for x in vector), [tuple(int(x) for x in w) for w in walls], False
    ideal = L.square(list(a)) == 0
    for E in path:
        E = [tuple(int(x) for x in w) for w in E]
        if not set(E) <= set(S):
            raise RuntimeError("The path takes an edge whose walls are not at the vertex")
        D = [w for w in S if w not in set(E)]
        if ideal:
            drops = [{S[i] for i in d} for d, _ in cusp_edges(L, imat(list(a)), [imat(list(w)) for w in S])]
            if set(D) not in drops:
                raise RuntimeError("The path takes a set of walls which is not an edge at a cusp")
        elif len(D) != 1:
            raise RuntimeError("The path takes a set of walls which is not an edge at a finite vertex")
        end, end_ideal, alpha, _ = _task_edge(L, None, a, E, D, S, cpp=cpp)
        if end_ideal:
            m = a if not ideal else tuple(x + y for x, y in zip(a, end))
            walls_z, _, _ = _task_cusp(L, end, E, m, cpp=cpp)
            S = [tuple(int(x) for x in w) for w in walls_z]
        else:
            S = E + [tuple(int(x) for x in alpha)]
        a, ideal = tuple(int(x) for x in end), end_ideal
    return a, S


def verify_certificate(cert: dict | str, cpp: bool = True) -> dict:
    """Checks a certificate of non-reflectivity (a dict, or the path of a JSON file): the walls at the root vertex
    are those of a chamber C at it; each path, retraced, leads to the stated vertex with the stated walls (so all the
    vertices are vertices of C); each bijection of walls gives an integral isometry g = B_x^-1 P_sigma B_r (so a
    symmetry of C); and the product of the word in these symmetries has infinite order. Returns {"verified": True,
    "matrix": the element of infinite order, "characteristic polynomial"}; raises RuntimeError otherwise."""
    if isinstance(cert, str):
        with open(cert) as f:
            cert = json.load(f)
    if cert.get("format") != CERTIFICATE_FORMAT:
        raise ValueError("Not a non-reflectivity certificate")
    L = Lattice(len(cert["gram"]), cert["gram"])
    A = fl.fmpz_mat(cert["gram"])
    if cert.get("type") == "geodesic":
        return _verify_geodesic(L, A, cert, cpp)
    root = cert["root"]
    if not _local_walls_ok(L, root["vector"], root["walls"]):
        raise RuntimeError("The walls at the root are not those of a chamber at it")
    elements = []
    for ident in cert["identifications"]:
        ends = []
        for side in ("x", "r"):
            data = ident[side]
            v, S = _follow(L, root["vector"], root["walls"], data["path"], cpp)
            if tuple(v) != tuple(data["vector"]) or set(map(tuple, S)) != set(tuple(w) for w in data["walls"]):
                raise RuntimeError("A path of the certificate does not lead to the stated vertex and walls")
            ends.append(data)
        x, r = ends
        k = len(x["walls"])
        sigma = ident["sigma"]
        Bx = fl.fmpq_mat(x["walls"] + [x["vector"]])
        Br = fl.fmpq_mat([r["walls"][sigma[i]] for i in range(k)] + [r["vector"]])
        g = Bx.inv() * Br
        if any(e.q != 1 for e in g.entries()):
            raise RuntimeError("An identification of the certificate is not integral")
        g = fl.fmpz_mat([[int(g[i, j]) for j in range(g.ncols())] for i in range(g.nrows())])
        if g * A * g.transpose() != A:
            raise RuntimeError("An identification of the certificate is not an isometry")
        elements.append(g)
    M = fl.fmpz_mat([[int(i == j) for j in range(L.rank)] for i in range(L.rank)])
    for i, e in cert["word"]:
        M = M * (elements[i] if e == 1 else _inverse(elements[i]))
    if not has_infinite_order(M):
        raise RuntimeError("The element of the certificate has finite order")
    return {"verified": True, "matrix": M, "characteristic polynomial": M.charpoly()}


def _verify_geodesic(L: Lattice, A: fl.fmpz_mat, cert: dict, cpp: bool) -> dict:
    """Checks a certificate of an infinite geodesic: the plane of the point a and the direction u is anisotropic of
    signature (1, 1), with u orthogonal to a; the translation g of its binary lattice (recomputed) extends to L by the
    identity on its orthogonal complement; and no mirror crosses the line over the period (a, g(a)] (trace_ray from a
    with no known walls must raise InfiniteGeodesic)."""
    a, u = imat(cert["point"]), imat(cert["direction"])
    p, c, q = L.square(a), L.product(a, u), L.square(u)
    if p <= 0 or c != 0 or q >= 0 or math.isqrt(-p * q) ** 2 == -p * q:
        raise RuntimeError("The plane of the certificate is not anisotropic of signature (1, 1), with u orthogonal to a")
    R = _river(L, a, u)
    basis = [[int(x) for x in r] for r in R["basis"]]
    N = [[int(x) for x in r] for r in L.complement(basis)]
    image = (np.array(R["g"], dtype=object) @ np.array(basis, dtype=object)).tolist()
    M = fl.fmpq_mat(basis + N).inv() * fl.fmpq_mat([[int(x) for x in r] for r in image] + N)
    if any(e.q != 1 for e in M.entries()):
        raise RuntimeError("The translation of the plane does not extend to the lattice")
    M = fl.fmpz_mat([[int(M[i, j]) for j in range(M.ncols())] for i in range(M.nrows())])
    if M * A * M.transpose() != A or not has_infinite_order(M):
        raise RuntimeError("The translation of the plane is not an isometry of infinite order")
    try:
        end = trace_ray(L, a, u, [], use_cpp=cpp)
    except InfiniteGeodesic:
        return {"verified": True, "matrix": M, "characteristic polynomial": M.charpoly()}
    raise RuntimeError(f"A mirror crosses the geodesic of the certificate (at {_key(end[0])})")
