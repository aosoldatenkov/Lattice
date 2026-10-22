"""The Coxeter diagram of the walls of a chamber, kept up to date while the walls are found, and its symmetries.

WallDiagram holds the walls r_1, ..., r_k of a chamber C (or the walls found so far) with their Gram matrix
G_ij = (r_i, r_j), computed incrementally (one row per new wall), and the labels of the Coxeter diagram.

Symmetries. An isometry of L preserving C permutes its walls and preserves G. Conversely, if the walls span V, a
permutation s of the walls with G_{s(i) s(j)} = G_ij defines a unique isometry g of V with g(r_i) = r_{s(i)} (the
linear relations among the walls are those among the rows of G, the form being nondegenerate); g preserves the
cone {x : (x, r_i) >= 0}, and it is a symmetry of C in O(L) iff it is integral (g(L) = L) and preserves the time
orientation. So
    Sym(C) = {s in Aut(G) : g(L) = L, g preserves the future cone},
where Aut(G) is the automorphism group of the complete graph on the walls with the vertices colored by the norms
and the edges by the products, computed by nauty (pynauty; the edge colors are encoded in the bits of layers of
copies of the vertices). Sym(C) is the stabilizer of the pair (L, future cone) under the action of Aut(G) on the
rational lattices of V and the two time orientations; its generators are the Schreier generators of the orbit of
this pair (usually of size 1 or 2). With all the walls of C, O^+(L) = W(L) Sym(C).

With only some of the walls of C known, Sym(G) is the symmetry group of the polyhedron P of the known walls; an
element g of it is a symmetry of C as soon as it maps one vertex v of C to a vertex of C with the same walls
g(walls at v), because gC and C then share a vertex and its local cone (see docs/EdgeTracing.md).
"""
from __future__ import annotations
import math
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple
import numpy as np
import flint as fl
from lattice._matrix import IMat, imat
from lattice.lattice import Lattice
from lattice.roots import Coxeter_label

try:
    import pynauty
except ImportError:  # optional dependency (pip install lattice[symmetry])
    pynauty = None


class WallDiagram:
    """The walls of a chamber of L with their Gram matrix and the labels of their Coxeter diagram, extended by
    add() as walls are found. The walls are oriented towards the chamber; their indices are their positions."""

    def __init__(self, L: Lattice, walls: Sequence = ()):
        self.L = L
        self.n = L.rank
        self.walls: List[Tuple[int, ...]] = []
        self.index: Dict[Tuple[int, ...], int] = {}
        self.gram: List[List[int]] = []
        self._A = np.array(L.A.tolist(), dtype=object)
        self.add(walls)

    def __len__(self) -> int:
        return len(self.walls)

    def add(self, walls: Sequence) -> List[int]:
        """Adds walls (the known ones are skipped); returns their indices."""
        idx = []
        for w in walls:
            key = tuple(int(x) for x in w)
            if key not in self.index:
                wa = self._A @ np.array(key, dtype=object)
                row = [int(x) for x in np.array(self.walls, dtype=object).reshape(-1, self.n) @ wa] if self.walls else []
                for i, x in enumerate(row):
                    self.gram[i].append(x)
                self.gram.append(row + [int(np.dot(np.array(key, dtype=object), wa))])
                self.index[key] = len(self.walls)
                self.walls.append(key)
            idx.append(self.index[key])
        return idx

    def norms(self) -> List[int]:
        """The norms -(r, r) of the walls."""
        return [-self.gram[i][i] for i in range(len(self.walls))]

    def labels(self) -> Dict[Tuple[int, int], object]:
        """The labels of the Coxeter diagram, {(i, j): label} for the pairs i < j which are not orthogonal (labels
        as in roots.Coxeter_graph: 3, 4, 6, 'inf', or cos^2 > 1 for an ultraparallel pair)."""
        G, k = self.gram, len(self.walls)
        labels = {}
        for i in range(k):
            for j in range(i + 1, k):
                if G[i][j] != 0:
                    labels[(i, j)] = Coxeter_label(Fraction(G[i][j] ** 2, G[i][i] * G[j][j]))
        return labels

    def rank(self) -> int:
        return fl.fmpz_mat([list(w) for w in self.walls]).rank() if self.walls else 0

    # --- symmetries ---

    def nauty_graph(self, kind: str = "gram"):
        """The colored graph on the walls for nauty, cached (only the coloring changes with set_marked): with
        kind="gram", its automorphisms are the permutations preserving the Gram matrix (the nonzero products as
        edge colors 1, ..., m, encoded in the bits of b layers of copies of the walls: vertex (i, l) is i + l k;
        the copies of a wall are joined in a path); with kind="coxeter", one layer and the pairs with an angle or
        parallel (0 < cos^2 <= 1) as edges, whose automorphisms contain those of the Gram matrix. The vertices are
        colored by the layer and the norm (and the marked walls, see set_marked)."""
        if pynauty is None:
            raise ImportError("The symmetries need pynauty (pip install pynauty, or lattice[symmetry])")
        cache = self.__dict__.setdefault("_nauty", {})
        if kind in cache and cache[kind][2] == len(self.walls):
            return cache[kind][0]
        k, G = len(self.walls), self.gram
        if kind == "gram":
            values = sorted({G[i][j] for i in range(k) for j in range(i + 1, k)} - {0})
            code = {v: c + 1 for c, v in enumerate(values)}
            b = max(1, len(values).bit_length())
        elif kind == "coxeter":
            code, b = None, 1
        else:
            raise ValueError("kind should be 'gram' or 'coxeter'")
        adj = {v: [] for v in range(b * k)}
        for i in range(k):
            for layer in range(b - 1):
                adj[i + layer * k].append(i + (layer + 1) * k)
            for j in range(i + 1, k):
                if code is not None:
                    c = code.get(G[i][j], 0)
                elif G[i][j] != 0 and G[i][j] ** 2 <= G[i][i] * G[j][j]:
                    c = 1
                else:
                    c = 0
                for layer in range(b):
                    if c >> layer & 1:
                        adj[i + layer * k].append(j + layer * k)
        g = pynauty.Graph(b * k, directed=False, adjacency_dict=adj)
        cache[kind] = (g, b, k)
        self.set_marked(g, (), kind)
        return g

    def set_marked(self, g, marked, kind: str = "gram") -> tuple:
        """Colors the vertices of the graph of nauty_graph(kind) by the layer, the norm and whether the wall is in
        marked; returns the sizes of the color classes (a part of the canonical form: nauty's certificate is the
        adjacency matrix of the canonically labeled graph)."""
        _, b, k = self.__dict__["_nauty"][kind]
        marked = set(marked)
        norms = self.norms()
        cells = []
        for layer in range(b):
            for d in sorted(set(norms)):
                for m in (True, False):
                    cell = {i + layer * k for i in range(k) if norms[i] == d and (i in marked) == m}
                    if cell:
                        cells.append(cell)
        g.set_vertex_coloring(cells)
        return tuple(len(c) for c in cells)

    def automorphisms(self, kind: str = "gram") -> Tuple[List[Tuple[int, ...]], int]:
        """Generators of the automorphism group of the graph of nauty_graph(kind) (by default Aut(G), the
        permutations s of the walls with G_{s(i) s(j)} = G_ij), as tuples of images, and its order; by nauty."""
        k = len(self.walls)
        if k <= 1:
            return [], 1
        g = self.nauty_graph(kind)
        self.set_marked(g, (), kind)
        gens, size1, size2, _, _ = pynauty.autgrp(g)
        perms = [tuple(p[:k]) for p in gens]
        if kind == "gram":
            for s in perms:
                if any(self.gram[s[i]][s[j]] != self.gram[i][j] for i in range(k) for j in range(i, k)):
                    raise RuntimeError("nauty returned a permutation not preserving the Gram matrix")
        return perms, round(size1 * 10 ** size2)

    def _basis(self) -> Tuple[List[int], fl.fmpq_mat]:
        """Indices of n linearly independent walls, and the inverse of their matrix."""
        chosen, rank = [], 0
        for i, w in enumerate(self.walls):
            if fl.fmpz_mat([list(self.walls[j]) for j in chosen + [i]]).rank() > rank:
                chosen.append(i)
                rank += 1
                if rank == self.n:
                    break
        if rank < self.n:
            raise ValueError(f"The walls span a space of rank {rank} < {self.n}: the symmetries are not determined")
        return chosen, fl.fmpq_mat(fl.fmpz_mat([list(self.walls[i]) for i in chosen])).inv()

    def matrix(self, s: Sequence[int], basis=None) -> fl.fmpq_mat:
        """The matrix M of the isometry with r_i M = r_{s(i)} (row vectors), for an automorphism s of G."""
        chosen, inv = basis if basis is not None else self._basis()
        return inv * fl.fmpq_mat(fl.fmpz_mat([list(self.walls[s[i]]) for i in chosen]))

    def symmetries(self, base: Sequence[int]) -> "ChamberSymmetry":
        """The symmetries of the polyhedron of the walls in O(L) (Sym(C) if all the walls of C are known); base is a
        vector in the future cone (e.g. the base point of the tracing), giving the time orientation."""
        perms, aut_order = self.automorphisms()
        basis = self._basis()
        A = fl.fmpq_mat(fl.fmpz_mat(self.L.A.tolist()))
        b = fl.fmpq_mat([[int(x) for x in base]])
        if (b * A * b.transpose())[0, 0] <= 0:
            raise ValueError("The base vector must have a positive square")
        k = len(self.walls)
        gens = [(s, self.matrix(s, basis)) for s in perms]
        for s, M in gens:
            if M * A * M.transpose() != A:
                raise RuntimeError("An automorphism of the Gram matrix does not give an isometry")

        def sign(M):
            return 1 if (b * M * A * b.transpose())[0, 0] > 0 else -1

        def key(M, eps):
            # The lattice spanned by the rows of M (its denominator, and the HNF of its multiple), and the
            # orientation
            d = math.lcm(*[int(M[i, j].q) for i in range(self.n) for j in range(self.n)])
            H = (fl.fmpz_mat([[int(M[i, j] * d) for j in range(self.n)] for i in range(self.n)])).hnf()
            return d, tuple(int(x) for x in H.entries()), eps

        # The orbit of (L, +) under Aut(G), with transversal elements (as permutations and matrices), and the
        # Schreier generators of the stabilizer: t_p s t_{p s}^{-1} (row action: first t_p, then s)
        identity = tuple(range(k))
        one = fl.fmpq_mat(fl.fmpz_mat([[int(i == j) for j in range(self.n)] for i in range(self.n)]))
        start = key(one, 1)
        transversal = {start: (identity, one, 1)}
        queue = [start]
        schreier = set()
        while queue:
            p = queue.pop()
            t, Mt, eps = transversal[p]
            for s, Ms in gens:
                ts = tuple(s[t[i]] for i in range(k))  # first t, then s
                Mts = Mt * Ms
                q = key(Mts, eps * sign(Ms))
                if q not in transversal:
                    transversal[q] = (ts, Mts, q[2])
                    queue.append(q)
                    continue
                u = transversal[q][0]
                u_inv = [0] * k
                for i in range(k):
                    u_inv[u[i]] = i
                g = tuple(u_inv[ts[i]] for i in range(k))
                if g != identity:
                    schreier.add(g)
        sym = []
        for g in sorted(schreier):
            M = self.matrix(g, basis)
            if any(M[i, j].q != 1 for i in range(self.n) for j in range(self.n)) or sign(M) != 1:
                raise RuntimeError("A Schreier generator is not a symmetry of the lattice")
            sym.append((g, imat([[int(M[i, j]) for j in range(self.n)] for i in range(self.n)])))
        return ChamberSymmetry(k, sym, aut_order, len(transversal))


class ChamberSymmetry:
    """The symmetries in O(L) of the polyhedron of the walls of a WallDiagram:
        generators: pairs (s, M), s a permutation of the walls (a tuple of images) and M the integral matrix of the
            isometry, r_i M = r_{s(i)} (row vectors);
        order: the order of the group; wall_orbits: the orbits of the walls;
        aut_order: the order of Aut(G) (the isometries of V permuting the walls), and index = aut_order / order
            (the size of the orbit of (L, the future cone) under Aut(G))."""

    def __init__(self, k: int, generators: List[Tuple[Tuple[int, ...], IMat]], aut_order: int, index: int):
        self.k = k
        self.generators = generators
        self.aut_order = aut_order
        self.index = index
        from sympy.combinatorics import Permutation, PermutationGroup
        perms = [Permutation(list(s)) for s, _ in generators] or [Permutation(list(range(k)))]
        self.group = PermutationGroup(perms) if k else None
        self.order = int(self.group.order()) if k else 1
        if self.order * index != aut_order:
            raise RuntimeError("The orders of the groups do not match the size of the orbit")
        self.wall_orbits = sorted((sorted(o) for o in self.group.orbits()), key=lambda o: o[0]) if k else []

    def __repr__(self) -> str:
        sizes = sorted((len(o) for o in self.wall_orbits), reverse=True)
        return f"ChamberSymmetry(order {self.order}, {len(self.generators)} generators, wall orbits {sizes})"
