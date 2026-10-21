# Edge tracing: the current implementation

This note describes how `lattice.vinberg.trace` finds the fundamental chamber of a reflective
Lorentzian lattice by following the edges of the chamber, and why the result is correct by
construction. Unlike Vinberg's algorithm (see [Vinberg.md](Vinberg.md)), the method does not
enumerate roots by their distance from a base point, and unlike the wall refinement of `RunTest`
it does not compute the faces of intermediate polyhedra with cdd: it walks along the 1-skeleton of
the chamber itself, and the cost is proportional to the size of the final chamber.

Code references:

| Part | Where |
|---|---|
| Traversal, local walls, start vertex, parallel scheduler | `src/lattice/vinberg/trace.py` (`trace_chamber`, `ChamberTracer`, `trace_ray`) |
| Roots crossing a segment | `src/lattice/vinberg/edges.py` (`edge_crossing_roots`) |
| The per-edge core in C++ (§7) | `src/cpp/edge_module.cpp` (module `edge_cpp`: `crossing_roots`, `trace_edge`, `vertex_directions`) |
| Roots through a cusp | `src/lattice/vinberg/walls.py` (`cusp_root_classes`, `cusp_walls`) |
| Conway's river of a binary lattice | `src/lattice/binary.py` (`BinLattice`) |
| Root system at a point (height 0 of a search), used for the start vertex | `src/lattice/vinberg/certify.py` (`roots_by_height`) |
| Independent certificate of a list of walls (reference) | `src/lattice/vinberg/certify.py` (`certify_polytope`) |

## 1. Conventions

As in [Vinberg.md](Vinberg.md) §1: $L$ has signature $(1, n)$, $(u, v) = u A v^T$, roots have
$(r, r) < 0$ and $|(r, r)| \leq 2 \exp(L)$, and the walls of a chamber $C$ are oriented towards it,
$C = \{x : (x, r) \geq 0\}$, so that two walls have $(r, r') \geq 0$. A **finite vertex** of $C$ is a
point $v$ with $(v, v) > 0$, an **ideal vertex** (a cusp) an isotropic $z$; both are stored as
primitive vectors of $L$, oriented to the future. An **edge** is a 1-dimensional face; it lies on
$n - 1$ walls, and its plane $\Pi$ (the orthogonal complement of these walls) has signature $(1, 1)$.

## 2. The principle

**Lemma.** No mirror of $W$ crosses the relative interior of an edge of a chamber $C$.

*Proof.* Near a point $y$ in the relative interior of an edge $e$, $C$ is the wedge of the $n - 1$
walls containing $e$, times a line along $e$. A mirror through $y$ transversal to $e$ divides this
neighbourhood, so it meets the interior of $C$, which is impossible for a mirror. $\square$

Hence an edge of $C$, followed from one of its ends, ends exactly at the **first mirror crossing its
line**, or at the ideal point of its line if no mirror crosses it before. If $C$ has finite volume, its
1-skeleton (the finite and ideal vertices and the edges) is finite and connected, so following all the
edges from one vertex finds every vertex and every edge of $C$, and every wall (each wall contains a
vertex). Every step is exact, so the result is **correct by construction**: no separate certification
(as in `certify_polytope`) is needed.

The end of an edge is determined by the first crossing mirror, but this mirror need not be a wall of $C$:
the new vertex often lies on several mirrors. The walls at the new vertex are therefore computed locally
(§5).

**Combinatorics.**

- A finite vertex $v$ lies on exactly $n$ walls: they are the simple roots of the finite root system
  $R_v$ of the roots orthogonal to $v$ (of rank $n$), for its chamber containing $C$. Its $n$ edges are
  obtained by dropping one wall; the edge without the wall $s_j$ has the direction $u_j \in v^\perp$ with
  $(u_j, s_i) = 0$ for $i \neq j$ and $(u_j, s_j) > 0$.
- At a cusp $z$, the walls form an affine Coxeter diagram with $k$ components and $n - 1 + k$ nodes (the
  walls span $z^\perp$). The edges are obtained by dropping one node from each component; the edge is the
  segment from $z$ to the other isotropic point $z'$ of its plane (or to a finite vertex before it).

## 3. Following an edge (`trace_ray`)

The edge is followed from a point $a$ (a finite vertex or a cusp) along $x(t) = a + t u$, $t > 0$. From a
finite vertex, $u \perp a$ is spacelike; from a cusp, $u = z'$ is the other isotropic vector of the plane,
so that $t \in (0, \infty)$ covers the segment $(a, z')$. Three cases, in this order:

1. **Predicted end.** For each known wall $w$ not through $a$ with $(w, a) > 0 > (w, u)$, the line meets
   $w^\perp$ at $t_w = (w, a) / |(w, u)|$. If the first such point $b$ lies in $\mathbb{H}^n$, the open
   segment $(a, b)$ is searched for crossing roots (§4). The edge ends at the first crossing, or at $b$.
   The known walls are only used to choose $b$; the search certifies the segment, so the result does not
   depend on which walls are known.
2. **Rational ideal end.** Otherwise, if the ideal end of the line is rational (an isotropic vector $z$;
   always the case from a cusp), the segment $(a, z)$ is searched; the edge ends at the first crossing,
   or at the cusp $z$. This search is finite: an isotropic binary lattice has only finitely many vectors
   of each negative norm (§4).
3. **Irrational ideal end** (then $a$ is finite and $\Pi$ is anisotropic). The edge must end at a crossing,
   at a distance not known in advance. The walk follows **Conway's river** of the binary lattice
   $\Pi \cap L$ (`_river_points`): the positive vectors on its bank, i.e. one period of the river
   (`BinLattice.river`) shifted by the powers of its automorphism (`BinLattice.shift`), ordered by their
   position $t$ along the line. They approach the ideal point; consecutive ones are close (one edge of the
   river), and their norms are small (bounded by the discriminant of $\Pi \cap L$). The overlapping
   segments $(c_{j-1}, c_{j+1})$ are searched one after the other, so that each point $c_j$ lies in the
   interior of a searched segment, until a crossing is found. The last bank vector of a period is the first
   one of the next; repeated points are skipped, since they would break the overlapping. For a chamber of
   finite volume the walk terminates; `max_steps` bounds it otherwise.

At a finite end, `trace_ray` also returns all the roots through the end with $(r, a) > 0$, from which the
new wall is computed (§5).

## 4. The roots crossing a segment (`edge_crossing_roots`)

Let $a, b$ be non-proportional vectors in the closed future light cone, $\Pi = \langle a, b \rangle$,
$N = \Pi^\perp$ (negative definite), $p = (a, a)$, $q = (b, b)$, $c = (a, b) > 0$. The mirror of a root $r$
crosses the open segment iff $(r, a)$ and $(r, b)$ have opposite signs; orient $r$ by
$A = (r, a) > 0 > B = (r, b)$.

**The plane part.** Write $r = r_\Pi + r_N$. The products with $\Pi$ only depend on $r_\Pi$, which is
determined by $(A, B)$. With the crossing point $x = |B| a + A b$,
$$(r_\Pi, r_\Pi) = -\frac{(x, x)}{c^2 - p q}, \qquad -2 \exp(L) \leq (r, r) \leq (r_\Pi, r_\Pi) < 0,$$
since $(r_N, r_N) \leq 0$. So the pairs $(A, B)$ of the roots crossing the segment are the points of the
lattice $\Lambda = \{((y, a), (y, b)) : y \in L\} \subset \mathbb{Z}^2$ (computed by an HNF, with lifts to
$L$) in the region $A > 0 > B$, $0 < (x, x) \leq 2 \exp(L) (c^2 - p q)$. This region is bounded: by an
ellipse if $p, q > 0$; if $q = 0$, then $B^2 p \leq 2 \exp c^2$ and $A \leq \exp c / |B|$; if $p = q = 0$,
then $A |B| \leq \exp c$. The enumeration runs over $B$ (multiples of the first HNF entry) and, for each,
over the arithmetic progression of admissible $A$.

**Promotion** (`_promote`). For each $(A, B)$, take $y \in L$ with these products; the roots with the same
plane part are $r = y + z$, $z \in L \cap N$, and
$-(y_N + z, y_N + z) = (r_\Pi, r_\Pi) - (r, r) \leq 2 \exp(L) + (r_\Pi, r_\Pi)$. This is a Fincke–Pohst
search in an LLL-reduced basis of $L \cap N$, centered at $-y_N$. The lift $y$ is first shifted by the
vector of $L \cap N$ nearest to its projection, so that the center lies in $[-1/2, 1/2]^{n-1}$ (a large
center would overflow `FPSearch` and lose precision). The bounds keep a margin of $0.5$ from the
attainable values, and the candidates are checked exactly (`_root_mask`). Every root found has
$(r, a) = A$ and $(r, b) = B$ by construction.

**Roots through the end** (`include_end=True`, $b$ finite). If no root crosses the open segment, the roots
through $b$ with the smallest $(r, a) > 0$ are returned instead: they have $B = 0$ and
$(x, x) = A^2 q \leq 2 \exp(L) (c^2 - p q)$, and $A$ is scanned upwards until roots are found.

**Cost.** The number of candidates $(A, B)$ grows like $\sqrt{c^2 - p q}$, i.e. with the hyperbolic length of
the segment and with the size of the vectors $a, b$ (it is not invariant under scaling of the lattice
vectors used as endpoints). This is why the descent (§5) uses short directions, and why the walk to an
irrational end uses the bank vectors of the river.

## 5. Walls at the vertices

**A new finite vertex** $v'$, reached along an edge with walls $W$ from $a$ (`_next_walls`). The walls at $v'$
are $W$ and one more simple root $\alpha$ of $R_{v'}$. The chamber of $R_{v'}$ containing $C$ contains
$\mathrm{dir} = K u' + y$ for $K$ large, where $u' = (v', v') a - (a, v') v'$ points back along the edge and
$y \in \operatorname{span}(W)$ has $(y, w) = 1$ for $w \in W$; the roots orthogonal to $u'$ are orthogonal to
the edge, hence in $\operatorname{span}(W)$. For a root $r$ through $v'$,
$(\mathrm{dir}, r) = K (v', v') (r, a) + (y, r)$, so $\alpha$, the first positive root independent of $W$ in
this ordering, is the root through $v'$ with the smallest $(r, a) > 0$, then the smallest $(y, r)$. All these
roots are returned by the edge search (the roots crossing at the end point, or the roots through the end
with `include_end`), so $R_{v'}$ itself is not computed. As a check, $\alpha$ must have non-negative products
with $W$.

**A new cusp** $z$, reached along an edge with walls $F$ (`_cusp_local_walls`). The walls at $z$ are the walls
of the chamber of the affine reflection group $W_z$ (the roots through $z$) containing $C$. They are found by
Vinberg's acceptance criterion with a base point $m$ on the incoming edge ($m = a$ if $a$ is finite, else
$a + z$) and $F$ as the walls at the base point: the mirrors of $W_z$ through $m$ are those orthogonal to the
edge, whose simple roots are $F$. The candidates are, for each class of parallel roots through $z$
(`cusp_root_classes`: the lifts $w + t\,\mathrm{step}$ of a vector of $z^\perp / z$), the one closest to $m$ on
its positive side (`cusp_walls`); the others are parallel to it with the same orientation, so their product
with it is $(w, w) < 0$ and they would be rejected (the rejection is valid by [Vinberg.md](Vinberg.md) §5).
In a class whose vector is primitive in $z^\perp / z$, all the lifts are roots, so no root test is needed.
The accepted walls must span $z^\perp$; otherwise the chamber has infinite volume at $z$.

**The start vertex** (`_descend`). From a base point $p$ (the base point chosen by `Vinberg` by default), with
the walls $F$ at $p$ (the simple roots of $R_p$ for the chamber containing a chamber vector), follow a
generic short direction in the face through $p$ (orthogonal to $p$ and $F$, a combination with small
coefficients of an LLL-reduced basis) to the first mirror. The new point lies on a face of higher
codimension; its walls are the simple roots of its root system for the chamber containing
$K u' + y$ (`_local_walls`, the general form of the rule above). Repeat until a vertex (finite, or a cusp) is
reached. A point with a root system of full rank, such as $e_0$ for $I_{1,n}$, is already a vertex.

## 6. The traversal and parallelism (`ChamberTracer.run`)

The main process keeps the registries (walls, vertices keyed by their primitive vectors, edges keyed by the
frozensets of their walls) and a queue of edges to trace. The work is done by tasks:

| Task | Input | Output |
|---|---|---|
| edge | start $a$, direction $u$, the reduced basis of $L \cap a^\perp$ (for a finite $a$), the walls of the edge, the known walls | the end, whether it is ideal, the new wall $\alpha$ at a finite end |
| vertex | a new finite vertex and its walls | the reduced basis of $L \cap v^\perp$ and the directions of its edges (one matrix inverse) |
| cusp | a new cusp, the walls of the incoming edge, a point $m$ on it | the walls at the cusp and the directions of its edges |

The complement $L \cap N$ of an edge, needed by the edge search, is the kernel of $y \mapsto (y, u)$ on the
reduced basis of $L \cap a^\perp$, computed in the edge task. An edge is submitted once, from the first end
that lists it; a vertex or cusp reached by several edges is computed once.

With `threads > 1` the tasks run in a `ProcessPoolExecutor` (at most $4 \times$ `threads` tasks in flight);
with `threads = 1` the same loop runs them in the main process. The order in which the tasks complete does
not change the result: the known walls given to an edge task only predict its end, which its own search
certifies.

**Consistency checks.** At the end, every edge listed at a vertex has been traced and ends at that vertex,
and every edge has two distinct ends whose walls contain the walls of the edge; finite vertices have $n$ walls,
and the walls at a new vertex contain those of the incoming edge. `max_vertices` stops the traversal of a
lattice which is not reflective (or whose chamber is too large).

## 7. The C++ core (`edge_cpp`)

The work on an edge is done in C++ when possible; the Python implementation remains the reference
(`ChamberTracer(cpp=False)`, `edge_crossing_roots(use_cpp=False)`) and the fallback.

- `crossing_roots(A, exp, a, b, include_end)`: the search of §4. The complement of the plane is the kernel of
  $(\cdot, a)$ on $\mathbb{Z}^{n+1}$, then of $(\cdot, b)$ (unimodular transformations from extended gcds), reduced
  by LLL; $\Lambda$ and the lifts come from a 2-dimensional HNF; the promotion searches use `FPSearch`
  (`fp_search.hpp`) with the center computed from the Cholesky factor of the complement; the root test is exact.
- `trace_edge(A, exp, a, u, walls, edge_walls)`: cases 1 and 2 of §3 and the new wall of §5. It returns
  `(0, end, wall)` for a finite end, `(1, cusp, None)` for a rational ideal end, and `(2, None, None)` for an
  irrational one, for which Python walks along the river (its segments are searched by `crossing_roots`). The new
  wall is chosen exactly: $y$ is replaced by $\det(G) \, G^{-1} \mathbf{1}$, $G$ the Gram matrix of the walls of
  the edge, by fraction-free elimination.
- `vertex_directions(A, v, S)`: the directions of the edges at a finite vertex, each the generator of the kernel
  of $(\cdot, v)$ and $(\cdot, s_i)$, $i \neq j$.

Integers are 64-bit, with 128-bit intermediate values; every operation that might overflow is checked
(`__builtin_*_overflow`) and raises `OverflowError`, and then the Python implementation, with arbitrary precision,
is used. The only floating-point steps are the LLL reduction of the complement (the row operations are exact and
unimodular, so only the quality of the reduction depends on the rounding; the integer Gram matrix is updated
exactly, and the Gram-Schmidt data are recomputed after each swap) and the Fincke-Pohst searches (bounds with a
margin of 0.5, and the candidates checked exactly). The cusps (§5) are still handled in Python.

## 8. Validation and performance

- $I_{1,n}$, $n \leq 14$: the same walls as `RunTest`, and the same numbers of vertices, cusps and edges as cdd
  (`tests/test_trace.py`).
- All 8595 lattices of Allcock's list (`experiments/allcock.py`): the same numbers of walls and the same
  cyclic sequences of norms and angles as Allcock's simple roots; 155 s in total with the C++ core (780 s in
  Python), at most 1.8 s per lattice.
- $I_{1,19}$: 50 walls, 27260 finite vertices, 581 cusps, 292340 edges, the same as `RunTest` + cdd +
  `certify_polytope`; about 20 minutes in one process before the optimizations below, 95 s with 16 worker
  processes after them.
- $I_{1,14}$ (89 vertices, 5 cusps, 704 edges): 1.1 s in one process in Python, 0.2 s with the C++ core.
- `tests/test_edge_cpp.py` compares the C++ core with the Python implementation on random segments, on the edges
  of chambers (with and without known walls), and on complete traversals; it also checks the fallback to Python
  beyond 64 bits.

The optimizations that matter: the new wall at a finite vertex from the roots found by the edge search
(instead of the root system of the vertex); the reduced complements of the vertices, from which the edge
directions and complements follow; vectorized products and root tests in the edge search and at the cusps;
Python integers rather than FLINT `fmpz` in numpy arrays (several times faster); and the walk along the river
for irrational ends (Allcock's 360.6: 34 s before, 0.2 s after); and the C++ core (§7), a factor of about 5.

## 9. Open points

- **Symmetry.** Large symmetric chambers (the 210 walls of $U \oplus D_{20}$, Borcherds' 17-dimensional example
  with 960 walls) may have too many edges for a full traversal. A group of isometries of $L$ preserving $C$,
  known in advance or computed from the final walls, would allow tracing one vertex and edge per orbit.
- **Cusps in C++.** With the C++ core, the walls at the cusps (`cusp_root_classes`, the acceptance, the edges at
  a cusp) are the largest remaining part in Python.
- **Coxeter diagrams.** Vertices and edges could be identified by their sets of walls alone (elliptic and
  parabolic subdiagrams), and the diagram gives free consistency checks (each edge has two ends, the f-vector,
  the Euler characteristic).
