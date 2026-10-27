# Lattice

A Python/C++ toolkit for computations with **integral lattices** (non-degenerate integral
symmetric bilinear forms), centred on the fundamental chamber `C` of the reflection group of a
hyperbolic (Lorentzian) lattice `L` of signature `(1, n)`: **Vinberg's algorithm**, the tracing
of the edges of `C`, and verifiable **certificates** that `L` is reflective (`C` has finitely many
walls) or not, with the structure of the symmetry group `Sym(C) = O⁺(L)/W(L)` of the chamber.

Lattice arithmetic is exact (NumPy `object` arrays of Python ints and
[python-flint](https://github.com/flintlib/python-flint)). Floating point is only used to drive
enumeration (Fincke–Pohst) and rendering, and candidate vectors are re-checked exactly.

Among the results: the chambers of the Bogachev–Perepechko list of reflective lattices, of
`U + D20(-1)` (210 walls) and of Borcherds' 18-dimensional lattice (960 walls) are certified by
orbits of vertices; `II_{1,25}` is certified non-reflective with `Sym(C)` fixing Conway's Weyl
vector (307 orbits of vertices: 284 finite ones and 23 cusps).

## Features

- **General lattices** (`lattice.lattice`): Gram-matrix validation, signature, parity,
  discriminant group and exponent; orthogonal sums (`L + M`), multiples (`3 * L`), rescaling
  (`L(-1)`); LLL reduction (definite, and indefinite via a positive-definite majorant); roots;
  saturation, index, orthogonal complements, fibre products, subquotients; even sublattice;
  square-free and strongly square-free associated lattices.
- **Binary forms** (`lattice.binary`): reduction and canonical forms of rank-2 lattices of every
  signature via Conway's topograph ("rivers" for indefinite forms), isomorphism testing,
  representation of zero, enumeration of represented values and roots, the automorphism
  ("shift") of the river, enumeration of binary lattices up to a discriminant bound.
- **Genus** (`lattice.genus`): p-adic and 2-adic Conway–Sloane symbols, string output in
  Conway–Sloane notation, and genus equality via the 2-adic oddity/sign-walking rules
  (Conway–Sloane, ch. 15, §7, Thm. 10).
- **Discriminant forms** (`lattice.discform`): the discriminant form, elements `u` with
  `b(u, u)` integral, the maximal such subgroups (a Bron–Kerbosch-style search over subgroups;
  also as an iterator, `iter_max_isospaces`) and the corresponding integral overlattices, with
  the embedding of `L` into them.
- **Standard lattices** (`lattice.catalogue`): `A_n, B_n, C_n, D_n, E_6..8`, `U(n)`, `I_{p,q}`,
  `II_{n,1}`, two constructions of the Leech lattice.
- **Root systems and polyhedra** (`lattice.roots`): Coxeter graphs, simple roots, decomposition
  of definite lattices into irreducibles, exact extremal rays of polyhedral cones (`pycddlib`).
- **Vector enumeration** (`lattice.enumeration`): ordered enumeration of `Z^n`, Fincke–Pohst
  search for vectors with `lbound < (x+b)^T A (x+b) <= ubound`. `FPSearch` is the C++
  implementation when it is built and the pure-Python `FPSearchPy` otherwise.
- **Vinberg's algorithm** (`lattice.vinberg`): root search by height, reduction to the
  fundamental chamber of the finite Weyl group at the base point, maintenance of the active
  walls, and a finite-volume test via the extremal rays of the polyhedral cone (`Vinberg`).
  `cusp_walls`, `geodesic_walls` and `RunTest` complete a partial set of walls by examining the
  non-positive rays of the current polyhedron. `RootSys`/`VSearch` are pure-Python reference
  implementations of the C++ search.
- **Edge tracing** (`lattice.vinberg.trace`, `edges`; see `docs/EdgeTracing.md`): `trace_chamber`
  finds the walls of `C` by following its edges from vertex to vertex (finite vertices and cusps),
  in parallel processes, with the per-edge core in C++; the walls found so far can be saved to a
  file. It also finds infinite geodesics in `C` (a certificate of non-reflectivity).
- **Coxeter diagrams and symmetries** (`lattice.vinberg.coxeter`): the diagram of the walls, its
  symmetries in `O(L)` (with nauty), and the types of local diagrams (`A_n`, ..., `E~_8`).
- **Certificates**, each written as JSON with an independent verifier:
  - `trace_chamber_by_orbits` (`orbits`): `L` is reflective — the walls of `C`, the generators of
    `Sym(C)`, and one vertex per orbit, whose edges are traced again by the verifier;
  - `search_nonreflective` (`nonreflective`): a symmetry of `C` of infinite order (a product of
    identifications of vertices), or a geodesic in `C` with irrational ends;
  - `quotient_chamber` (`quotient`): the vertices of `C` up to `Sym(C)` (finite vertices and cusps
    identified by canonical forms of their glue graphs), and the structure of `Sym(C)`: finite
    (`L` reflective), parabolic (a fixed cusp, the heights of the walls, a Weyl vector), virtually
    cyclic (the axis of a hyperbolic element) or non-elementary.
  - `vinberg.report.certificate_report` renders any certificate as a readable PDF: the lattice (Gram
    matrix, discriminant form, genus), the claim, the walls with the Coxeter diagram, the
    symmetries, the vertices with their local diagrams, the structure of `Sym(C)`.
- **Rendering** (`lattice.draw`): Coxeter diagrams as TikZ/PDF, SVG and interactive HTML (with a
  matrix view for large diagrams), and in 3D (WebGL); circle/line arrangements in the Poincaré
  disc exported to TikZ/PDF (fundamental polygons, tilings of rank-3 lattices).

## C++ extensions

Three pybind11 modules, built automatically by `uv sync` (scikit-build-core + CMake):

| Module                  | Source                                        | Contents |
|-------------------------|-----------------------------------------------|----------|
| `lattice.fp_search_cpp` | `src/cpp/fp_search.hpp`, `fp_search_module.cpp` | `FPSearch`: Fincke–Pohst enumeration (Eigen, double precision), with `batch_search`, `search_all`, `exhausted`. |
| `lattice.vsearch_cpp`   | `src/cpp/vsearch_module.cpp`                  | `RootSysCpp`: finite root systems, reflections into the positive Weyl chamber. `VSearchCpp`: multithreaded root search for Vinberg's algorithm, with `run`, `update_walls`, `get_walls`, `get_vecs`. Uses 64-bit integers (`int_class = long`); a GMP `mpz_class` variant is prepared in the source. |
| `lattice.edge_cpp`      | `src/cpp/edge_module.cpp`                     | The per-edge core of edge tracing: `crossing_roots`, `edge_direction`, `trace_edge`, `cusp_walls`. 64-bit integers with 128-bit intermediates; on a possible overflow the Python implementation is used. |

Without the extensions, everything except the root search of Vinberg's algorithm (`Vinberg`,
`certify.iter_roots`, `RenderTiling`) still works: Fincke–Pohst and edge tracing fall back to
Python.

## Setup

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), a C++17 compiler, CMake ≥ 3.18,
GMP with C++ bindings (`libgmp-dev` on Debian/Ubuntu) and, optionally, `pkg-config` (used to locate
GMP if present) and Eigen ≥ 3.4 (`libeigen3-dev`; if it is not found, CMake downloads Eigen 3.4.0).
On Debian/Ubuntu: `sudo apt install build-essential libgmp-dev` (plus `pkg-config libeigen3-dev` optionally).
The symmetries of chambers, the orbit and quotient certificates need
[pynauty](https://github.com/pdobsan/pynauty) (the optional dependency `lattice[symmetry]`, installed
with the dev group), and the PDF output needs `pdflatex` (TeX Live).

```bash
uv sync          # installs dependencies and builds/installs the package in editable mode
uv run pytest    # runs the test suite
```

Python changes are picked up immediately. Changes to `src/cpp/*`, `CMakeLists.txt` or
`pyproject.toml` trigger a rebuild on the next `uv sync` / `uv run`.

## Example

```python
from lattice import U_lat, E_lat, I_lat, Genus, Vinberg

L = U_lat() + E_lat(8)(-1)
print(L.info())                     # Even lattice of signature (1, 9), discriminant 1 ...
print(Genus(L).str())               # II_{1, 9}

L = I_lat(1, 4)                     # diag(1, -1, -1, -1, -1)
V = Vinberg(L, base=[1, 0, 0, 0, 0])
walls = V.run(iterations=5)         # "The fundamental domain has finite volume"
for w in walls:
    print(w, L.square(w))           # the 5 simple roots
```

Edge tracing, certificates and reports:

```python
from lattice import U_lat, A_lat, Lattice
from lattice.vinberg import trace_chamber
from lattice.vinberg.orbits import trace_chamber_by_orbits, verify_certificate
from lattice.vinberg.quotient import quotient_chamber, verify_quotient
from lattice.vinberg.report import certificate_report

L = U_lat() + A_lat(2)(-1) * 3
trace_chamber(L, walls_file="walls.json", threads=4)       # the 10 walls of the chamber
trace_chamber_by_orbits(L, "walls.json", certificate="cert.json")
verify_certificate("cert.json")                            # checked independently
certificate_report("cert.json")                            # cert.pdf

L = Lattice(1, [[7]]) + A_lat(3)(-1)                       # <7> + A_3(-1): not reflective
res = quotient_chamber(L, certificate="quotient.json")
print(res["structure"], res["salem polynomial"])           # virtually cyclic [1, -11, -8, -11, 1]
verify_quotient("quotient.json")
```

## Repository layout

| Path | Purpose |
|------|---------|
| `src/lattice/` | The Python package (see Features). `_matrix.py` holds the integer-matrix helpers (`imat`, FLINT conversions, `euclid`, ...). |
| `src/cpp/` | C++ sources of the extensions. |
| `CMakeLists.txt` | Build of the extensions (finds pybind11, Eigen, GMP). |
| `tests/` | pytest suite: a light set of regression checks against known results (`I_{1,n}`, `II_{1,9}` = E10, `E_n`, `D_n`, Leech, genus invariance, edge tracing, certificates and their tampering, ...). |
| `experiments/` | Research scripts: `allcock.py` (Allcock's rank-3 list), `reflective_tests.py` (certification of known reflective lattices: diagonal lattices, Apollonian, Borcherds, Vinberg, Bogachev–Perepechko, and `II_{1,25}`), `render_examples.py` (Apollonian packings, tilings), `run.py` (scratchpad). `reflective_tests.py` writes to `../output/Reflective/` (run it from `experiments/`), `render_examples.py` to `output/` (run it from the repository root); `allcock.py` finds its paths itself. |
| `notebooks/run.ipynb` | Working notebook. |
| `data/` | Local data for the experiments (git-ignored, see `data/README.md`). |
| `docs/Vinberg.md` | Vinberg's algorithm as implemented: the adapted basis, the root search by height, the selection of the walls. |
| `docs/EdgeTracing.md` | Edge tracing, the C++ core, the orbit certification, non-reflectivity, the chamber modulo its symmetries, the reports. |
| `docs/bibliography.md` | Annotated literature list: Vinberg, Nikulin, Borcherds, Conway, Allcock, K3 surfaces, classification of reflective lattices. |

Rendered output goes to `output/` by default (git-ignored).

## References

See [docs/bibliography.md](docs/bibliography.md). Key sources: E. B. Vinberg, *Hyperbolic
reflection groups* (1985); J. H. Conway and N. J. A. Sloane, *Sphere Packings, Lattices and
Groups*, ch. 15; V. V. Nikulin, *Integral symmetric bilinear forms and some of their geometric
applications* (1979); D. Allcock, *The reflective Lorentzian lattices of rank 3* (Mem. AMS 2012;
arXiv 2010) and *An alternative to Vinberg's algorithm* (2021).

## License

MIT — see [LICENSE](LICENSE).
