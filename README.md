# Lattice

A Python/C++ toolkit for computations with **integral lattices** (non-degenerate integral
symmetric bilinear forms), centred on **Vinberg's algorithm** for computing the fundamental
polyhedron of the reflection group of a hyperbolic (Lorentzian) lattice of signature `(1, n)`.

Lattice arithmetic is exact (NumPy `object` arrays of Python ints and
[python-flint](https://github.com/flintlib/python-flint)). Floating point is only used to drive
enumeration (Fincke–Pohst) and rendering, and candidate vectors are re-checked exactly.

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
  `b(u, u)` integral, maximal such subgroups and the corresponding integral overlattices.
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
- **Rendering** (`lattice.draw`): circle/line arrangements in the Poincaré disc exported to
  TikZ/PDF (fundamental polygons, tilings of rank-3 lattices).

## C++ extensions

Two pybind11 modules, built automatically by `uv sync` (scikit-build-core + CMake):

| Module                  | Source                                        | Contents |
|-------------------------|-----------------------------------------------|----------|
| `lattice.fp_search_cpp` | `src/cpp/fp_search.hpp`, `fp_search_module.cpp` | `FPSearch`: Fincke–Pohst enumeration (Eigen, double precision), with `batch_search`, `search_all`, `exhausted`. |
| `lattice.vsearch_cpp`   | `src/cpp/vsearch_module.cpp`                  | `RootSysCpp`: finite root systems, reflections into the positive Weyl chamber. `VSearchCpp`: multithreaded root search for Vinberg's algorithm, with `run`, `update_walls`, `get_walls`, `get_vecs`. Uses 64-bit integers (`int_class = long`); a GMP `mpz_class` variant is prepared in the source. |

Without the extensions, everything except `Vinberg` and `RenderTiling` still works (Fincke–Pohst
falls back to Python).

## Setup

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), a C++17 compiler, CMake ≥ 3.18,
`pkg-config`, GMP with C++ bindings (`libgmp-dev` on Debian/Ubuntu) and, optionally, Eigen ≥ 3.4
(`libeigen3-dev`; if it is not found, CMake downloads Eigen 3.4.0).

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
print(Genus(L).str())               # II_{1, 9}()

L = I_lat(1, 4)                     # diag(1, -1, -1, -1, -1)
V = Vinberg(L, base=[1, 0, 0, 0, 0])
walls = V.run(iterations=5)         # "The fundamental domain has finite volume"
for w in walls:
    print(w, L.square(w))           # the 5 simple roots
```

## Repository layout

| Path | Purpose |
|------|---------|
| `src/lattice/` | The Python package (see Features). `_matrix.py` holds the integer-matrix helpers (`imat`, FLINT conversions, `euclid`, ...). |
| `src/cpp/` | C++ sources of the extensions. |
| `CMakeLists.txt` | Build of the extensions (finds pybind11, Eigen, GMP). |
| `tests/` | pytest suite: a light set of regression checks against known results (`I_{1,n}`, `II_{1,9}` = E10, `E_n`, `D_n`, Leech, genus invariance, ...). |
| `experiments/` | Research scripts: `allcock.py` (Allcock's rank-3 list), `reflective_tests.py`, `render_examples.py` (Apollonian packings, tilings), `run.py` (scratchpad). Run them from the repository root. |
| `notebooks/run.ipynb` | Working notebook. |
| `data/` | Local data for the experiments (git-ignored, see `data/README.md`). |
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
