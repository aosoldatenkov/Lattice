# Data

Data files used by the scripts in `experiments/` and `notebooks/`. Only Allcock's list `allcock` is tracked, with this README. The data set `allcock.jsonl` (about 12 MB) and its metadata are rebuilt from the list by `python experiments/allcock.py build` (about 5 minutes); the run results and the older files are local.

The lattices come from the preprint of D. Allcock, *The reflective Lorentzian lattices of rank 3* (arXiv 2010; Mem. AMS 2012), which lists 8595 reflective lattices. The TeX source `docs/rk3monolithic.tex` is also a Perl script that prints the list: `perl docs/rk3monolithic.tex all > data/allcock`.

| File | Contents |
|------|----------|
| `allcock` | The list as printed by the Perl script: one lattice per line, `[IP, EDs, roots, W, L]` in PARI syntax (see below). |
| `allcock.jsonl` | The data set: one JSON record per lattice, written by `python experiments/allcock.py build`. |
| `allcock.jsonl.meta.json` | How the data set was built: source file and its SHA-256, date, git commit, failures. |
| `allcock_runs/*.jsonl` | Results of Vinberg's algorithm, one JSON record per lattice and run, written by `python experiments/allcock.py run`. |
| `in`, `out` | Older outputs (Gram matrices after LLL; `[Gram matrix, number of walls, W, L]`), superseded by `allcock.jsonl`. |

JSON is used because the entries of Allcock's Gram matrices reach 10^21, beyond 64-bit integers; Python's `json` reads and writes integers of any size. JSON lines can be appended to, read one record at a time, and inspected with `grep` or `jq`.

## Allcock's format

`[IP, EDs, roots, W, L]`: `IP` is the inner product matrix, of signature (2, 1); `EDs = [a, b, c]` are the elementary divisors (c | b | a), the dual lattice being spanned by (1/a, 0, 0), (0, 1/b, 0), (0, 0, 1/c); the columns of `roots` are the simple roots; `W` is the number of the Weyl group and `L` the number of the lattice among those with that Weyl group (the lattice L_{W.L} of the paper).

## The data set `allcock.jsonl`

Records are in the order of Allcock's list. The library's convention is signature (1, 2), so the Gram matrices of the forms below are the negatives of Allcock's (in particular, the roots have negative squares).

| Field | Contents |
|-------|----------|
| `id` | `"W.L"`, unique |
| `index` | the line number in `allcock` (1-based) |
| `weyl`, `number`, `nwalls` | W, L, and the number of simple roots |
| `allcock` | the original data: `gram` (Allcock's, signature (2, 1)), `eds`, `roots` (as rows), `weyl`, `number` |
| `chamber` | the cyclic sequence `[n_0, m_01, n_1, m_12, ...]` of the norms n_i of the simple roots and the angles π/m between consecutive walls (m = 0 for parallel walls), in a canonical form up to rotations and reflections of the polygon |
| `invariants` | `det`, `dgroup`, `parity`, `genus` (Conway-Sloane symbol, not canonical at 2) |
| `forms` | the lattice in three bases: `allcock` (Allcock's basis), `lll` (after repeated LLL reduction), `reduced` (a Lorentzian basis: e_1, e_2 span the complement of a positive vector; equal to `lll` if none was found) |

Each form has `gram` (the Gram matrix T (−A) T^T), `basis` (the matrix T: its rows are the new basis vectors in Allcock's coordinates), `roots` (Allcock's simple roots in the new coordinates, i.e. roots · T^{−1}) and `size` (the largest |entry| of `gram`); `lll` also has `rounds`, and `reduced` has `lorentzian`. All of this is verified exactly when the data set is built.

## Vinberg runs `allcock_runs/*.jsonl`

One file per run (`<form>-<timestamp>.jsonl` by default), appended line by line, so an interrupted run loses nothing and can be continued (`--out` with the same file skips the lattices already done). A record has `id`, `index`, `form`, `params`, `timestamp`, `commit`, `seconds`, and `status`:

| Status | Meaning |
|--------|---------|
| `ok` | finite volume, and the walls have the same `chamber` as Allcock's simple roots |
| `mismatch` | finite volume, but a different number of walls or a different `chamber` |
| `unfinished` | the iterations ran out (or, with `--certify`, the walls were not certified) |
| `overflow` | the search raised `OverflowError` (64-bit bounds; typically for the `allcock` form) |
| `error` | any other exception, in `error` |

When the run finishes, the record also has `walls` (in the coordinates of the form), `walls_allcock` (in Allcock's coordinates), `nwalls`, `chamber`, `matches`, `certified`, `searched_height`, `base` (the base point, in the coordinates of the form), `base_norm` (its square) and `base_walls` (the number of walls through it). The walls generally bound a different chamber than Allcock's roots (one conjugate under the Weyl group), so they are compared through `chamber`, not directly.

In Python: `load_dataset()`, `select(...)`, `run_vinberg(...)`, `load_results()`, `latest(...)` and `summary(...)` in `experiments/allcock.py`.
