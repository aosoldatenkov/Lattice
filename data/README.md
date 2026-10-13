# Data

Local data files used by the scripts in `experiments/` and `notebooks/`. They are deliberately not tracked; only this README is.

The lattices come from the preprint of D. Allcock, *The reflective Lorentzian lattices of rank 3* (arXiv 2010; Mem. AMS 2012), which lists 8595 reflective lattices. The files here are that list after two stages of LLL reduction.

| File | Contents |
|------|----------|
| `allcock` | The list as taken from the preprint, one lattice per line, beginning with its Gram matrix. |
| `in` | Gram matrices, one per line. |
| `out` | `[Gram matrix, number of walls, Weyl group no., lattice no.]`, as written by `experiments/allcock.py: Allcock_list`. Read by `CheckAllcock` and `Render_Allcock_chambers`. |
