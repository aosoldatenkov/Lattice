#!/usr/bin/env python3

import random as rnd
import time
import math
import os
import subprocess
from itertools import product
import numpy as np
from lattice import *  # scratch script: the whole public API
from lattice import fp_search_cpp, vsearch_cpp
from lattice.vinberg import certify_polytope, trace_chamber
from lattice.vinberg.orbits import trace_chamber_by_orbits
from lattice.vinberg.quotient import quotient_chamber
from lattice.vinberg.nonreflective import search_nonreflective
from lattice.draw import RenderTiling, CoxeterDiagram, draw_Coxeter_diagram, draw_Coxeter_diagram_3d
from lattice.vinberg.trace import load_walls
from reflective_tests import *
from lattice._matrix import *


def TestReflections():
    L = E_lat(8) + E_lat(8)
    print(L.info())
    print(L.A)
    roots = []
    fps = fp_search_cpp.FPSearch(np.array(L.A.tolist(), dtype=float), np.zeros(L.rank, dtype=float), 0, 2.5)
    for v in fps.search_all():
        if L.is_root(v):
            roots.append(v)
    print(len(roots))
    R = vsearch_cpp.RootSysCpp(L.A.tolist(), roots)
    print(R.sroots)
    start = time.perf_counter()
    count = 0
    while True:
        v = [rnd.randint(1, 10 ** 6) for _ in range(L.rank)]
        # r = fl.fmpz_mat(R.find_reflection(v))
        # vr = fl.fmpz_mat(1, L.rank, v) * r
        # chamber = R.closed_chamber(vr.tolist()[0])
        vr = R.reflect(v)
        chamber = R.closed_chamber(vr)
        if not all(x > 0 for x in chamber):
            print("Error")
            break
        count += 1
        print(f"Speed: {count / (time.perf_counter() - start):10.2f} vecs/sec", end='\r')

# CertApollonian(10)
# res = load_walls("../output/Reflective/Apollonian10_walls.json")
# L = Lattice(len(res['gram']), imat(res['gram']))
# D = draw_Coxeter_diagram_3d(L, res["walls"], "../output/A.html", method="stress")

L = Lattice(3, imat_diag([3, -4, -7]))
res = quotient_chamber(L, base=[1, 0, 0], progress=True, threads=24)
geodesic = res['axis']['lattice']
v, d = L.dual_vec(L.complement(geodesic)[0])
basis = concat_rows(geodesic, v)
M = Lattice(3, L.batch_prod(basis, basis))
print(M.A)
tikzcode = RenderTiling(M, [1, 0, 0])
lines = []
lines.append("\\documentclass[tikz]{standalone}\n")
lines.append("\\begin{document}\n")
lines.extend(tikzcode)
lines.append("\\end{document}\n")
fname = "tiling_diag(3,-4,-7).tex"
with open(os.path.join("../output", fname), "w", encoding="utf-8") as file_out:
    file_out.writelines(lines)
result = subprocess.run(["pdflatex", "\\nonstopmode\\input", fname], cwd="../output",
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.PIPE)
if result.returncode != 0:
    raise RuntimeError(f"PDFLaTeX compilation failed: {result.stderr.decode('utf-8')}")

# count = 0
# upper = 7
# lattices = []
# for diag in product(range(1, upper), range(-upper, 0), range(-upper, 0)):
#     if math.gcd(*diag) != 1:
#         continue
#     L = Lattice(3, imat_diag(diag))
#     res = quotient_chamber(L, base=[1, 0, 0], threads=24)
#     if not res['reflective'] and res['structure'] == 'virtually cyclic':
#         count += 1
#         print(count, end='\r')
#         print(L.A)
#         lattices.append(L)


# fname = "../output/" + symbol
# res = trace_chamber(L,
#                     symmetry=True,
#                     progress=True,
#                     threads=24,
#                     walls_file=fname + "_walls.json")
# trace_chamber_by_orbits(L, fname + "_walls.json",
#                         certificate = fname + ".json",
#                         threads=24,
#                         progress=True)

# compl = [[[2, 0], [0, 6]], [[4, 2], [2, 4]]]
# for b in compl:
#     C = Lattice(2, b)
#     print(C.info())
#     D = DiscForm(L + C)
#     print(D.Ared)
#     iso = D.list_max_isospaces()
#     for s in iso:
#         M = D.overlattice(s)
#         if M.parity == 0:
#             print(M.info())
#             DD = DiscForm(M)
#             print(DD.iso)
#             print(DD.Ared)

# print(L.info())
# print(L.A)
# V = Vinberg(L, h_batch=1)
# V.print_info()
# walls = V.run(root_batch=1, use_reflections=False)
# for w in walls:
#     print(w, L.square(w))
# end = time.perf_counter()
# print("Total execution time: " + str(datetime.timedelta(seconds=(end - start))))

# M = Leech_lat_alt()
# A = M.lll()
# L = Lattice(24, A)
# FPS = fp_search_cpp.FPSearch(np.array(M.A.tolist(), dtype=float), np.zeros(M.rank, dtype=float), 0, 4.5)
# vecs = FPS.search_all()
# print(len(vecs))