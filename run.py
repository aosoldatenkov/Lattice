#!/usr/bin/env python3

import flint as fl
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import time
from BinLattice import *
from Lattice import *
from LatticeUtils import *
from DiscForm import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *
from Allcock import *
import fp_search_cpp
import vsearch_cpp


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

L = I_lat(1, 21)
basis = L.even_sublattice()
L = Lattice(22, L.batch_prod(basis, basis))
print(L.info())

compl = [[[2, 0], [0, 6]], [[4, 2], [2, 4]]]
for b in compl:
    C = Lattice(2, b)
    print(C.info())
    D = DiscForm(L + C)
    iso = D._list_max_isospaces()
    for s in iso:
        gens = [D.iso[i] for i in s] + D.D.tolist()
        basis = Lattice.image(gens)
        fl_bas = fl.fmpz_mat(basis)
        A, denom = (fl_bas * D.A * fl_bas.transpose()).numer_denom()
        M = Lattice(24, A.tolist())
        if M.parity == 0:
            print(M.info())
            DD = DiscForm(M)
            print(DD.iso)
            print(DD.A)

# print(L.info())
# print(L.A)
# V = Vinberg(L, h_batch=1)
# V.print_info()
# walls = V.run(root_batch=1, use_reflections=False)
# for w in walls:
#     print(w, L.square(w))
# end = time.perf_counter()
# print("Total execution time: " + str(datetime.timedelta(seconds=(end - start))))

# M = Leech_lat()
# print(M.A)
# FPS = fp_search_cpp.FPSearch(np.array(M.A.tolist(), dtype=float), np.zeros(M.rank, dtype=float), 0, 4.5)
# vecs = FPS.search_all()
# print(len(vecs))

# count = 0
# while True:
#     count += 1
#     engine.run(root_batch=10000000, use_reflections=False)
#     engine.update_walls()
#     walls = engine.get_walls()
#     print(f"Iteration {count}; {len(walls)} walls", end='\r')
#     rays, lines = get_extremal_rays(walls, L.A)
#     if len(lines) == 0:
#         squares = [(fl.fmpq_mat(1, L.rank, ray) * L.A * fl.fmpq_mat(L.rank, 1, ray))[0, 0] for ray in rays]
#         if all(x >= 0 for x in squares):
#             print("\nThe fundamental domain has finite volume")
#             print("Walls:")
#             for w in walls:
#                 print(w)
#             break