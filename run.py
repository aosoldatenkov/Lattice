#!/usr/bin/env python3

import matplotlib.pyplot as plt
import random as rnd
import time
from Commons import *
from BinLattice import *
from Lattice import *
from LatticeUtils import *
from DiscForm import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *
from Allcock import *
from Circle import *
from ReflectiveTests import *
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


G = imat([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
          [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
          [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
gens = [[2 * int(i == j) for i in range(16)] for j in range(16)]
for v in product(range(2), repeat=5):
    gens.append([x % 2 for x in (imat(v) @ G).tolist()])
basis = Lattice.image(imat(gens))
K = Lattice(16, basis @ basis.transpose() // 2)
L = U_lat() + K(-1)
print(L.info())
RunTest(L, base=[1, 1] + [0] * 16, h_batch=1, fps_batch=10000, n_iter=2)

# rank = 22
# L = D_lat(20)(-1) + U_lat()
# base = [0] * 20 + [1, 1]
# V = Vinberg(L, base=base, h_batch=1, fps_batch=10000)
# V.print_info()
# walls = V.run(max_iterations=5)
# for w in walls:
#     print(w, L.square(w))
# rays, lines = get_extremal_rays(walls, L.A)
# print(len(rays), len(lines))
# squares = [(fl.fmpq_mat(1, L.rank, ray) * L.A * fl.fmpq_mat(L.rank, 1, ray))[0, 0] for ray in rays]
# for i, r in enumerate(rays):
#     print(r, squares[i])
# group = Allcock_group(Coxeter_graph(L, walls), len(walls))
# print(f"The Weyl group: {group}")

# rank = 9
# L = Lattice(rank, [[1 - 2 * int(i == j) for j in range(rank)] for i in range(rank)])
# print(L.A)
# print(L.info())
# V = Vinberg(L, base=[1] * rank, h_batch=20, fps_batch=10000)
# V.print_info()
# walls = V.run(max_iterations=5)
# for w in walls:
#     print(w, L.square(w))
# rays, lines = get_extremal_rays(walls, L.A)
# print(len(rays), len(lines))
# squares = [(fl.fmpq_mat(1, L.rank, ray) * L.A * fl.fmpq_mat(L.rank, 1, ray))[0, 0] for ray in rays]
# for i, r in enumerate(rays):
#     print(r, squares[i])

# V = vsearch_cpp.VSearchCpp(L.A.tolist(), [1] + [0] * (rank - 1), 2.1 * L.exp, 50, False, True)
# V.init_chamber([0] + [i for i in range(1, rank)])
# count = 0
# while True:
#     count += V.run(100000, 10000)
#     vecs = V.get_vecs()
#     print(count, end='\r')

# L = D_lat(20) + U_lat()
# print(L(-1).info())
# D = DiscForm(L(-1))
# print(D.Ared)

# L = I_lat(1, 21)
# basis = L.even_sublattice()
# L = Lattice(22, L.batch_prod(basis, basis))
# print(L.info())
# D = DiscForm(L)
# print(D.Ared)

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