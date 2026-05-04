#!/usr/bin/env python3

import flint as fl
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import time
import datetime
from BinLattice import BinLattice, int_seq
from Lattice import *
from LatticeUtils import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *
import fp_search_cpp
import vsearch_cpp

class DiscForm:

    def __init__(self, L):
        M = np.array(L.A.tolist(), dtype=object)
        D, _, R = smith_normal_form(M)
        Dual = fl.fmpq_mat(R.tolist()) * fl.fmpz_mat(D.tolist()).inv()
        self.A = Dual.transpose() * L.A * Dual
        self.D = D
        self.rank = L.rank
        self.iso = self._list_iso()

    def _list_iso(self):
        isotropic = []
        for u in product(*[range(self.D[i, i]) for i in range(self.rank)]):
            if all(x == 0 for x in u):
                continue
            u_mat = fl.fmpz_mat(1, self.rank, u)
            _, denom =(u_mat * self.A * u_mat.transpose()).numer_denom()
            if denom == 1:
                isotropic.append(u)
        return isotropic
    
    def _list_max_isospaces(self):
        d = len(self.iso)
        if d == 0:
            return []
        max_isospaces = []
        I = fl.fmpq_mat(self.iso)
        P = I * self.A * I.transpose()
        is_integral = [[P[i, j].denom() == 1 for j in range(d)] for i in range(d)]
        def dfs(current):
            nonlocal max_isospaces
            maximal = True
            for i in range(current[-1] + 1, d):
                if all(is_integral[i][j] for j in current):
                    maximal = False
                    current.append(i)
                    dfs(current)
                    current.pop()
            if maximal:
                current_set = set(current)
                if any([current_set.issubset(m) for m in max_isospaces]):
                    return
                max_isospaces.append(current_set)
        for i in range(d):
            current = [i]
            dfs(current)
        return max_isospaces

def CheckAllcock():
    start = time.perf_counter()
    with open('in', "r") as f:
        lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
        for j, l in enumerate(lattices):
            print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
            lstart = time.perf_counter()
            L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])(-1)
            print(L.info())
            print(L.A)
            V = Vinberg(L, h_batch=100)
            V.print_info()
            walls = V.run(root_batch=1000000, use_reflections=False)
            lend = time.perf_counter()
            for w in walls:
                print(w, L.square(w))
            print("Vinberg's algorithm execution time: " + str(datetime.timedelta(seconds=(lend - lstart))))
    end = time.perf_counter()
    print("Total execution time: " + str(datetime.timedelta(seconds=(end - start))))

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

# start = time.perf_counter()
# L = I_lat(1, 16)
# print(L.info())
# print(L.A)
# V = Vinberg(L, h_batch=1)
# V.print_info()
# walls = V.run(root_batch=1, use_reflections=False)
# for w in walls:
#     print(w, L.square(w))
# end = time.perf_counter()
# print("Total execution time: " + str(datetime.timedelta(seconds=(end - start))))

L = Leech_lat()
print(len(irred_decomp(L)))

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