#!/usr/bin/env python3

import flint as fl
import matplotlib.pyplot as plt
import numpy as np
from BinLattice import BinLattice, int_seq
import math
import sympy as sp
from Lattice import *
from collections import defaultdict
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
    with open('in', "r") as f:
        lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
    for j, l in enumerate(lattices):
            print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
            L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])(-1)
            print(L.info())
            print(L.A)
            #print(Vinberg2(L, root_batch=100))
            V = Vinberg(L)
            V.print_info()
            print(V.run(10 ** 100, root_batch=100))


with open('in', "r") as f:
    lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
    for j, l in enumerate(lattices[2802:2803]):
        print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
        L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])(-1)
        print(L.info())
        print(L.A)
        V = Vinberg(L, h_batch=100)
        V.print_info()
        print(V.run(root_batch=1000000))
        
# L = I_lat(1, 14)
# print(L.info())
# print(L.A)
# V = Vinberg(L, h_batch=10)
# V.print_info()
# print(V.run(root_batch=1))

# A_np = np.array(L.A.tolist(), dtype=np.int64)
# engine = vsearch_cpp.VSearchCpp(A_np, 1, 1)

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