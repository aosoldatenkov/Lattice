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
    


# with open('in', "r") as f:
#     lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
# for j, l in enumerate(lattices[2080:]):
#         print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
#         L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])(-1)
#         print(L.info())
#         print(L.A)
#         #print(Vinberg2(L, root_batch=100))
#         V = Vinberg(L)
#         V.print_info()
#         print(V.run(10 ** 100, batch_size=100))

L = Leech_lat_alt()
L = Lattice(L.rank, L.lll())
print(L.info())
print(L.A)
count = 0
for v in fincke_pohst_search(np.array(L.A.tolist(), dtype=float), np.zeros(L.rank), 0, 4.5):
    if L.square(v) == 4:
        count += 1
        print(f"{count}", end='\r')
print()


# A = -np.array(L.batch_prod(basis, basis), dtype=float)
# count = 0
# roots = set()
# for v in fincke_pohst_search(A, np.zeros(21), 2.5, 4.5):
#     if M.is_root(v):
#         roots.add(tuple(v))
#     count += 1
#     print(f"{count} vectors, {len(roots)} roots", end="\r")
#     if count >= 100000:
#         break
# print()

# A = -np.array(L.batch_prod(basis[:2], basis[:2]), dtype=float)
# count = 0
# xx = []
# yy = []
# cc = []
# roots = set()
# level = 1
# step = 10
# while True:
#     for v in fincke_pohst_search(A, np.array([1000, 1000], dtype=float), step * (level - 1) ** 2 - 0.5, step * level ** 2 + 0.5):
#         count += 1
#         print(f"{count} vectors, level {level}", end="\r")
#         xx.append(v[0])
#         yy.append(v[1])
#         cc.append((level ** 3) % 20)
#     level += 1
#     if count >= 100000:
#         break

# x = np.array(xx)
# y = np.array(yy)
# fig, ax = plt.subplots(1)
# ax.scatter(x, y, s=1, c=cc)
# plt.show()

# count = 0
# mindd = []
# maxdd = []
# maxd = 0
# mind = 10 ** 100
# bl = BatchGenerator(M.rank, d = 10, max_size = 10 ** 8)
# for v in bl.vectors():
#     # prod = L.product(base, v)
#     # if prod < 0:
#     #     continue
#     count += 1
#     print(f"{count} roots", end='\r')
#     d = float(abs(M.square(v)))
#     mind = min(mind, d)
#     maxd = max(maxd, d)
#     if count % 10 ** 6 == 0:
#         mindd.append(mind)
#         maxdd.append(maxd)
#         mind = 10 ** 100
#         maxd = 0
#     if count >= 10 ** 8:
#         break

# t = np.arange(len(maxdd))
# fig, ax = plt.subplots(1)
# ax.fill_between(t, np.array(maxdd), np.array(mindd), facecolor='C0', alpha=0.4)
# plt.show()
# Vinberg(L, root_batch=100000)
  