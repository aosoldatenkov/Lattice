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

def split_Un(L):
    nmin = abs(L.disc)
    best = None
    for u in int_seq(L.rank, nonzero=True, length=100000):
        if math.gcd(*u) != 1:
            continue
        if L.square(u) == 0:
            v, n = L.dual_vec(u)
            if L.square(v) % 2 != 0:
                compl = L.complement([u])
                even = True
                for w in compl:
                    if L.square(w) % 2 != 0:
                        even = False
                        v = [v[i] + w[i] for i in range(L.rank)]
                        break
                if not even:
                    continue
            t = L.square(v) // 2
            v = [n * v[i] - t * u[i] for i in range(L.rank)]
            compl = L.complement([u, v])
            if L.index([u, v] + compl) == 1 and n < nmin:
                best = [u, v] + compl
                nmin = n
    return best


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
    

def Vinberg2(L, root_batch = 1000):
    # Check the signature
    if L.signature[0] != 1:
        raise ValueError("The lattice should be of signature (1, n)")
    
    # Check if the given basis is suitable
    if L.A[0, 0] > 0:
        base = [1] + [0] * (L.rank - 1)
    else:
        # If the first vector is not positive, change the basis
        for base in int_seq(L.rank, nonzero=True):
            if math.gcd(*base) != 1:
                continue
            if L.square(base) > 0:
                break
    
    # Step 1: find the walls passing through the base point
    roots = defaultdict(list)
    axis, d = L.dual_vec(base)
    if L.product(base, axis) != d:
        raise ValueError("Error computing the dual vector")
    compl = L.complement([base])
    basis = [axis] + compl
    M = Lattice(L.rank, L.batch_prod(basis, basis))
    C = Lattice(L.rank - 1, L.batch_prod(compl, compl))
    A = np.array(C.A.tolist(), dtype = float)
    b = np.array(M.A.tolist()[0][1:], dtype = float)
    b = b @ np.linalg.inv(A)
    axis = [axis[i] - sum(compl[j][i] * int(b[j]) for j in range(L.rank - 1)) for i in range(L.rank)]
    basis = [axis] + compl
    M = Lattice(L.rank, L.batch_prod(basis, basis))
    print(f"Using {axis} with square {int(M.A[0, 0])} and divisibility {d} as the main axis")
    for u in fincke_pohst_search(-A, np.zeros(L.rank - 1), 0.5, 2 * L.exp + 0.5):
        v = [0] + [int(x) for x in u]
        if M.is_root(v):
            roots[0].append(v)
    print(f"Found {len(roots[0])} walls passing through the base point")

    # Step 2: Prepare for the search of roots of increasing height
    # b = np.array(M.A.tolist()[0][1:], dtype = float)
    # b = b @ np.linalg.inv(A)
    # s = float(M.A[0, 0]) - b @ A @ b.transpose()
    b = fl.fmpz_mat(1, M.rank - 1, M.A.tolist()[0][1:]) * C.A.inv()
    s = M.A[0, 0] - (b * C.A * b.transpose())[0, 0]
    if s < 0:
        raise ValueError("s is negative")
    
    def Walls(L, roots):
        distances = sorted(roots.keys())
        walls = []
        for d in distances:
            if d == 0:
                walls = simple_roots(L, roots[0])
                continue
            for r in roots[d]:
                if all(L.product(r, w) >= 0 for w in walls):
                    walls.append(r)
        return walls

    count_v = count_r = 0
    walls = []
    for h in range(1, 10 ** 100):
        bnum, bden = (h * b).numer_denom()
        bound = s * h ** 2
        for u in fincke_pohst_search(-A, np.array([float(x) / float(bden) for x in bnum.tolist()[0]], dtype = float), \
                                    0.5 + float(bound.p) / float(bound.q), 2 * L.exp + 0.5 + float(bound.p) / float(bound.q)):
            print(f"Height {h}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
            count_v += 1
            v = [h] + u
            sq = M.square(v)
            if not M.is_root(v) or sq > 0:
                continue
            count_r += 1
            print(f"Height {h}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
            prod = h * d
            roots[fl.fmpq(prod ** 2, abs(sq))].append(v)
            if count_r % root_batch != 0:
                continue
            walls = Walls(M, roots)
            print(f"Height {h}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
            rays, lines = get_extremal_rays(walls, M.A)
            if len(lines) == 0:
                squares = [(fl.fmpq_mat(1, M.rank, ray) * M.A * fl.fmpq_mat(M.rank, 1, ray))[0, 0] for ray in rays]
                if all(x >= 0 for x in squares):
                    print("\nThe fundamental domain has finite volume")
                    B = fl.fmpz_mat(basis)
                    W = fl.fmpz_mat(walls)
                    return (W * B).tolist()

# with open('in', "r") as f:
#     lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
# for j, l in enumerate(lattices[2080:]):
#         print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
#         L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])(-1)
#         print(L.info())
#         print(L.A)
#         #print(Vinberg2(L, root_batch=100))
#         V = Vinberg_(L)
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
  