#!/usr/bin/env python3

import flint as fl
from BinLattice import BinLattice, int_seq
import math
from Lattice import Lattice
from collections import defaultdict
from LatticeUtils import *

def root_search(L):
    m = 2 * L.exp
    k = int(L.square([1] + [0] * (L.rank - 1)))
    check = {}
    skip = set()
    def decomp(n):
        nonlocal m, k
        t = int(math.sqrt(-n / k))
        result = []
        while t >= 0 and (val := k * t**2 + n) >= -m:
            if val < 0 and m % val == 0:
                result.append(t)
            t -= 1
        return result
    for v in int_seq(L.rank - 1, nonzero=True):
        n = int(L.square([0] + v))
        if n in skip:
            continue
        elif n not in check:
            d = decomp(n)
            if len(d) == 0:
                skip.add(n)
                continue
            check[n] = d
        for t in check[n]:
            u = [t] + v
            if math.gcd(*u) != 1:
                continue
            if L.is_root(u):
                yield u
            

def Hyp():
    d = 10
    I = Lattice(1, [[1]])
    #L = I(9) + A_lat(3)(-1)
    #L = I(30) + I(-1) + A_lat(2)(-1)
    L = I + I(-1) * d
    print(L.info())
    base = [1] + [0] * d
    roots = defaultdict(list)
    count = 0
    for v in root_search(L):
        roots[fl.fmpq(L.product(base, v) ** 2, abs(L.square(v)))].append(v)
        count += 1
        print(f"Found {count} roots\r", end='')
        if count >= 3000000:
            break
    #for v in int_seq(d + 1, nonzero=True):
    #    if math.gcd(*v) != 1:
    #        continue
    #    if L.is_root(v) and L.square(v) < 0 and L.product(base, v) >= 0:
    #        roots[fl.fmpq(L.product(base, v) ** 2, abs(L.square(v)))].append(v)
    #        count += 1
    #        print(f"Found {count} roots\r", end='')
    #        if count >= 5000:
    #            break
    print()
    distances = sorted(roots.keys())
    #print(f"Distances: {distances}")
    dir = list(range(d + 1))
    walls = [] if len(roots[0]) == 0 else simple_roots(L, roots[0], dir)
    print(walls)
    for d in distances:
        if d == 0:
            continue
        for r in roots[d]:
            if all(L.product(r, w) >= 0 for w in walls):
                walls.append(r)
    print(f"Found {len(walls)} walls")
    print(walls)
    print("Computing extremal rays...")
    rays, lines = get_extremal_rays(walls, L.A)
    print(rays)
    if len(lines) > 0:
        print("The cone is not pointed, it contains lines. Lines:")
        for l in lines:
            print(l)
    else:
        s = [(fl.fmpq_mat(1, L.rank, ray) * L.A * fl.fmpq_mat(L.rank, 1, ray))[0, 0] for ray in rays]
        print(s)
        if all(x >= 0 for x in s):
            print("The fundamental domain has finite volume")

Hyp()