import numpy as np
import math
from collections import defaultdict
from Lattice import *
from LatticeUtils import *
from IntVectors import *


class Vinberg:

    def __init__(self, L: Lattice):
        # The lattice L should be of signature (1, n)
        if L.signature[0] != 1:
            raise ValueError("The lattice should be of signature (1, n)")
        self.L = L

        # Check if the given basis is suitable
        if L.A[0, 0] > 0:
            self.base = [1] + [0] * (L.rank - 1)
        else:
            # If the first vector is not positive, change the basis
            for u in int_seq(L.rank, nonzero=True):
                if math.gcd(*u) != 1:
                    continue
                if L.square(u) > 0:
                    self.base = u
                    break
        
        self._init_basis()
        self._init_chamber()

    def _init_basis(self):
        rank = self.L.rank
        axis, d = self.L.dual_vec(self.base)
        if self.L.product(self.base, axis) != d:
            raise ValueError("Error computing the dual vector")
        compl = self.L.complement([self.base])
        self.basis = [axis] + compl
        self.M = Lattice(rank, self.L.batch_prod(self.basis, self.basis))
        self.C = Lattice(rank - 1, self.L.batch_prod(compl, compl))
        self.b = fl.fmpz_mat(1, rank - 1, self.M.A.tolist()[0][1:]) * self.C.A.inv()
        self.s = self.M.A[0, 0] - (self.b * self.C.A * self.b.transpose())[0, 0]
        if self.s <= 0:
            raise ValueError("Error initializing the basis: s is non-positive")

    def _init_chamber(self):
        self.walls = defaultdict(list)
        self.roots = defaultdict(list)
        A = np.array(self.C.A.tolist(), dtype = float)
        for u in fincke_pohst_search(-A, np.zeros(self.L.rank - 1), 0.5, 2 * self.L.exp + 0.5):
            v = [0] + [int(x) for x in u]
            if self.M.is_root(v):
                self.roots[0].append(v)
        self.walls[0] = simple_roots(self.L, self.roots[0])
        self.h = 0
        
    def print_info(self):
        print(f"Using {self.basis[0]} with square {int(self.M.A[0, 0])} as the main axis")
        print(f"Found {len(self.walls[0])} walls passing through the base point")

    def update_walls(self):
        new_roots = defaultdict(list)
        new_walls = defaultdict(list)
        new_walls[0] = self.walls[0]
        distances = sorted(self.roots.keys())
        for d in distances:
            if d == 0:
                continue
            for r in self.roots[d]:
                if all(self.L.product(r, w) >= 0 for w in self.walls):
                    new_walls.append(r)