import numpy as np
import math
from collections import defaultdict
from Lattice import *
from LatticeUtils import *
from IntVectors import *
from FPSearch import *
import fp_search_cpp

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
        axis, self.d = self.L.dual_vec(self.base)
        if self.L.product(self.base, axis) != self.d:
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
        r = []
        A = np.array(self.C.A.tolist(), dtype = float)
        for u in fincke_pohst_search(-A, np.zeros(self.L.rank - 1), 0.5, 2 * self.L.exp + 0.5):
            v = [0] + [int(x) for x in u]
            if self.M.is_root(v):
                r.append(v[1:])
        self.walls[0] = [[0] + v for v in simple_roots(self.C, r)]
        
    def print_info(self):
        print(f"Using {self.base} as the base point")
        print(f"Using the basis {self.basis}")
        print("The Gram matrix in this basis:")
        print(self.M.A)
        if len(self.walls[0]) > 0:
            B = fl.fmpz_mat(self.basis)
            W = fl.fmpz_mat(self.walls[0])
            print(f"{len(self.walls[0])} walls passing through the base point: {(W * B).tolist()}")
        else:
            print("No walls pass through the base point")

    def update_walls(self):
        new_walls = defaultdict(list)
        new_walls[0] = self.walls[0].copy()
        wall_list = self.walls[0]
        distances = sorted(list(self.roots.keys()))
        for d in distances:
            if d == 0:
                continue
            for r in self.roots[d]:
                if all(self.M.product(r, w) >= 0 for w in wall_list):
                    new_walls[d].append(r)
                    wall_list.append(r)
        self.walls = new_walls
        self.roots = new_walls.copy()

    def run(self, max_height : int, root_batch = 10000, height_batch = 3):
        count_v = count_r = 0
        A = np.array(self.C.A.tolist(), dtype = float)
        walls = []
        for h in range(1, max_height, height_batch):
            FPS = []
            for i in range(height_batch):
                bnum, bden = ((h + i) * self.b).numer_denom()
                bound = self.s * (h + i) ** 2
                #FPS = FPSearch((-self.C.A).tolist(), [float(x) / float(bden) for x in bnum.tolist()[0]], \
                #               0.5 + float(bound.p) / float(bound.q), 2 * self.L.exp + 0.5 + float(bound.p) / float(bound.q))
                FPS.append(fp_search_cpp.FPSearch(np.array((-self.C.A).tolist(), dtype=float),
                                                    np.array([float(x) / float(bden) for x in bnum.tolist()[0]], dtype=float),
                                                    0.5 + float(bound.p) / float(bound.q),
                                                    2 * self.L.exp + 0.5 + float(bound.p) / float(bound.q)))
            while True:
                vecs = []
                for i in range(height_batch):
                    vecs.extend([h + i] + u for u in FPS[i].batch_search(10 ** 5))
                if not vecs:
                    break
                for v in vecs:
                    print(f"Heights {h} -- {h + height_batch - 1}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
                    count_v += 1
                    #v = [h] + u
                    sq = self.M.square(v)
                    if not self.M.is_root(v) or sq > 0:
                        continue
                    count_r += 1
                    print(f"Height {h}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
                    prod = h * self.d
                    self.roots[fl.fmpq(prod ** 2, abs(sq))].append(v)
                    if count_r % root_batch != 0:
                        continue
                    self.update_walls()
                    walls = sum(self.walls.values(), start=[])
                    print(f"Height {h}, {count_v} vectors, {count_r} roots, {len(walls)} walls", end = "\r")
                    rays, lines = get_extremal_rays(walls, self.M.A)
                    if len(lines) == 0:
                        squares = [(fl.fmpq_mat(1, self.M.rank, ray) * self.M.A * fl.fmpq_mat(self.M.rank, 1, ray))[0, 0] for ray in rays]
                        if all(x >= 0 for x in squares):
                            print("\nThe fundamental domain has finite volume")
                            B = fl.fmpz_mat(self.basis)
                            W = fl.fmpz_mat(walls)
                            return (W * B).tolist()
                        
    def probe(self, max_height : int):
        count_v = {}
        A = np.array(self.C.A.tolist(), dtype = float)
        for h in range(1, max_height):
            count_v[h] = 0
            bnum, bden = (h * self.b).numer_denom()
            bound = self.s * h ** 2
            FPS = fp_search_cpp.FPSearch(np.array((-self.C.A).tolist(), dtype=float),
                                         np.array([float(x) / float(bden) for x in bnum.tolist()[0]], dtype=float),
                                         0.5 + float(bound.p) / float(bound.q),
                                         2 * self.L.exp + 0.5 + float(bound.p) / float(bound.q))
            while True:
                vecs = FPS.batch_search(10 ** 7)
                if not vecs:
                    break
                count_v[h] += len(vecs)
                print(count_v, end='\r')