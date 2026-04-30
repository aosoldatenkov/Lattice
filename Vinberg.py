import numpy as np
import math
from collections import defaultdict
from Lattice import *
from LatticeUtils import *
from IntVectors import *
from FPSearch import *
from VSearch import *
import fp_search_cpp
import vsearch_cpp

class Vinberg:

    def __init__(self, L: Lattice, h_batch: int = 3, fps_batch: int = 10 ** 3, use_reflections: bool = True):
        # The lattice L should be of signature (1, n)
        if L.signature[0] != 1:
            raise ValueError("The lattice should be of signature (1, n)")
        self.L = L
        self.h_batch = h_batch
        self.fps_batch = fps_batch
        self.use_reflections = use_reflections
        self._init_basis()

    def _init_basis(self):
        # Check if the given basis is suitable
        compl = [[int(i == j) for j in range(self.L.rank)] for i in range(1, self.L.rank)]
        base = self.L.complement(compl)
        if len(base) == 1 and self.L.square(base[0]) > 0:
            self.base = base[0]
            axis = [1] + [0] * (self.L.rank - 1)
            if self.L.product(self.base, axis) <= 0:
                print(self.L.product(self.base, axis))
                axis[0] = -1
        else:
            # If the first vector is not positive, change the basis
            for u in int_seq(self.L.rank, nonzero=True):
                if math.gcd(*u) != 1:
                    continue
                if self.L.square(u) > 0:
                    self.base = u
                    break
            axis, _ = self.L.dual_vec(self.base)
            compl = self.L.complement([self.base])
        
        # for u in int_seq(self.L.rank, nonzero=True):
        #     if math.gcd(*u) != 1:
        #         continue
        #     if self.L.square(u) > 0:
        #         self.base = u
        #         break
        # axis, _ = self.L.dual_vec(self.base)
        # compl = self.L.complement([self.base])
        self.basis = [axis] + compl
        B, _ = fl.fmpz_mat(self.basis).inv().numer_denom()
        self.M = Lattice(self.L.rank, self.L.batch_prod(self.basis, self.basis))
        self.VS = vsearch_cpp.VSearchCpp(self.M.A.tolist(), (fl.fmpz_mat(1, self.L.rank, self.base) * B).tolist()[0], self.M.exp, self.h_batch)
        # self.VS = VSearch(self.M.A, self.M.exp, h_batch=self.h_batch, fps_batch=self.fps_batch)

    def print_info(self):
        print(f"Using {self.base} as the base point")
        print(f"Using the basis {self.basis}")
        print("The Gram matrix in this basis:")
        print(self.M.A)
        # print(f"Root system at the base point: {len(self.VS.R.pos_roots)} positive roots")
        # B = fl.fmpz_mat(self.basis)
        # print(f"{len(self.VS.R.sroots)} walls passing through the base point: {[r * B for r in self.VS.R.sroots]}")

    def run(self, root_batch = 1000, max_iterations = 5000):
        count = 0
        while True:
            count += 1
            if count > max_iterations:
                print("\nReached the maximum number of iterations")
                walls = self.VS.get_walls()
                B = fl.fmpz_mat(self.basis)
                W = fl.fmpz_mat(walls)
                return (W * B).tolist()
            self.VS.run(root_batch=root_batch, use_reflections=False)
            self.VS.update_walls()
            walls = self.VS.get_walls()
            # self.VS.run(root_batch=root_batch, use_reflections=self.use_reflections)
            # walls = self.VS.R.sroots + sum(self.VS.walls.values(), start=[])
            # walls = [[int(x) for x in w.tolist()[0]] for w in walls]
            print(f"Iteration {count}; {len(walls)} walls", end='\r')
            if len(walls) == 0:
                continue
            rays, lines = get_extremal_rays(walls, self.M.A)
            if len(lines) == 0:
                squares = [(fl.fmpq_mat(1, self.L.rank, ray) * self.M.A * fl.fmpq_mat(self.L.rank, 1, ray))[0, 0] for ray in rays]
                if all(x >= 0 for x in squares):
                    print("\nThe fundamental domain has finite volume")
                    B = fl.fmpz_mat(self.basis)
                    W = fl.fmpz_mat(walls)
                    return (W * B).tolist()
                    


class VinbergOld:

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

    def run(self, max_height: int, root_batch = 10000, height_batch = 5):
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
                    self.roots[fl.fmpq(v[0] ** 2, abs(sq))].append(v)
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