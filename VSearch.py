import flint as fl
import numpy as np
import random as rnd
import math
from collections import deque, defaultdict
import fp_search_cpp

class RootSys:

    def __init__(self, A: fl.fmpz_mat, roots: list[list[int]], base: list[int] = None, max_rank: int = -1, cache_size: int = 2 ** 16):
        self.A = A
        self.eye = fl.fmpz_mat([[int(i == j) for i in range(A.ncols())] for j in range(A.ncols())])
        self.max_rank = self.A.ncols() if max_rank == -1 else max_rank
        b = fl.fmpz_mat(1, self.A.ncols(), base) if base else None
        self._simple_roots([fl.fmpz_mat(1, self.A.ncols(), r) for r in roots], b)
        self.cache_c = deque(maxlen = cache_size)
        self.cache_r = deque(maxlen = cache_size)
        self.cache_hit = 0
        self.cache_miss = 0
        
    def _simple_roots(self, roots, base = None):
        self.sroots = []
        if len(roots) == 0:
            return
        if base and any((base * self.A * r.transpose())[0, 0] == 0 for r in roots):
            base = None
        if base is None:
            # Pick a direction that is not orthogonal to any root. This will be used to determine which roots are positive.
            rnd.seed(int(roots[0][0, 0]))
            r = self.A.ncols()
            while True:
                b = fl.fmpz_mat(1, r, [rnd.randint(1, 10 ** 6) for _ in range(r)])
                if all((b * self.A * r.transpose())[0, 0] != 0 for r in roots):
                    break
            self.base = b
        else:
            self.base = base
        self.pos_roots = [r for r in roots if (self.base * self.A * r.transpose())[0, 0] > 0]
        self.pos_roots.sort(key=lambda r: (self.base * self.A * r.transpose())[0, 0])
        slist = []
        for r in self.pos_roots:
            slist.append(r.tolist()[0])
            smatr = fl.fmpz_mat(slist)
            M = smatr * self.A * smatr.transpose()
            if M.det() != 0:
                self.sroots.append(r)
            else:
                slist.pop()
            if len(self.sroots) >= self.max_rank:
                break
        self.M = self.A * fl.fmpz_mat([r.tolist()[0] for r in self.pos_roots]).transpose()

    def weyl_group(self):
        chambers = {self.chamber(self.base): self.eye}
        new = deque([self.base])
        while len(new) > 0:
            b = new.popleft()
            for s in self.sroots:
                new_b = b * self.reflection(s)
                c = self.chamber(new_b)
                if c not in self.chambers:
                    chambers[c] = self.reflection(s) * self.chambers[self.chamber(b)]
                    new.append(new_b)
        return chambers

    def reflection(self, r):
        return self.eye - 2 * self.A * r.transpose() * r / (r * self.A * r.transpose())[0, 0]
    
    def chamber(self, v):
        def sign(a):
            return 1 if a > 0 else (-1 if a < 0 else 0)
        prod = v * self.M
        return tuple(sign(x) for x in prod.tolist()[0])
        #return tuple(sign((v * self.A * r.transpose())[0, 0]) for r in self.pos_roots)
    
    def closed_chamber(self, v):
        def sign(a):
            return 1 if a >= 0 else -1
        prod = v * self.M
        return tuple(sign(x) for x in prod.tolist()[0])
        #return tuple(sign((v * self.A * r.transpose())[0, 0]) for r in self.pos_roots)
    
    def find_reflection(self, v):
        c = self.closed_chamber(v)
        if c in self.cache_c:
            self.cache_hit += 1
            return self.cache_r[self.cache_c.index(c)]
        r = self.eye
        while any(x < 0 for x in c):
            height = sum(1 for j in range(len(self.pos_roots)) if c[j] < 0)
            for i in [j for j in range(len(self.pos_roots)) if c[j] < 0]:
                new_c = self.closed_chamber(v * self.reflection(self.pos_roots[i]))
                if new_c in self.cache_c:
                    self.cache_hit += 1
                    return r * self.reflection(self.pos_roots[i]) * self.cache_r[self.cache_c.index(new_c)]
                new_height = sum(1 for j in range(len(self.pos_roots)) if new_c[j] < 0)
                if new_height < height:
                    c = new_c
                    r = r * self.reflection(self.pos_roots[i])
                    v = v * self.reflection(self.pos_roots[i])
                    break
        self.cache_miss += 1
        self.cache_c.append(c)
        self.cache_r.append(r)
        return r
    

class VSearch:

    def __init__(self, A: fl.fmpz_mat, exp: int, h_batch: int = 1):
        self.A = A
        self.exp = exp
        self.C = fl.fmpz_mat([v[1:] for v in A.tolist()[1:]])
        self.rank = A.ncols()
        self.walls = defaultdict(list)
        self.roots = defaultdict(list)
        self._init_chamber()
        self.h = 0
        self.h_batch = h_batch
        self.FPS = []
        for _ in range(h_batch):
            self.FPS.append(self._init_fps())

    def _init_chamber(self):
        self.b = fl.fmpz_mat(1, self.rank - 1, self.A.tolist()[0][1:]) * self.C.inv()
        self.s = self.A[0, 0] - (self.b * self.C * self.b.transpose())[0, 0]
        if self.s <= 0:
            raise ValueError("Error initializing the basis: s is non-positive")
        FPS = fp_search_cpp.FPSearch(np.array((-self.C).tolist(), dtype=float),
                                    np.zeros(self.rank - 1, dtype=float),
                                    0, 2 * self.exp + 0.5)
        while True:
            vecs = FPS.batch_search(10 ** 6)
            if not vecs:
                break
            roots = []
            for u in vecs:
                v = fl.fmpz_mat(1, self.rank, [0] + [int(x) for x in u])
                if self._is_root(v):
                    roots.append(v)
        self.R = RootSys(self.A, roots)
        self.walls = defaultdict(list)

    def _is_root(self, r):
        prod = r * self.A
        sq = int((prod * r.transpose())[0, 0])
        if sq == 0:
            return False
        a = 2 * math.gcd(*[int(x) for x in prod.tolist()[0]])
        return a % sq == 0

    def update_walls(self):
        new_walls = defaultdict(list)
        wall_list = self.R.sroots.copy()
        distances = sorted(list(self.roots.keys()))
        for d in distances:
            if d == 0:
                continue
            for r in self.roots[d]:
                if all((w * self.A * r.transpose())[0, 0] >= 0 for w in wall_list):
                    new_walls[d].append(r)
                    wall_list.append(r)
        self.walls = new_walls
        self.roots = new_walls.copy()

    def _init_fps(self):
        self.h += 1
        bnum, bden = (self.h * self.b).numer_denom()
        bound = self.s * self.h ** 2
        return (self.h, fp_search_cpp.FPSearch(np.array((-self.C).tolist(), dtype=float),
                                               np.array([float(x) / float(bden) for x in bnum.tolist()[0]], dtype=float),
                                               0.5 + float(bound.p) / float(bound.q),
                                               2 * self.exp + 0.5 + float(bound.p) / float(bound.q)))
    
    def _run_fps(self):
        vecs = []
        for i in range(self.h_batch):
            if self.FPS[i][1].exhausted():
                self.FPS[i] = self._init_fps()
            vv = self.FPS[i][1].batch_search(10 ** 3)
            vecs.extend([fl.fmpz_mat(1, self.rank, [self.FPS[i][0]] + v) for v in vv])
        return vecs

    def run(self, root_batch = 10000):
        count_v = count_r = 0
        while True:
            vecs = self._run_fps()
            for v in vecs:
                count_v += 1
                sq = (v * self.A * v.transpose())[0, 0]
                if not self._is_root(v) or sq > 0:
                    continue
                v = v * self.R.find_reflection(v)
                count_r += 1
                print(count_v, count_r, end='\r')
                self.roots[fl.fmpq(int(v[0, 0]) ** 2, abs(int(sq)))].append(v)
            if count_r >= root_batch:
                self.update_walls()
                break
