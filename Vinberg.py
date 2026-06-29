from collections import defaultdict
from Commons import *
from Lattice import *
from LatticeUtils import *
from IntVectors import *
from FPSearch import *
from VSearch import *
import fp_search_cpp
import vsearch_cpp

class Vinberg:

    def __init__(self, L: Lattice, base: list[int] | IMat = None, h_batch: int = 3, fps_batch: int = 10 ** 3):
        # The lattice L should be of signature (1, n)
        if L.signature[0] != 1:
            raise ValueError("The lattice should be of signature (1, n)")
        self.L = L
        self.h_batch = h_batch
        self.fps_batch = fps_batch
        if base != None and L.square(base) > 0:
            base = imat(base)
            gcd = math.gcd(*base.tolist())
            self.base = base // gcd
            axis, _ = self.L.dual_vec(base)
            compl = self.L.complement([base])
            self.basis = concat_rows(axis, compl)
        else:
            b = sorted(self.list_bases(1), key=lambda x: x[0])
            self.base = b[0][1]
            self.basis = b[0][2]
        self._init_basis()

    def _init_basis(self):
        B, _ = imat2flz(self.basis).inv().numer_denom()
        self.M = Lattice(self.L.rank, self.L.batch_prod(self.basis, self.basis))
        self.VS = vsearch_cpp.VSearchCpp(self.M.A.tolist(), (imat2flz(self.base) * B).tolist()[0], 2.1 * self.M.exp + 0.5, self.h_batch, True, True)
        # self.VS = VSearch(self.M.A, self.M.exp, h_batch=self.h_batch, fps_batch=self.fps_batch)

    def list_bases(self, n):
        bases = []
        # Check if the given basis is suitable
        compl = imat_diag([1] * self.L.rank)[1:, :]
        base = self.L.complement(compl)
        if nrows(base) == 1 and self.L.square(base) > 0:
            base = base.flatten()
            axis = imat([1] + [0] * (self.L.rank - 1))
            s = float(self.L.square(base)) / float(base[0] ** 2)
            bases.append((s, base, concat_rows(axis, compl)))
        # Attempt to find other bases by searching for positive vectors with small coefficients
        for u in int_seq(self.L.rank, nonzero=True):
            if len(bases) >= n:
                break
            if math.gcd(*u) != 1:
                continue
            if self.L.square(u) > 0:
                axis, d = self.L.dual_vec(u)
                compl = self.L.complement(u)
                bases.append((float(d ** 2) / float(self.L.square(u)), imat(u), concat_rows(axis,compl)))
        return bases

    def print_info(self):
        print(f"Using {self.base} as the base point")
        print(f"Using the basis\n{self.basis}")
        print("The Gram matrix in this basis:")
        print(self.M.A)
        # print(f"Root system at the base point: {len(self.VS.R.pos_roots)} positive roots")
        # B = fl.fmpz_mat(self.basis)
        # print(f"{len(self.VS.R.sroots)} walls passing through the base point: {[r * B for r in self.VS.R.sroots]}")

    def run(self, root_batch = 1000, max_iterations = 50000):
        count = 0
        while True:
            count += 1
            if count > max_iterations:
                print("\nReached the maximum number of iterations")
                walls = imat(self.VS.get_walls())
                return list_rows(walls @ self.basis)
            self.VS.run(root_batch, self.fps_batch)
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
                squares = [(ray * self.M.A_fl * ray.transpose())[0, 0] for ray in rays]
                if all(x >= 0 for x in squares):
                    print("\nThe fundamental domain has finite volume")
                    return list_rows(walls @ self.basis)
