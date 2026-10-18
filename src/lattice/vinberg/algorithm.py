import math
from fractions import Fraction
from lattice._matrix import IMat, imat, imat_zero, imat_diag, imat2flz, nrows, concat_rows, list_rows
from lattice.lattice import Lattice
from lattice.roots import get_extremal_rays
from lattice.enumeration import int_seq
try:
    from lattice import vsearch_cpp
except ImportError:  # the C++ extension is not built
    vsearch_cpp = None

class Vinberg:

    def __init__(self, L: Lattice, base: list[int] | IMat = None, chamber: list[int] | IMat = None, h_batch: int = 3,
                 fps_batch: int = 10 ** 3):
        """Prepares Vinberg's algorithm for the lattice L of signature (1, n), n > 0.

        Args:
            base: the base point, a vector with (base, base) > 0; if None, a base point is chosen
                by list_bases().
            chamber: a vector choosing the chamber at the base point (see docs/Vinberg.md).
            h_batch: the number of heights searched in parallel (the worker threads of the search).
            fps_batch: the batch size of the Fincke-Pohst search of each worker.
        """

        if vsearch_cpp is None:
            raise ImportError("Vinberg's algorithm requires the vsearch_cpp extension; run `uv sync` to build it")
        if L.signature[0] != 1 or L.signature[1] == 0:
            raise ValueError("The lattice should be of signature (1, n) with n > 0")
        self.L = L
        self.h_batch = h_batch
        self.fps_batch = fps_batch

        # The search covers the vectors v with 1 <= -(v, v) <= bound_factor * exp.
        # All roots satisfy -(v, v) <= 2 exp, so the default 2 finds all of them; a larger
        # factor only slows the search down, and a smaller one may miss roots (then the
        # walls cannot be certified).
        self.bound_factor = 2
        
        if base is not None and L.square(base) <= 0:
            raise ValueError("The base point must have a positive square")
        if base is not None:
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
        self.chamber = imat(chamber) if chamber is not None else imat_zero(1, L.rank)
        self._init_basis()

    def _init_basis(self):
        # The search orients the roots of positive height by (v, base) = h (e_0, base) > 0
        if self.L.product(self.basis[0], self.base) <= 0:
            raise ValueError("The first basis vector must have a positive product with the base point")
        B, _ = imat2flz(self.basis).inv().numer_denom()
        self.M = Lattice(self.L.rank, self.L.batch_prod(self.basis, self.basis))
        # The values -(v, v) are integers; the bound keeps a margin of 0.5 from them (docs/Vinberg.md, 3.3)
        bound = math.floor(self.bound_factor * self.M.exp) + 0.5
        self.VS = vsearch_cpp.VSearchCpp(self.M.A.tolist(), (imat2flz(self.base) * B).tolist()[0], (imat2flz(self.chamber) * B).tolist()[0],
                                         bound, self.h_batch, True, True)
        self.base_walls = [imat(w) @ self.basis for w in self.VS.get_walls()]
        self.base_walls.sort(key = lambda x: self.L.product(x, self.chamber))
        self.active_walls = [(w, 0) for w in self.base_walls]
        # self.VS = VSearch(self.M.A, self.M.exp, h_batch=self.h_batch, fps_batch=self.fps_batch)

    def list_bases(self, n):
        bases = []
        # Check if the given basis is suitable
        compl = imat_diag([1] * self.L.rank)[1:, :]
        base = self.L.complement(compl)
        if nrows(base) == 1 and self.L.square(base) > 0:
            base = base.flatten()
            axis = imat([1] + [0] * (self.L.rank - 1))
            # The sign of the complement is arbitrary; the search needs (e_0, base) > 0
            if self.L.product(axis, base) < 0:
                base = -base
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
                compl = self.L.complement([u])
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

    def run(self, root_batch = 1000, iterations = 1, certify = False):
        """Runs at most the given number of iterations of the search, and returns the active walls.
        Stops when the polyhedron bounded by the active walls has finite volume. Rejected roots are
        never walls of the chamber, but an accepted root is only certain to be a wall once all the
        roots closer to the base point are known (see docs/Vinberg.md). With certify = True, the run
        only stops once all the heights up to required_height() have also been searched completely;
        then the active walls are exactly the walls of the chamber, and self.certified is set.
        Certification requires bound_factor >= 2, so that the search finds all the roots."""
        if certify and self.bound_factor < 2:
            raise ValueError("Certification requires bound_factor >= 2")
        self.certified = False
        count = 0
        while True:
            count += 1
            if count > iterations:
                print("\nReached the maximal number of iterations")
                return [w[0] for w in self.active_walls]
            self.VS.run(root_batch, self.fps_batch)
            self.VS.update_walls()
            walls = self.VS.get_walls()
            # imat([]) has shape (0,), not (0, n); see the shape conventions in _matrix.py
            walls = imat(walls) if walls else imat_zero(0, self.L.rank)
            self.update_walls(list_rows(walls @ self.basis))
            print(f"Iteration {count}; {len(self.active_walls)} active walls", end='\r')
            if self.update_rays():
                if not certify:
                    print("\nThe fundamental domain has finite volume")
                    return [w[0] for w in self.active_walls]
                if self.VS.searched_height() >= self.required_height():
                    self.certified = True
                    print("\nThe fundamental domain has finite volume; all the walls are certified")
                    return [w[0] for w in self.active_walls]

    def required_height(self) -> int:
        """The height up to which the search must be complete for all the roots at most as far from
        the base point as the active walls to be known. A root r at height h (the first coordinate
        in self.basis) has (r, base) = h (e_0, base) and |(r, r)| <= 2 exp, so its key
        (r, base)^2 / |(r, r)| is at least h^2 (e_0, base)^2 / (2 exp)."""
        keys = [w[1] for w in self.active_walls if w[1] > 0]
        if not keys:
            return 0
        d0 = self.L.product(self.basis[0], self.base)
        bound = Fraction(2 * self.L.exp) * max(keys) / (d0 * d0)
        return math.isqrt(bound.numerator // bound.denominator)

    def update_rays(self):
        if len(self.active_walls) == 0:
            return False
        self.rays, self.lines = get_extremal_rays([w[0] for w in self.active_walls], self.L.A)
        if len(self.lines) == 0:
            squares = [(ray * self.L.A_fl * ray.transpose())[0, 0] for ray in self.rays]
            if all(x >= 0 for x in squares):
                return True
        return False
                
    def update_walls(self, new_walls):
        wall_list = []
        for w in new_walls:
            if not self.L.is_root(w):
                continue
            p = self.L.product(w, self.base)
            if p == 0:
                continue
            if p < 0:
                w, p = -w, -p
            q = self.L.square(w)
            if q >= 0:
                continue
            wall_list.append((imat(w), Fraction(p * p, -q)))
        wall_list.extend([w for w in self.active_walls if w[1] > 0])
        wall_list.sort(key=lambda w: w[1])
        self.active_walls = [(w, 0) for w in self.base_walls]
        for w in wall_list:
            active = True
            for v in self.active_walls:
                if self.L.product(w[0], v[0]) < 0:
                    active = False
                    break
            if active:
                self.active_walls.append(w)
