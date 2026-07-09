from Commons import *
from BinLattice import *
from Lattice import *
from LatticeUtils import *
from DiscForm import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *
from Allcock import *
from Circle import *


def cusp_walls(L: Lattice, v: IMat, base: IMat) -> List[IMat]:
    if L.signature[0] != 1:
        raise ValueError("The lattice must be of signature (1, n)")
    if L.square(v) != 0:
        raise ValueError("The vector v must be isotropic")
    if L.square(base) <= 0:
        raise ValueError("The base vector must be positive")
    v = imat(v) // math.gcd(*v.tolist())
    if L.product(v, base) < 0:
        v = -v
    basis = L.subquotient(L.complement(v))
    dual, a = L.dual_vec(v)
    if nrows(basis) != L.rank - 2:
        raise ValueError("Error constructing subquotient")
    M = Lattice(nrows(basis), L.batch_prod(basis, basis))
    A, T = M.lll(transform=True)
    basis = imat(list_rows(T @ basis))
    M = Lattice(M.rank, A)
    FPS = fp_search_cpp.FPSearch(np.array(-A, dtype=float), np.zeros(M.rank, dtype=float), 0, 2 * M.exp + 1)
    vecs = FPS.search_all()
    roots = []
    for u in vecs:
        if not M.is_root(u):
            continue
        w = imat(u) @ basis
        b = L.square(w)
        c, x, _, _, _ = euclid(2 * a, b)
        y = 2 * L.product(dual, w)
        if y % c != 0:
            continue
        w = w - (y // c) * x * v
        delta = abs(b // c)
        x, y = divmod(-L.product(w, base), (delta * L.product(v, base)))
        if y == 0:
            shift = [x - 1, x, x + 1]
        else:
            shift = [x, x + 1]
        for k in shift:
            roots.append(w + k * delta * v)
    return roots

def geodesic_walls(L: Lattice, basis: IMat, base: IMat, nshifts: int = 5) -> List[IMat]:
    if nrows(basis) != 2:
        raise ValueError("The sublattice has to be of rank 2")
    M = L.batch_prod(basis, basis)
    B = BinLattice(M[0, 0], M[1, 1], M[0, 1])
    if B.signature != (1, 1):
        # raise ValueError("The sublattice has to be of signature (1, 1)")
        return []
    shifts = [basis]
    if B.shift is not None:
        S = imat_diag([1] * 2)
        for i in range(nshifts):
            S = S @ B.shift
            shifts.append(S @ basis)            
        S = imat_diag([1] * 2)
        T, _ = inv2x2(B.shift)
        for i in range(nshifts):
            S = S @ T
            shifts.append(S @ basis)
    vecs = B.list_negative(2 * L.exp)
    roots = []
    for s in vecs.keys():
        if (2 * L.exp) % s != 0:
            continue
        for v, T in product(vecs[s], shifts):
            w = imat(v) @ T
            if L.square(w) >= 0:
                raise ValueError("Positive root!")
            if not L.is_root(w):
                continue
            d = L.product(w, base)
            if d == 0:
                continue
            if d < 0:
                w = -w
            roots.append((w, Fraction(d * d, abs(int(s)))))
    roots.sort(key=lambda x: x[1])
    if len(roots) == 0:
        return []
    walls = [roots[0][0]]
    for r in roots:
        if L.product(r[0], walls[0]) > 0:
            walls.append(r[0])
            break
    return walls



def RunTest(L: Lattice, base: IMat = None, chamber: IMat = None, h_batch: int = 1, fps_batch: int = 10 ** 3, n_iter: int = 1):
    V = Vinberg(L, base=base, chamber=chamber, h_batch=h_batch, fps_batch=fps_batch)
    V.run(iterations=n_iter)
    while not V.update_rays():
        print(len(V.active_walls), "active walls")
        print(len(V.rays), "rays")
        squares = [(ray * L.A_fl * ray.transpose())[0, 0] for ray in V.rays]
        for a, (r, s) in enumerate(zip(V.rays, squares)):
            if s >= 0:
                continue
            ray, _ = flq2imat(r)
            new_walls = []
            if L.is_root(ray):
                new_walls.append(ray if V.L.product(ray, V.base) > 0 else -ray)
            adj = [w[0] for w in V.active_walls if L.product(ray, w[0]) == 0]
            for i in range(len(adj)):
                compl = L.complement([adj[j] for j in range(len(adj)) if i != j])
                if nrows(compl) != 2:
                    continue
                M = L.batch_prod(compl, compl)
                Lcompl = BinLattice(M[0, 0], M[1, 1], M[0, 1])
                if Lcompl.zero:
                    for v in Lcompl.zero:
                        w = imat(v) @ compl
                        if L.product(w, V.base) < 0:
                            continue
                        new_walls.extend(cusp_walls(L, w, V.base))
                new_walls.extend(geodesic_walls(L, compl, V.base, nshifts=3))
            V.update_walls(new_walls)
            print(f"{a}-th ray of {len(V.rays)}; {len(V.active_walls)} active walls", end='\r')
        print()
        print("Updating rays...")
    print(len(V.active_walls), "walls")
    print(len(V.rays), "rays")
    print("Finite volume")
    ncusps = sum([1 for ray in V.rays if (ray * L.A_fl * ray.transpose())[0, 0] == 0])
    print(ncusps, "cusps")
    print("Walls:")
    for v in V.active_walls:
        print(v[0], v[1])

def TestDiagLat(rank, d = 1):
    L = I_lat(1, 0)(d) + I_lat(0, rank - 1)
    print(L.info())
    RunTest(L, base=[1] + [0] * (rank - 1), chamber=[rank ** 2 - i ** 2 for i in range(rank)], h_batch=1, fps_batch=10000, n_iter=3)

def TestEvenUnimod():
    L = U_lat() + E_lat(8)(-1) + E_lat(8)(-1)
    print(L.info())
    RunTest(L, base=[1, 1] + [0] * 16, h_batch=1, fps_batch=10000, n_iter=3)

def TestVLat():
    L = E_lat(8)(-1) + E_lat(8)(-1) + A_lat(2)(-1) + U_lat()
    print(L.info())
    RunTest(L, base=[0] * 18 + [1, 1], chamber=[101 - i ** 2 for i in range(20)], h_batch=1, fps_batch=10000, n_iter=3)