import math
from fractions import Fraction
from typing import List
import numpy as np
from lattice._matrix import IMat, imat, imat2flz, nrows, flq2imat, inv2x2, euclid
from lattice.lattice import Lattice
from lattice.binary import BinLattice
from lattice.enumeration import FPSearch
from lattice.vinberg.algorithm import Vinberg


def cusp_root_classes(L: Lattice, v: IMat) -> List[tuple]:
    """The roots of L orthogonal to the primitive isotropic vector v, as a list of classes (w, step, m): the
    roots of a class are the primitive vectors among w + t step, t in Z, where step is a multiple of v,
    and primitivity depends only on t modulo divisors of m (so it holds for some t in any m consecutive
    values, or for none).

    The roots orthogonal to v are r = w + k v, where w lifts a vector u of the negative definite
    lattice M = v^perp / v with (u, u) = (r, r), so 1 <= -(u, u) <= 2 exp(L) (exp(M) may be smaller).
    For x in v^perp, (r, x) = (u, x mod v), so (u, u) must divide 2 (u, M); u need not be primitive in
    M (e.g. r = 2 w' + v). For the vector dual with (dual, v) = div(v), (u, u) must divide
    2 ((w, dual) + k div(v)), which holds for k in a residue class modulo delta, and step = delta v.
    A prime dividing r divides the content m of u, and divides w + t step for t in one residue class
    modulo it at most."""
    if L.signature[0] != 1:
        raise ValueError("The lattice must be of signature (1, n)")
    if L.square(v) != 0:
        raise ValueError("The vector v must be isotropic")
    v = imat(v) // math.gcd(*[int(x) for x in v])
    basis = L.subquotient(L.complement([v]))
    dual, a = L.dual_vec(v)
    if nrows(basis) != L.rank - 2:
        raise ValueError("Error constructing subquotient")
    M = Lattice(nrows(basis), L.batch_prod(basis, basis))
    A, T = M.lll(transform=True)
    basis = T @ basis
    M = Lattice(M.rank, A)
    # The bounds keep a margin of 0.5 from the integer values of -(u, u)
    FPS = FPSearch(np.array(-A, dtype=float), np.zeros(M.rank, dtype=float), 0.5, 2 * L.exp + 0.5)
    classes = []
    for u in FPS.search_all():
        u = imat([int(x) for x in u])
        b = -M.square(u)
        if (2 * math.gcd(*[int(x) for x in u @ M.A])) % b != 0:
            continue
        w = u @ basis
        # Solve 2 div(v) k = -2 (w, dual) mod b
        c, x, _, _, _ = euclid(2 * a, b)
        y = 2 * L.product(dual, w)
        if y % c != 0:
            continue
        w = w - (y // c) * x * v
        classes.append((w, (b // c) * v, math.gcd(*[int(z) for z in u])))
    return classes


def cusp_walls(L: Lattice, v: IMat, base: IMat) -> List[IMat]:
    """Candidate walls through the cusp v: roots of L orthogonal to the isotropic vector v, near the
    hyperplane of the base point. For each class of cusp_root_classes(), the roots r = w + t step closest
    to the hyperplane base^perp are returned: the one with the smallest (r, base) >= 0 (and the next one if
    (r, base) = 0) and the one with the largest (r, base) < 0. Primitivity of r depends on t, so several t
    may be tried."""
    if L.square(base) <= 0:
        raise ValueError("The base vector must be positive")
    v = imat(v) // math.gcd(*[int(x) for x in v])
    if L.product(v, base) < 0:
        v = -v
    roots = []
    for w, step, m in cusp_root_classes(L, v):
        if L.product(step, base) < 0:
            step = -step
        # r_t = w + t step has (r_t, base) = (w, base) + t (step, base); t0 is the smallest t with a nonnegative value
        t0 = -(L.product(w, base) // L.product(step, base))
        for direction in (1, -1):
            t = t0 if direction == 1 else t0 - 1
            for _ in range(m + 1):
                r = w + t * step
                if L.is_root(r):
                    roots.append(r)
                    if L.product(r, base) != 0:
                        break
                t += direction
    return roots


def geodesic_walls(L: Lattice, basis: IMat, base: IMat, nshifts: int = 1) -> List[IMat]:
    """Candidate walls crossing an edge perpendicularly: the roots of L in the rank 2 sublattice spanned
    by basis, of signature (1, 1), whose mirrors cut the geodesic of this plane closest to the projection
    of the base point on either side. Returns the closest one and the closest one on the other side
    (at most two roots), or [] if the sublattice is not of signature (1, 1).

    The negative vectors of the binary lattice are listed up to its automorphism (the shift along the
    river) by BinLattice.list_negative. For each orbit, the shift k bringing its mirror closest to the
    base point is found by walking along the orbit (|(r, base)| is unimodal in k), and the shifts within
    nshifts of it are taken: the closest mirrors of an orbit on either side are among them for nshifts >= 1."""
    if nrows(basis) != 2:
        raise ValueError("The sublattice has to be of rank 2")
    M = L.batch_prod(basis, basis)
    B = BinLattice(M[0, 0], M[1, 1], M[0, 1])
    if B.signature != (1, 1):
        return []
    vecs = B.list_negative(2 * L.exp)
    if B.shift is not None:
        S = imat(B.shift)
        S_inv, _ = inv2x2(B.shift)

    def orbit(v):
        """The vectors v S^k (in the coordinates of L) for the shifts k near the base point."""
        if B.shift is None:
            return [imat(v) @ basis]
        cur, prev = imat(v), None
        # Walk in the direction in which |(v S^k, base)| decreases
        d0 = abs(L.product(cur @ basis, base))
        U = S if abs(L.product((cur @ S) @ basis, base)) < d0 else S_inv
        while True:
            nxt = cur @ U
            if abs(L.product(nxt @ basis, base)) >= abs(L.product(cur @ basis, base)):
                break
            cur = nxt
        result = [cur @ basis]
        for T in (S, S_inv):
            w = cur
            for _ in range(nshifts):
                w = w @ T
                result.append(w @ basis)
        return result

    roots = []
    for s in vecs.keys():
        # The norm of a root divides 2 exp
        if (2 * L.exp) % s != 0:
            continue
        for v in vecs[s]:
            for w in orbit(v):
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
        # Mirrors on opposite sides of the projection of the base point have a positive product
        if L.product(r[0], walls[0]) > 0:
            walls.append(r[0])
            break
    return walls


def RunTest(L: Lattice, base: IMat = None, chamber: IMat = None, h_batch: int = 1, fps_batch: int = 10 ** 3,
            n_iter: int = 1, max_passes: int = 1000) -> List[IMat]:
    """Vinberg's algorithm with wall refinement: runs n_iter iterations of the search, then, while the active
    walls do not bound a polyhedron of finite volume, looks for further walls near the extremal rays of their
    cone. For an isotropic ray (a cusp), the candidates are the roots through it (cusp_walls). For a ray outside
    the light cone, they are the ray itself if it is a root, and for each edge through the ray (spanned with an
    adjacent ray), the roots of the rank 2 lattice of the edge crossing it perpendicularly (geodesic_walls) and
    the roots through the cusps at its ends. If a pass adds no walls, or the cone is not pointed, the root search
    is continued. Returns the active walls.

    The candidates are not found in the order of their distance from the base point, so the result is not
    certified: the active walls bound a polyhedron of finite volume containing the chamber, which may be a
    union of several chambers."""
    V = Vinberg(L, base=base, chamber=chamber, h_batch=h_batch, fps_batch=fps_batch)
    V.run(iterations=n_iter)
    passes = 0
    while not V.update_rays():
        passes += 1
        if passes > max_passes:
            print(f"Stopped after {max_passes} passes")
            break
        before = {tuple(int(x) for x in w[0]) for w in V.active_walls}
        if len(V.lines) > 0 or not V.rays:
            # The walls do not bound a pointed cone yet
            V.run(iterations=max(n_iter, 1))
            continue
        print(len(V.active_walls), "active walls")
        print(len(V.rays), "rays")
        # The walls and rays of this pass: the walls added below do not change them
        walls = [w[0] for w in V.active_walls]
        rays = []
        for r in V.rays:
            ray = flq2imat(r)[0][0]
            rays.append(ray // math.gcd(*[int(x) for x in ray]))
        incidence, adjacency = V.incidence, V.adjacency
        for a, ray in enumerate(rays):
            s = L.square(ray)
            if s > 0:
                continue
            # Skip the rays cut off by the walls added in this pass
            if any(L.product(ray, w[0]) < 0 for w in V.active_walls):
                continue
            new_walls = []
            if s == 0:
                new_walls.extend(cusp_walls(L, ray, V.base))
            else:
                if L.is_root(ray):
                    new_walls.append(ray)
                # The edges through the ray: the 2-faces spanned with the adjacent rays (from cdd)
                for b in adjacency[a]:
                    common = [walls[j] for j in incidence[a] & incidence[b]]
                    compl = L.complement(common)
                    if nrows(compl) != 2:
                        continue
                    M = L.batch_prod(compl, compl)
                    Lcompl = BinLattice(M[0, 0], M[1, 1], M[0, 1])
                    for z in Lcompl.zero:
                        w = imat(z) @ compl
                        if L.product(w, V.base) < 0:
                            continue
                        new_walls.extend(cusp_walls(L, w, V.base))
                    new_walls.extend(geodesic_walls(L, compl, V.base))
            V.update_walls(new_walls)
            print(f"{a}-th ray of {len(rays)}; {len(V.active_walls)} active walls", end='\r')
        print()
        if {tuple(int(x) for x in w[0]) for w in V.active_walls} == before:
            # No new walls near the rays: continue the root search
            V.run(iterations=max(n_iter, 1))
        print("Updating rays...")
    print(len(V.active_walls), "walls")
    print(len(V.rays), "rays")
    if V.update_rays():
        print("Finite volume")
        ncusps = sum([1 for ray in V.rays if (ray * L.A_fl * ray.transpose())[0, 0] == 0])
        print(ncusps, "cusps")
    print("Walls:")
    for v in V.active_walls:
        print(v[0], v[1])
    return [w[0] for w in V.active_walls]
