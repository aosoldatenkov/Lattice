"""The roots whose mirrors cross a given edge (or segment) in hyperbolic space.

Let L be a lattice of signature (1, n), and a, b non-proportional vectors of L in the closed future light
cone ((a, a), (b, b) >= 0, (a, b) > 0), the endpoints (finite or ideal) of a segment of the geodesic in the
plane Pi = <a, b>. The mirror of a root r crosses the open segment iff (r, a) and (r, b) have opposite signs.

Write r = r_Pi + r_N with r_N in N = Pi^perp, which is negative definite. The products with Pi only depend
on r_Pi, which is determined by A = (r, a) and B = (r, b). With p = (a, a), q = (b, b), c = (a, b) and the
crossing point x = |B| a + A b (for A > 0 > B),
    (r_Pi, r_Pi) = -(x, x) / (c^2 - p q),
and (r, r) <= (r_Pi, r_Pi) since (r_N, r_N) <= 0. So the roots crossing the segment, oriented by A > 0 > B,
have (A, B) in the lattice Lambda = {((y, a), (y, b)) : y in L} with
    0 < (x, x) <= 2 exp(L) (c^2 - p q),
a finite set (for p, q >= 0 the region is bounded in the quadrant A > 0 > B: if p, q > 0 it is bounded by
an ellipse; if q = 0, then B^2 p <= 2 exp c^2 and A <= exp c / |B|; if p = q = 0, then A |B| <= exp c).

For each such (A, B), take y in L with ((y, a), (y, b)) = (A, B). The roots with these products are
r = y + z with z in L cap N, and (r, r) = (r_Pi, r_Pi) + (y_N + z, y_N + z), where y_N is the projection of
y to N. So z runs over a Fincke-Pohst search in L cap N centered at -y_N, with
-(y_N + z, y_N + z) <= 2 exp + (r_Pi, r_Pi); the candidates are then checked with Lattice.is_root.

To find roots crossing an edge from a vertex a towards a vector b outside the light cone (e.g. during the
search for walls), pass a point of the edge in the light cone instead of b, such as k a + b for k large
enough that (k a + b, k a + b) >= 0: the result consists of the roots crossing the edge between a and it.
"""
from __future__ import annotations
import math
from fractions import Fraction
from typing import List, Tuple
import numpy as np
import flint as fl
from lattice._matrix import IMat, imat, imat2flz, flz2imat, nrows
from lattice.lattice import Lattice
from lattice.enumeration import FPSearch


def _ceil_div(x: Fraction | int, y: Fraction | int) -> int:
    q = Fraction(x) / Fraction(y)
    return -((-q.numerator) // q.denominator)


def edge_crossing_roots(L: Lattice, a: IMat, b: IMat) -> List[Tuple[IMat, Fraction]]:
    """All the roots of L whose mirrors cross the open segment between a and b (see the module docstring),
    oriented so that (r, a) > 0 > (r, b), as pairs (root, t) sorted by t = (r, a) / |(r, b)|: the crossing
    point is proportional to a + t b, so t increases from a to b."""
    a, b = imat([int(x) for x in a]), imat([int(x) for x in b])
    p, q, c = L.square(a), L.square(b), L.product(a, b)
    if p < 0 or q < 0 or c <= 0:
        raise ValueError("a and b must lie in the closed future light cone")
    D = c * c - p * q  # -det of the Gram matrix of Pi, positive
    if D <= 0:
        raise ValueError("a and b must not be proportional")
    exp = L.exp
    n = L.rank
    # Lambda: the image of L under y -> ((y, a), (y, b)), with a basis in the form (B, A) = s (b1, a1) + t (0, a2)
    # and lifts of the basis vectors to L
    images = L.A_fl * imat2flz(imat([b.tolist(), a.tolist()])).transpose()  # rows ((e_i, b), (e_i, a))
    H, T = images.hnf(transform=True)
    b1, a1, a2 = int(H[0, 0]), int(H[0, 1]), int(H[1, 1])
    g1, g2 = flz2imat(T)[0], flz2imat(T)[1]
    if b1 < 0:
        b1, a1, g1 = -b1, -a1, -g1
    if a2 < 0:
        a2, g2 = -a2, -g2
    # The complement N cap L in an LLL-reduced basis (it is negative definite, so the reduction is exact),
    # its Gram matrix and inverse (for the projections)
    Nb = L.complement([a, b])
    if nrows(Nb) != n - 2:
        raise ValueError("Error constructing the orthogonal complement of the edge")
    if n > 2:
        _, U = Lattice(n - 2, L.batch_prod(Nb, Nb)).lll(transform=True)
        Nb = U @ Nb
    GN = L.batch_prod(Nb, Nb)
    GN_inv = fl.fmpq_mat(imat2flz(GN)).inv() if n > 2 else None
    Q = np.array(-GN, dtype=float) if n > 2 else None
    bound = 2 * exp * D  # on (x, x) = A^2 q + 2 A |B| c + B^2 p
    result = []
    s = -1
    while True:
        B = s * b1
        absB = -B
        # Even A = 1 violates the bound from here on (the left side increases with |B|)
        if q + 2 * absB * c + absB * absB * p > bound:
            break
        # The A > 0 with A^2 q + 2 A |B| c <= bound - B^2 p, in the progression s a1 + a2 Z
        rest = bound - absB * absB * p
        if q > 0:
            # A <= (-|B| c + sqrt(|B|^2 c^2 + q rest)) / q
            A_max = (math.isqrt(absB * absB * c * c + q * rest) - absB * c) // q
        else:
            A_max = rest // (2 * absB * c)
        if A_max >= 1:
            t_min = _ceil_div(1 - s * a1, a2)
            t_max = (A_max - s * a1) // a2
            for t in range(t_min, t_max + 1):
                A = s * a1 + t * a2
                y = s * g1 + t * g2
                r_pi2 = Fraction(-(A * A * q + 2 * A * absB * c + absB * absB * p), D)
                result.extend(_promote(L, y, r_pi2, Nb, GN_inv, Q, exp))
        s -= 1
    # Orient, check and sort
    roots = []
    seen = set()
    for r in result:
        key = tuple(int(x) for x in r)
        if key in seen:
            continue
        seen.add(key)
        A, B = L.product(r, a), L.product(r, b)
        if A > 0 > B:
            roots.append((r, Fraction(A, -B)))
    roots.sort(key=lambda x: x[1])
    return roots


def _promote(L, y, r_pi2, Nb, GN_inv, Q, exp) -> List[IMat]:
    """The roots y + z, z in L cap N (basis Nb), with (y + z, y + z) >= -2 exp."""
    if GN_inv is None:
        return [y] if L.is_root(y) else []
    # The coordinates of the projection of y to N in the basis Nb: (y, Nb) GN^{-1}. Subtracting the nearest
    # vector of L cap N from y does not change the coset y + (L cap N), and brings the center of the
    # search into [-1/2, 1/2]^(n-2): the lift y itself can be large (it comes from an HNF transformation),
    # and a large center would overflow the coordinates of FPSearch and lose precision
    yN = fl.fmpq_mat(imat2flz(imat([L.batch_prod(imat([y]), Nb)[0].tolist()]))) * GN_inv
    coords = [Fraction(int(yN[0, i].p), int(yN[0, i].q)) for i in range(yN.ncols())]
    nearest = [math.floor(x + Fraction(1, 2)) for x in coords]
    y = y - imat(nearest) @ Nb
    center = [float(x - k) for x, k in zip(coords, nearest)]
    # -(y_N + z)^2 = (r_Pi, r_Pi) - (r, r) <= 2 exp + (r_Pi, r_Pi); its values lie in (r_Pi, r_Pi) + Z, so the
    # bound keeps a margin of 0.5 from them, and the lower bound -0.5 admits z = -y_N
    ubound = float(2 * exp + r_pi2) + 0.5
    if ubound < 0:
        return []
    roots = []
    for k in FPSearch(Q, np.array(center), -0.5, ubound).search_all():
        r = y + imat([int(x) for x in k]) @ Nb
        if L.is_root(r):
            roots.append(r)
    return roots
