"""Brute-force discriminant forms, independent of lattice.discform and lattice.genus.

The discriminant form of a lattice L with Gram matrix A is computed from sympy's Smith normal
form with exact fractions, and isomorphisms of discriminant forms are found by a backtracking
search over the images of the generators. By Nikulin's theorem [Nikulin, Integral symmetric
bilinear forms and some of their applications, Cor. 1.9.4], two even lattices are in the same
genus iff they have the same signature and isomorphic discriminant quadratic forms. Since the
genus is invariant under rescaling, odd lattices L, M are compared via the even lattices
L(2), M(2). This gives an oracle for the comparison of genera, which does not use the
Conway-Sloane symbols at all.
"""
from fractions import Fraction
from itertools import product
import sympy as sp
from sympy.matrices.normalforms import smith_normal_decomp
from lattice.lattice import Lattice


class BruteDiscForm:
    """The discriminant group L^∨/L = ⊕ Z/d_i of the lattice with Gram matrix A, with the
    Gram matrix of the generators of order d_i > 1 as exact fractions (not reduced)."""

    def __init__(self, A):
        n = len(A)
        M = sp.Matrix(A)
        D, P, R = smith_normal_decomp(M, domain=sp.ZZ)  # D = P M R
        d = [int(D[i, i]) for i in range(n)]
        idx = [i for i in range(n) if abs(d[i]) > 1]
        self.orders = [abs(d[i]) for i in idx]
        # The vectors R e_i / d_i (columns) form a basis of L^∨
        gens = [[Fraction(int(R[r, i]), d[i]) for r in range(n)] for i in idx]
        for g in gens:
            assert all(sum(g[r] * A[r][s] for r in range(n)).denominator == 1 for s in range(n))
        self.gram = [[sum(g[r] * A[r][s] * h[s] for r in range(n) for s in range(n)) for h in gens] for g in gens]
        self.elements = list(product(*[range(o) for o in self.orders]))

    def value(self, x, y=None):
        """The rational number (x, y) for lifts of x, y to L^∨ (y = x by default)."""
        y = x if y is None else y
        k = len(x)
        return sum(x[i] * y[j] * self.gram[i][j] for i in range(k) for j in range(k))

    def b(self, x, y):
        """The discriminant bilinear form b(x, y) in [0, 1)."""
        return self.value(x, y) % 1

    def q(self, x):
        """The discriminant quadratic form q(x) in [0, 2), well defined if L is even."""
        return self.value(x) % 2

    def isotropic(self):
        """The nonzero elements x with b(x, x) = 0."""
        return [x for x in self.elements if any(x) and self.b(x, x) == 0]

    def is_isomorphic(self, other) -> bool:
        """Checks whether the quadratic forms are isomorphic (both lattices must be even)."""
        if sorted(self.orders) != sorted(other.orders):
            return False
        if sorted(self.q(x) for x in self.elements) != sorted(other.q(x) for x in other.elements):
            return False
        k = len(self.orders)
        gens = [tuple(int(i == j) for j in range(k)) for i in range(k)]
        # Candidate images of the generator g_i: elements y with d_i y = 0 and q(y) = q(g_i).
        # A homomorphism preserving q is injective, as b is nondegenerate, hence bijective.
        cand = [[y for y in other.elements
                 if all((self.orders[i] * c) % m == 0 for c, m in zip(y, other.orders)) and other.q(y) == self.q(gens[i])]
                for i in range(k)]
        images = []

        def search(i):
            if i == k:
                return True
            for y in cand[i]:
                if all(other.b(y, images[j]) == self.b(gens[i], gens[j]) for j in range(i)):
                    images.append(y)
                    if search(i + 1):
                        return True
                    images.pop()
            return False
        return search(0)


def same_genus(L: Lattice, M: Lattice) -> bool:
    """Checks whether L and M are in the same genus by comparing discriminant quadratic forms."""
    if L.rank != M.rank or L.signature != M.signature or L.parity != M.parity:
        return False
    if L.parity:
        L, M = L(2), M(2)
    return BruteDiscForm(L.A.tolist()).is_isomorphic(BruteDiscForm(M.A.tolist()))
