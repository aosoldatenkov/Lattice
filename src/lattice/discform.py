"""The discriminant bilinear form of a lattice, its isotropic subgroups and integral overlattices."""
from __future__ import annotations
from itertools import product
from typing import List, Set
import flint as fl
from lattice._matrix import imat, imat2flz, flz2imat
from lattice.lattice import Lattice


class DiscForm:
    """The discriminant group A_L = L^∨/L of a lattice L with the bilinear form
    b: A_L x A_L -> Q/Z induced by the form of L.

    Let P * A * R = D be the Smith normal form of the Gram matrix A (P, R in GL(n, Z),
    D = diag(d_1, ..., d_n) with d_1 | ... | d_n), and let f_1, ..., f_n be the basis of L
    given by the columns of R. The vectors v_i = f_i / d_i form a basis of L^∨, since
    A^{-1} = R D^{-1} P and the columns of A^{-1} span L^∨. Thus L is spanned by the d_i v_i,
    and A_L = ⊕ Z/d_i with generators v_i. Elements of A_L are written as coordinate
    vectors in the generators v_i with d_i > 1.

    Only the bilinear form b is kept (the diagonal entries are reduced mod 1, not mod 2),
    so "isotropic" means b(u, u) = 0 in Q/Z: the isotropic subgroups H correspond to the
    integral (but not necessarily even) overlattices of L, via H -> the preimage of H in L^∨.

    Attributes:
        rank: the rank n of L.
        A: the Gram matrix of L^∨ in the basis v_1, ..., v_n, as a flint.fmpq_mat.
        D: the diagonal matrix of the Smith normal form, as an IMat.
        ord: the orders d_i > 1 of the nontrivial generators of A_L.
        Ared: the matrix of b on the nontrivial generators, with entries reduced to [0, 1).
        iso: the list of nonzero isotropic elements, filled by list_iso(), or None.
    """

    def __init__(self, L: Lattice):
        L._compute_snf()
        R, D = L._snf_R, L._snf_D
        # The columns of Dual are the vectors v_i = R e_i / d_i
        Dual = fl.fmpq_mat(R) * fl.fmpq_mat(D).inv()
        self.A = Dual.transpose() * L.A_fl * Dual
        self.D = flz2imat(D)
        self.rank = L.rank
        self._init_A_red()
        self.iso = None

    def _init_A_red(self) -> None:
        """Computes the orders of the nontrivial generators and the reduced form Ared on them."""
        # The invariant factors are increasing, so the trivial ones (d_i = 1) come first
        i0 = min([self.rank] + [i for i in range(self.rank) if self.D[i, i] > 1])
        self.ord = [int(self.D[i, i]) for i in range(i0, self.rank)]

        def reduce(x):
            return fl.fmpq(x.numer() % x.denom(), x.denom())
        self.Ared = fl.fmpq_mat([[reduce(x) for x in l[i0:]] for l in self.A.tolist()[i0:]])

    def list_iso(self) -> List[List[int]]:
        """Lists the nonzero isotropic elements u of A_L, i.e. those with b(u, u) = 0 in Q/Z,
        as coordinate vectors with 0 <= u_i < d_i. The result is cached in self.iso.
        The search runs over the whole group, so it is only practical for small groups."""
        isotropic = []
        n = len(self.ord)
        for u in product(*[range(self.ord[i]) for i in range(n)]):
            if all(x == 0 for x in u):
                continue
            u_mat = fl.fmpz_mat(1, n, u)
            # u Ared u^T = b(u, u) mod Z, as the off-diagonal terms appear twice
            _, denom = (u_mat * self.Ared * u_mat.transpose()).numer_denom()
            if denom == 1:
                isotropic.append(list(u))
        self.iso = isotropic
        return isotropic

    def list_max_isospaces(self) -> List[Set[int]]:
        """Lists the maximal isotropic subgroups H of A_L. Each subgroup is returned as the
        set of indices in self.iso of its nonzero elements. If A_L has no nonzero isotropic
        elements (so that L has no proper integral overlattices), returns [].

        A subset of isotropic elements spans an isotropic subgroup iff the elements are
        pairwise orthogonal, b(u, v) = 0 in Q/Z, and the set of nonzero elements of an
        isotropic subgroup is such a subset. So the maximal isotropic subgroups correspond
        to the maximal cliques of the graph on self.iso with edges between orthogonal
        elements, and the latter are found by a depth-first search."""
        if self.iso is None:
            self.list_iso()
        d = len(self.iso)
        if d == 0:
            return []
        max_isospaces = []
        I = fl.fmpq_mat(self.iso)
        P = I * self.Ared * I.transpose()
        is_integral = [[P[i, j].denom() == 1 for j in range(d)] for i in range(d)]

        def dfs(current):
            # current is an increasing list of indices of pairwise orthogonal elements;
            # the search visits all such lists in lexicographic order
            maximal = True
            for i in range(current[-1] + 1, d):
                if all(is_integral[i][j] for j in current):
                    maximal = False
                    current.append(i)
                    dfs(current)
                    current.pop()
            if maximal:
                # current may still be extendable by a smaller index; in that case,
                # a maximal clique containing it is lexicographically smaller, so it
                # has already been found
                current_set = set(current)
                if any(current_set.issubset(m) for m in max_isospaces):
                    return
                max_isospaces.append(current_set)
        for i in range(d):
            dfs([i])
        return max_isospaces

    def overlattice(self, iso_gens: List[int] | Set[int]) -> Lattice:
        """Returns the overlattice of L generated by L and the isotropic elements with the
        given indices in self.iso, as a lattice in a new basis. Raises ValueError if these
        elements are not pairwise orthogonal, i.e. the overlattice is not integral."""
        if self.iso is None:
            self.list_iso()
        # The generators in the coordinates of the basis v_i of L^∨: the elements of A_L,
        # padded by zeros for the trivial generators, and the basis d_i v_i of L
        n0 = self.rank - len(self.ord)
        gens = [[0] * n0 + self.iso[i] for i in sorted(iso_gens)] + self.D.tolist()
        basis = imat2flz(Lattice.image(imat(gens)))
        A, denom = (basis * self.A * basis.transpose()).numer_denom()
        if denom != 1:
            raise ValueError("The isotropic elements must be pairwise orthogonal.")
        return Lattice(self.rank, A.tolist())
