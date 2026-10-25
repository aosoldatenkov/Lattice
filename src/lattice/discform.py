"""The discriminant bilinear form of a lattice, its isotropic subgroups and integral overlattices."""
from __future__ import annotations
import math
from typing import Iterator, List, Set, Tuple
import flint as fl
import numpy as np
from lattice._matrix import IMat, imat, imat2flz, flz2imat
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
        dual: the vectors v_1, ..., v_n in the standard basis e_1, ..., e_n of L (rows), as a flint.fmpq_mat.
        embedding: the embedding L -> L^∨: the coordinates of e_1, ..., e_n in the basis v_1, ..., v_n
            (rows: e_j = sum_i embedding[j, i] v_i, i.e. embedding = (D R^{-1})^T), as an IMat.
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
        self.dual = Dual.transpose()
        # e_j = sum_i (D R^{-1})_{ij} v_i, since the matrix with the columns v_i is R D^{-1}
        E, _ = (fl.fmpq_mat(D) * fl.fmpq_mat(R).inv()).numer_denom()
        self.embedding = flz2imat(E.transpose())
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

    def _integral_form(self) -> Tuple[int, np.ndarray]:
        """(N, B): the common denominator N of the entries of Ared, and the integer matrix B = N Ared, so that
        b(u, v) = u B v^T / N mod 1 (int64 if the products fit, else Python ints)."""
        N = math.lcm(1, *[int(x.denom()) for x in self.Ared.entries()])
        m = len(self.ord)
        dtype = np.int64 if m * N * max(self.ord + [1]) < 2 ** 62 else object
        B = np.array([[int(self.Ared[i, j] * N) for j in range(m)] for i in range(m)], dtype=dtype).reshape(m, m)
        return N, B

    def list_iso(self) -> List[List[int]]:
        """Lists the nonzero isotropic elements u of A_L, i.e. those with b(u, u) = 0 in Q/Z,
        as coordinate vectors with 0 <= u_i < d_i, in lexicographic order. The result is cached
        in self.iso. The search runs over the whole group (vectorized), so it is only practical
        for groups of moderate size."""
        m = len(self.ord)
        if m == 0:
            self.iso = []
            return self.iso
        N, B = self._integral_form()
        # All the elements, in lexicographic order (the first coordinate varies slowest)
        U = np.indices(self.ord).reshape(m, -1).T.astype(B.dtype)
        # u B u^T = N b(u, u) mod N, as the off-diagonal terms appear twice
        isotropic = (((U @ B) % N) * U).sum(axis=1) % N == 0
        isotropic[0] = False
        self.iso = [[int(x) for x in u] for u in U[isotropic]]
        return self.iso

    def list_max_isospaces(self) -> List[Set[int]]:
        """Lists the maximal isotropic subgroups H of A_L, sorted. Each subgroup is returned as
        the set of indices in self.iso of its nonzero elements. If A_L has no nonzero isotropic
        elements (so that L has no proper integral overlattices), returns []. See
        iter_max_isospaces for the search, and to get the subgroups as they are found."""
        return sorted(self.iter_max_isospaces(), key=sorted)

    def iter_max_isospaces(self) -> Iterator[Set[int]]:
        """Yields the maximal isotropic subgroups H of A_L one by one, as they are found (in the
        order of the search, not sorted), each as the set of indices in self.iso of its nonzero
        elements. Nothing is yielded if A_L has no nonzero isotropic elements.

        The candidates at an isotropic subgroup H are the isotropic v orthogonal to H
        (b(v, H) = 0 in Q/Z) and not in H; then <H, v> is isotropic, and H is maximal iff
        there are none. A maximal M containing H consists of H and candidates, so the
        candidates orthogonal to all the candidates (forced) are in M, else M could be
        extended by them. The search, from H = 0, adds the forced candidates, then branches
        on candidates v: the branch of v excludes the candidates branched on before it (as in
        the Bron-Kerbosch algorithm for maximal cliques), and is cut if <H, v> or the forced
        candidates contain an excluded one (the candidates w with <H, w> = <H, x> for an
        excluded x are skipped at once). With a pivot p among the candidates, M contains p or a
        candidate not orthogonal to p (else M + p would be isotropic), so only these are
        branched on. Every maximal subgroup is found exactly once: in the branch of the first
        of them it contains."""
        if self.iso is None:
            self.list_iso()
        d = len(self.iso)
        if d == 0:
            return
        N, B = self._integral_form()
        m = len(self.ord)
        ords = np.array(self.ord, dtype=B.dtype)
        strides = np.array([math.prod(self.ord[i + 1:]) for i in range(m)], dtype=B.dtype)
        iso = np.array(self.iso, dtype=B.dtype).reshape(d, m)
        codes = (iso @ strides).astype(np.int64)
        index = np.full(math.prod(self.ord), -1, dtype=np.int64)       # code -> index in self.iso
        index[codes] = np.arange(d)
        G = (iso @ B) % N                       # b(u, w) = G[u] w^T / N mod 1

        def extend(H, in_H, i):
            """<H, v> for v = self.iso[i]: the union of the cosets H + k v, k < o the order of v modulo H; also
            the indices in self.iso of its elements outside H, and of the generators of <H, v> modulo H (the cosets
            with gcd(k, o) = 1)."""
            v = iso[i]
            cosets, w = [H], v % ords
            while not in_H[int(w @ strides)]:
                cosets.append((H + w) % ords)
                w = (w + v) % ords
            H2 = np.concatenate(cosets)
            o = len(cosets)
            gens = np.concatenate([cosets[k] for k in range(1, o) if math.gcd(k, o) == 1])
            return H2, index[(H2[len(H):] @ strides).astype(np.int64)], index[(gens @ strides).astype(np.int64)]

        def search(H, in_H, orth, excluded):
            # in_H: the mask of the elements of H (by code); orth: of the isotropic elements orthogonal to H
            while True:
                C = np.flatnonzero(orth & ~in_H[codes])
                if not len(C):
                    yield {int(i) for i in index[(H @ strides).astype(np.int64)] if i >= 0}
                    return
                O = (G[C] @ iso[C].T) % N == 0      # orthogonality among the candidates
                forced = C[O.all(axis=1)]
                if not len(forced):
                    break
                if excluded[forced].any():
                    return                          # no maximal subgroup avoids them
                for i in forced:
                    if not in_H[codes[i]]:
                        H, new, _ = extend(H, in_H, i)
                        in_H[codes[new]] = True
                        orth = orth & ((G @ iso[i]) % N == 0)
            # The pivot p: a maximal M contains p or a candidate not orthogonal to p (else M + p is isotropic), so
            # only these are branched on; p minimizes their number (p excluded: only the latter)
            free = ~excluded[C]
            counts = (~O & free).sum(axis=1) + free
            p = int(np.argmin(counts))
            branch = C[(~O[p] | (np.arange(len(C)) == p)) & free]
            excluded = excluded.copy()
            # dead: the candidates w with <H, w> = <H, x> for an excluded x (so <H, w> contains x)
            dead = excluded.copy()
            for x in C[excluded[C]]:
                dead[extend(H, in_H, x)[2]] = True
            for i in branch:
                if not dead[i]:
                    H2, new, gens = extend(H, in_H, i)
                    if not excluded[new].any():
                        in_H2 = in_H.copy()
                        in_H2[codes[new]] = True
                        yield from search(H2, in_H2, orth & ((G @ iso[i]) % N == 0), excluded)
                    excluded[i] = True
                    dead[gens] = True

        in_H = np.zeros(len(index), dtype=bool)
        in_H[0] = True
        yield from search(np.zeros((1, m), dtype=B.dtype), in_H, np.ones(d, dtype=bool), np.zeros(d, dtype=bool))

    def overlattice(self, iso_gens: List[int] | Set[int],
                    embedding: bool = False) -> Lattice | Tuple[Lattice, IMat]:
        """Returns the overlattice M of L generated by L and the isotropic elements with the
        given indices in self.iso, as a lattice in a new basis. With embedding=True, returns
        the pair (M, E), where the rows of E are the coordinates of the standard basis vectors
        e_1, ..., e_n of L in the basis of M (so that E A_M E^T = A_L). Raises ValueError if
        these elements are not pairwise orthogonal, i.e. the overlattice is not integral."""
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
        M = Lattice(self.rank, A.tolist())
        if not embedding:
            return M
        # The coordinates X of e_j in the basis of M: X basis = embedding, integral since L is in M
        X, denom = (fl.fmpq_mat(imat2flz(self.embedding)) * fl.fmpq_mat(basis).inv()).numer_denom()
        assert denom == 1
        return M, flz2imat(X)
