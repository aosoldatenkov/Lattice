"""Orthogonal decomposition of a definite lattice into indecomposable sublattices.

A definite lattice L is decomposable if L = L_1 ⊕ L_2 with nonzero orthogonal sublattices, and
indecomposable otherwise. The decomposition of L into indecomposable sublattices is unique, not
only up to isometry but as a set of sublattices [Eichler 1952]. Kneser's constructive proof
[Kneser 1954] uses the indecomposable vectors: v != 0 is decomposable if v = a + b with nonzero
orthogonal a, b in L. Each indecomposable vector lies in one of the summands, the indecomposable
vectors generate L, and the summands are spanned by the connected components of the graph on the
indecomposable vectors with edges between non-orthogonal vectors.

The algorithm implemented here is Algorithm 1 of [Hemkemeier-Vallentin 1998], which avoids testing
all pairs of vectors for decomposability: the vectors of a generating system S of L consisting of
all vectors of norm at most a bound are processed by increasing norm, and a vector already lying in
the orthogonal sum of the sublattices constructed so far is skipped. A decomposable vector v = a + b
is skipped, because a and b are shorter, so they are processed earlier and lie in the sum. A vector
that is not skipped is thus indecomposable, and it is joined with all the sublattices it is not
orthogonal to.

References:
    M. Eichler, Note zur Theorie der Kristallgitter, Math. Ann. 125 (1952), 51-55.
    M. Kneser, Zur Theorie der Kristallgitter, Math. Ann. 127 (1954), 105-106.
    B. Hemkemeier, F. Vallentin, On the decomposition of lattices, ECCC TR98-052 (1998);
        also in B. Hemkemeier, Algorithmische Konstruktionen von Gittern, Dissertation,
        Universität Dortmund (2003), arXiv:math/0411134.
"""
from typing import List
import numpy as np
import flint as fl
from lattice._matrix import IMat, imat, imat2flz, flz2imat
from lattice.lattice import Lattice
from lattice.enumeration import FPSearch


def _in_span(B: fl.fmpz_mat, G_inv: fl.fmpq_mat, A: fl.fmpz_mat, v: fl.fmpz_mat) -> bool:
    """Whether the row v lies in the lattice spanned by the rows of B, given the inverse G_inv of the
    Gram matrix B A B^T (the rows of B are linearly independent)."""
    x = v * A * B.transpose() * G_inv
    if any(c.denom() != 1 for c in x.entries()):
        return False
    return x * fl.fmpq_mat(B) == fl.fmpq_mat(v)


def irred_decomp(L: Lattice) -> List[List[List[int]]]:
    """Decomposes the positive or negative definite lattice L into indecomposable orthogonal sublattices.

    Returns the list of the summands, each given by a basis (a list of row vectors in the coordinates
    of L), LLL-reduced with respect to the form. The summands are pairwise orthogonal and their sum is
    L. See the module docstring for the algorithm."""
    if L.signature not in [(0, L.rank), (L.rank, 0)]:
        raise ValueError("The lattice must be either positive or negative definite")
    if L.rank == 0:
        return []
    if L.signature[0] == 0:
        L = L(-1)

    # Work in an LLL-reduced basis; the vectors of norm at most the largest norm of its basis
    # vectors form a generating system S
    B, basis = L.A_fl.lll(rep='gram', transform=True)
    M = Lattice(L.rank, B.tolist())
    A = M.A_fl
    ubound = float(max(M.A[i, i] for i in range(M.rank)))
    fps = FPSearch(np.array(M.A, dtype=float), np.zeros(M.rank, dtype=float), 0.5, ubound + 0.5)
    vecs = [imat([int(x) for x in v]) for v in fps.search_all()]
    vecs.sort(key=lambda x: M.square(x))

    sublat = []  # the bases of the indecomposable sublattices constructed so far, as fmpz_mat
    span, G_inv = None, None  # their concatenation and the inverse of its Gram matrix
    for v in vecs:
        w = imat2flz(v)
        # A vector of the current orthogonal sum is skipped (in particular, every decomposable vector)
        if span is not None and _in_span(span, G_inv, A, w):
            continue
        # v is indecomposable: join it with all the sublattices it is not orthogonal to
        joined = [b for b in sublat if any(x != 0 for x in (w * A * b.transpose()).entries())]
        rest = [b for b in sublat if not any(x != 0 for x in (w * A * b.transpose()).entries())]
        gens = flz2imat(w).tolist() + [row for b in joined for row in flz2imat(b).tolist()]
        new = imat2flz(Lattice.image(imat(gens)))
        sublat = rest + [new]
        span = fl.fmpz_mat([row for b in sublat for row in b.tolist()])
        G_inv = fl.fmpq_mat(span * A * span.transpose()).inv()
        if span.nrows() == M.rank and abs(span.det()) == 1:
            break

    result = []
    for b in sublat:
        # An LLL-reduced basis of the summand with respect to the form, in the coordinates of L
        _, T = (b * A * b.transpose()).lll(rep='gram', transform=True)
        result.append((T * b * basis).tolist())
    return [[[int(x) for x in row] for row in b] for b in result]
