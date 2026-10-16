"""Helpers for integer matrices stored in several formats.

Formats in use
--------------
* IMat: a numpy array with dtype=object, so the entries are arbitrary-precision
  Python ints (or flint.fmpz, see below) and never overflow.
* Python lists: List[List[int]] for matrices, List[int] for single vectors.
* flint.fmpz_mat / flint.fmpq_mat: used for exact linear algebra
  (HNF, SNF, LLL, inverses, nullspaces).

Matrices are lists of row vectors; "row" is always the first index.

Shape convention (empty matrices, single rows)
----------------------------------------------
The shape of an IMat is part of its meaning and never depends on the number
of rows:

* A matrix is a 2-d IMat of shape (r, c), including the degenerate shapes
  (0, c), (r, 0) and (1, c). With rows as vectors, shape (0, n) is the empty
  set of vectors in Z^n (e.g. a basis of the zero sublattice); shape (n, 0)
  is n vectors in Z^0, typically the transpose of the former.
* A vector is a 1-d IMat of shape (n,).
* Every function returning a matrix returns a 2-d IMat of the exact shape;
  in particular, flz2imat() / flq2imat() map an r x c flint matrix to shape
  (r, c), also for r = 0 and r = 1. To get a vector from a 1 x n flint
  matrix, take row 0 explicitly: flz2imat(m)[0].
* Functions accepting a matrix also accept a 1-d IMat of shape (n,), read as
  the 1 x n matrix with that row. Consequently imat([]) (shape (0,)) means a
  single row of length 0: nrows() returns 1 and imat2flz() gives a 1 x 0
  fmpz_mat. To represent "no rows", use imat_zero(0, n), which keeps n.
* Python lists are always read as lists of rows, so [] has no rows
  (nrows([]) == 0, imat2flz([]) is 0 x 0) and a flat List[int] is NOT a
  single vector. Wrap a single vector as [v] or pass it as an IMat.

Entry types
-----------
fmpz_mat.tolist() returns flint.fmpz entries, so IMats produced by flz2imat()
and flq2imat() contain fmpz rather than int (the denominator returned by
flq2imat() is an fmpz as well). They mix freely with Python ints in arithmetic;
convert with int() where a genuine int is required.
"""
import numpy as np
import flint as fl
from typing import List, Tuple

IMat = np.ndarray

def imat(data, **kwargs):
    return np.array(data, dtype=object, **kwargs)

def imat_zero(nlines, ncols, **kwargs):
    return np.zeros((nlines, ncols), dtype=object, **kwargs)

def imat_diag(data, **kwargs):
    n = len(data)
    if n == 0:
        return imat_zero(0, 0, **kwargs)
    return np.array([[0] * i + [data[i]] + [0] * (n - i - 1) for i in range(n)], dtype=object, **kwargs)

def flz2imat(m, **kwargs):
    # tolist() of a 0 x c matrix is [], which loses c
    if m.nrows() == 0:
        return imat_zero(0, m.ncols(), **kwargs)
    return imat(m.tolist(), **kwargs)

def flq2imat(m, **kwargs):
    mz, d = m.numer_denom()
    return flz2imat(mz, **kwargs), d

def imat2flz(m):
    if type(m) is not IMat:
        if len(m) == 0:
            return fl.fmpz_mat(0, 0)
        m = imat(m)
    shape = m.shape[:2] if len(m.shape) > 1 else (1, m.shape[0])
    return fl.fmpz_mat(*shape, m.flatten().tolist())

def nrows(m: List[List[int]] | IMat) -> int:
    if type(m) is IMat:
        return m.shape[0] if m.ndim > 1 else 1
    return len(m)

def concat_rows(m1: List[List[int]] | IMat, m2: List[List[int]] | IMat) -> IMat:
    if type(m1) is IMat:
        m1 = m1.tolist() if m1.ndim > 1 else [m1.tolist()]
    if type(m2) is IMat:
        m2 = m2.tolist() if m2.ndim > 1 else [m2.tolist()]
    return imat(m1 + m2)

def list_rows(m: List[List[int]] | IMat) -> List[IMat]:
    if type(m) is not IMat:
        return [imat(r) for r in m]
    if m.ndim == 1:
        return [m]
    return [m[i, :] for i in range(m.shape[0])]

def inv2x2(m: List[List[int]] | IMat) -> Tuple[IMat, int]:
    """Returns the pair (adj(m), det(m)) for a 2x2 matrix m, so that m^{-1} = adj(m) / det(m)."""
    m = imat(m)
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    return imat([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]]), det

def euclid(a: int, b: int) -> Tuple[int, int, int, int, int]:
    """Transforms the 1x2 matrix (a, b) with integral entries into
    the standard form (c, 0), where c is the GCD of a, b.
    Returns the tuple (c, x, y, z, w) representing the corresponding
    2x2 transformation matrix with columns (x y) and (z w),
    such that a * x + b * y = c and a * z + b * w = 0.
    When a and b are nonnegative, it is guaranteed that
    c is also nonnegative"""
    if abs(b) > abs(a):
        x, y, z, w = 0, 1, 1, 0
        a, b = b, a
    else:
        x, y, z, w = 1, 0, 0, 1
    while b != 0:
        u = a // b
        a, b = b, a % b
        x, y, z, w = z, w, x - u * z, y - u * w
    return a, x, y, z, w
