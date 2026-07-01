import math
import numpy as np
import sympy as sp
import flint as fl
from typing import List, Tuple

IMat = np.ndarray

def imat(data, **kwargs):
    return np.array(data, dtype=object, **kwargs)

def imat_zero(nlines, ncols, **kwargs):
    return np.zeros((nlines, ncols), dtype=object, **kwargs)

def imat_diag(data, **kwargs):
    n = len(data)
    return np.array([[0] * i + [data[i]] + [0] * (n - i - 1) for i in range(n)], dtype=object, **kwargs)

def flz2imat(m, **kwargs):
    mlist = m.tolist()
    return np.array(mlist if len(mlist) > 1 else mlist[0], dtype=object, **kwargs)

def flq2imat(m, **kwargs):
    mz, d = m.numer_denom()
    mlist = mz.tolist()
    return np.array(mlist if len(mlist) > 1 else mlist[0], dtype=object, **kwargs), d

def imat2flz(m):
    if type(m) is not IMat:
        m = imat(m)
    shape = m.shape[:2] if len(m.shape) > 1 else (1, m.shape[0])
    return fl.fmpz_mat(*shape, m.flatten())

def nrows(m: List[int] | IMat) -> int:
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
