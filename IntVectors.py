from itertools import product
import math
from collections import defaultdict
from typing import List, Optional, Iterator
import flint as fl
from Lattice import *

def int_seq(dim: int, signs: Optional[List[int]] = None, nonzero: bool = False, length: int = -1) -> Iterator[List[int]]:
    """Lists all integer sequences of length dim, with given restrictions on signs"""
    if signs is None:
        signs = [0] * dim
    
    v = [0] * dim
    if nonzero:
        head = 0
        width = 1
        v[0] = 1
    else:
        head = dim - 1
        width = 0
        
    n = length
    while n != 0:
        out = [(v[i] >> 1) - (v[i] & 1) * (v[i] | 1) for i in range(dim)]
        
        # Verify sign constraints
        if all(out[j] * signs[j] >= 0 for j in range(dim)):
            yield out
            n -= 1
            
        if head == dim - 1:
            width += 1
            v[head] = 0
            v[0] = width
            head = 0
        else:
            v[head + 1] += 1
            w = v[head]
            v[head] = 0
            v[0] = w - 1 if w > 0 else 0
            head = 0 if v[0] > 0 else head + 1


def int_seq_r(dim: int, max_rr: int = 10 ** 100) -> Iterator[tuple[int]]:
    """Lists all integer sequences of length dim, sorted by their squared length, up to max_rr"""
    vectors = defaultdict(set)
    vectors[0].add(tuple([0] * dim))
    distances = {0}
    while True:
        min_dist = max_rr + 1 if not distances else min(distances)
        if min_dist > max_rr:
            break
        v = vectors[min_dist].pop()
        if len(vectors[min_dist]) == 0:
            distances.remove(min_dist)
        for i in range(dim):
            dist = min_dist + 2 * abs(v[i]) + 1
            distances.add(dist)
            if v[i] == 0:
                new_v = {tuple(s if j == i else v[j] for j in range(dim)) for s in [-1, 1]}
                vectors[dist] |= new_v
            else:
                sign = 1 if v[i] > 0 else -1
                new_v = tuple(v[j] + (sign if j == i else 0) for j in range(dim))
                vectors[dist].add(new_v)
        yield v


def fincke_pohst_search(A: np.ndarray, b: np.array, lbound: float, ubound: float) -> Iterator[List[int]]:
    """
    Finds all integer vectors x such that lbound < (x + b)^T A (x + b) <= ubound.
    Uses Cholesky decomposition for O(sqrt(bound)^rank) performance.
    """
    rank = A.shape[0]
    # Cholesky decomposition: M = R^T R (R is upper triangular)
    # We add a tiny epsilon to the diagonal for numerical stability on float matrices
    R = np.linalg.cholesky(A + np.eye(rank) * 1e-10).T 
    # State arrays for the DFS
    x = [0] * rank
    dist = np.zeros(rank)
    # Start at the last coordinate (bottom of the tree)
    k = rank - 1
    # Compute the center and bounds for the current coordinate
    z = np.zeros(rank)
    x[k] = math.floor(math.sqrt(ubound / (R[k, k]**2)) - b[k])
    while True:
        # Calculate the partial distance
        d = dist[k] + R[k, k]**2 * (x[k] + b[k] - z[k])**2
        if d <= ubound:
            if k == 0:
                # We reached a leaf node (a complete valid vector)
                if d > lbound: # Check the lower bound
                    yield x
                elif x[k] > z[k]:
                     x[k] = math.floor(2 * z[k] - x[k]) + 1
                # Backtrack
                x[k] -= 1
            else:
                # Move down the tree
                k -= 1
                dist[k] = dist[k + 1] + R[k + 1, k + 1]**2 * (x[k + 1] + b[k + 1] - z[k + 1])**2
                # Update the center z for the new level
                z[k] = -sum(R[k, j] * (x[j] + b[j]) for j in range(k + 1, rank)) / R[k, k]
                # Set the upper bound for this coordinate
                x[k] = math.floor(math.sqrt((ubound - dist[k]) / (R[k, k]**2)) + z[k] - b[k])
        else:
            # Prune the branch and step back up
            k += 1
            if k == rank:
                break
            x[k] -= 1

class BatchGenerator:

    def __init__(self, dim: int, d: int = 3, symmetric: bool = True, max_size: Optional[int] = None):
        self.dim = dim
        self.d = d
        self.bsize = self.d ** dim
        if max_size and self.bsize > max_size:
            self.d = max(1, int(math.pow(max_size, 1 / dim)))
            self.bsize = self.d ** dim
        interval = range(-((self.d - 1) // 2), 1 + self.d // 2) if symmetric else range(self.d)
        self.block = list(product(interval, repeat=dim))

    def blocks(self) -> Iterator[List[List[int]]]:
        for v in int_seq_r(self.dim):
            base = tuple(v[i] * self.d for i in range(self.dim))
            b = [tuple(base[j] + self.block[i][j] for j in range(self.dim)) for i in range(self.bsize)]
            yield b

    def vectors(self) -> Iterator[List[int]]:
        for block in self.blocks():
            for v in block:
                yield v


class VectorSearch:

    def __init__(self, L: Lattice):
        self.L = L

    def roots(self):
        """Lists all roots of the lattice"""
        bl = BatchGenerator(self.L.rank, d = 10, max_size = 10 ** 8)
        for v in bl.vectors():
            if math.gcd(*v) != 1:
                continue
            if self.L.is_root(v):
                yield v
        
    def roots_neg(self):
        """Lists the negative roots of a Lorentzian lattice in the positive
        half-space defined by the product with the first basis element.
        The first basis element should be positive"""

        e0 = [1] + [0] * (self.L.rank - 1)
        a = int(self.L.square(e0))
        if a <= 0:
            raise ValueError("The lattice is not in a standard form")
        m = 2 * self.L.exp
        check = {}
        skip = set()

        def decompose(b, c):
            d = b ** 2 - a * c
            if d < 0:
                return []
            t1, t2 = math.ceil(-b / a), math.ceil((-b + math.sqrt(d)) / a)
            result = []
            for t in range(t1, t2 + 1):
                val = a * (t ** 2) + 2 * b * t + c
                if val < -m or val >= 0:
                    continue
                if m % val == 0:
                    result.append(t)
            return result
        
        bl = BatchGenerator(self.L.rank - 1, d = 10, max_size = 10 ** 8)
        for v in bl.vectors():
            e = [0] + list(v)
            b = int(self.L.product(e, e0))
            c = int(self.L.square(e))
            if (b, c) in skip:
                continue
            elif (b, c) not in check:
                t = decompose(b, c)
                if len(t) == 0:
                    skip.add((b, c))
                    continue
                check[(b, c)] = t
            for t in check[(b, c)]:
                u = [t] + list(v)
                if math.gcd(*u) != 1:
                    continue
                if self.L.is_root(u):
                    yield u