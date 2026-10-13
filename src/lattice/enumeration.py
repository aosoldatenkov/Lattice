from itertools import product
import math
from collections import defaultdict
from typing import Any, List, Optional, Iterator
import numpy as np

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
                elif x[k] + b[k] > z[k]:
                    x[k] = min(x[k], math.ceil(2 * (z[k] - b[k]) - x[k]) + 1)
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


class FPSearchPy:
    """
    Finds all integer vectors x such that lbound < (x + b)^T A (x + b) <= ubound.
    Uses Cholesky decomposition for O(sqrt(bound)^rank) performance.
    """

    def __init__(self, A: List[List[Any]], b: list[Any], lbound: float, ubound: float):
        self.rank = len(A)
        self.A = np.array(A, dtype = float)
        self.b = np.array(b, dtype = float)
        self.lbound = lbound
        self.ubound = ubound
        self.R = np.linalg.cholesky(self.A + np.eye(self.rank) * 1e-10).T
        self.x = [0] * self.rank
        self.dist = np.zeros(self.rank)
        self.k = self.rank - 1
        self.z = np.zeros(self.rank)
        self.x[self.k] = math.floor(math.sqrt(ubound / (self.R[self.k, self.k] ** 2)) - self.b[self.k])
        self._iter = self.iterate()

    def iterate(self):
        if self.k == self.rank:
            return
        while True:
            # Calculate the partial distance
            d = self.dist[self.k] + self.R[self.k, self.k]**2 * (self.x[self.k] + self.b[self.k] - self.z[self.k])**2
            if d <= self.ubound:
                if self.k == 0:
                    # We reached a leaf node (a complete valid vector)
                    if d > self.lbound: # Check the lower bound
                        yield self.x
                    elif self.x[self.k] + self.b[self.k] > self.z[self.k]:
                        self.x[self.k] = min(self.x[self.k], math.ceil(2 * (self.z[self.k] - self.b[self.k]) - self.x[self.k]) + 1)
                    self.x[self.k] -= 1
                else:
                    # Move down the tree
                    self.k -= 1
                    self.dist[self.k] = self.dist[self.k + 1] + self.R[self.k + 1, self.k + 1]**2 * (self.x[self.k + 1] + self.b[self.k + 1] - self.z[self.k + 1])**2
                    # Update the center z for the new level
                    self.z[self.k] = -sum(self.R[self.k, j] * (self.x[j] + self.b[j]) for j in range(self.k + 1, self.rank)) / self.R[self.k, self.k]
                    # Set the upper bound for this coordinate
                    self.x[self.k] = math.floor(math.sqrt((self.ubound - self.dist[self.k]) / (self.R[self.k, self.k]**2)) + self.z[self.k] - self.b[self.k])
            else:
                # Prune the branch and step back up
                self.k += 1
                if self.k == self.rank:
                    break
                self.x[self.k] -= 1

    def batch_search(self, size):
        vecs = []
        for v in self._iter:
            vecs.append(v.copy())
            if len(vecs) >= size:
                break
        return vecs

    def search_all(self):
        return [v.copy() for v in self._iter]

    def exhausted(self):
        return self.k == self.rank


try:
    from lattice.fp_search_cpp import FPSearch
except ImportError:  # the C++ extension is not built: fall back to the reference implementation
    FPSearch = FPSearchPy
