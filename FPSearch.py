import math
from typing import List, Any
import numpy as np

class FPSearch:
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
        for v in self.iterate():
            vecs.append(v.copy())
            if len(vecs) >= size:
                break
        return vecs
