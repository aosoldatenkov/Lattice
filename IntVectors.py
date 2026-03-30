from itertools import product
import math
from collections import defaultdict
from typing import List, Optional, Iterator

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