#!/usr/bin/env python3

from BinLattice import BinLattice, int_seq
from collections import defaultdict
import math
from itertools import product

ll = defaultdict(list)
max_d = 1000
max_a = int(2. * math.sqrt(max_d) / math.sqrt(3)) + 1
# Listing the indefinite lattices that represent zero
max_h = int(math.sqrt(max_d)) + 1
for h, b in [(h, b) for h in range(1, max_h) for b in range(2 * h + 1)]:
    lat = BinLattice(0, b, h)
    if all(lat.is_isomorphic(l) == False for l in ll[lat.disc]):
        ll[lat.disc].append(lat)
# Listing the positive and the rest of the indefinite lattices
for a, h in [(a, h) for a in range(-max_a, max_a) for h in range(abs(a)) if a != 0]:
    max_b = int((max_d + h * h) / abs(a)) + 1
    for b in range(-max_b, max_b):
        if abs(a * b - h * h) > max_d:
            continue
        lat = BinLattice(a, b, h)
        if lat.signature not in [(2, 0), (1, 1)]:
            continue
        if all(lat.is_isomorphic(l) == False for l in ll[lat.disc]):
            ll[lat.disc].append(lat)

for k in sorted(ll.keys()):
    print(f'{k:3}: ', ', '.join(f'({l.can.a}, {l.can.b}, {l.can.h})' for l in ll[k]))

print()
print(sum(len(ll[d]) for d in ll.keys()), 'isomorphism classes of lattices listed')

# for d in sorted(ll.keys(), reverse=True):
#     for l in ll[d]:
#         if len(l.zero) == 0 and not l.list_roots():
#             print(f"Found a lattice with no roots: a={l.can.a}, b={l.can.b}, h={l.can.h}, disc={l.disc}")

# for a, b, h in int_seq(3, signs = [0, 0, 1], length = 1000000):
#     if abs(a * b - h * h) > max_d:
#         continue
#     lat = BinLattice(a, b, h)
#     if lat.signature not in [(2, 0), (1, 1)]:
#         continue
#     if all(lat.is_isomorphic(l) == False for l in ll[lat.disc]):
#         print('Lattice missing:')
#         print(lat.info())