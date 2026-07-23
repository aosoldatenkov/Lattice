from __future__ import annotations
from Lattice import *
from Commons import *
from itertools import product
from collections import namedtuple
from typing import Dict, Tuple
from sympy.ntheory import factorint
from sympy.functions.combinatorial.numbers import legendre_symbol

ComPAdic = namedtuple('ComPAdic', ['sign', 'dim'])
Com2Adic = namedtuple('Com2Adic', ['type', 'tr', 'sign', 'dim'])

class Genus:

    def __init__(self, L: Lattice):
        primes = factorint(L.exp)
        self.sym = {p: self.symbol(L.A, p, primes[p]) for p in primes}
        self.signature = L.signature
        self.parity = L.parity

    def __eq__(self, other: Genus):
        if self.signature != other.signature or self.parity != other.parity:
            return False
        if sorted(self.sym.keys()) != sorted(other.sym.keys()):
            return False
        for p in self.sym.keys():
            a = self.sym[p]
            b = other.sym[p]
            if sorted(a.keys()) != sorted(b.keys()):
                return False
            if any([a[e].dim != b[e].dim for e in a.keys()]):
                return False
            signdiff = [e for e in a.keys() if a[e].sign != b[e].sign]
            if p > 2 and len(signdiff) > 0:
                return False
            if p == 2:
                if len(signdiff) % 2 != 0:
                    return False
                if any([a[e].type != b[e].type for e in a.keys()]):
                    return False
                typeII = [e for e in a.keys() if a[e].type == 'II'] + [e for e in range(max(a.keys()) + 2) if e not in a.keys()]
                for m in typeII:
                    lhs = sum([a[e].tr - b[e].tr for e in range(m) if e in a.keys()])
                    rhs = 4 * sum([min(x, m) for x in signdiff], start=0)
                    if (rhs - lhs) % 8 != 0:
                        return False
        return True

    def str(self, p = None):
        if p is None:
            s = "I_{"  if self.parity else "II_{"
            s += f"{self.signature[0]}, {self.signature[1]}" + "}("
            s += ":".join([self.str(p) for p in sorted(self.sym.keys())]) + ")"
            return s
        if p not in self.sym:
            return ""
        if p > 2:
            return "".join([f"{p ** e}^" + "{" + f"{self.sym[p][e].sign * self.sym[p][e].dim}" + "}" for e in sorted(self.sym[p].keys())])
        s = ""
        for e in sorted(self.sym[2].keys()):
            if self.sym[2][e].type == 'I':
                s += f"{2 ** e}_" + "{" + f"{self.sym[2][e].tr}" + "}^{" + f"{self.sym[2][e].sign * self.sym[2][e].dim}" + "}"
            else:
                s += f"{2 ** e}" + "^{" + f"{self.sym[2][e].sign * self.sym[2][e].dim}" + "}"
        return s

    @staticmethod
    def symbol(A: IMat, p: int, e: int) -> Dict[int, ComPAdic | Com2Adic]:
        
        M = A.copy()
        r = nrows(M)
        if r == 0:
            return {}
        if p == 2:
            q = p ** (e + 3)
        else:
            q = p ** (e + 1)
        scale = 0
        R = M % p
        while not any(R.flatten().tolist()):
            scale += 1
            if scale > e:
                raise ValueError("The exponent is too small or the matrix is singular")
            M = M // p
            R = M % p
        
        def p_invert(n, q):
            _, m, _, _, _ = euclid(n, q)
            if (m * n) % q != 1:
                raise ValueError("p-adic inversion failed")
            return m
        
        def sign(n, p):
            if p > 2:
                return legendre_symbol(n, p)
            return 1 if n % 8 in [1, 7] else -1
        
        def exch(M, i, j):
            x = M[i, :].copy()
            M[i, :] = M[j, :]
            M[j, :] = x
            x = M[:, i].copy()
            M[:, i] = M[:, j]
            M[:, j] = x

        def type_I(M, i0):
            exch(M, 0, i0)
            M %= q
            a = p_invert(M[0, 0], q)
            M[1:, 1:] -= a * (M[1:, 0:1] @ M[0:1, 1:])
            s = Genus.symbol(M[1:, 1:], p, e)
            s = {x + scale: s[x] for x in s.keys()}
            if p > 2:
                if scale in s:
                    s[scale] = ComPAdic(s[scale].sign * sign(M[0, 0], p), s[scale].dim + 1)
                else:
                    s[scale] = ComPAdic(sign(M[0, 0], p), 1)
            else:
                if scale in s and s[scale].type == 'I':
                    s[scale] = Com2Adic('I', (M[0, 0] + s[scale].tr) % 8, s[scale].sign * sign(M[0, 0], 2), s[scale].dim + 1)
                elif scale in s and s[scale].type == 'II':
                    s[scale] = Com2Adic('I', M[0, 0] % 8, s[scale].sign * sign(M[0, 0], 2), s[scale].dim + 1)
                else:
                    s[scale] = Com2Adic('I', M[0, 0] % 8, sign(M[0, 0], 2), 1)
            return s
        
        def type_II(M, i0, j0):
            exch(M, 0, i0)
            exch(M, 1, j0)
            M %= q
            det = M[0, 0] * M[1, 1] - M[0, 1] ** 2
            B = p_invert(det % q, q) * imat([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
            M[2:, 2:] -= M[2:, 0:2] @ B @ M[0:2, 2:]
            s = Genus.symbol(M[2:, 2:], p, e)
            s = {x + scale: s[x] for x in s.keys()}
            if scale in s and s[scale].type == 'I':
                s[scale] = Com2Adic('I', s[scale].tr, sign(det, 2) * s[scale].sign, s[scale].dim + 2)
            elif scale in s and s[scale].type == 'II':
                s[scale] = Com2Adic('II', 0, sign(det, 2) * s[scale].sign, s[scale].dim + 2)
            else:
                s[scale] = Com2Adic('II', 0, sign(det, 2), 2)
            return s
        
        if r == 1:
            return {scale: ComPAdic(sign(M[0, 0], p), 1)} if p > 2 else {scale: Com2Adic('I', M[0, 0] % 8, sign(M[0, 0], 2), 1)}    
        i0 = min([r] + [i for i in range(r) if R[i, i] != 0])
        if i0 < r:
            return type_I(M, i0)
        i0, j0 = min([(i, j) for i, j in product(range(r), repeat=2) if R[i, j] != 0])
        if p > 2:
            M[i0, :] += M[j0, :]
            M[:, i0] += M[:, j0]
            return type_I(M, i0)
        else:
            return type_II(M, i0, j0)
        
