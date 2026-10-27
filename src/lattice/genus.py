"""Genus symbols of integral lattices in the notation of Conway-Sloane.

Reference: [CS] J. H. Conway, N. J. A. Sloane, Sphere Packings, Lattices and Groups, Chapter 15.

Two lattices are in the same genus iff they have the same signature and are isomorphic over
Z_p for every prime p. For p not dividing the discriminant, the p-adic form is unimodular and
determined by the rank, the determinant and (for p = 2) the parity, so only the primes p
dividing the discriminant are stored.

Over Z_p, a form has a Jordan decomposition f = ⊕_q f_q, where f_q = q * (unimodular form)
is the constituent of scale q = p^e. The p-adic symbol records for each scale:
  * p odd: the dimension n_q and the sign ε_q = (u_q / p), the Legendre symbol of the unit part
    u_q = det(f_q) / q^{n_q};
  * p = 2: in addition, the type (I if f_q / q is odd, II if it is even) and, for type I,
    the oddity t_q (the trace of a diagonalization of f_q / q, taken mod 8);
    the sign is ε_q = (2 / u_q), which is +1 iff u_q = ±1 mod 8.
For p odd, (n_q, ε_q) are invariants of the form. For p = 2, the dimensions and types are
invariants, but the signs and oddities depend on the choice of the decomposition; see
Genus.__eq__ for the comparison of 2-adic symbols.
"""
from __future__ import annotations
from itertools import product
from collections import namedtuple
from typing import Dict
from sympy.ntheory import factorint
from sympy.functions.combinatorial.numbers import legendre_symbol
from lattice._matrix import IMat, imat, nrows, euclid
from lattice.lattice import Lattice

# A Jordan constituent for p odd: the sign ε_q = ±1 and the dimension n_q
ComPAdic = namedtuple('ComPAdic', ['sign', 'dim'])
# A Jordan constituent for p = 2: the type 'I' or 'II', the oddity t_q mod 8 (0 for type II),
# the sign ε_q = ±1 and the dimension n_q
Com2Adic = namedtuple('Com2Adic', ['type', 'tr', 'sign', 'dim'])


class Genus:
    """The genus of an integral lattice.

    Attributes:
        sym: a dictionary {p: {e: constituent}} over the primes p dividing the discriminant
            (equivalently, the exponent of the discriminant group), where e runs over the
            exponents of the scales p^e of the nonzero Jordan constituents, starting from
            e = 0 for the unimodular part; the constituents are ComPAdic for p odd and
            Com2Adic for p = 2.
        signature: the signature (n_+, n_-).
        parity: True if the lattice is odd, False if it is even.
    """

    def __init__(self, L: Lattice):
        primes = factorint(L.exp)
        # The largest scale of a Jordan constituent is the p-part of the exponent of L^∨/L
        self.sym = {p: self.symbol(L.A, p, primes[p]) for p in primes}
        self.signature = L.signature
        self.parity = L.parity

    def __eq__(self, other: Genus) -> bool:
        """Checks whether the genera coincide.

        For p odd, the p-adic forms are isomorphic iff the dimensions and the signs agree
        for all scales. For p = 2, the comparison does not use the canonical 2-adic symbol
        of [CS, Ch. 15, Sec. 7.6] (its classical definition has problems pointed out by
        Allcock), but the criterion for the equivalence of two Jordan decompositions
        f = ⊕ f_{2^e}, f' = ⊕ f'_{2^e} of [CS, Ch. 15, Thm. 10]. In the form implemented
        here, if the constituents of f and f' have the same dimensions and types, then
        f and f' are isomorphic iff
          (a) prod_e ε_e = prod_e ε'_e, and
          (b) for every m such that f_{2^m} is of type II or zero (a zero constituent
              counts as type II), the quantity
                  O_m(f) = sum_{e < m} t_e + 4 * sum_{e: ε_e = -1} min(e, m)   (mod 8)
              coincides with O_m(f').
        Condition (a) holds automatically for lattices of the same determinant. For m
        larger than all scales, O_m(f) = sum_e t_e + 4 * #{e odd: ε_e = -1} is the oddity
        of f, and the conditions for larger m coincide with that for m = max(e) + 1.
        If f_{2^m} and f_{2^{m+1}} are both of type II, then (b) for m and m + 1 says that
        the product of the signs ε_e for e > m is the same for f and f' (signs cannot walk
        between different trains).

        The quantities in (a) and (b) are invariant under the moves relating Jordan
        decompositions of the same form [CS, Ch. 15, Sec. 7.5]: oddity fusion (only the total
        oddity of a compartment, a maximal run of type I constituents of consecutive scales,
        is an invariant; a compartment never contains m, so it lies entirely below or above
        m), and sign walking (changing the signs of f_{2^e} and f_{2^{e+1}}, one of which is
        of type I, together with the total oddity of a compartment containing one of them
        by 4; this changes O_m(f) by 4 (min(e, m) + min(e + 1, m)) + 4 [the compartment lies
        below m], which is 0 mod 8 in all cases)."""
        if not isinstance(other, Genus):
            return NotImplemented
        if self.signature != other.signature or self.parity != other.parity:
            return False
        if sorted(self.sym.keys()) != sorted(other.sym.keys()):
            return False
        for p in self.sym.keys():
            a = self.sym[p]
            b = other.sym[p]
            # The scales and dimensions of the constituents are invariants for all p
            if sorted(a.keys()) != sorted(b.keys()):
                return False
            if any([a[e].dim != b[e].dim for e in a.keys()]):
                return False
            signdiff = [e for e in a.keys() if a[e].sign != b[e].sign]
            if p > 2 and len(signdiff) > 0:
                return False
            if p == 2:
                # Condition (a)
                if len(signdiff) % 2 != 0:
                    return False
                # The types are invariants
                if any([a[e].type != b[e].type for e in a.keys()]):
                    return False
                # Condition (b) for the scales m of type II, including the zero constituents
                # up to max(e) + 1. The sign terms enter with a factor 4, so the difference
                # O_m(f) - O_m(f') only depends on the parity of the sum over the scales
                # with different signs.
                typeII = [e for e in a.keys() if a[e].type == 'II'] + [e for e in range(max(a.keys()) + 2) if e not in a.keys()]
                for m in typeII:
                    lhs = sum([a[e].tr - b[e].tr for e in range(m) if e in a.keys()])
                    rhs = 4 * sum([min(x, m) for x in signdiff], start=0)
                    if (rhs - lhs) % 8 != 0:
                        return False
        return True

    def str(self, p = None):
        """Returns the genus symbol in LaTeX notation, e.g. II_{1, 9} for U + E8(-1) (no brackets for a
        unimodular lattice), or I_{2, 1}(1_{4}^{-2}4_{7}^{1}:1^{-2}3^{1}) for diag(1, -4, 3), or the p-adic symbol if p is given
        (the empty string if p does not divide the discriminant). Each constituent is written
        as q^{ε n} for p odd, q_{t}^{ε n} for p = 2 and type I, and q^{ε n} for p = 2 and
        type II, including the unimodular constituent q = 1. The 2-adic part is written for
        the Jordan decomposition found by symbol(), not in a canonical form, so lattices of
        the same genus can have different strings."""
        if p is None:
            s = "I_{"  if self.parity else "II_{"
            s += f"{self.signature[0]}, {self.signature[1]}" + "}("
            s += ":".join([self.str(p) for p in sorted(self.sym.keys())]) + ")"
            return s if len(self.sym.keys()) > 0 else s[:-2]
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
        """Computes the p-adic symbol of the nondegenerate form with Gram matrix A, given
        that the scales of its Jordan constituents are at most p^e (e.g. p^e is the p-part
        of the exponent of the discriminant group). Returns the dictionary {k: constituent}
        over the exponents k of the scales p^k of the nonzero constituents.

        The Jordan decomposition is found recursively: after dividing A by the largest
        power p^s dividing all its entries, a 1 x 1 block with a unit entry (type I), or for
        p = 2 an even 2 x 2 block with a unit determinant (type II), is split off by taking
        the Schur complement, and the constituent of scale p^s of the result is updated.
        The computations are done modulo q = p^{e+1} (p odd) or 2^{e+3}: a form is determined
        up to isomorphism by its Gram matrix modulo q. Each division by p loses one p-adic
        digit, but the total scale never exceeds p^e, so at least one (p odd) or three (p = 2)
        digits remain for the unit parts, enough to determine their square classes."""
        M = A.copy()
        r = nrows(M)
        if r == 0:
            return {}
        if p == 2:
            q = p ** (e + 3)
        else:
            q = p ** (e + 1)
        # The scale of the next constituent, relative to the scale of A
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
            """The sign of a p-adic unit n: the Legendre symbol (n / p) for p odd,
            and the Kronecker symbol (2 / n) for p = 2."""
            if p > 2:
                return legendre_symbol(n, p)
            return 1 if n % 8 in [1, 7] else -1

        def exch(M, i, j):
            """Swaps the basis vectors i and j."""
            x = M[i, :].copy()
            M[i, :] = M[j, :]
            M[j, :] = x
            x = M[:, i].copy()
            M[:, i] = M[:, j]
            M[:, j] = x

        def type_I(M, i0):
            """Splits off the 1 x 1 block M[i0, i0], a p-adic unit."""
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
                # Adding an odd 1 x 1 block makes the constituent odd
                if scale in s and s[scale].type == 'I':
                    s[scale] = Com2Adic('I', (M[0, 0] + s[scale].tr) % 8, s[scale].sign * sign(M[0, 0], 2), s[scale].dim + 1)
                elif scale in s and s[scale].type == 'II':
                    s[scale] = Com2Adic('I', M[0, 0] % 8, s[scale].sign * sign(M[0, 0], 2), s[scale].dim + 1)
                else:
                    s[scale] = Com2Adic('I', M[0, 0] % 8, sign(M[0, 0], 2), 1)
            return s

        def type_II(M, i0, j0):
            """For p = 2, splits off the 2 x 2 block on the basis vectors i0, j0, with even
            diagonal and odd off-diagonal entries (so its determinant is a 2-adic unit)."""
            exch(M, 0, i0)
            exch(M, 1, j0)
            M %= q
            det = M[0, 0] * M[1, 1] - M[0, 1] ** 2
            B = p_invert(det % q, q) * imat([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
            M[2:, 2:] -= M[2:, 0:2] @ B @ M[0:2, 2:]
            s = Genus.symbol(M[2:, 2:], p, e)
            s = {x + scale: s[x] for x in s.keys()}
            # An even block does not change the type or the oddity of the constituent
            if scale in s and s[scale].type == 'I':
                s[scale] = Com2Adic('I', s[scale].tr, sign(det, 2) * s[scale].sign, s[scale].dim + 2)
            elif scale in s and s[scale].type == 'II':
                s[scale] = Com2Adic('II', 0, sign(det, 2) * s[scale].sign, s[scale].dim + 2)
            else:
                s[scale] = Com2Adic('II', 0, sign(det, 2), 2)
            return s

        if r == 1:
            return {scale: ComPAdic(sign(M[0, 0], p), 1)} if p > 2 else {scale: Com2Adic('I', M[0, 0] % 8, sign(M[0, 0], 2), 1)}
        # A diagonal entry that is a unit gives a 1 x 1 block
        i0 = min([r] + [i for i in range(r) if R[i, i] != 0])
        if i0 < r:
            return type_I(M, i0)
        # All diagonal entries are divisible by p, so some off-diagonal entry M[i0, j0]
        # with i0 < j0 is a unit
        i0, j0 = min([(i, j) for i, j in product(range(r), repeat=2) if R[i, j] != 0])
        if p > 2:
            # Replacing the basis vector i0 by the sum of the vectors i0 and j0 gives the
            # diagonal entry M[i0, i0] + 2 M[i0, j0] + M[j0, j0] = 2 M[i0, j0], a unit
            M[i0, :] += M[j0, :]
            M[:, i0] += M[:, j0]
            return type_I(M, i0)
        else:
            return type_II(M, i0, j0)
