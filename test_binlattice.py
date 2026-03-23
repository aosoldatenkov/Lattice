#!/usr/bin/env python3

import math
import unittest
from BinLattice import BinBasis, BinLattice, int_seq

class TestIntSeq(unittest.TestCase):
    def test_int_seq_generation(self):
        """Test the bitwise integer sequence generator."""
        seq = list(int_seq(dim=2, signs=[1, 1], nonzero=True, length=3))
        # Expected to generate valid positive sequence components
        self.assertEqual(len(seq), 3)
        self.assertTrue(all(v[0] >= 0 and v[1] >= 0 for v in seq))
        # Test if the generation lists all integer pairs in a fixed range, each one exactly once
        pairs = {(i, j): 0 for i in range(-100, 100) for j in range(-100, 100)}
        for s in int_seq(2, length = len(pairs.keys()) * 10):
            if (s[0], s[1]) in pairs:
                pairs[(s[0], s[1])] += 1
        self.assertTrue(all(pairs[k] == 1 for k in pairs))

class TestBinBasis(unittest.TestCase):
    def setUp(self):
        self.basis = BinBasis(e=(1, 0), f=(0, 1), a=1, b=2, h=3)

    def test_flip(self):
        flipped = self.basis.flip()
        self.assertEqual(flipped.e, (1, 0))
        self.assertEqual(flipped.f, (0, -1))
        self.assertEqual(flipped.h, -3)

    def test_swap(self):
        swapped = self.basis.swap()
        self.assertEqual(swapped.e, (0, 1))
        self.assertEqual(swapped.f, (1, 0))
        self.assertEqual(swapped.a, 2)
        self.assertEqual(swapped.b, 1)

    def test_negate(self):
        negated = self.basis.negate()
        self.assertEqual(negated.a, -1)
        self.assertEqual(negated.b, -2)
        self.assertEqual(negated.h, -3)

    def test_sum(self):
        self.assertEqual(self.basis.sum(), (1, 1))

class TestBinLattice(unittest.TestCase):
    def test_zero_lattice(self):
        """Test a completely degenerate zero lattice."""
        lattice = BinLattice(0, 0, 0)
        self.assertEqual(lattice.signature, (0, 0))
        self.assertEqual(lattice.disc, 0)

    def test_definite_lattice_A1_A1(self):
        """Test a positive definite lattice: x^2 + y^2."""
        # a=1, b=1, h=0 -> disc = 1 > 0
        lattice = BinLattice(1, 1, 0)
        self.assertEqual(lattice.signature, (2, 0))
        self.assertEqual(lattice.disc, 1)
        self.assertIsNotNone(lattice.can)
        self.assertEqual(lattice.min_positive(), 1)
        
    def test_definite_lattice_A2(self):
        """Test the A2 root lattice equivalent form: x^2 - xy + y^2."""
        # To get the norm x^2 - xy + y^2, we need a=1, b=1.
        # However, the code defines prod as a*u0*v0 + b*u1*v1 + h*(u0*v1 + v0*u1).
        # So 2h represents the cross term. If h is restricted to integers, 
        # let's test a=2, b=2, h=-1 (disc = 3).
        lattice = BinLattice(2, 2, -1)
        self.assertEqual(lattice.signature, (2, 0))
        self.assertEqual(lattice.disc, 3)
        self.assertTrue(lattice.is_root((1, 0)))

    def test_hyperbolic_plane_U(self):
        """Test the indefinite hyperbolic plane U: 2xy."""
        # a=0, b=0, h=1 -> disc = -1 < 0
        lattice = BinLattice(0, 0, 1)
        self.assertEqual(lattice.signature, (1, 1))
        self.assertEqual(lattice.disc, -1)
        
        # The river should be empty because the basis vectors are isotropic (a=0, b=0)
        self.assertEqual(len(lattice.river), 0)
        
        # U represents zero
        self.assertGreater(len(lattice.zero), 0)
        self.assertIn((1, 0), lattice.zero) 
        self.assertIn((0, 1), lattice.zero)

    def test_indefinite_lattice_general(self):
        """Test an indefinite lattice: x^2 + 4xy + y^2."""
        # a=1, b=1, h=2 -> disc = 1 - 4 = -3 < 0
        lattice = BinLattice(1, 1, 2)
        self.assertEqual(lattice.signature, (1, 1))
        self.assertEqual(lattice.disc, -3)
        self.assertIsNotNone(lattice.can)

    def test_isomorphic_lattices(self):
        """Test the isomorphism check between two equivalent representations."""
        lattice1 = BinLattice(1, 1, 0)  # x^2 + y^2
        lattice2 = BinLattice(1, 1, 0)
        self.assertTrue(lattice1.is_isomorphic(lattice2))
        
        lattice3 = BinLattice(2, 2, 0)  # 2x^2 + 2y^2
        self.assertFalse(lattice1.is_isomorphic(lattice3))

    def test_list_roots(self):
        """Test root generation for a known definite lattice."""
        lattice = BinLattice(1, 1, 0)
        roots = lattice.list_roots()
        
        # For x^2 + y^2, vectors squared to 1 are (1,0), (0,1), (-1,0), (0,-1)
        # Due to canonical normalization, we expect positive orientations in the keys
        self.assertIn(1, roots)
        self.assertTrue(len(roots[1]) > 0)

    def test_lattice_consistency(self):
        """Test the BinLattice class for a large set of inputs"""
        lattices = [BinLattice(a, b, h) for a, b, h in int_seq(3, signs = [0, 0, 1], nonzero = True, length = 10000)]
        self.assertTrue(all(l.is_isomorphic(BinLattice(l.can.a, l.can.b, l.can.h)) for l in lattices))

    def test_values_consistency(self):
        """Test that BinLattice.list_positive() and BinLattice.list_negative() list all values up to a certain limit."""
        lat = BinLattice(2, 3, 5)
        vals1 = list(lat.list_positive(100).keys())
        vals2 = [lat.sqr(s) for s in int_seq(2, signs=[1, 0], nonzero=True, length=10000) if abs(math.gcd(*s)) == 1]
        self.assertTrue(all(v in vals1 for v in [v for v in vals2 if 100 >= v > 0]))
        vals1 = list(lat.list_negative(100).keys())
        self.assertTrue(all(v in vals1 for v in [v for v in vals2 if -100 <= v < 0]))

if __name__ == '__main__':
    unittest.main()