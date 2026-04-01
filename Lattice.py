from __future__ import annotations
import math
import numpy as np
from hsnf import smith_normal_form
import flint as fl
from typing import List, Tuple

class Lattice:
    def __init__(self, rank: int, prod: List[List[int]]):
        self.rank = rank
        self.A = fl.fmpz_mat(prod)
        
        if self.A.det() == 0:
            raise ValueError("Gram matrix must be non-degenerate.")
        if self.A.nrows() != self.rank or self.A.ncols() != self.rank:
            raise ValueError("Gram matrix dimensions must match the specified rank.")
        if self.A.transpose() != self.A:
            raise ValueError("Gram matrix must be symmetric.")
            
        self._compute_exponent()
        self._init_basis()
        self._snf_L_inv = None
        self._snf_D = None
        self._snf_R = None
        
    def _compute_exponent(self) -> None:
        B = self.A.snf()
        self.dgroup = [int(B[i, i]) for i in range(self.rank)]
        self.disc = self.A.det()
        self.exp = max(self.dgroup)

    def _compute_snf(self) -> None:
        if self._snf_L_inv is not None:
            return
        M = np.array(self.A.tolist(), dtype=object)
        D, L, R = smith_normal_form(M)
        L_inv, _denom = fl.fmpz_mat(L.tolist()).inv().numer_denom()
        self._snf_L_inv = L_inv
        self._snf_D = fl.fmpz_mat(D.tolist())
        self._snf_R = fl.fmpz_mat(R.tolist())

    def _init_basis(self) -> None:
        # Convert to float64 safely for np.linalg
        A_float = np.array(self.A.tolist(), dtype=np.float64)
        eival, eivec = np.linalg.eigh(A_float)
        
        # Sort descending to group positive eigenvalues first
        idx = np.argsort(eival)[::-1]
        eival = eival[idx]
        eivec = eivec[:, idx]
        
        pos_count = np.sum(eival > 0)
        self.signature = (int(pos_count), self.rank - int(pos_count))
        
        eival_abs = np.sqrt(np.abs(eival))
        self.norm_basis = eivec / eival_abs

    def __add__(self, other: Lattice) -> Lattice:
        """Computes the orthogonal direct sum of two lattices."""
        A_list = self.A.tolist()
        B_list = other.A.tolist()
        # Build block diagonal matrix
        new_prod = [row + [0] * other.rank for row in A_list] + \
                   [[0] * self.rank + row for row in B_list]
        return Lattice(self.rank + other.rank, new_prod)
    
    def __mul__(self, other: int) -> Lattice:
        """Creates an orthogonal direct sum of 'other' copies of the lattice."""
        if not isinstance(other, int) or other <= 0:
            raise ValueError("Multiplier must be a positive integer.")
        A_list = self.A.tolist()
        new_prod = []
        for i in range(other):
            new_prod += [[0] * (self.rank * i) + row + [0] * (self.rank * (other - i - 1)) for row in A_list]
        return Lattice(self.rank * other, new_prod)
    
    def __rmul__(self, other: int) -> Lattice:
        return self.__mul__(other)
    
    def __call__(self, d: int) -> Lattice:
        """Scales the quadratic form by a constant d."""
        A_scaled = (self.A * d).tolist()
        return Lattice(self.rank, A_scaled)

    def info(self) -> str:
        if self.signature == (0, 0):
            return 'Zero lattice'
        parity = 'Even' if all(self.A[i, i] % 2 == 0 for i in range(self.rank)) else 'Odd'
        lines = [
            f"{parity} lattice of signature {self.signature}, discriminant {self.disc} and exponent {self.exp}",
            f"Discriminant group: {self.dgroup}"
        ]
        return '\n'.join(lines) + '\n'
    
    def _lll_indefinite(self) -> None:
        """
        Applies LLL reduction to an indefinite lattice
        using a positive-definite majorant metric.
        """
        # 1. Convert to float for spectral decomposition
        A_float = np.array(self.A.tolist(), dtype=np.float64)
        
        # 2. Build the positive-definite majorant (M)
        eival, eivec = np.linalg.eigh(A_float)
        M_float = eivec @ np.diag(np.abs(eival)) @ eivec.T
        
        # 3. Scale up to preserve precision in integer arithmetic
        # We scale M by 10^12, meaning the Cholesky basis scales by 10^6
        scale = 1e12 
        eps = np.eye(self.rank) * 1e-10 # Ensure numerical strict positive-definiteness
        
        # 4. Get the "basis" of the Majorant via Cholesky decomposition
        # np.linalg.cholesky returns a lower triangular matrix where rows are basis vectors
        L_float = np.linalg.cholesky(M_float * scale + eps)
        
        # 5. Round to integer basis matrix and convert to FLINT
        L_int_list = np.round(L_float).astype(int).tolist()
        L_fmpz = fl.fmpz_mat(L_int_list)
        
        # 6. Apply LLL to this basis to extract the unimodular transformation U
        # FLINT's transform=True returns a tuple: (reduced_matrix, transformation_matrix)
        _, U = L_fmpz.lll(transform=True)
        
        # 7. Apply the transformation to the ORIGINAL exact indefinite Gram matrix
        # The new Gram matrix is U * A * U^T
        B = U * self.A * U.transpose()
        return B.tolist()
        
    def lll(self) -> List[List[int]]:
        """Returns the LLL-reduced Gram matrix for positive-definite lattices."""
        match self.signature:
            case (self.rank, 0):
                return self.A.lll(rep='gram').tolist()
            case (0, self.rank):
                return (-self.A).lll(rep='gram').tolist()
            case _:
                return self._lll_indefinite()
        
    def product(self, u: List[int], v: List[int]) -> int:
        return (fl.fmpz_mat(1, self.rank, u) * self.A * fl.fmpz_mat(self.rank, 1, v))[0, 0]
    
    def batch_prod(self, u: List[List[int]], v: List[List[int]]) -> List[List[int]]:
        return (fl.fmpz_mat(u) * self.A * fl.fmpz_mat(v).transpose()).tolist()

    def square(self, u: List[int]) -> int:
        return self.product(u, u)
    
    def is_root(self, u: List[int]) -> bool:
        """Checks if u is a root, i.e. if the reflection in u preserves the lattice."""
        norm_sq = self.square(u)
        if norm_sq == 0:
            return False
        umat = fl.fmpz_mat(1, self.rank, u)
        v = umat * self.A
        d = 2 * math.gcd(*v.tolist()[0])
        return d % norm_sq == 0
        
    def saturate(self, gens: List[List[int]]) -> List[List[int]]:
        """Given a set of generators for a sublattice, returns a basis for the saturated sublattice spanned by the generators."""
        H, T = fl.fmpz_mat(gens).transpose().hnf(transform=True)
        B, _denom = T.inv().numer_denom() # T in GL(n, Z), so denom is 1
        # Finds number of rows in H that are not entirely zero
        H_list = H.tolist()
        H_len = min(len(gens), self.rank)
        num_independent = sum(1 for i in range(H_len) if any(H_list[i][j] != 0 for j in range(H_len)))
        B_transposed = B.transpose().tolist()
        return [B_transposed[i] for i in range(num_independent)]
    
    def index(self, gens: List[List[int]]) -> int:
        """Given a set of generators for a sublattice, returns the index of the sublattice spanned by the generators."""
        S = fl.fmpz_mat(gens).snf()
        return math.prod([S[i, i] for i in range(len(gens)) if S[i, i] != 0])
    
    def complement(self, gens: List[List[int]]) -> List[List[int]]:
        """Given a set of generators for a sublattice, returns a basis for its orthogonal complement."""
        G = fl.fmpz_mat(gens) * self.A
        K, n = G.nullspace()
        return self.saturate(K.transpose().tolist()[:n])
    
    def dual_vec(self, u: List[int]) -> Tuple[List[int], int]:
        """Given a vector u, finds its divisibility d in the dual lattice and a vector v, such that (u, v) = d.
        Returns the pair (v, d)."""
        B = self.A * fl.fmpz_mat(self.rank, 1, u)
        H, T = B.hnf(transform=True)
        return T.tolist()[0], H[0, 0]
    
    def make_primitive(self) -> Lattice:
        d = math.gcd(*self.dgroup)
        return Lattice(self.rank, [[a // d for a in row] for row in self.A.tolist()]) if d > 1 else self
    
    def clear_squares(self) -> Lattice:
        """Returns the square-free overlattice of the lattice"""
        def max_square(n: int) -> int:
            s = 1
            for k in range(1, int(math.sqrt(n)) + 1):
                if n % (k * k) == 0:
                    s = k
            return s
        self._compute_snf()
        D0 = fl.fmpz_mat(self._snf_D)
        for i in range(self.rank):
            D0[i, i] = max_square(self._snf_D[i, i])
        new_A, _ = (D0.inv() * self._snf_R.transpose() * self.A * self._snf_R * D0.inv()).numer_denom()
        return Lattice(self.rank, new_A.tolist())
    
    def make_ssf(self) -> Lattice:
        """Returns the associated strongly square-free lattice"""
        factor = 1
        for i in range(self.rank // 2 + 1):
            factor *= (self.dgroup[i] // factor)
        return self(factor).clear_squares()