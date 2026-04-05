from __future__ import annotations
import math
import numpy as np
import sympy as sp
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
        # L * M * R = D
        D, L, R = smith_normal_form(M)
        L_inv, _denom = fl.fmpz_mat(L.tolist()).inv().numer_denom()
        self._snf_L_inv = L_inv
        self._snf_D = fl.fmpz_mat(D.tolist())
        self._snf_R = fl.fmpz_mat(R.tolist())

    def _init_basis(self) -> None:
        """Uses sympy to compute the signature of the lattice via a symbolic computation
        to avoid numerical instability issues with np.linalg.eigh."""
        A_sp = sp.Matrix(self.A.tolist())
        eival = A_sp.eigenvals()
        # Sympy does not always correctly handle inequalities with symbolic expressions,
        # so we use a numerical approximation as a fallback in those cases.
        try:
            n_pos = sum(eival[a] for a in eival if a.as_real_imag()[0] > 0)
        except:
            n_pos = sum(eival[a] for a in eival if sp.N(a, 50).as_real_imag()[0] > 0)
        self.signature = (n_pos, self.rank - n_pos)
        self.parity = any(self.A[i, i] % 2 for i in range(self.rank))
        # # Convert to float64 safely for np.linalg
        # A_float = np.array(self.A.tolist(), dtype=np.float64)
        # eival, eivec = np.linalg.eigh(A_float)
        
        # # Sort descending to group positive eigenvalues first
        # idx = np.argsort(eival)[::-1]
        # eival = eival[idx]
        # eivec = eivec[:, idx]
        
        # pos_count = np.sum(eival > 0)
        # self.signature = (int(pos_count), self.rank - int(pos_count))
        
        # eival_abs = np.sqrt(np.abs(eival))
        # self.norm_basis = eivec / eival_abs

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
        parity = 'Odd' if self.parity else 'Even'
        lines = [
            f"{parity} lattice of signature {self.signature}, discriminant {self.disc} and exponent {self.exp}",
            f"Discriminant group: {self.dgroup}"
        ]
        return '\n'.join(lines)
    
    def _lll_indefinite_np(self) -> None:
        """
        Applies LLL reduction to an indefinite lattice
        using a positive-definite majorant metric.
        Uses numpy for the spectral decomposition and Cholesky decomposition,
        and FLINT for the integer lattice operations.
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

    def _lll_indefinite_sp(self) -> None:
        """
        Applies LLL reduction to an indefinite lattice
        using a positive-definite majorant metric.
        Uses sympy to reduce numerical instability issues with np.linalg.eigh,
        and sympy's cholesky for the majorant basis. Finally, uses sympy's LLL
        to get the transformation matrix. This is much slower than the numpy version,
        but more stable for large discriminants.
        """
        A_sp = sp.Matrix(self.A.tolist())
        P, D = A_sp.diagonalize()
        D = sp.N(D, 50).as_real_imag()[0]
        P = sp.N(P, 50).as_real_imag()[0]
        M = P * D.applyfunc(lambda x: abs(x)) * P.inv()
        M = M * 1e15
        M = M.applyfunc(lambda x: int(x.round()))
        B = (M.cholesky(hermitian=True)).evalf()
        B = B.applyfunc(lambda x: int(x.round()))
        _, U = B.lll_transform()
        return [[int(x) for x in row] for row in (U * A_sp * U.transpose()).tolist()]
        
    def lll(self) -> List[List[int]]:
        """Returns the LLL-reduced Gram matrix. In the case of an indefinite lattice,
        uses a positive-definite majorant."""
        match self.signature:
            case (self.rank, 0):
                return self.A.lll(rep='gram').tolist()
            case (0, self.rank):
                return (-self.A).lll(rep='gram').tolist()
            case _:
                return self._lll_indefinite_np()
        
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

    @staticmethod
    def saturate(gens: List[List[int]]) -> List[List[int]]:
        """Given a set of generators for a sublattice, returns a basis for the saturated sublattice
        spanned by the generators."""
        rank = len(gens[0])
        H, T = fl.fmpz_mat(gens).transpose().hnf(transform=True)
        B, _denom = T.inv().numer_denom() # T in GL(n, Z), so denom is 1
        # Finds number of rows in H that are not entirely zero
        H_len = min(len(gens), rank)
        num_independent = sum(1 for i in range(H_len) if any(H[i, j] != 0 for j in range(H_len)))
        B_transposed = B.transpose().tolist()
        return [B_transposed[i] for i in range(num_independent)]
    
    @staticmethod
    def index(gens: List[List[int]]) -> int:
        """Given a set of generators for a sublattice, returns the index of the sublattice
        spanned by the generators."""
        S = fl.fmpz_mat(gens).snf()
        return math.prod([S[i, i] for i in range(len(gens)) if S[i, i] != 0])
    
    @staticmethod
    def fibre_product(A1: list[list[int]], A2: list[list[int]]) -> list[list[int]]:
        n1 = len(A1)
        n2 = len(A2)
        if n1 == 0 or n2 == 0:
            return []
        M = fl.fmpz_mat(A1 + A2).transpose()
        K, r = M.nullspace()
        _H, T = K.hnf(transform=True)
        B, _denom = T.inv().numer_denom()
        C = B.transpose().tolist()[:r]
        return [v[:n1] for v in C], [v[n1 : n1 + n2] for v in C]

    @staticmethod
    def image(A: list[list[int]]) -> list[list[int]]:
        """Given a matrix A, returns a basis for the Z-span of its rows."""
        L = fl.fmpz_mat(A)
        B = L * L.transpose()
        H, T = B.hnf(transform=True)
        rank = sum(1 for i in range(len(A)) if any(H[i, j] != 0 for j in range(len(A))))
        return (T * L).tolist()[:rank]
        # M = np.array(A, dtype=object).transpose()
        # D, L, _R = smith_normal_form(M)
        # L_inv, _denom = fl.fmpz_mat(L.tolist()).inv().numer_denom()
        # n = min(M.shape)
        # r = sum(1 for i in range(n) if D[i, i] != 0)
        # T_list = (L_inv * fl.fmpz_mat(D.tolist())).transpose().tolist()
        # return [T_list[i] for i in range(r)]
    
    def complement(self, gens: List[List[int]]) -> List[List[int]]:
        """Given a set of generators for a sublattice, returns a basis for its orthogonal complement."""
        G = fl.fmpz_mat(gens) * self.A
        K, n = G.nullspace()
        return self.saturate(K.transpose().tolist()[:n])
    
    def subquotient(self, gens: List[List[int]]) -> List[List[int]]:
        """Given a set of generators for a sublattice L, returns a basis for the lattice
        that represents the quotient of L by the kernel of the quadratic form induced on L."""
        L = fl.fmpz_mat(gens)
        B = L * self.A * L.transpose()
        H, T = B.hnf(transform=True)
        rank = sum(1 for i in range(len(gens)) if any(H[i, j] != 0 for j in range(len(gens))))
        return (T * L).tolist()[:rank]
    
    def even_sublattice(self) -> List[List[int]]:
        """Returns a basis for the biggest even sublattice of the lattice."""
        if self.parity:
            basis, _ = self.fibre_product([[self.A[i, i] % 2] for i in range(self.rank)], [[2]])
            return basis
        else:
            return [[int(i == j) for i in range(self.rank)] for j in range(self.rank)]

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