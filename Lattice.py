from __future__ import annotations
from Commons import *
from hsnf import smith_normal_form
from typing import List, Tuple


class Lattice:
    
    def __init__(self, rank: int, prod: List[List[int]] | IMat):
        self.rank = rank
        if type(prod) is IMat:
            self.A = prod.copy()
        else:
            self.A = imat(prod)
        self.A_fl = imat2flz(self.A)
        
        if self.A_fl.det() == 0:
            raise ValueError("Gram matrix must be non-degenerate.")
        if self.A_fl.nrows() != self.rank or self.A_fl.ncols() != self.rank:
            raise ValueError("Gram matrix dimensions must match the specified rank.")
        if self.A_fl.transpose() != self.A_fl:
            raise ValueError("Gram matrix must be symmetric.")
            
        self._compute_exponent()
        self._init_basis()
        self._snf_L_inv = None
        self._snf_D = None
        self._snf_R = None
        
    def _compute_exponent(self) -> None:
        B = self.A_fl.snf()
        self.dgroup = [int(B[i, i]) for i in range(self.rank)]
        self.disc = int(self.A_fl.det())
        self.exp = max(self.dgroup)

    def _compute_snf(self) -> None:
        if self._snf_L_inv is not None:
            return
        # L * A * R = D
        D, L, R = smith_normal_form(self.A)
        L_inv, _ = imat2flz(L).inv().numer_denom()
        self._snf_L_inv = L_inv
        self._snf_D = imat2flz(D)
        self._snf_R = imat2flz(R)

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
        return Lattice(self.rank, self.A * d)

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
        A_float = np.array(self.A, dtype=np.float64)
        
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
        B = U * self.A_fl * U.transpose()
        return B.tolist()

    def _lll_indefinite_sp(self, precision = 50, rescale = 1e15) -> None:
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
        D = sp.N(D, precision).as_real_imag()[0]
        P = sp.N(P, precision).as_real_imag()[0]
        M = P * D.applyfunc(lambda x: abs(x)) * P.inv()
        M = M * rescale
        M = M.applyfunc(lambda x: int(x.round()))
        B = (M.cholesky(hermitian=True)).evalf()
        B = B.applyfunc(lambda x: int(x.round()))
        _, U = B.lll_transform()
        return [[int(x) for x in row] for row in (U * A_sp * U.transpose()).tolist()]
        
    def lll(self) -> IMat:
        """Returns the LLL-reduced Gram matrix. In the case of an indefinite lattice,
        uses a positive-definite majorant."""
        match self.signature:
            case (self.rank, 0):
                return flz2imat(self.A_fl.lll(rep='gram'))
            case (0, self.rank):
                return flz2imat((-self.A_fl).lll(rep='gram'))
            case _:
                return self._lll_indefinite_np()
        
    def product(self, u: List[int] | IMat, v: List[int] | IMat) -> int:
        try:
            x = int(imat(u) @ self.A @ imat(v).transpose())
        except:
            x = (imat(u) @ self.A @ imat(v).transpose()).flatten()[0]
        return x
    
    def batch_prod(self, u: IMat, v: IMat) -> IMat:
        return u @ self.A @ v.transpose()

    def square(self, u: List[int] | IMat) -> int:
        return self.product(u, u)
    
    def is_root(self, u: List[int] | IMat) -> bool:
        """Checks if u is a root, i.e. if the reflection in u preserves the lattice."""
        norm_sq = self.square(u)
        if norm_sq == 0:
            return False
        v = u @ self.A
        d = 2 * math.gcd(*v.tolist())
        return d % norm_sq == 0

    @staticmethod
    def saturate(gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice, returns a basis for the saturated sublattice
        spanned by the generators."""
        M = imat2flz(gens)
        H, T = M.transpose().hnf(transform=True)
        B, _ = T.inv().numer_denom() # T in GL(n, Z), so denom is 1
        # Finds number of rows in H that are not entirely zero
        H_len = min(len(gens), M.ncols())
        num_independent = sum(1 for i in range(H_len) if any(H[i, j] != 0 for j in range(H_len)))
        B_tr = flz2imat(B.transpose())
        return B_tr[:num_independent, :]
    
    @staticmethod
    def index(gens: List[List[int]] | IMat) -> int:
        """Given a set of generators for a sublattice, returns the index of the sublattice
        spanned by the generators."""
        S = imat2flz(gens).snf()
        return math.prod([int(S[i, i]) for i in range(len(gens)) if S[i, i] != 0])
    
    @staticmethod
    def fibre_product(A1: List[List[int]] | IMat, A2: List[List[int]] | IMat) -> tuple[IMat, IMat]:
        n1 = nrows(A1)
        n2 = nrows(A2)
        if n1 == 0 or n2 == 0:
            return []
        M = imat2flz(concat_rows(A1, A2)).transpose()
        K, r = M.nullspace()
        _, T = K.hnf(transform=True)
        B, _ = T.inv().numer_denom()
        C = B.transpose().tolist()[:r]
        return imat([v[:n1] for v in C]), imat([v[n1 : n1 + n2] for v in C])

    @staticmethod
    def image(A: List[List[int]] | IMat) -> IMat:
        """Given a matrix A, returns a basis for the Z-span of its rows."""
        L = imat2flz(A)
        B = L * L.transpose()
        H, T = B.hnf(transform=True)
        rank = sum(1 for i in range(L.nrows()) if any(H[i, j] != 0 for j in range(L.nrows())))
        return imat((T * L).tolist()[:rank])
    
    def complement(self, gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice, returns a basis for its orthogonal complement."""
        G = imat2flz(gens) * self.A_fl
        K, n = G.nullspace()
        return self.saturate(imat(K.transpose().tolist()[:n]))
    
    def subquotient(self, gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice L, returns a basis for the lattice
        that represents the quotient of L by the kernel of the quadratic form induced on L."""
        L = imat2flz(gens)
        B = L * self.A_fl * L.transpose()
        H, T = B.hnf(transform=True)
        rank = sum(1 for i in range(L.nrows()) if any(H[i, j] != 0 for j in range(L.nrows())))
        return imat((T * L).tolist()[:rank])
    
    def even_sublattice(self) -> IMat:
        """Returns a basis for the biggest even sublattice of the lattice."""
        if self.parity:
            basis, _ = self.fibre_product(imat([[self.A[i, i] % 2] for i in range(self.rank)]), imat([[2]]))
            return basis
        else:
            return imat_diag([1] * self.rank)

    def dual_vec(self, u: List[int] | IMat) -> Tuple[IMat, int]:
        """Given a vector u, finds its divisibility d in the dual lattice and a vector v, such that (u, v) = d.
        Returns the pair (v, d)."""
        B = self.A_fl * imat2flz(u).transpose()
        H, T = B.hnf(transform=True)
        return imat(T.tolist()[0]), int(H[0, 0])
    
    def make_primitive(self) -> Lattice:
        d = math.gcd(*self.dgroup)
        return Lattice(self.rank, self.A // d) if d > 1 else self
    
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
        new_A, _ = (D0.inv() * self._snf_R.transpose() * self.A_fl * self._snf_R * D0.inv()).numer_denom()
        return Lattice(self.rank, new_A.tolist())
    
    def make_ssf(self) -> Lattice:
        """Returns the associated strongly square-free lattice"""
        factor = 1
        for i in range(self.rank // 2 + 1):
            factor *= (self.dgroup[i] // factor)
        return self(factor).clear_squares()