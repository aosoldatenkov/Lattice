from __future__ import annotations
import math
import operator
import numpy as np
import sympy as sp
import flint as fl
from sympy.matrices.normalforms import smith_normal_decomp
from typing import List, Tuple
from lattice._matrix import IMat, imat, imat_zero, imat_diag, imat2flz, flz2imat, nrows, concat_rows


def _sp2flz(M: sp.Matrix) -> fl.fmpz_mat:
    return fl.fmpz_mat(M.rows, M.cols, [int(x) for x in M])


class Lattice:
    """An integral lattice L = Z^n with a nondegenerate symmetric bilinear form.

    The form is (u, v) = u A v^T, where A is the Gram matrix in the standard basis
    e_1, ..., e_n. Vectors are row vectors of coordinates in this basis, and a set of
    vectors (generators or a basis of a sublattice) is a matrix whose rows are the vectors;
    see lattice/_matrix.py for the conventions on matrix shapes.

    Attributes:
        rank: the rank n.
        A, A_fl: the Gram matrix as an IMat and as a flint.fmpz_mat.
        signature: the pair (n_+, n_-).
        parity: True if L is odd (some (v, v) is odd), False if L is even.
        dgroup: the invariant factors d_1 | d_2 | ... | d_n of the discriminant group
            L^∨/L ≅ ⊕ Z/d_i (the diagonal of the Smith normal form of A, including the 1s).
        disc: |det A|, the order of the discriminant group.
        exp: the exponent d_n of the discriminant group (1 for a unimodular lattice).
    """

    def __init__(self, rank: int, prod: List[List[int]] | IMat):
        """Creates the lattice of the given rank with Gram matrix prod: an n x n integral
        symmetric matrix of nonzero determinant, as a list of rows or an IMat. For rank 1,
        the 1-d form [a] is also accepted. Raises ValueError if the matrix is not n x n,
        not symmetric or degenerate, and TypeError if its entries are not integers."""
        self.rank = rank
        A = imat(prod)
        if rank == 0 and A.size == 0:
            A = imat_zero(0, 0)
        if rank == 1 and A.ndim < 2:
            A = A.reshape(1, 1)
        if A.shape != (rank, rank):
            raise ValueError("Gram matrix dimensions must match the specified rank.")
        # Python ints never overflow; operator.index rejects floats, fractions and strings
        self.A = imat([operator.index(x) for x in A.flat]).reshape(A.shape)
        self.A_fl = imat2flz(self.A)

        if self.A_fl.det() == 0:
            raise ValueError("Gram matrix must be non-degenerate.")
        if self.A_fl.transpose() != self.A_fl:
            raise ValueError("Gram matrix must be symmetric.")

        self._compute_exponent()
        self._init_basis()
        self._snf_L_inv = None
        self._snf_D = None
        self._snf_R = None

    def _compute_exponent(self) -> None:
        """Computes the invariant factors, the order and the exponent of the discriminant group."""
        B = self.A_fl.snf()
        self.dgroup = [int(B[i, i]) for i in range(self.rank)]
        self.disc = abs(int(self.A_fl.det()))
        self.exp = max(self.dgroup, default=1)

    def _compute_snf(self) -> None:
        """Computes and caches the Smith normal form L * A * R = D with L, R in GL(n, Z),
        storing L^{-1}, D and R as flint matrices."""
        if self._snf_L_inv is not None:
            return
        D, L, R = smith_normal_decomp(sp.Matrix(self.A.tolist()), domain=sp.ZZ)
        L_inv, _ = _sp2flz(L).inv().numer_denom()
        self._snf_L_inv = L_inv
        self._snf_D = _sp2flz(D)
        self._snf_R = _sp2flz(R)

    def _init_basis(self) -> None:
        """Computes the signature and the parity of the lattice.
        The signature is computed exactly, avoiding numerical instability issues with np.linalg.eigh:
        all roots of the characteristic polynomial of A are real and nonzero, so by Descartes' rule
        of signs, the number of positive roots is the number of sign changes in its coefficients."""
        coeffs = [c for c in self.A_fl.charpoly().coeffs() if c != 0]
        n_pos = sum(1 for a, b in zip(coeffs, coeffs[1:]) if (a > 0) != (b > 0))
        self.signature = (n_pos, self.rank - n_pos)
        self.parity = any(self.A_fl[i, i] % 2 for i in range(self.rank))

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
        """Same as self * other."""
        return self.__mul__(other)

    def __call__(self, d: int) -> Lattice:
        """Scales the quadratic form by a constant d, returning L(d)."""
        return Lattice(self.rank, self.A * d)

    def info(self) -> str:
        """Returns a human-readable summary of the main invariants."""
        if self.signature == (0, 0):
            return 'Zero lattice'
        parity = 'Odd' if self.parity else 'Even'
        lines = [
            f"{parity} lattice of signature {self.signature}, discriminant {self.disc} and exponent {self.exp}",
            f"Discriminant group: {self.dgroup}"
        ]
        return '\n'.join(lines)

    def _lll_indefinite_np(self, transform: bool = False) -> IMat | Tuple[IMat, IMat]:
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
        return flz2imat(B) if not transform else (flz2imat(B), flz2imat(U))

    def _lll_indefinite_sp(self, precision = 50, rescale = 1e15, transform: bool = False) -> List[List[int]] | Tuple[IMat, IMat]:
        """
        Applies LLL reduction to an indefinite lattice
        using a positive-definite majorant metric.
        Uses sympy to reduce numerical instability issues with np.linalg.eigh,
        and sympy's cholesky for the majorant basis. Finally, uses sympy's LLL
        to get the transformation matrix. This is much slower than the numpy version,
        but more stable for large discriminants. Returns the reduced Gram matrix as a list;
        with transform=True, returns the pair (M, U) of IMats with M = U A U^T.
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
        M = [[int(x) for x in row] for row in (U * A_sp * U.transpose()).tolist()]
        if transform:
            return imat(M), imat([[int(x) for x in row] for row in U.tolist()])
        return M

    def lll(self, transform: bool = False) -> IMat | Tuple[IMat, IMat]:
        """Returns an LLL-reduced Gram matrix M of the lattice; with transform=True,
        returns the pair (M, T), where T is in GL(n, Z) and M = T A T^T (the rows of T
        are the new basis vectors). Definite lattices are reduced exactly by FLINT.
        In the case of an indefinite lattice, uses a positive-definite majorant computed
        in floating point: M is still exactly equivalent to A, but the quality of the
        reduction depends on the numerics."""
        match self.signature:
            case (self.rank, 0):
                if transform:
                    M, T = self.A_fl.lll(transform=True, rep='gram')
                    return flz2imat(M), flz2imat(T)
                return flz2imat(self.A_fl.lll(rep='gram'))
            case (0, self.rank):
                if transform:
                    M, T = (-self.A_fl).lll(transform=True, rep='gram')
                    return flz2imat(-M), flz2imat(T)
                return flz2imat(-(-self.A_fl).lll(rep='gram'))
            case _:
                return self._lll_indefinite_np(transform)

    def product(self, u: List[int] | IMat, v: List[int] | IMat) -> int:
        """Returns the product (u, v) of two vectors (1-d, or 1 x n matrices)."""
        return int(np.ravel(imat(u) @ self.A @ imat(v).transpose())[0])

    # From this rank on, batch_prod multiplies with FLINT, which is faster than numpy
    # with Python ints despite the conversions; for smaller ranks the conversions dominate
    FLINT_MIN_RANK = 16

    def batch_prod(self, u: IMat, v: IMat) -> IMat:
        """Returns the matrix of products (u_i, v_j) of the rows of u and v."""
        if self.rank < self.FLINT_MIN_RANK:
            return u @ self.A @ v.transpose()
        u, v = imat(u), imat(v)
        M = imat2flz(u) * self.A_fl * imat2flz(v).transpose()
        # Same shape as u @ A @ v^T, also for 1-d u or v (a scalar if both are 1-d)
        result = imat([int(x) for x in M.entries()]).reshape(u.shape[:-1] + v.shape[:-1])
        return result[()] if result.ndim == 0 else result

    def square(self, u: List[int] | IMat) -> int:
        """Returns the square (u, u) of a vector u."""
        return self.product(u, u)

    def is_root(self, u: List[int] | IMat) -> bool:
        """Checks if u is a root: a primitive vector with (u, u) != 0, such that the reflection
        in u preserves the lattice, i.e. (u, u) divides 2 (u, x) for all x in L.
        Non-primitive vectors are never roots, even if the reflection in them is integral."""
        u = imat(u).flatten()
        if math.gcd(*u.tolist()) != 1:
            return False
        norm_sq = self.square(u)
        if norm_sq == 0:
            return False
        v = u @ self.A
        d = 2 * math.gcd(*v.tolist())
        return d % norm_sq == 0

    @staticmethod
    def saturate(gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice, returns a basis for the saturated sublattice
        spanned by the generators, i.e. for the intersection of their Q-span with Z^n.
        The number of basis vectors equals the rank of gens."""
        M = imat2flz(gens)
        H, T = M.transpose().hnf(transform=True)
        B, _ = T.inv().numer_denom() # T in GL(n, Z), so denom is 1
        # The nonzero rows of H come first, and there are rank(M) of them
        return flz2imat(B.transpose())[:M.rank()]

    @staticmethod
    def index(gens: List[List[int]] | IMat) -> int:
        """Given a set of generators for a sublattice, returns the index of the sublattice
        spanned by the generators in its saturation (the product of the nonzero
        invariant factors of gens). For n linearly independent vectors in Z^n, this is
        the absolute value of their determinant."""
        S = imat2flz(gens).snf()
        return math.prod([int(S[i, i]) for i in range(min(S.nrows(), S.ncols())) if S[i, i] != 0])

    @staticmethod
    def fibre_product(A1: List[List[int]] | IMat, A2: List[List[int]] | IMat) -> tuple[IMat, IMat]:
        """Given integer matrices A1 (n1 x k) and A2 (n2 x k), i.e. maps Z^n1 -> Z^k and
        Z^n2 -> Z^k acting on row vectors, returns a basis of their fibre product
        {(x, y) in Z^n1 x Z^n2 : x A1 = y A2}, as the pair of matrices (B1, B2) whose
        rows are the components x and y of the basis vectors."""
        n1 = nrows(A1)
        n2 = nrows(A2)
        if n1 + n2 == 0:
            return imat_zero(0, 0), imat_zero(0, 0)
        M = imat2flz(concat_rows(A1, A2)).transpose()
        K, r = M.nullspace()
        _, T = K.hnf(transform=True)
        B, _ = T.inv().numer_denom()
        # The rows of C form a basis of {(x, y) : x A1 + y A2 = 0}
        C = flz2imat(B.transpose())[:r]
        return C[:, :n1], -C[:, n1:]

    @staticmethod
    def image(A: List[List[int]] | IMat) -> IMat:
        """Given a matrix A, returns a basis for the Z-span of its rows."""
        L = imat2flz(A)
        B = L * L.transpose()
        H, T = B.hnf(transform=True)
        # A row of T * L is zero iff the corresponding row of T * B is zero
        rank = sum(1 for i in range(L.nrows()) if any(H[i, j] != 0 for j in range(L.nrows())))
        return flz2imat(T * L)[:rank]

    def complement(self, gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice, returns a basis for its orthogonal complement
        {x in L : (x, g) = 0 for all g in gens}."""
        G = imat2flz(gens) * self.A_fl
        K, n = G.nullspace()
        return self.saturate(flz2imat(K.transpose())[:n])

    def subquotient(self, gens: List[List[int]] | IMat) -> IMat:
        """Given a set of generators for a sublattice L, returns a basis for the lattice
        that represents the quotient of L by the kernel of the quadratic form induced on L.
        The basis vectors lie in L, and their images form a basis of the quotient."""
        L = imat2flz(gens)
        B = L * self.A_fl * L.transpose()
        H, T = B.hnf(transform=True)
        # The rows of T * L beyond the rank of B span the kernel of the form on L
        rank = sum(1 for i in range(L.nrows()) if any(H[i, j] != 0 for j in range(L.nrows())))
        return flz2imat(T * L)[:rank]

    def even_sublattice(self) -> IMat:
        """Returns a basis for the biggest even sublattice {x in L : (x, x) is even}.
        It has index 2 if L is odd, and coincides with L if L is even."""
        if self.parity:
            # (x, x) = sum x_i^2 A_ii = sum x_i A_ii mod 2
            basis, _ = self.fibre_product(imat([[self.A[i, i] % 2] for i in range(self.rank)]), imat([[2]]))
            return basis
        else:
            return imat_diag([1] * self.rank)

    def dual_vec(self, u: List[int] | IMat) -> Tuple[IMat, int]:
        """Given a vector u, finds its divisibility d in the dual lattice and a vector v, such that (u, v) = d.
        Returns the pair (v, d). Here d >= 0 generates the ideal (u, L) in Z, so that u / d is
        a primitive vector of the dual lattice; for u = 0, d = 0."""
        B = self.A_fl * imat2flz(u).transpose()
        H, T = B.hnf(transform=True)
        return imat(T.tolist()[0]), int(H[0, 0])

    def make_primitive(self) -> Lattice:
        """Returns the lattice L(1/d), where d is the GCD of the entries of the Gram matrix,
        or the lattice itself if d = 1."""
        d = math.gcd(*self.dgroup)
        return Lattice(self.rank, self.A // d) if d > 1 else self

    def clear_squares(self) -> Lattice:
        """Returns the square-free overlattice of the lattice.

        In a basis f_1, ..., f_n of L adapted to the Smith normal form, the dual lattice
        is spanned by the vectors f_i^* / d_i, and the overlattice is spanned by the
        vectors f_i / s_i, where s_i^2 is the largest square dividing d_i. It is integral,
        of index prod s_i, and its discriminant group has square-free invariant factors.
        The overlattice does not depend on the choice of the basis."""
        def max_square(n: int) -> int:
            return math.prod(int(p) ** (e // 2) for p, e in fl.fmpz(n).factor())
        self._compute_snf()
        D0 = fl.fmpz_mat(self._snf_D)
        for i in range(self.rank):
            D0[i, i] = max_square(self._snf_D[i, i])
        new_A, _ = (D0.inv() * self._snf_R.transpose() * self.A_fl * self._snf_R * D0.inv()).numer_denom()
        return Lattice(self.rank, new_A.tolist())

    def make_ssf(self) -> Lattice:
        """Returns the associated strongly square-free lattice.

        A lattice of rank n is strongly square-free if the invariant factors of its
        discriminant group are square-free, and for every prime p at most n/2 of them
        are divisible by p. The result is obtained by clearing the squares, and then,
        for the primes p with more than n/2 invariant factors divisible by p,
        by rescaling by p and clearing the squares again; this replaces the p-rank k
        of the discriminant group by n - k."""
        M = self.clear_squares()
        factor = 1
        for p, _ in fl.fmpz(M.exp).factor():
            if 2 * sum(1 for d in M.dgroup if d % p == 0) > self.rank:
                factor *= int(p)
        return M(factor).clear_squares() if factor > 1 else M
