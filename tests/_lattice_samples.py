"""Deterministic samples of Gram and generator matrices shared by the tests of lattice.lattice."""
import random
from lattice._matrix import imat, imat_diag
from lattice.lattice import Lattice

# Invariant factors with square divisors, to exercise clear_squares and make_ssf
SQUAREFUL = [1, 1, 2, 3, 4, 5, 6, 8, 9, 12, 18, 25, 27, 36, 50]


def unimodular(rng, n):
    """A pseudo-random matrix in GL(n, Z), built from elementary row operations."""
    T = [[int(i == j) for j in range(n)] for i in range(n)]
    for _ in range(3 * n if n > 1 else 0):
        i, j = rng.sample(range(n), 2)
        c = rng.randint(-2, 2)
        T[i] = [a + c * b for a, b in zip(T[i], T[j])]
    if n and rng.random() < 0.5:
        T[0] = [-a for a in T[0]]
    return imat(T)


def conjugate(A, T):
    """The Gram matrix T A T^T of the basis given by the rows of T."""
    return T @ imat(A) @ T.T


def random_symmetric(rng, n, bound):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = A[j][i] = rng.randint(-bound, bound)
    return A


def nondegenerate(make, count):
    """The first count nondegenerate Gram matrices produced by make()."""
    result = []
    while len(result) < count:
        A = make()
        try:
            Lattice(len(A), A)
        except ValueError:
            continue
        result.append(A)
    return result


def random_grams(seed, count, ranks, bound):
    """Random nondegenerate symmetric matrices; mostly indefinite for rank >= 3."""
    rng = random.Random(seed)
    return nondegenerate(lambda: random_symmetric(rng, rng.choice(ranks), bound), count)


def random_definite(seed, count, ranks, bound, sign=1):
    """Gram matrices sign * B B^T of n random vectors in Z^(n+1)."""
    rng = random.Random(seed)

    def make():
        n = rng.choice(ranks)
        B = imat([[rng.randint(-bound, bound) for _ in range(n + 1)] for _ in range(n)])
        return (sign * B @ B.T).tolist()
    return nondegenerate(make, count)


def random_squareful(seed, count, ranks, signs=(1, -1)):
    """Gram matrices U D U^T, where U is unimodular and D is diagonal with entries
    +-d for d in SQUAREFUL, so that the discriminant groups are far from square-free."""
    rng = random.Random(seed)

    def make():
        n = rng.choice(ranks)
        D = imat_diag([rng.choice(signs) * rng.choice(SQUAREFUL) for _ in range(n)])
        return conjugate(D, unimodular(rng, n)).tolist()
    return nondegenerate(make, count)


def random_generators(seed, count, max_rows, ranks, bound):
    """Random generator matrices k x n, often with dependent or non-primitive rows."""
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        n = rng.choice(ranks)
        k = rng.randint(0, max_rows)
        rows = []
        for _ in range(k):
            if rows and rng.random() < 0.3:
                # A combination of the previous rows
                rows.append([sum(rng.randint(-2, 2) * r[j] for r in rows) for j in range(n)])
            else:
                c = rng.choice([1, 1, 2, 3])
                rows.append([c * rng.randint(-bound, bound) for _ in range(n)])
        result.append((rows, n))
    return result


def ident(A):
    """A short test id: the shape of a matrix (pytest makes repeated ids unique)."""
    if isinstance(A, int):
        return f"n{A}"
    return "x".join(str(d) for d in imat(A).shape) if len(A) else "empty"
