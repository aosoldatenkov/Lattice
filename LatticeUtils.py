import flint as fl
import sympy as sp
from itertools import product
from Lattice import Lattice
from BinLattice import BinLattice
from IntVectors import *
from typing import List, Iterator
import numpy as np
import math
import cdd
import cdd.gmp
from fractions import Fraction
import re
import random as rnd


def get_extremal_rays(roots: list[list[int]], gram_matrix: fl.fmpz_mat) -> list[list[Fraction]]:
    """
    Given a list of roots (normal vectors) and the ambient Gram matrix,
    returns the exact extremal rays of the resulting polyhedral cone.
    """
    rank = gram_matrix.nrows()
    # 1. Set up the H-representation for cdd.
    # Create the cdd matrix list
    # The first column is 'b' (which is 0 because we are building a cone starting at the origin)
    h_rep_data = []
    for root in roots:
        # Calculate the linear functional vector for this root: v = root^T * G
        root_mat = fl.fmpz_mat(1, rank, root)
        functional = root_mat * gram_matrix
        functional_list = functional.tolist()[0]
        # Format for cdd: [0, v_1, v_2, ..., v_n]
        row = [Fraction(0, 1)] + [Fraction(int(val), 1) for val in functional_list]
        h_rep_data.append(row)
    # 2. Initialize the cdd Matrix in exact rational mode
    mat = cdd.gmp.matrix_from_array(h_rep_data, rep_type=cdd.RepType.INEQUALITY)
    # 3. Use the polyhedron factory function
    poly = cdd.gmp.polyhedron_from_matrix(mat)
    # 4. Extract the V-representation
    v_rep = cdd.gmp.copy_generators(poly)
    extreme_rays = []
    for row in v_rep.array:
        ray_type = row[0]
        if ray_type == 0:
            # Extract just the vector components
            ray = [fl.fmpq(row[j].numerator, row[j].denominator) for j in range(1, rank + 1)]
            extreme_rays.append(ray)
    return extreme_rays, v_rep.lin_set


def order_vert(graph, v):
    """Given a graph and a subset of its vertices, extracts the longest possible
    simple path from that set of vertices, removes those vertices, and then
    recursively does the same for the remaining vertices.
    Returns the concatenation of the paths found at each step."""
    if not v:
        return []
    v_set = set(v)  # O(1) lookup for whether a node is in 'v'
    best_path = []
    def dfs(current_node, current_path, current_set):
        nonlocal best_path
        # Update best path if we found a longer one
        if len(current_path) > len(best_path):
            best_path = list(current_path)
        for neighbor in graph.get(current_node, []):
            if neighbor in v_set and neighbor not in current_set:
                # Backtracking: Add to path, explore, then remove
                current_set.add(neighbor)
                current_path.append(neighbor)
                dfs(neighbor, current_path, current_set)
                current_path.pop()
                current_set.remove(neighbor)
    # Start a DFS from every node in the current subset
    for start_node in v:
        dfs(start_node, [start_node], {start_node})
    # Find the remainder and recurse
    best_set = set(best_path)
    rest = [i for i in v if i not in best_set]
    return best_path + order_vert(graph, rest)


def simple_roots(L, roots, dir = None):
    """Given a lattice L, a list of roots in L, and an optional direction dir,
    returns a list of simple roots for the root system spanned by the given roots.
    If dir is not provided, it will be chosen automatically."""
    if len(roots) == 0:
        return []
    if L.signature not in [(L.rank, 0), (0, L.rank)]:
        # Check if the sublattice spanned by the roots is sign-definite. If not, we can't pick simple roots.
        Sublat = L.saturate(roots)
        M = Lattice(len(Sublat), L.batch_prod(Sublat, Sublat))
        if M.signature not in [(M.rank, 0), (0, M.rank)]:
            raise ValueError("The roots do not span a sign-definite sublattice, so we cannot pick simple roots.")
    if dir is None:
        # Pick a direction that is not orthogonal to any root. This will be used to determine which roots are positive.
        rnd.seed(roots[0][0])
        while True:
            dir = [rnd.randint(1, 10 ** 6) for _ in range(L.rank)]
            if all(L.product(dir, r) != 0 for r in roots):
                break
    pos_roots = [r for r in roots if L.product(dir, r) > 0]
    pos_roots.sort(key=lambda r: L.product(dir, r))
    simple = []
    for r in pos_roots:
        M = fl.fmpz_mat(L.batch_prod(simple + [r], simple + [r]))
        if M.det() != 0:
            simple.append(r)
        if len(simple) == L.rank:
            break
    return simple


def majorant(A_matrix: np.ndarray) -> np.ndarray:
    """Constructs the positive-definite majorant of an indefinite Gram matrix."""
    eival, eivec = np.linalg.eigh(A_matrix)
    # Take absolute values of eigenvalues to force positive-definiteness
    eival_abs = np.abs(eival)
    # Reconstruct the matrix: M = Q * |Lambda| * Q^T
    M = eivec @ np.diag(eival_abs) @ eivec.T
    return M

def A_lat(n):
    I = Lattice(1, [[1]])
    R = I * (n + 1)
    E = [[int(i == j) - int(i == j + 1) for i in range(n + 1)] for j in range(n)]
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def B_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = [[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[int(i == n - 1) for i in range(n)]]
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def C_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = [[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[2 * int(i == n - 1) for i in range(n)]]
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def D_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = [[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[int(i == n - 1 or i == n - 2) for i in range(n)]]
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def E_lat(n):
    if n == 4:
        return A_lat(4)
    elif n == 5:
        return D_lat(5)
    elif n not in [6, 7, 8]:
        raise ValueError("n must be 6, 7, or 8")
    A = A_lat(n - 1).A.tolist()
    A = [[2, 0, 0, -1] + [0] * (n - 4)] + [[-int(i == 2)] + A[i] for i in range(n - 1)]
    return Lattice(n, A)

def U_lat(n = 1):
    return Lattice(2, [[0, n], [n, 0]])

def I_lat(p, q):
    if p > 0 and q > 0:
        return Lattice(1, [[1]]) * p + Lattice(1, [[-1]]) * q
    elif p > 0 and  q <= 0:
        return Lattice(1, [[1]]) * p
    elif q > 0:
        return Lattice(1, [[-1]]) * q
    else:
        raise ValueError("At least one of p or q must be positive.")


def II_lat_n_1(n):
    if (n - 1) % 8 != 0:
        raise ValueError("n - 1 must be divisible by 8")
    L = I_lat(n, 1)
    basis = [[int(i == j) for j in range(n)] + [-1] for i in range(1, n)]
    basis += [[0] * n + [2]]
    basis += [[1] * n + [-1]]
    A = L.batch_prod(basis, basis)
    A[n][n - 1] //= 2
    A[n - 1][n] //= 2
    A[n][n] //= 4
    return Lattice(n + 1, A)

def Leech_lat():
    M = II_lat_n_1(25)
    v = [i + 1 for i in range(24)] + [185, 0]
    compl = M.complement([v])
    basis = M.subquotient(compl)
    return Lattice(len(basis), M.batch_prod(basis, basis))

def Leech_lat_alt():
    M = I_lat(24, 1)
    v = [2 * i + 1 for i in range(1, 24)] + [51, 145]
    basis = M.complement([v])
    return Lattice(len(basis), M.batch_prod(basis, basis))


def E_lat_test(n):
    if n == 4:
        return A_lat(4)
    elif n == 5:
        return D_lat(5)
    elif n not in [6, 7, 8]:
        raise ValueError("n must be 6, 7, or 8")
    n_roots = {6: 72, 7: 126, 8: 240}
    I = Lattice(1, [[1]])
    L = I(-1) + I * n
    v = [-3] + [1] * n
    compl = L.complement([v])
    A = L.batch_prod(compl, compl)
    M = Lattice(len(compl), A)
    M = Lattice(len(compl), M.lll())
    roots = []
    for v in int_seq(n, nonzero=True):
        if M.is_root(v):
            roots.append(v)
        if len(roots) == n_roots[n]:
            break
    for v in int_seq(n, nonzero=True):
        if all(M.product([n + v[i] for i in range(n)], r) != 0 for r in roots):
            dir = [n + v[i] for i in range(n)]
            break
    sim = simple_roots(M, roots, dir)
    A = M.batch_prod(sim, sim)
    graph = {i: [j for j in range(len(sim)) if A[i][j] != 0] for i in range(len(sim))}
    order = order_vert(graph, list(graph.keys()))
    sim = [sim[i] for i in order]
    A = M.batch_prod(sim, sim)
    return Lattice(len(sim), A)


def lorentz_basis_3d(L, bound = 10000):
    if L.rank != 3:
        raise ValueError("Lattice must be 3-dimensional")
    if L.signature == (2, 1):
        L = Lattice(3, (-L.A).tolist())
    if L.signature != (1, 2):
        raise ValueError("Lattice must have signature (1, 2) or (2, 1)")
    
    def decompose(L, u):
        compl = fl.fmpz_mat(L.complement([u]))
        v = L.dual_vec(u)[0]
        if L.index(compl.tolist() + [v]) != 1:
            raise ValueError("Error constructing decomposition")
        M = compl * L.A * compl.transpose()
        B = BinLattice(M[0, 0], M[1, 1], M[0, 1])
        if B.signature != (0, 2):
            raise ValueError("Error: the complement is not sign-definite")
        basis = fl.fmpz_mat([list(B.can.e), list(B.can.f)])
        compl = basis * compl
        M = compl * L.A * compl.transpose()
        B = fl.fmpz_mat(1, 3, v) * L.A * compl.transpose()
        M_fl = np.array(M.tolist(), dtype=np.float64)
        B_fl = np.array(B.tolist(), dtype=np.float64)
        T = -np.linalg.inv(M_fl) @ B_fl.transpose()
        T_int = fl.fmpz_mat(np.round(T).astype(int).tolist())
        for delta in product([0, 1, -1], repeat=2):
            v = (fl.fmpz_mat(1, 3, v) + (fl.fmpz_mat(1, 2, delta) + T_int.transpose()) * compl).tolist()[0]
            if L.square(v) > 0:
                return [v] + compl.tolist()
        return None

    for u in int_seq(3, nonzero=True, length=bound):
        if math.gcd(*u) != 1:
            continue
        if L.square(u) > 0:
            basis = decompose(L, u)
            if basis is None:
                continue
            return basis
    return None


def Allcock_list(fin, fout):
    with open(fin, "r") as f:
        lattices = [re.findall(r'-?\d+', line.strip())[:9] for line in f.readlines()]
    with open(fout, "w") as f:
        for i, l in enumerate(lattices):
            print('#' * 50 + f"{i + 1:^7}" + '#' * 50)
            L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])
            print(L.info())
            print('Gram matrix:')
            print(L.A)
            if L.signature != (2, 1):
                print("Lattice does not have signature (2, 1), skipping...")
                continue
            print("LLL-reduced Gram matrix:")
            try:
                M = Lattice(3, L.lll())
                reduced = True
            except:
                reduced = False
            if not reduced:
                try:
                    M = Lattice(3, L._lll_indefinite_sp())
                    reduced = True
                except:
                    print("Error occurred while computing LLL reduction. Lattice number", i + 1, file=f)
                    print(L.A.tolist(), file=f)
                    continue
            print(M.A)
            print("Constructing hyperbolic basis...")
            basis = lorentz_basis_3d(M)
            if not basis:
                print("No hyperbolic basis found, skipping...")
                print(M.A.tolist(), file=f)
                continue
            N = Lattice(3, M.batch_prod(basis, basis))
            print(N.info())
            print(N.A)
            print(N.A.tolist(), file=f)

