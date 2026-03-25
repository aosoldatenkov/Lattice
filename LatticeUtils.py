import flint as fl
from Lattice import Lattice
from BinLattice import BinLattice, int_seq
from typing import List, Iterator
import numpy as np
import math
import cdd
import cdd.gmp
from fractions import Fraction


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
    """Given a graph and a subset of its vertices, extracts the longest possible simple path from the given set of vertices,
    removes those vertices, and then recursively does the same for the remaining vertices. Returns the concatenation of the paths found at each step."""
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
    if L.signature not in [(L.rank, 0), (0, L.rank)]:
        # Check if the sublattice spanned by the roots is sign-definite. If not, we can't pick simple roots.
        Sublat = L.saturate(roots)
        M = Lattice(len(Sublat), L.batch_prod(Sublat, Sublat))
        if M.signature not in [(M.rank, 0), (0, M.rank)]:
            raise ValueError("The roots do not span a sign-definite sublattice, so we cannot pick simple roots.")
    if dir is None:
        # Pick a direction that is not orthogonal to any root. This will be used to determine which roots are positive.
        for dir in int_seq(L.rank, nonzero=True):
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


def fincke_pohst_search(M: np.ndarray, bound: float) -> Iterator[List[int]]:
    """
    Finds all integer vectors x such that x^T M x <= bound.
    Uses Cholesky decomposition for O(sqrt(bound)^rank) performance.
    """
    rank = M.shape[0]
    # Cholesky decomposition: M = R^T R (R is upper triangular)
    # We add a tiny epsilon to the diagonal for numerical stability on float matrices
    R = np.linalg.cholesky(M + np.eye(rank) * 1e-10).T 
    # Store found vectors
    results = []
    # State arrays for the DFS
    x = np.zeros(rank)
    dist = np.zeros(rank)
    # Start at the last coordinate (bottom of the tree)
    k = rank - 1
    # Compute the center and bounds for the current coordinate
    z = np.zeros(rank)
    z[k] = 0.0
    x[k] = math.floor(math.sqrt((bound - dist[k]) / (R[k, k]**2)) + z[k])
    while True:
        # Calculate the partial distance
        d = bound - dist[k] - R[k, k]**2 * (x[k] - z[k])**2
        if d >= 0:
            if k == 0:
                # We reached a leaf node (a complete valid vector)
                v = np.round(x).astype(int)
                if np.any(v): # Ignore the zero vector
                    yield [int(coord) for coord in v]
                # Backtrack
                x[k] -= 1
            else:
                # Move down the tree
                k -= 1
                dist[k] = dist[k + 1] + R[k + 1, k + 1]**2 * (x[k + 1] - z[k + 1])**2
                # Update the center z for the new level
                z[k] = -sum(R[k, j] * x[j] for j in range(k + 1, rank)) / R[k, k]
                # Set the upper bound for this coordinate
                x[k] = math.floor(math.sqrt((bound - dist[k]) / (R[k, k]**2)) + z[k])
        else:
            # Prune the branch and step back up
            k += 1
            if k == rank:
                break # Tree exhausted
            x[k] -= 1
    return results

def A_lat(n):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
        if i > 0:
            A[i][i - 1] = -1
        if i < n - 1:
            A[i][i + 1] = -1
    return Lattice(n, A)

def E_lat(n):
    if n not in [6, 7, 8]:
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
            print(f'Found {len(roots)} of {n_roots[n]} roots\r', end='')
            roots.append(v)
        if len(roots) == n_roots[n]:
            break
    print('Choosing simple roots...')
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