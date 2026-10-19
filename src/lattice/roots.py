import random as rnd
from fractions import Fraction
from typing import List
import numpy as np
import flint as fl
import cdd
import cdd.gmp
from lattice._matrix import IMat, imat, imat2flz, nrows
from lattice.lattice import Lattice


def Coxeter_graph(L, roots, reorder: bool = False):
    """Returns the Coxeter graph of the given roots of L, as a dictionary {(i, j): label} over the pairs
    i < j, where the label of the pair of roots r, s depends on cos^2 = (r, s)^2 / ((r, r)(s, s)): the
    mirrors meet at the angle pi / m for cos^2 = 0, 1/4, 1/2, 3/4 (label m = 2, 3, 4, 6), are parallel for
    cos^2 = 1 (label 'inf'), and are ultraparallel for cos^2 > 1 (label: the Fraction cos^2). For roots of
    a lattice 4 cos^2 is an integer, so there are no other values.

    By default, i and j are the indices of the roots in the list. With reorder=True, they are instead the
    positions in the ordering of order_vert() of the graph in which the roots with cos^2 <= 1 are adjacent:
    for the walls of a polygon (rank 3), this is their cyclic order. This search is exponential in general."""
    rmat = imat(roots)
    nroots = len(roots)
    A = L.batch_prod(rmat, rmat)
    C = [[Fraction(int(A[i, j] ** 2), int(A[i, i] * A[j, j])) for j in range(nroots)] for i in range(nroots)]
    if reorder:
        graph = {i: [j for j in range(nroots) if i != j and C[i][j] <= 1] for i in range(nroots)}
        path = order_vert(graph, list(range(nroots)))
        ord = [path.index(i) for i in range(nroots)]
    else:
        ord = list(range(nroots))
    edges = {}
    for i in range(nroots):
        for j in range(i + 1, nroots):
            cos = C[i][j]
            pair = (min(ord[i], ord[j]), max(ord[i], ord[j]))
            if cos == 1:
                edges[pair] = 'inf'
            elif cos == 0:
                edges[pair] = 2
            elif cos == Fraction(1, 4):
                edges[pair] = 3
            elif cos == Fraction(2, 4):
                edges[pair] = 4
            elif cos == Fraction(3, 4):
                edges[pair] = 6
            elif cos > 1:
                edges[pair] = cos
    return edges


def get_extremal_rays(roots: list[list[int]] | list[IMat], gram_matrix: IMat) -> tuple[list[fl.fmpq_mat], frozenset]:
    """
    Given a list of roots (normal vectors) and the ambient Gram matrix, computes exactly (cdd, GMP) the
    generators of the cone {x : (x, r) >= 0 for all roots r}. Returns the pair (rays, lin_set): the
    generators as 1 x n fmpq_mats, and the set of the indices of those spanning the lineality space.
    The cone is pointed iff lin_set is empty; then rays are its extremal rays.
    """
    rank = nrows(gram_matrix)
    # 1. Set up the H-representation for cdd.
    # Create the cdd matrix list
    # The first column is 'b' (which is 0 because we are building a cone starting at the origin)
    h_rep_data = []
    for root in roots:
        # Calculate the linear functional vector for this root: v = root^T * G
        functional = imat(root) @ gram_matrix
        # Format for cdd: [0, v_1, v_2, ..., v_n]
        row = [Fraction(0, 1)] + [Fraction(int(val), 1) for val in functional.tolist()]
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
            ray = fl.fmpq_mat(1, rank, [fl.fmpq(row[j].numerator, row[j].denominator) for j in range(1, rank + 1)])
            extreme_rays.append(ray)
    return extreme_rays, v_rep.lin_set


def order_vert(graph, v):
    """Given a graph and a subset of its vertices, extracts the longest possible
    simple path from that set of vertices, removes those vertices, and then
    recursively does the same for the remaining vertices.
    Returns the concatenation of the paths found at each step.
    The search for a longest path is exhaustive, hence exponential: use it for small graphs only."""
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


def simple_roots(L: Lattice, roots: List[IMat] | List[List[int]], dir = None) -> List[IMat] | List[List[int]]:
    """Given a lattice L, a list of roots in L, and an optional direction dir,
    returns a list of simple roots for the root system spanned by the given roots.
    If dir is not provided, it will be chosen automatically (reproducibly, without changing the
    state of the global random generator).
    The positive roots (those with (dir, r) > 0) are taken by increasing (dir, r), and kept if they are
    linearly independent of those kept before. This gives the simple roots if the list contains all
    the roots of the root system and dir is orthogonal to none of them; for a partial list, the
    result may contain non-simple roots."""
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
        rng = rnd.Random(int(roots[0][0]))
        while True:
            dir = imat([rng.randint(1, 10 ** 6) for _ in range(L.rank)])
            if all(L.product(dir, r) != 0 for r in roots):
                break
    pos_roots = [r for r in roots if L.product(dir, r) > 0]
    pos_roots.sort(key=lambda r: L.product(dir, r))
    simple = []
    for r in pos_roots:
        extended = imat(simple + [r])
        M = imat2flz(L.batch_prod(extended, extended))
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
