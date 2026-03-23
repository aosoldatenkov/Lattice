import flint as fl
from Lattice import Lattice
from BinLattice import BinLattice, int_seq

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