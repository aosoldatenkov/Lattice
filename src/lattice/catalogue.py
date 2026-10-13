from lattice._matrix import imat
from lattice.lattice import Lattice
from lattice.enumeration import int_seq
from lattice.roots import simple_roots, order_vert


def A_lat(n):
    I = Lattice(1, [[1]])
    R = I * (n + 1)
    E = imat([[int(i == j) - int(i == j + 1) for i in range(n + 1)] for j in range(n)])
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def B_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = imat([[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[int(i == n - 1) for i in range(n)]])
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def C_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = imat([[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[2 * int(i == n - 1) for i in range(n)]])
    A = R.batch_prod(E, E)
    return Lattice(n, A)

def D_lat(n):
    I = Lattice(1, [[1]])
    R = I * n
    E = imat([[int(i == j) - int(i == j + 1) for i in range(n)] for j in range(n - 1)] +\
        [[int(i == n - 1 or i == n - 2) for i in range(n)]])
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
    A = L.batch_prod(imat(basis), imat(basis))
    A[n][n - 1] //= 2
    A[n - 1][n] //= 2
    A[n][n] //= 4
    return Lattice(n + 1, A)

def Leech_lat():
    M = II_lat_n_1(25)
    v = [i + 1 for i in range(24)] + [185, 0]
    compl = M.complement([v])
    basis = M.subquotient(compl)
    return Lattice(basis.shape[0], M.batch_prod(basis, basis))

def Leech_lat_alt():
    M = I_lat(24, 1)
    v = [2 * i + 1 for i in range(1, 24)] + [51, 145]
    basis = M.complement([v])
    return Lattice(basis.shape[0], M.batch_prod(basis, basis))


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
    M = Lattice(compl.shape[0], A)
    M = Lattice(compl.shape[0], M.lll())
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
    sim = imat(simple_roots(M, roots, dir))
    A = M.batch_prod(sim, sim)
    graph = {i: [j for j in range(len(sim)) if A[i][j] != 0] for i in range(len(sim))}
    order = order_vert(graph, list(graph.keys()))
    sim = imat([sim[i] for i in order])
    A = M.batch_prod(sim, sim)
    return Lattice(sim.shape[0], A)
