import flint as fl
import numpy as np
import time
import datetime
from BinLattice import BinLattice, int_seq
from Lattice import *
from LatticeUtils import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *


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
        lattices = [re.findall(r'-?\d+', line.strip()) for line in f.readlines()]
    with open(fout, "w") as f:
        for i, l in enumerate(lattices):
            print('#' * 50 + f"{i + 1:^7}" + '#' * 50)
            L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])
            Lno = int(l[-1])
            Wno = int(l[-2])
            if (len(l) - 14) % 3 != 0:
                print(f"Error parsing line {i}")
            Nwalls = (len(l) - 14) // 3
            print(L.info())
            print('Gram matrix:')
            print(L.A)
            if L.signature != (2, 1):
                print(f"The lattice {i + 1} does not have Lorentzian signature, skipping...")
                continue
            L = L(-1)
            for i in range(3):
                try:
                    M = Lattice(3, L.lll())
                    reduced = True
                except:
                    reduced = False
                if not reduced:
                    rescale = [1e15, 1e16, 1e17]
                    for s in rescale:
                        try:
                            M = Lattice(3, L._lll_indefinite_sp(rescale=s))
                            reduced = True
                            break
                        except:
                            continue
                    if not reduced:    
                        print("Error occurred while computing LLL reduction. Lattice number", i + 1)
                        sleep(1)
                        M = L
                L = M
            print("LLL-reduced Gram matrix:")
            print(M.A)
            print("Constructing Lorentzian basis...")
            basis = lorentz_basis_3d(M)
            if not basis:
                print("No Lorentzian basis found, using the LLL basis")
                N = M
            else:
                N = Lattice(3, M.batch_prod(basis, basis))
            print(N.info())
            print(N.A)
            print([N.A.tolist()] + [Nwalls, Wno, Lno], file=f)

def Allcock_group(graph, n):
    return [graph[(min(i, (i + 1) % n), max(i, (i + 1) % n))] for i in range(n)]

def Allcock_group_compare(g1, g2):
    if len(g1) != len(g2):
        return False
    gg = g2 + g2
    comp = [all(g1[i] == gg[i + j] for i in range(len(g1))) for j in range(len(g2))] + \
           [all(g1[-i] == gg[i + j] for i in range(len(g1))) for j in range(len(g2))]
    if any(comp):
        return True
    return False

def CheckAllcock():
    groups = {}
    start = time.perf_counter()
    with open('out', "r") as f:
        lattices = [re.findall(r'-?\d+', line.strip()) for line in f.readlines()]
        for j, l in enumerate(lattices):
            print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
            lstart = time.perf_counter()
            L = Lattice(3, [[int(x) for x in l[i:i+3]] for i in range(0, 9, 3)])
            nwalls = int(l[-3])
            Wno = int(l[-2])
            print(L.info())
            print(L.A)
            V = Vinberg(L, h_batch=100)
            V.print_info()
            walls = V.run(root_batch=1000000, use_reflections=False)
            lend = time.perf_counter()
            print("Vinberg's algorithm execution time: " + str(datetime.timedelta(seconds=(lend - lstart))))
            for w in walls:
                print(w, L.square(w))
            if len(walls) != nwalls:
                print(f"Wrong number of walls, {nwalls} expected")
                break
            group = Allcock_group(Coxeter_graph(L, walls), len(walls))
            print(f"The Weyl group: {group}")
            if Wno not in groups:
                groups[Wno] = group
            else:
                if not Allcock_group_compare(groups[Wno], group):
                    print(f"The Weyl group does not match")
                    break
    end = time.perf_counter()
    print("Total execution time: " + str(datetime.timedelta(seconds=(end - start))))