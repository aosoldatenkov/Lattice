import flint as fl
import numpy as np
from Lattice import *
from LatticeUtils import *
from IntVectors import *

class DiscForm:

    def __init__(self, L):
        M = np.array(L.A.tolist(), dtype=object)
        D, _, R = smith_normal_form(M)
        Dual = fl.fmpq_mat(R.tolist()) * fl.fmpz_mat(D.tolist()).inv()
        self.A = Dual.transpose() * L.A * Dual
        self.D = D
        self.rank = L.rank
        self.iso = self._list_iso()

    def _list_iso(self):
        isotropic = []
        for u in product(*[range(self.D[i, i]) for i in range(self.rank)]):
            if all(x == 0 for x in u):
                continue
            u_mat = fl.fmpz_mat(1, self.rank, u)
            _, denom =(u_mat * self.A * u_mat.transpose()).numer_denom()
            if denom == 1:
                isotropic.append(list(u))
        return isotropic
    
    def _list_max_isospaces(self):
        d = len(self.iso)
        if d == 0:
            return []
        max_isospaces = []
        I = fl.fmpq_mat(self.iso)
        P = I * self.A * I.transpose()
        is_integral = [[P[i, j].denom() == 1 for j in range(d)] for i in range(d)]
        def dfs(current):
            nonlocal max_isospaces
            maximal = True
            for i in range(current[-1] + 1, d):
                if all(is_integral[i][j] for j in current):
                    maximal = False
                    current.append(i)
                    dfs(current)
                    current.pop()
            if maximal:
                current_set = set(current)
                if any([current_set.issubset(m) for m in max_isospaces]):
                    return
                max_isospaces.append(current_set)
        for i in range(d):
            current = [i]
            dfs(current)
        return max_isospaces
    