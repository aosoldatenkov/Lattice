from Commons import *
from Lattice import *
from LatticeUtils import *
from IntVectors import *

class DiscForm:

    def __init__(self, L):
        D, _, R = smith_normal_form(L.A)
        Dual = fl.fmpq_mat(R.tolist()) * fl.fmpz_mat(D.tolist()).inv()
        self.A = Dual.transpose() * L.A_fl * Dual
        self.D = D
        self.rank = L.rank
        self._init_A_red()
        self.iso = None

    def _init_A_red(self):
        i0 = min([self.rank] + [i for i in range(self.rank) if self.D[i, i] > 1])
        self.ord = [int(self.D[i, i]) for i in range(i0, self.rank)]
        def reduce(x):
            return fl.fmpq(x.numer() % x.denom(), x.denom())
        self.Ared = fl.fmpq_mat([[reduce(x) for x in l[i0:]] for l in self.A.tolist()[i0:]])
        
    def list_iso(self):
        isotropic = []
        n = len(self.ord)
        for u in product(*[range(self.ord[i]) for i in range(n)]):
            if all(x == 0 for x in u):
                continue
            u_mat = fl.fmpz_mat(1, n, u)
            _, denom =(u_mat * self.Ared * u_mat.transpose()).numer_denom()
            if denom == 1:
                isotropic.append(list(u))
        self.iso = isotropic
        return isotropic
    
    def list_max_isospaces(self):
        if not self.iso:
            self.list_iso()
        d = len(self.iso)
        n = len(self.ord)
        if d == 0:
            return []
        max_isospaces = []
        I = fl.fmpq_mat(self.iso)
        P = I * self.Ared * I.transpose()
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
    
    def overlattice(self, iso_gens):
        gens = [[0] * (self.rank - len(self.ord)) + self.iso[i] for i in iso_gens] + self.D.tolist()
        basis = imat2flz(Lattice.image(imat(gens)))
        A, _ = (basis * self.A * basis.transpose()).numer_denom()
        return Lattice(self.rank, A.tolist())