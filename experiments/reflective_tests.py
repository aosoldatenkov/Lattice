"""Vinberg's algorithm with wall refinement (RunTest) on a few higher-rank lattices."""
from lattice import A_lat, E_lat, U_lat, I_lat, RunTest


def TestDiagLat(rank, d = 1):
    L = I_lat(1, 0)(d) + I_lat(0, rank - 1)
    print(L.info())
    RunTest(L, base=[1] + [0] * (rank - 1), chamber=[rank ** 2 - i ** 2 for i in range(rank)], h_batch=1, fps_batch=10000, n_iter=3)

def TestEvenUnimod():
    L = U_lat() + E_lat(8)(-1) + E_lat(8)(-1)
    print(L.info())
    RunTest(L, base=[1, 1] + [0] * 16, h_batch=1, fps_batch=10000, n_iter=3)

def TestVLat():
    L = E_lat(8)(-1) + E_lat(8)(-1) + A_lat(2)(-1) + U_lat()
    print(L.info())
    RunTest(L, base=[0] * 18 + [1, 1], chamber=[101 - i ** 2 for i in range(20)], h_batch=1, fps_batch=10000, n_iter=3)
