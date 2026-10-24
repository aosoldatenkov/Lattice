"""Vinberg's algorithm with wall refinement (RunTest) on a few higher-rank lattices.
Edge tracing with chamber certification for the known families
of reflective lattices of rank greater than 3"""

from itertools import product
from lattice import *
from lattice.vinberg import certify_polytope, trace_chamber
from lattice.vinberg.orbits import trace_chamber_by_orbits
from lattice.draw import CoxeterDiagram, draw_Coxeter_diagram, draw_Coxeter_diagram_3d

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

def print_walls(L, walls):
    for w in walls:
        print(w, L.square(w))

def CertDiag(rank, d = 1, max_walls = None):
    L = I_lat(1, 0)(d) + I_lat(0, rank - 1)
    print(L.info())
    fname = "../output/Reflective/Diag(" + str(d) + ",-1x" + str(rank - 1) +")"
    res = trace_chamber(L, base=[1] + [0] * (rank - 1),
                        chamber=[rank ** 2 - i ** 2 for i in range(rank)],
                        symmetry=True,
                        progress=True,
                        threads=24,
                        walls_file=fname + "_walls.json",
                        max_walls=max_walls)
    print_walls(L, res['walls'])
    trace_chamber_by_orbits(L, fname + "_walls.json",
                            certificate = fname + "_cert.json",
                            threads=24,
                            progress=True)

def CertAllDiag():
    dr = sum([[(1, r) for r in range(3, 21)],
            [(2, r) for r in range(3, 16)],
            [(3, r) for r in range(3, 15)],
            [(5, r) for r in range(3, 10)],
            [(6, 3), (7, 3), (7, 4), (10, 3), (10, 4),
            (11, 3), (11, 4), (11, 5), (13, 3),
            (14, 3), (15, 3), (15, 4), (17,3), (17, 4),
            (19, 3), (23, 3), (30, 3), (33, 3), (39, 3), (51, 3)]], start=[])
    for d, r in dr:
        CertDiag(r, d)

def CertBorcherds():
    # D20 + U
    L = D_lat(20)(-1) + U_lat()
    print(L.info())
    fname = "../output/Reflective/D20U"
    res = trace_chamber(L, base=[0] * 20 + [1, 1],
                        chamber=[22 ** 3 - i ** 3 for i in range(22)],
                        symmetry=True,
                        progress=True,
                        threads=24,
                        max_walls=210,
                        walls_file=fname + "_walls.json")
    print_walls(L, res['walls'])
    trace_chamber_by_orbits(L, fname + "_walls.json",
                            certificate = fname + "_cert.json",
                            threads=24,
                            progress=True)
    # Borcherds rank 18
    G = imat([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    gens = [[2 * int(i == j) for i in range(16)] for j in range(16)]
    for v in product(range(2), repeat=5):
        gens.append([x % 2 for x in (imat(v) @ G).tolist()])
    basis = Lattice.image(imat(gens))
    K = Lattice(16, basis @ basis.transpose() // 2)
    L = K(-1) + U_lat()
    print(L.info())
    fname = "../output/Reflective/Borcherds18"
    res = trace_chamber(L, base=[0] * 16 + [1, 1],
                        chamber=[18 ** 3 - i ** 3 for i in range(18)],
                        symmetry=True,
                        progress=True,
                        threads=24,
                        max_walls=960,
                        walls_file=fname + "_walls.json")
    print_walls(L, res['walls'])
    trace_chamber_by_orbits(L, fname + "_walls.json",
                            certificate = fname + "_cert.json",
                            threads=24,
                            progress=True)

def CertVinberg():
    L = E_lat(8)(-1) + E_lat(8)(-1) + A_lat(2)(-1) + U_lat()
    print(L.info())
    fname = "../output/Reflective/Vinberg20"
    res = trace_chamber(L, base=[0] * 18 + [1, 1],
                        chamber=[101 - i ** 2 for i in range(20)],
                        symmetry=True,
                        progress=True,
                        threads=24,
                        walls_file=fname + "_walls.json",
                        max_walls=36)
    trace_chamber_by_orbits(L, fname + "_walls.json",
                            certificate = fname + ".json",
                            threads=24,
                            progress=True)

def CertApollonian(rank):
    L = Lattice(rank, [[1 - 2 * int(i == j) for j in range(rank)] for i in range(rank)])
    print(L.info())
    fname = "../output/Reflective/Apollonian" + str(rank)
    res = trace_chamber(L, base=[1] * rank,
                        symmetry=True,
                        progress=True,
                        threads=24,
                        walls_file=fname + "_walls.json")
    trace_chamber_by_orbits(L, fname + "_walls.json",
                            certificate = fname + ".json",
                            threads=24,
                            progress=True)
    

def CertLattice(L, fname, expected=None, threads=24, **kwargs):
    """Traces the chamber of L (saving the walls), certifies it by orbits (saving the certificate), and compares the
    number of walls with the expected one; returns the result of the certification."""
    print(L.info())
    trace_chamber(L, symmetry=True, progress=True, threads=threads, walls_file=fname + "_walls.json", **kwargs)
    res = trace_chamber_by_orbits(L, fname + "_walls.json", certificate=fname + "_cert.json", threads=threads,
                                  progress=True)
    if expected is not None:
        print(f"{res['walls']} walls, expected {expected}: " + ("agrees" if res["walls"] == expected else "DISAGREES"))
    return res

def CertBogachevPerepechko(threads=24):
    """The reflective lattices of Table 1 of Bogachev, Perepechko, Vinberg's algorithm for hyperbolic lattices
    (Mat. Zametki 103:5, 2018), with their numbers of walls. The paper uses the signature (n, 1): [-k] + A_3 there is
    <k> + A_3(-1) here, [1] is <-1>, and U + [36] + [6] is U + <-36> + <-6>.
    The computations here agree with the table except for [-8] + A_3 (5 walls, the table has 7) and [-10] + A_3 (7 walls,
    the table has 12); in both cases the walls are also certified by certify_polytope and found by Vinberg's algorithm."""
    A3, A2 = A_lat(3)(-1), A_lat(2)(-1)
    diag = lambda *v: Lattice(len(v), [[v[i] if i == j else 0 for j in range(len(v))] for i in range(len(v))])
    table = [(f"[-{k}]+A3", diag(k) + A3, n) for k, n in
             [(1, 4), (2, 5), (3, 5), (4, 4), (5, 6), (6, 6), (8, 7), (9, 9), (10, 12), (12, 5), (15, 12)]]
    table += [("U+[36]+[6]", U_lat() + diag(-36, -6), 15)]
    table += [(f"[-{k}]+[1]+A2", diag(k, -1) + A2, n) for k, n in
              [(1, 4), (2, 6), (3, 5), (4, 5), (5, 7), (6, 8), (7, 11), (8, 6), (9, 5), (10, 11), (15, 15), (30, 20)]]
    # Also stated reflective in the paper (without the numbers of walls): [-2] + A_2 + [1] + ... + [1], n <= 6
    # (signature (n, 1)); here also n = 7
    for m in range(0, 6):
        table.append((f"[-2]+A2+[1]x{m}", diag(2, *([-1] * m)) + A2, None))
    summary = []
    for name, L, n in table:
        res = CertLattice(L, "../output/Reflective/BogachevPerepechko " + name, n, threads=threads)
        summary.append((name, n, res["walls"]))
    print("\nlattice              paper  certified")
    for name, n, w in summary:
        print(f"{name:20s} {'-' if n is None else n:>5}  {w:>9}" + ("" if n is None or n == w else "   <- differs"))
