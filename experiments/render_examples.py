"""Rendering experiments: Apollonian packings (PLY) and tilings of Allcock's rank-3 lattices (TikZ/PDF)."""
import os
import time
import datetime
import math
import subprocess
from itertools import product
import numpy as np
import flint as fl
from lattice import Lattice, Vinberg, Coxeter_graph, int_seq
from lattice import vsearch_cpp
from lattice.draw import Circle, CircleArrangement, Lorentz_ONB, RenderTiling
from allcock import Allcock_group, DATASET, load_dataset


def Apollonian(outdir="output"):
    rank = 4
    L = Lattice(rank, [[1 - 2 * int(i == j) for j in range(rank)] for i in range(rank)])
    base = [1] * rank
    axis, d = L.dual_vec(base)
    compl = L.complement([base])
    basis = [axis] + compl
    L = Lattice(rank, L.batch_prod(basis, basis))
    print(L.A)
    B, d = fl.fmpq_mat(basis).inv().numer_denom()
    base = (fl.fmpz_mat(1, rank, base) * B).tolist()[0]
    onb = np.linalg.inv(Lorentz_ONB(L, base))
    V = vsearch_cpp.VSearchCpp(L.A.tolist(), base, 1.5, 50, False, True)
    vecs = []
    while len(vecs) < 5 * 10 ** 6:
        V.run(10000, 10000)
        vecs.extend(V.get_vecs())
        print(len(vecs), 'vectors found', end='\r')
    print()
    CA = CircleArrangement()
    scale = 300
    min_r = 0.1
    count_discs = count_holes = 0
    for v in vecs:
        if L.square(v) == -1:
            w = np.array(v, dtype=float) @ onb
            C = Circle(w[0] - w[1], -2 * w[2], -2 * w[3], w[0] + w[1], disc=1, color='white')
            if not C.is_bounded() and not C.is_line():
                depth = 1 / abs(C.r)
                count_holes += 1
            elif not C.is_line():
                depth = -1 / abs(C.r)
                count_discs += 1
            else:
                depth = 0
            if C.r * scale > min_r:
                if not C.is_bounded():
                    C.set_attr(color='black')
                CA.add_circle(C, depth)    
    frame = CA.find_frame()
    #pdf = PDFPicture()
    #pdf.append(CA.tikz_out(frame=frame, r_min=min_r, background='black'))
    count = sum(1 for c in CA.circles if c[0].is_bounded())
    with open(os.path.join(outdir, "Apollo.ply"), "w", encoding="ASCII") as file_out:
        print("ply", file=file_out)
        print("format ascii 1.0", file=file_out)
        print(f"element vertex {count}", file=file_out)
        print("property float x", file=file_out)
        print("property float y", file=file_out)
        print("property float z", file=file_out)
        print("property float r", file=file_out)
        print("end_header", file=file_out)
        for c in CA.circles:
            if c[0].is_bounded():
                print(f"{c[0].x:.16f} {c[0].y:.16f} 0.0 {c[0].r:.16f}", file=file_out)
    #pdf.print('tikzcode.tex')

def Apollonian3D(outdir="output"):
    rank = 4
    L = Lattice(rank, [[1 - 2 * int(i == j) for j in range(rank)] for i in range(rank)])
    base = [1] * rank
    axis, d = L.dual_vec(base)
    compl = L.complement([base])
    basis = [axis] + compl
    L = Lattice(rank, L.batch_prod(basis, basis))
    print(L.A)
    B, d = fl.fmpq_mat(basis).inv().numer_denom()
    base = (fl.fmpz_mat(1, rank, base) * B).tolist()[0]
    onb = np.linalg.inv(Lorentz_ONB(L, base))
    V = vsearch_cpp.VSearchCpp(L.A.tolist(), base, 1.5, 50, False, True)
    vecs = []
    while len(vecs) < 5 * 10 ** 6:
        V.run(10000, 10000)
        vecs.extend(V.get_vecs())
        print(len(vecs), 'vectors found', end='\r')
    print()
    print("Sorting...")
    vecs_onb = sorted([np.array(v, dtype=float) @ onb for v in vecs if L.square(v) == -1], key=lambda x: x[0])
    print(f"{len(vecs_onb)} roots found")
    spheres = []
    for v in vecs_onb:
        if v[0] < Circle.epsilon:
            continue
        x, y, z = v[1] / v[0], v[2] / v[0], v[3] / v[0]
        r = 1 / v[0]
        if any((s[0] - x)**2 + (s[1] - y)**2 + (s[2] - z)**2 < s[3]**2 for s in spheres):
            continue
        spheres.append([x, y, z, r])
        print(len(spheres), end='\r')
    print()
    spheres = spheres[4:]
    with open(os.path.join(outdir, "Apollo3d.ply"), "w", encoding="ASCII") as file_out:
        print("ply", file=file_out)
        print("format ascii 1.0", file=file_out)
        print(f"element vertex {len(spheres)}", file=file_out)
        print("property float x", file=file_out)
        print("property float y", file=file_out)
        print("property float z", file=file_out)
        print("property float r", file=file_out)
        print("end_header", file=file_out)
        for s in spheres:
            print(f"{s[0]:.16f} {s[1]:.16f} {s[2]:.16f} {s[3]:.16f}", file=file_out)


def Render_Allcock_chambers(fin=DATASET, outdir="output", form="reduced"):
    lattices = load_dataset(fin)

    lines = []
    lines.append("\\documentclass[tikz]{standalone}\n")
    lines.append("\\begin{document}\n")
    groups = {}

    for j, rec in enumerate(lattices):
        print('#' * 50 + f"{j + 1:^7}" + '#' * 50)
        lstart = time.perf_counter()
        L = Lattice(3, rec["forms"][form]["gram"])
        nwalls = rec["nwalls"]
        Wno = rec["weyl"]
        print(L.info())
        print(L.A)
        V = Vinberg(L, h_batch=100)
        V.print_info()
        walls = V.run(root_batch=1000000)
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
            print("Rendering tiling no", Wno)
            for u in int_seq(3, nonzero=True):
                if math.gcd(*u) != 1:
                    continue
                if L.square(u) > 0 and all(L.product(u, w) > 0 for w in walls):
                    base = u
                    break
            B = L.batch_prod(walls, walls)
            max_prod = -1
            i0, j0 = 0, 1
            for i, j in product(range(len(walls)), repeat=2):
                a = int(B[i][j] ** 2)
                b = int(B[i][i] * B[j][j])
                if a >= b:
                    continue
                if max_prod < a / b:
                    max_prod = a / b
                    i0, j0 = i, j
            if max_prod == -1:
                base = V.base
            else:
                base = L.complement([walls[i0], walls[j0]])[0]
            lines.extend(RenderTiling(L, base))

    lines.append("\\end{document}\n")
    fname = "pattern2.tex"
    with open(os.path.join(outdir, fname), "w", encoding="utf-8") as file_out:
        file_out.writelines(lines)
    result = subprocess.run(["pdflatex", "\\nonstopmode\\input", fname], cwd=outdir,
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"PDFLaTeX compilation failed: {result.stderr.decode('utf-8')}")
