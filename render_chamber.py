import os
from BinLattice import *
from Lattice import *
from LatticeUtils import *
from DiscForm import *
from IntVectors import *
from Vinberg import *
from FPSearch import *
from VSearch import *
from Allcock import *
from Circle import *
import fp_search_cpp
import vsearch_cpp


def Lorentz_ONB(L: Lattice, axis: List[int]):
    if L.rank < 2 or L.signature[0] != 1:
        raise ValueError("The lattice has to be of signature (1, n) with n > 0")
    if (s := L.square(axis)) <= 0:
        raise ValueError("The axis has to be a positive vector")
    axis_np = np.array(axis, dtype=np.float64) / np.sqrt(float(s))
    C = L.complement([axis])
    C_np = np.array(C, dtype=np.float64)
    M = L.batch_prod(C, C)
    M_np = np.array(M, dtype=np.float64)
    eival, eivec = np.linalg.eigh(M_np)
    eival_abs = np.sqrt(np.abs(eival))
    eivec_norm = eivec / eival_abs
    norm_basis = eivec_norm.T @ C_np
    return np.array([axis_np] + [norm_basis[i, :] for i in range(L.rank - 1)], dtype=np.float64)


def Apollonian():
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
    os.chdir('output')
    count = sum(1 for c in CA.circles if c[0].is_bounded())
    with open("Apollo.ply", "w", encoding="ASCII") as file_out:
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

def Apollonian3D():
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
    os.chdir('output')
    with open("Apollo3d.ply", "w", encoding="ASCII") as file_out:
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


def RenderChamber(L, walls, fname):
    if L.rank != 3:
        raise ValueError("The lattice has to be of rank 3")
    #A = np.array(L.A, dtype=np.float64)
    base = [sum(w[i] for w in walls) for i in range(3)]
    if L.square(base) <= 0 or not all(L.product(base, w) > 0 for w in walls):
        for u in int_seq(3, nonzero=True):
            if math.gcd(*u) != 1:
                continue
            if L.square(u) > 0 and all(L.product(u, w) > 0 for w in walls):
                base = u
                break
    print(base)
    onb = np.linalg.inv(Lorentz_ONB(L, base))
    CA = CircleArrangement()
    CA.add_circle(Circle(-1, 0, 0, 1, disc=1, color='black'), depth=1)
    #CA.add_circle(Circle(1, 0, 0, 1, disc=1, color='black!30!red'), depth=1)
    for v in walls:
        w = np.array(v, dtype=float) @ onb
        # if w[0] < 0:
        #     w *= -1
        C = Circle(w[0], -2 * w[1], -2 * w[2], w[0], disc=1, color='white')
        CA.add_circle(C, 0)
        C = Circle(w[0], -2 * w[1], -2 * w[2], w[0], color='black')
        CA.add_circle(C, 0.1)
    os.chdir('output')
    pdf = PDFPicture()
    pdf.append(CA.tikz_out(frame=Circle(1, 0, 0, -1), background='white!60!red'))
    pdf.print(f'{fname}.tex')

def RenderTiling(L, base):
    if L.rank != 3:
        raise ValueError("The lattice has to be of rank 3")
    axis, _ = L.dual_vec(base)
    compl = L.complement([base])
    basis = [axis] + compl
    L = Lattice(3, L.batch_prod(basis, basis))
    print(L.A)
    B, _ = fl.fmpq_mat(basis).inv().numer_denom()
    base = (fl.fmpz_mat(1, 3, base) * B).tolist()[0]
    onb = np.linalg.inv(Lorentz_ONB(L, base))
    V = vsearch_cpp.VSearchCpp(L.A.tolist(), base, 2.1 * L.exp + 0.5, 50, False, True)
    vecs = []
    while len(vecs) < 3 * 10 ** 3:
        V.run(1000, 10000)
        vecs.extend(V.get_vecs())
        print(len(vecs), 'roots found', end='\r')
    print()
    width = 300
    height = 300
    w, h = width // 2, height // 2
    scale = max(w, h)
    lines = []
    lines.append("\\begin{tikzpicture}\n")
    lines.append(f"  \\fill[fill=white] (-{w}pt,-{h}pt) rectangle ({w}pt,{h}pt);\n")
    lines.append(f"  \\fill[fill=black] (0pt, 0pt) circle ({scale:.16f}pt);\n")
    lines.append(f"  \\clip (0pt, 0pt) circle ({scale:.16f}pt);\n")
    lines.append("  \\begin{scope}[blend group=difference]\n")
    half_spaces = []
    for u in vecs:
        if math.gcd(*u) != 1:
            continue
        v = np.array(u, dtype=float) @ onb
        if v[0] < -Circle.epsilon:
            continue
        if v[0] < Circle.epsilon:
            if u in half_spaces or [-x for x in u] in half_spaces:
                continue
            half_spaces.append(u)
            n = math.sqrt(v[1] ** 2 + v[2] ** 2)
            nx, ny = -v[1] / n, -v[2] / n
            W = math.sqrt(w ** 2 + h ** 2)
            lines.append(f"  \\fill[fill=white] ({-ny * W:.16f}pt, {nx * W:.16f}pt)" + 
                        f" -- ({ny * W:.16f}pt, {-nx * W:.16f}pt) -- ({(ny + 2 * nx) * W:.16f}pt, {-(nx - 2 * ny) * W:.16f}pt)" +
                        f" -- ({-(ny - 2 * nx) * W:.16f}pt, {(nx + 2 * ny) * W:.16f}pt) -- cycle;\n")
            continue
        x = (v[1] / v[0]) * scale
        y = (v[2] / v[0]) * scale
        r = (math.sqrt(abs(L.square(u))) / v[0]) * scale
        lines.append(f"  \\fill[fill=white] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
    lines.append(f"  \\fill[fill=black!80] (0pt, 0pt) circle ({scale:.16f}pt);\n")
    lines.append("  \\end{scope}\n")
    lines.append("\\end{tikzpicture}\n")
    return lines


def Render_Allcock_chambers():
    with open('out', "r") as f:
        lattices = [re.findall(r'-?\d+', line.strip()) for line in f.readlines()]

    lines = []
    lines.append("\\documentclass[tikz]{standalone}\n")
    lines.append("\\begin{document}\n")
    groups = {}

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
    os.chdir('output')
    fname = "pattern2.tex"
    with open(fname, "w", encoding="utf-8") as file_out:
        file_out.writelines(lines)
    result = subprocess.run(["pdflatex", "\\nonstopmode\\input", fname], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"PDFLaTeX compilation failed: {result.stderr.decode('utf-8')}")
