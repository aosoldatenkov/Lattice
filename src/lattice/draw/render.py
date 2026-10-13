import os
import math
from typing import List
import numpy as np
import flint as fl
from lattice.lattice import Lattice
from lattice.enumeration import int_seq
from lattice.draw.circle import Circle, CircleArrangement, PDFPicture


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


def RenderChamber(L, walls, fname, outdir="output"):
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
    pdf = PDFPicture()
    pdf.append(CA.tikz_out(frame=Circle(1, 0, 0, -1), background='white!60!red'))
    pdf.print(os.path.join(outdir, f'{fname}.tex'))

def RenderTiling(L, base):
    if L.rank != 3:
        raise ValueError("The lattice has to be of rank 3")
    from lattice import vsearch_cpp
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
