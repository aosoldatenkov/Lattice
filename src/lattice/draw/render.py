import os
import math
from typing import List
import numpy as np
from lattice.lattice import Lattice
from lattice.enumeration import int_seq
from lattice.draw.circle import Circle, CircleArrangement, PDFPicture
from lattice._matrix import *


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

def RenderTiling(L: Lattice, base, max_roots: int = 3000, max_height: int = None, threads: int = 1,
                 width: int = 300, height: int = 300) -> List[str]:
    """The tiling of the hyperbolic plane by the mirrors of the roots of L (of rank 3), as TikZ lines: the Poincare
    disc centred at the positive vector base, with the regions colored alternately (blend group=difference), so that
    the chambers form a checkerboard. The roots are listed by height (r, base) >= 0 (certify.iter_roots), up to
    max_height or until max_roots are found; a root r with (r, base) > 0 is drawn as the circle orthogonal to the
    boundary with center r'/r_0 and radius sqrt(-(r, r))/r_0 (r = (r_0, r') in an orthonormal basis with first
    vector along base), one with (r, base) = 0 as a line through the center."""
    from lattice.vinberg.certify import iter_roots
    if L.rank != 3:
        raise ValueError("The lattice has to be of rank 3")
    base = [int(x) for x in np.ravel(base)]
    if L.square(base) <= 0:
        raise ValueError("The base point has to be a positive vector")
    onb = np.linalg.inv(Lorentz_ONB(L, base))
    roots = []
    for batch in iter_roots(L, imat(base), max_height if max_height is not None else 10 ** 9, threads=threads):
        roots.extend([int(x) for x in r] for r in batch)
        if len(roots) >= max_roots:
            break
    w, h = width // 2, height // 2
    scale = max(w, h)
    W = math.sqrt(w ** 2 + h ** 2)
    lines = ["\\begin{tikzpicture}\n",
             f"  \\fill[fill=white] (-{w}pt,-{h}pt) rectangle ({w}pt,{h}pt);\n",
             f"  \\fill[fill=black] (0pt, 0pt) circle ({scale:.16f}pt);\n",
             f"  \\clip (0pt, 0pt) circle ({scale:.16f}pt);\n",
             "  \\begin{scope}[blend group=difference]\n"]
    lines_through_center = set()
    for r in roots:
        v = np.array(r, dtype=float) @ onb
        if v[0] < -Circle.epsilon:
            continue
        if v[0] < Circle.epsilon:
            # Height 0: both r and -r are listed, a single line through the center (a half-plane)
            if tuple(r) in lines_through_center or tuple(-x for x in r) in lines_through_center:
                continue
            lines_through_center.add(tuple(r))
            n = math.sqrt(v[1] ** 2 + v[2] ** 2)
            nx, ny = -v[1] / n, -v[2] / n
            lines.append(f"  \\fill[fill=white] ({-ny * W:.16f}pt, {nx * W:.16f}pt)"
                         f" -- ({ny * W:.16f}pt, {-nx * W:.16f}pt) -- ({(ny + 2 * nx) * W:.16f}pt, {-(nx - 2 * ny) * W:.16f}pt)"
                         f" -- ({-(ny - 2 * nx) * W:.16f}pt, {(nx + 2 * ny) * W:.16f}pt) -- cycle;\n")
            continue
        x, y = v[1] / v[0] * scale, v[2] / v[0] * scale
        radius = math.sqrt(-L.square(r)) / v[0] * scale
        lines.append(f"  \\fill[fill=white] ({x:.16f}pt, {y:.16f}pt) circle ({radius:.16f}pt);\n")
    lines += [f"  \\fill[fill=black!80] (0pt, 0pt) circle ({scale:.16f}pt);\n",
              "  \\end{scope}\n",
              "\\end{tikzpicture}\n"]
    return lines
