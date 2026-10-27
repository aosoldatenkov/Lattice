"""Readable reports of the certificates (and of the walls files) of lattice.vinberg, as PDF documents (LaTeX).

certificate_report(cert, path) recognizes the kind of its input:
  - "lattice.vinberg.orbits certificate" (trace_chamber_by_orbits): L is reflective; the walls of the chamber C, the
    generators of Sym(C), and one vertex per orbit of Sym(C);
  - "lattice.vinberg.nonreflective certificate", of type "symmetry" (search_nonreflective: identifications of
    vertices of C and a product of them of infinite order) or "geodesic" (a geodesic in C with irrational ends);
  - "lattice.vinberg.quotient certificate" (quotient_chamber): the vertices of C up to Sym(C), and the structure of
    Sym(C) (finite: L reflective; parabolic; virtually cyclic; non-elementary);
  - a walls file of trace_chamber(walls_file=...) (not a certificate: the walls found so far).
The report starts with the lattice (Gram matrix, discriminant group and form, genus), then states the claim of the
certificate and lists its content: walls (with the Coxeter diagram if it is small), symmetries (permutations of the
walls, matrices, characteristic polynomials), representatives of the vertices with the types of their local
diagrams, and the analysis of Sym(C). It is built from the certificate alone, without retracing anything (the values
derived from it, such as norms, heights and diagram types, are recomputed from its vectors); with verify=True the
certificate is also checked by its verification function, and the outcome is reported. Indices of walls and
representatives start at 0, as in the certificate files.
"""
from __future__ import annotations
import datetime
import json
import math
import os
import subprocess
import tempfile
import time
from collections import Counter
from typing import Dict, List, Sequence
import flint as fl
from lattice.lattice import Lattice
from lattice.discform import DiscForm
from lattice.genus import Genus
from lattice.vinberg.coxeter import WallDiagram

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{longtable,booktabs,array}
\usepackage{tikz}
\usepackage{adjustbox}
\usepackage{xcolor}
\setcounter{MaxMatrixCols}{100}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\newcolumntype{V}[1]{>{\raggedright\arraybackslash}p{#1}}
"""


# --- LaTeX helpers ---

def _esc(text: str) -> str:
    """Text with the special characters of LaTeX escaped."""
    out = []
    for c in str(text):
        out.append({"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "^": r"\^{}", "#": r"\#",
                    "%": r"\%", "&": r"\&", "$": r"\$", "~": r"\~{}"}.get(c, c))
    return "".join(out)


def _tt(text: str) -> str:
    """Typewriter text, breakable after the dots and underscores."""
    return "\\texttt{" + _esc(text).replace(".", ".\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}") + "}"


def _vec(v) -> str:
    """A vector in math mode, breakable after the commas."""
    return "(" + ",\\allowbreak ".join(str(int(x)) for x in v) + ")"


def _num(q) -> str:
    """A rational number (fmpq, int, or a string "p/q")."""
    q = fl.fmpq(*map(int, q.split("/"))) if isinstance(q, str) else fl.fmpq(q)
    if q.q == 1:
        return str(int(q.p))
    sign = "-" if q < 0 else ""
    return f"{sign}\\tfrac{{{abs(int(q.p))}}}{{{int(q.q)}}}"


def _matrix(M, small: bool = None) -> str:
    """A matrix (rows of numbers) in display math, scaled down to the width of the text if needed."""
    rows = [list(r) for r in M]
    if not rows:
        return "$()$"
    body = " \\\\ ".join(" & ".join(_num(x) for x in r) for r in rows)
    size = "\\scriptsize" if small or (small is None and len(rows[0]) > 12) else ""
    return (f"\\begin{{center}}\\begin{{adjustbox}}{{max width=\\textwidth}}{size}"
            f"$\\begin{{pmatrix}} {body} \\end{{pmatrix}}$\\end{{adjustbox}}\\end{{center}}")


def _poly(coeffs, var: str = "x") -> str:
    """A polynomial with the integer coefficients coeffs (constant first)."""
    terms = []
    for e in range(len(coeffs) - 1, -1, -1):
        c = int(coeffs[e])
        if c == 0:
            continue
        mono = "" if e == 0 else (var if e == 1 else f"{var}^{{{e}}}")
        coef = str(abs(c)) if abs(c) != 1 or e == 0 else ""
        terms.append(("-" if c < 0 else "+") + coef + mono)
    s = "".join(terms) or "0"
    return s[1:] if s.startswith("+") else s


def _factored_charpoly(M: fl.fmpz_mat) -> str:
    """The characteristic polynomial of M, factored over Q, with the type of each factor (cyclotomic or Salem)."""
    _, factors = M.charpoly().factor()
    parts = []
    for f, e in sorted(factors, key=lambda fe: (fe[0].degree(), str(fe[0]))):
        c = f.coeffs()
        p = f"({_poly(c)})" if len(c) > 2 or e > 1 or len(factors) > 1 else _poly(c)
        parts.append(p + (f"^{{{e}}}" if e > 1 else ""))
    return " ".join(parts)


def _type(types: Sequence[str]) -> str:
    """The type of a Coxeter diagram (WallDiagram.type) in math mode: A_1 + G_2^3, \\tilde E_6."""
    counts = Counter(types)
    out = []
    for t in dict.fromkeys(types):
        letter, n = t.split("_")
        name = f"\\tilde{{{letter[0]}}}_{{{n}}}" if letter.endswith("~") else f"{letter}_{{{n}}}"
        if letter == "?":
            name = f"\\mathrm{{?}}_{{{n}}}"
        out.append(name + (f"^{{{counts[t]}}}" if counts[t] > 1 else ""))
    return " + ".join(out) if out else "\\emptyset"


def _cycles(perm: Sequence[int]) -> str:
    """A permutation of 0, ..., k - 1 in cycle notation (math mode)."""
    seen, cycles = set(), []
    for i in range(len(perm)):
        if i in seen or perm[i] == i:
            seen.add(i)
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = perm[j]
        cycles.append("(" + "\\,".join(str(x) for x in c) + ")")
    return "\\allowbreak ".join(cycles) if cycles else "\\mathrm{id}"


def _longtable(header: Sequence[str], spec: str, rows: List[Sequence[str]], max_rows: int = None) -> str:
    """A longtable (cells already in LaTeX), truncated to max_rows rows."""
    shown = rows if max_rows is None or len(rows) <= max_rows else rows[:max_rows]
    lines = [f"\\begin{{longtable}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead"]
    lines += [" & ".join(r) + " \\\\" for r in shown]
    if len(shown) < len(rows):
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{\\ldots{{}} and {len(rows) - len(shown)} "
                     f"more}}}} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    return "\n".join(lines)


def _orbits(k: int, perms) -> List[int]:
    """The orbit number of each of 0, ..., k - 1 under the permutations (numbered by their smallest element)."""
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for p in perms:
        for i in range(k):
            a, b = find(i), find(p[i])
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = sorted({find(i) for i in range(k)})
    number = {r: t for t, r in enumerate(roots)}
    return [number[find(i)] for i in range(k)]


# --- the lattice ---

def _lattice_section(L: Lattice, base=None, chamber=None) -> List[str]:
    out = ["\\section{The lattice}"]
    parity = "odd" if L.parity else "even"
    det = int(L.A_fl.det())
    out.append(f"$L = \\mathbb{{Z}}^{{{L.rank}}}$ with the Gram matrix $A$ below: an {parity} lattice of rank "
               f"{L.rank}, signature $({L.signature[0]}, {L.signature[1]})$ and determinant ${det}$. Vectors are "
               f"rows of coordinates in the standard basis $e_1, \\ldots, e_{{{L.rank}}}$, with $(u, v) = u A v^T$.")
    out.append(_matrix(L.A.tolist()))
    D = DiscForm(L)
    if not D.ord:
        out.append("The discriminant group $L^\\vee / L$ is trivial: $L$ is unimodular.")
    else:
        group = " \\oplus ".join(f"\\mathbb{{Z}}/{d}" for d in D.ord)
        out.append(f"\\textbf{{Discriminant group}} $A_L = L^\\vee / L \\cong {group}$ (order {L.disc}, exponent "
                   f"{L.exp}), with generators $g_1, \\ldots, g_{{{len(D.ord)}}}$:")
        i0 = L.rank - len(D.ord)
        rows = []
        for t, d in enumerate(D.ord):
            g = [D.dual[i0 + t, j] for j in range(L.rank)]
            rows.append([f"$g_{{{t + 1}}}$", f"${d}$", "$(" + ",\\allowbreak ".join(_num(x) for x in g) + ")$"])
        out.append(_longtable(["", "order", "in the coordinates of $L$"], "llV{0.7\\textwidth}", rows))
        # The discriminant form: b(g_i, g_j) in Q/Z off the diagonal, and q(g_i) = (g_i, g_i) in Q/2Z (even L) or
        # b(g_i, g_i) in Q/Z (odd L) on the diagonal
        mod = 1 if L.parity else 2
        F = [[None] * len(D.ord) for _ in D.ord]
        for a in range(len(D.ord)):
            for b in range(len(D.ord)):
                x = D.A[i0 + a, i0 + b]
                m = mod if a == b else 1
                F[a][b] = fl.fmpq(x.p % (m * x.q), x.q)
        what = ("$q(g_i) = (g_i, g_i) \\bmod 2\\mathbb{Z}$ on the diagonal" if mod == 2 else
                "$b(g_i, g_i) \\bmod \\mathbb{Z}$ on the diagonal")
        out.append(f"\\textbf{{Discriminant form}}: $b(g_i, g_j) = (g_i, g_j) \\bmod \\mathbb{{Z}}$, and {what}:")
        out.append(_matrix(F))
    try:
        G = Genus(L)
        out.append(f"\\textbf{{Genus}} (Conway--Sloane symbol; the 2-adic part is not normalized): "
                   f"${G.str()}$.")
    except Exception as e:                          # the genus symbol is informative only
        out.append(f"\\textbf{{Genus}}: not computed ({_esc(e)}).")
    if base is not None:
        out.append(f"\\textbf{{Base point}} of the chamber: ${_vec(base)}$, of norm ${L.square(list(base))}$" +
                   (f"; chamber vector ${_vec(chamber)}$." if chamber is not None else "."))
    return out


# --- walls and vertices ---

def _walls_rows(L: Lattice, walls, extra=None) -> List[List[str]]:
    rows = []
    for i, w in enumerate(walls):
        row = [str(i), f"${_vec(w)}$", f"${L.square(list(w))}$"]
        if extra is not None:
            row += extra(i, w)
        rows.append(row)
    return rows


def _diagram_figure(L: Lattice, walls, max_nodes: int) -> List[str]:
    if not walls or len(walls) > max_nodes:
        return []
    from lattice.draw.diagram import CoxeterDiagram
    D = CoxeterDiagram(L, walls)
    legend = ", ".join(f"\\textcolor{{c{n}}}{{$\\bullet$}} $(r, r) = {-n}$" for n in sorted(D.color))
    lines = [line.rstrip("\n") for line in D.tikz()]
    colors = [line for line in lines if line.startswith("\\definecolor")]     # also for the legend
    return [*colors, "\\begin{center}",
            "\\begin{adjustbox}{max width=\\textwidth, max totalheight=0.6\\textheight}",
            "\\begin{tikzpicture}", *[line for line in lines if line not in colors], "\\end{tikzpicture}",
            "\\end{adjustbox}\\\\[0.5em]",
            f"\\parbox{{0.9\\textwidth}}{{\\small The Coxeter diagram of the walls (nodes numbered as in the "
            f"table; simple, double, triple edges: angles $\\pi/3, \\pi/4, \\pi/6$; thick: parallel; dashed: "
            f"ultraparallel). {legend}.}}",
            "\\end{center}"]


def _local_type(L: Lattice, walls) -> str:
    return f"${_type(WallDiagram(L, walls).type())}$"


# --- the kinds of certificates ---

def _orbits_report(L, cert, opts) -> List[str]:
    walls, group, counts = cert["walls"], cert["group"], cert["counts"]
    k = len(walls)
    perms = [g["perm"] for g in group["generators"]]
    orbit = _orbits(k, perms)
    out = ["\\section{The claim}",
           f"$L$ is \\textbf{{reflective}}. The fundamental chamber $C$ of the reflection group $W(L)$ containing the "
           f"base point has {counts['walls']} walls, {counts['vertices']} finite vertices, {counts['cusps']} cusps "
           f"(ideal vertices) and {counts['edges']} edges. Its symmetry group $\\mathrm{{Sym}}(C) = O^+(L)/W(L)$ has "
           f"order {group['order']} (index {group['index']} in the group of the {group['aut_order']} automorphisms "
           f"of its Gram matrix); the vertices form {counts['orbits']} orbits under it.",
           "\\section{The walls}",
           f"The {k} walls $r$ of $C$ (roots with $(x, r) \\ge 0$ on $C$), in {max(orbit) + 1} orbits of "
           f"$\\mathrm{{Sym}}(C)$:"]
    out += _diagram_figure(L, walls, opts["diagram"])
    out.append(_longtable(["\\#", "$r$", "$(r, r)$", "orbit"], "rV{0.62\\textwidth}rr",
                          _walls_rows(L, walls, lambda i, w: [str(orbit[i])]), opts["max_rows"]))
    out += ["\\section{The symmetries}",
            f"$\\mathrm{{Sym}}(C)$ (order {group['order']}) is generated by the following isometries $g$ of $L$, "
            f"given as permutations $s$ of the walls ($r_i g = r_{{s(i)}}$) and as matrices (row action):"]
    for t, g in enumerate(group["generators"]):
        out.append(f"$g_{{{t}}}$: $s = {_cycles(g['perm'])}$")
        if L.rank <= opts["max_matrix"]:
            out.append(_matrix(g["matrix"]))
    if not group["generators"]:
        out.append("(none: $\\mathrm{Sym}(C)$ is trivial)")
    out += ["\\section{The vertices}",
            f"One vertex per orbit of $\\mathrm{{Sym}}(C)$ (cusps: isotropic, with an affine local diagram), with the "
            f"walls through it and the size of its orbit:"]
    rows = []
    for i, r in enumerate(cert["representatives"]):
        rows.append([str(i), "cusp" if r["ideal"] else "finite", f"${_vec(r['vector'])}$",
                     f"${L.square(r['vector'])}$", _local_type(L, [walls[j] for j in r["walls"]]),
                     "$" + ",\\allowbreak ".join(str(j) for j in r["walls"]) + "$", str(r["orbit"])])
    out.append(_longtable(["\\#", "", "vertex", "norm", "diagram", "walls", "orbit"],
                          "rlV{0.3\\textwidth}rlV{0.2\\textwidth}r", rows, opts["max_rows"]))
    return out


def _symmetry_matrix(L: Lattice, x, r, sigma) -> fl.fmpz_mat:
    """g = B_x^-1 P_sigma B_r (nonreflective.verify_certificate), or None if it is not integral."""
    k = len(x["walls"])
    g = fl.fmpq_mat(x["walls"] + [x["vector"]]).inv() * \
        fl.fmpq_mat([r["walls"][sigma[i]] for i in range(k)] + [r["vector"]])
    if any(e.q != 1 for e in g.entries()):
        return None
    return fl.fmpz_mat([[int(g[i, j]) for j in range(g.ncols())] for i in range(g.nrows())])


def _element_analysis(L: Lattice, M: fl.fmpz_mat, opts) -> List[str]:
    from lattice.vinberg.nonreflective import classify_element
    out = []
    if L.rank <= opts["max_matrix"]:
        out.append(_matrix([[int(M[i, j]) for j in range(M.ncols())] for i in range(M.nrows())]))
    out.append(f"Its characteristic polynomial is ${_factored_charpoly(M)}$.")
    kind, c = classify_element(L, M)
    if kind == "hyperbolic":
        f = next(p for p, _ in M.charpoly().factor()[1] if p.is_cyclotomic() == 0)
        lam = max(abs(complex(r)) for r, _ in fl.fmpz_poly(f.coeffs()).complex_roots())
        out.append(f"It is \\textbf{{hyperbolic}}: the factor ${_poly(f.coeffs())}$ is not cyclotomic (a Salem "
                   f"factor, with the root $\\lambda \\approx {lam:.6g} > 1$), so it has infinite order; it "
                   f"translates its axis by $\\log \\lambda$.")
    elif kind == "parabolic":
        out.append(f"It is \\textbf{{parabolic}} (unipotent up to a finite order, not of finite order), fixing the "
                   f"cusp $c = {_vec(c)}$.")
    else:
        out.append("It has \\textbf{finite order}.")
    return out


def _nonreflective_report(L, cert, opts) -> List[str]:
    if cert.get("type") == "geodesic":
        a, u = cert["point"], cert["direction"]
        p, q = L.square(a), L.square(u)
        out = ["\\section{The claim}",
               "$L$ is \\textbf{not reflective}: the chamber $C$ contains a whole geodesic with irrational ends, so "
               "it has infinite volume (its symmetry group contains the translation of that geodesic, of infinite "
               "order).",
               "\\section{The geodesic}",
               f"The geodesic through the point $a = {_vec(a)}$, of norm ${p}$, in the direction $u = {_vec(u)}$, of "
               f"norm ${q}$, with $(a, u) = {L.product(a, u)}$. The plane spanned by $a$ and $u$ has signature $(1, 1)$ "
               f"and discriminant $-(a, a)(u, u) = {-p * q}$, not a square: it contains no isotropic vector, and the "
               f"ends of the geodesic are irrational. The stabilizer of the plane in $O(L)$ contains a translation $g$ "
               f"of the geodesic (on the lattice of the plane, extended by the identity on its orthogonal complement); "
               f"the certificate states that no mirror of a root crosses the geodesic over one period, from $a$ to "
               f"$g(a)$, so none crosses it at all, and the geodesic lies in a chamber."]
        try:
            from lattice.vinberg.trace import _river
            from lattice._matrix import imat
            R = _river(L, imat(a), imat(u))
            basis = [[int(x) for x in r] for r in R["basis"]]
            out.append(f"The lattice of the plane has the basis ${_vec(basis[0])}, {_vec(basis[1])}$, with the Gram "
                       f"matrix")
            out.append(_matrix([[L.product(x, y) for y in basis] for x in basis]))
            out.append("and $g$ acts on it (in this basis, row action) by")
            out.append(_matrix([[int(x) for x in row] for row in R["g"]]))
        except Exception:
            pass
        return out
    root, idents = cert["root"], cert["identifications"]
    out = ["\\section{The claim}",
           "$L$ is \\textbf{not reflective}: the chamber $C$ has a symmetry of infinite order. Each identification "
           "below is an integral isometry $g = B_x^{-1} P_\\sigma B_r$ mapping a vertex $x$ of $C$ to a vertex $r$ "
           "of $C$ and the walls at $x$ to those at $r$ ($B$: the matrix of the walls and the vertex), hence a "
           "symmetry of $C$ (it maps the local cone of $C$ at $x$ to that at $r$); the vertices are reached from the "
           "root by the stated paths of edges, and a product of these symmetries has infinite order.",
           "\\section{The root}",
           f"The root vertex $x_0 = {_vec(root['vector'])}$, of norm ${L.square(root['vector'])}$, with the "
           f"{len(root['walls'])} walls (local diagram {_local_type(L, root['walls'])}):"]
    out.append(_longtable(["\\#", "$r$", "$(r, r)$"], "rV{0.7\\textwidth}r", _walls_rows(L, root["walls"])))
    out.append("\\section{The identifications}")
    elements = []
    for t, ident in enumerate(idents):
        x, r, sigma = ident["x"], ident["r"], ident["sigma"]
        out.append(f"\\subsection*{{Identification {t}}}")
        for name, v in (("x", x), ("r", r)):
            out.append(f"${name} = {_vec(v['vector'])}$, of norm ${L.square(v['vector'])}$, local diagram "
                       f"{_local_type(L, v['walls'])}, reached from the root by a path of {len(v['path'])} "
                       f"edge{'' if len(v['path']) == 1 else 's'}.")
        out.append(f"The bijection of the walls: $\\sigma = {_cycles(sigma)}$ (wall $i$ of $x$ to wall $\\sigma(i)$ "
                   f"of $r$, in the order of the certificate).")
        g = _symmetry_matrix(L, x, r, sigma)
        elements.append(g)
        if g is None:
            out.append("\\textcolor{red}{The isometry of this identification is not integral.}")
        elif L.rank <= opts["max_matrix"]:
            out.append(f"$g_{{{t}}} =$")
            out.append(_matrix([[int(g[i, j]) for j in range(g.ncols())] for i in range(g.nrows())]))
    word = cert["word"]
    out.append("\\section{The element of infinite order}")
    product = " ".join(f"g_{{{i}}}" + ("^{-1}" if e == -1 else "") for i, e in word)
    out.append(f"The product $g = {product}$ is")
    if all(g is not None for g in elements):
        from lattice.vinberg.nonreflective import _inverse
        M = fl.fmpz_mat([[int(i == j) for j in range(L.rank)] for i in range(L.rank)])
        for i, e in word:
            M = M * (elements[i] if e == 1 else _inverse(elements[i]))
        out += _element_analysis(L, M, opts)
    return out


def _quotient_report(L, cert, opts) -> List[str]:
    reps, idents, claim = cert["representatives"], cert["identifications"], cert["structure"]
    s = claim["structure"]
    cusp = claim.get("cusp")
    n_cusps = sum(r["ideal"] for r in reps)
    out = ["\\section{The claim}"]
    if s == "finite":
        order = f" of order {claim['order']}" if "order" in claim else ""
        out.append(f"$L$ is \\textbf{{reflective}}: $\\mathrm{{Sym}}(C) = O^+(L)/W(L)$ is finite{order}, and the "
                   f"chamber $C$ has {claim.get('walls', '?')} walls (the orbit of the walls at the representatives "
                   f"below under the generators).")
    elif s == "parabolic":
        out.append(f"$L$ is \\textbf{{not reflective}}: $\\mathrm{{Sym}}(C)$ is infinite and fixes the cusp "
                   f"$c = {_vec(cusp)}$ (a parabolic element and all the generators fix it); so $\\mathrm{{Sym}}(C)$ "
                   f"is virtually abelian, of rank at most ${L.rank - 2}$.")
    elif s == "virtually cyclic":
        out.append("$L$ is \\textbf{not reflective}: $\\mathrm{Sym}(C)$ is infinite and preserves a geodesic, the "
                   "axis of a hyperbolic symmetry $h$ (all the generators preserve its pair of ends): it is virtually "
                   "cyclic.")
    elif s == "non-elementary":
        out.append("$L$ is \\textbf{not reflective}: $\\mathrm{Sym}(C)$ is infinite and non-elementary (it contains "
                   "elements of infinite order with disjoint fixed points at infinity, hence a free group of rank "
                   "2).")
    else:
        out.append(f"Structure: {_esc(s)}.")
    out.append(f"The certificate lists {len(reps)} vertices of $C$, one per orbit of $\\mathrm{{Sym}}(C)$ "
               f"({len(reps) - n_cusps} finite, {n_cusps} cusps), and {len(idents)} identifications: each traced edge "
               f"of a representative ends at a representative or at its image by a stated symmetry. The "
               f"identifications and the stabilizers of the representatives generate $\\mathrm{{Sym}}(C)$; its "
               f"structure is decided from these generators.")
    out.append("\\section{The structure of $\\mathrm{Sym}(C)$}")
    walls = list(dict.fromkeys(tuple(w) for r in reps for w in r["walls"]))
    if s == "parabolic":
        heights = sorted({(-L.square(list(w)), L.product(cusp, list(w))) for w in walls})
        out.append(f"The fixed cusp $c = {_vec(cusp)}$. The heights $(c, r)$ of the walls at the representatives "
                   f"are the heights of all the walls of $C$ (as $\\mathrm{{Sym}}(C)$ fixes $c$); by norm $-(r, r)$: " +
                   ", ".join(f"$({d}, {h})$" for d, h in heights) + ".")
        if all(h >= 0 for _, h in heights):
            ratios = {fl.fmpq(h) / fl.fmpq(d, 2) for d, h in heights}
            if len(ratios) == 1 and 0 not in ratios:
                ratio = next(iter(ratios))
                multiple = "\\rho" if ratio == 1 else f"{_num(ratio)}\\,\\rho"
                out.append(f"They are proportional to $-(r, r)/2$: $c$ is a multiple of a \\textbf{{lattice Weyl "
                           f"vector}} $\\rho$ of norm 0, with $(\\rho, r) = -(r, r)/2$ for all the walls "
                           f"($c = {multiple}$).")
            else:
                out.append("They are bounded and $\\ge 0$: $c$ is a \\textbf{generalized lattice Weyl vector}.")
        zero = [list(w) for w in walls if L.product(cusp, list(w)) == 0]
        rank = fl.fmpz_mat(zero).rank() if zero else 0
        out.append(f"The walls through $c$ have rank {rank}.")
    if s == "virtually cyclic" and "salem polynomial" in claim:
        out.append(f"The hyperbolic element $h$ has the eigenvalue $\\lambda \\approx {claim['lambda']:.6g}$, a root "
                   f"of ${_poly(claim['salem polynomial'])}$; it translates its axis by $\\log\\lambda$" +
                   (", and some generator swaps the ends of the axis (an extension of the infinite dihedral group)."
                    if claim.get("reverses") else "."))
        if "hyperbolic element" in claim and L.rank <= opts["max_matrix"]:
            out.append("$h =$")
            out.append(_matrix(claim["hyperbolic element"]))
        axis = claim.get("axis")
        if axis:
            d = len(axis["lattice"])
            if d == L.rank:
                out.append("The axis lies in no proper rational subspace (the Salem factor has the degree of $L$).")
            else:
                what = ("the lattice of the plane of the axis" if d == 2 else
                        f"the lattice of the rational hull of the axis (rank {d}; the plane of the axis is not "
                        f"rational)")
                out.append(f"A basis of {what}: " + ", ".join(f"${_vec(v)}$" for v in axis["lattice"]) + ".")
            out.append("The ends of the axis (isotropic, scaled to $(v_+, v_-) = 1$) are $v_\\pm \\approx " +
                       ", ".join(f"({', '.join(f'{x:.4g}' for x in axis[k])})" for k in ("plus", "minus")) + "$.")
    if s == "non-elementary" and "moves the axis" in claim:
        out.append(f"A hyperbolic element $h$ (the word {_esc(claim['word'])} in the generators) and the generator "
                   f"{claim['moves the axis']}, which moves its axis.")
    if s == "finite" and "order" in claim:
        out.append(f"$\\mathrm{{Sym}}(C)$ is finite of order {claim['order']}.")
    out.append("\\section{The representatives}")
    out.append("The vertices of $C$ up to $\\mathrm{Sym}(C)$, each reached from its parent (another representative) "
               "along an edge; ``edges'': the number of edges traced (one per orbit of its stabilizer).")
    rows = []
    for i, r in enumerate(reps):
        S = r["walls"]
        parent = "--" if r["parent"] is None else str(r["parent"][0])
        rows.append([str(i), "cusp" if r["ideal"] else "finite", f"${_vec(r['vector'])}$",
                     f"${L.square(r['vector'])}$", _local_type(L, S), str(len(r["traced"])), parent])
    out.append(_longtable(["\\#", "", "vertex", "norm", "local diagram", "edges", "parent"],
                          "rlV{0.38\\textwidth}rlrr", rows, opts["max_rows"]))
    out.append("\\section{The walls at the representatives}")
    out.append(f"The {len(walls)} distinct walls at the representatives: every wall of $C$ is in the "
               f"$\\mathrm{{Sym}}(C)$-orbit of one of them" + (" (heights $(c, r)$ for the fixed cusp $c$)."
                                                              if s == "parabolic" else "."))
    extra = (lambda i, w: [f"${L.product(cusp, list(w))}$"]) if s == "parabolic" else None
    out.append(_longtable(["\\#", "$r$", "$(r, r)$"] + (["$(c, r)$"] if extra else []),
                          "rV{0.62\\textwidth}r" + ("r" if extra else ""), _walls_rows(L, walls, extra),
                          opts["max_rows"]))
    return out


def _walls_file_report(L, data, opts) -> List[str]:
    walls = data["walls"]
    out = ["\\section{The walls}",
           f"A walls file of {_tt('trace_chamber')}: the {len(walls)} walls of the chamber $C$ found so far "
           f"(\\emph{{not a certificate}}: certify them with {_tt('trace_chamber_by_orbits')}, or find the structure of "
           f"$\\mathrm{{Sym}}(C)$ with {_tt('quotient_chamber')})."]
    out += _diagram_figure(L, walls, opts["diagram"])
    out.append(_longtable(["\\#", "$r$", "$(r, r)$"], "rV{0.7\\textwidth}r", _walls_rows(L, walls), opts["max_rows"]))
    return out


# --- verification ---

def _verify(kind: str, cert: dict, threads: int) -> dict:
    if kind == "orbits":
        from lattice.vinberg.orbits import verify_certificate
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cert, f)
        try:
            return verify_certificate(f.name, threads=threads)
        finally:
            os.unlink(f.name)
    if kind == "nonreflective":
        from lattice.vinberg.nonreflective import verify_certificate
        return verify_certificate(cert)
    from lattice.vinberg.quotient import verify_quotient
    return verify_quotient(cert, threads=threads)


VERIFY = {"orbits": "lattice.vinberg.orbits.verify_certificate(path)",
          "nonreflective": "lattice.vinberg.nonreflective.verify_certificate(path)",
          "quotient": "lattice.vinberg.quotient.verify_quotient(path)"}

TITLES = {"orbits": "Certificate of reflectivity (orbits of vertices)",
          "nonreflective": "Certificate of non-reflectivity",
          "quotient": "Certificate of the structure of Sym(C) (the chamber modulo its symmetries)",
          "walls": "Walls of a chamber (not a certificate)"}


def certificate_kind(cert: dict) -> str:
    """"orbits", "nonreflective", "quotient" or "walls" (a walls file), from the format of the data."""
    fmt = cert.get("format", "")
    for kind in ("orbits", "nonreflective", "quotient"):
        if fmt.startswith(f"lattice.vinberg.{kind} certificate"):
            return kind
    if not fmt and "walls" in cert and "gram" in cert:
        return "walls"
    raise ValueError(f"Not a certificate of lattice.vinberg (format {fmt!r})")


def certificate_report(cert: dict | str, path: str = None, title: str = None, verify: bool = False,
                       threads: int = 1, diagram: int = 120, max_rows: int = None, max_matrix: int = 16,
                       compile: bool = True) -> str:
    """Writes a readable report of a certificate of lattice.vinberg (or of a walls file), a dict or a JSON file, to
    path (.pdf; by default the path of the file with the extension .pdf), as LaTeX compiled with pdflatex (the .tex
    is kept next to it); returns the path of the PDF (or of the .tex without compile). See the module docstring.
        verify: also check the certificate (with threads processes), and report the outcome;
        diagram: the largest number of walls for which the Coxeter diagram is drawn;
        max_rows: truncate the long tables to this many rows (None: all);
        max_matrix: the largest rank for which the matrices of the symmetries are printed."""
    source = cert if isinstance(cert, str) else None
    if source is not None:
        with open(source) as f:
            cert = json.load(f)
    if path is None:
        if source is None:
            raise ValueError("No output path")
        path = os.path.splitext(source)[0] + ".pdf"
    kind = certificate_kind(cert)
    L = Lattice(len(cert["gram"]), cert["gram"])
    opts = {"diagram": diagram, "max_rows": max_rows, "max_matrix": max_matrix}
    body = _lattice_section(L, cert.get("base"), cert.get("chamber"))
    body += {"orbits": _orbits_report, "nonreflective": _nonreflective_report, "quotient": _quotient_report,
             "walls": _walls_file_report}[kind](L, cert, opts)
    if kind != "walls":
        body.append("\\section{Verification}")
        body.append(f"The certificate (format {_tt(cert['format'])}) is checked by {_tt(VERIFY[kind])}, which "
                    f"recomputes everything it states from its data.")
        if verify:
            t = time.perf_counter()
            try:
                _verify(kind, cert, threads)
                body.append(f"\\textbf{{Verified}} on {datetime.date.today().isoformat()} "
                            f"({time.perf_counter() - t:.2f} s).")
            except Exception as e:
                body.append(f"\\textcolor{{red}}{{\\textbf{{The verification failed}}: {_esc(e)}}}")
    title = title or TITLES[kind]
    subtitle = f"\\\\ \\large \\texttt{{{_esc(os.path.basename(source))}}}" if source else ""
    tex = os.path.splitext(path)[0] + ".tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write(PREAMBLE)
        f.write(f"\\title{{{_esc(title)}{subtitle}}}\n\\date{{{datetime.date.today().isoformat()}}}\n")
        f.write("\\author{}\n\\begin{document}\n\\maketitle\n")
        f.write("\n\n".join(body))
        f.write("\n\\end{document}\n")
    if not compile:
        return tex
    return _compile(tex)


def _compile(tex: str) -> str:
    """Runs pdflatex twice (for the widths of the long tables) next to tex; returns the path of the PDF."""
    folder, name = os.path.dirname(os.path.abspath(tex)), os.path.basename(tex)
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name], cwd=folder,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            log = result.stdout.decode("utf-8", errors="replace")
            error = [line for line in log.splitlines() if line.startswith("!")]
            raise RuntimeError(f"pdflatex failed on {tex}: {error[:3] or log[-1500:]}")
    for ext in (".aux", ".log", ".out"):
        try:
            os.remove(os.path.splitext(tex)[0] + ext)
        except OSError:
            pass
    return os.path.splitext(tex)[0] + ".pdf"
