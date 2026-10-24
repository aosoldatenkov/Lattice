"""Pictures of Coxeter diagrams: a layout computed once, and two renderings, TikZ (a PDF, for small diagrams and
papers) and a self-contained interactive HTML page (for large diagrams).

Conventions (Vinberg): a node per wall r, colored by its norm -(r, r); for two walls r, s with
cos^2 = (r, s)^2 / ((r, r)(s, s)) (see roots.Coxeter_graph):
    cos^2 = 0 (orthogonal, m = 2): no edge;
    cos^2 = 1/4, 1/2, 3/4 (angle pi / m, m = 3, 4, 6): a simple, double or triple edge;
    cos^2 = 1 (parallel): a thick edge;
    cos^2 > 1 (ultraparallel): a dashed edge, labeled by cos^2 in the tooltips.

The layout (diagram_layout) is stress majorization (SMACOF) on the shortest-path distances of the diagram, started
from classical multidimensional scaling: deterministic, and symmetric diagrams come out with visible symmetry.
Large diagrams of reflective lattices have mostly ultraparallel pairs: their edges count as longer in the layout
(by the distance between the mirrors), and are hidden by default when there are many. The HTML page also has a matrix view (the labels of all the pairs,
in the order of a one-dimensional scaling), which stays readable for hundreds or thousands of walls.
"""
from __future__ import annotations
import html
import json
import math
import os
from fractions import Fraction
from typing import Dict, List, Sequence
import numpy as np
from lattice.lattice import Lattice
from lattice.vinberg.coxeter import WallDiagram
from lattice.draw.circle import PDFPicture

# Colors of the norms (Okabe-Ito), in the order of increasing -(r, r)
_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
_ANGLES = (3, 4, 6)


class CoxeterDiagram:
    """The Coxeter diagram of the walls roots of L (a list, or a vinberg.coxeter.WallDiagram, e.g. the diagram of a
    ChamberTracer), with a layout; render with tikz(), pdf(), svg() and html(),
    or draw(path) by the extension of path (.tex, .pdf, .svg, .html).
        labels: {(i, j): label} for the pairs i < j that are not orthogonal (as in Coxeter_graph);
        norms: the norms -(r, r) of the walls;
        xy: the layout, an array of k points (median length of the non-dashed edges 1)."""

    def __init__(self, L: Lattice, roots: Sequence | WallDiagram, layout: np.ndarray = None, iterations: int = 300):
        diagram = roots if isinstance(roots, WallDiagram) else WallDiagram(L, roots)
        self.roots = [list(r) for r in diagram.walls]
        self.k = len(self.roots)
        self.norms = diagram.norms()
        self.labels = diagram.labels()
        self.xy = np.asarray(layout, dtype=float) if layout is not None else diagram_layout(self.k, self.labels,
                                                                                           iterations)
        values = sorted(set(self.norms))
        self.color = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(values)}

    def _spacing(self) -> float:
        """The median distance from a node to the nearest other node, in layout units (dense diagrams are scaled
        up by it, so that the nodes do not overlap)."""
        if self.k < 2:
            return 1.0
        d = np.sqrt(((self.xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        return max(float(np.median(d.min(axis=1))), 1e-6)

    def _ultraparallel(self, ultraparallel) -> bool:
        if ultraparallel is None:
            return sum(1 for lab in self.labels.values() if isinstance(lab, Fraction)) <= 500
        return ultraparallel

    def _node_title(self, i: int) -> str:
        return f"wall {i}: r = {tuple(self.roots[i])}, (r, r) = {-self.norms[i]}"

    @staticmethod
    def _edge_title(i: int, j: int, lab) -> str:
        if lab == 'inf':
            return f"{i} - {j}: parallel"
        if isinstance(lab, Fraction):
            return f"{i} - {j}: ultraparallel, cos^2 = {lab}"
        return f"{i} - {j}: angle pi/{lab}"

    # --- TikZ ---

    def tikz(self, highlight: Sequence[int] = (), numbers: bool = None, ultraparallel: bool = None) -> List[str]:
        """The lines of a tikzpicture (without the environment); highlight: walls drawn with a ring."""
        numbers = self.k <= 60 if numbers is None else numbers
        lines = [f"\\definecolor{{c{n}}}{{HTML}}{{{c[1:]}}}\n" for n, c in self.color.items()]
        scale = max(1.0, 0.5 / self._spacing())  # in cm: a typical edge, and at least 0.5 cm between nodes
        p = {i: f"({scale * self.xy[i, 0]:.3f},{scale * self.xy[i, 1]:.3f})" for i in range(self.k)}
        show_up = self._ultraparallel(ultraparallel)
        for (i, j), lab in sorted(self.labels.items(), key=lambda e: isinstance(e[1], Fraction), reverse=True):
            if isinstance(lab, Fraction):
                if show_up:
                    lines.append(f"\\draw[dashed, gray] {p[i]} -- {p[j]};\n")
            elif lab == 'inf':
                lines.append(f"\\draw[line width=1.6pt] {p[i]} -- {p[j]};\n")
            elif lab == 3:
                lines.append(f"\\draw {p[i]} -- {p[j]};\n")
            elif lab == 4:
                lines.append(f"\\draw[double, double distance=1.2pt] {p[i]} -- {p[j]};\n")
            else:
                lines.append(f"\\draw[double, double distance=2pt] {p[i]} -- {p[j]};\n\\draw {p[i]} -- {p[j]};\n")
        for i in range(self.k):
            lines.append(f"\\fill[c{self.norms[i]}] {p[i]} circle (2.5pt);\n")
            if i in highlight:
                lines.append(f"\\draw[red, thick] {p[i]} circle (4pt);\n")
            if numbers:
                lines.append(f"\\node[font=\\tiny, above right=-1pt] at {p[i]} {{{i}}};\n")
        return lines

    def pdf(self, path: str, **kwargs) -> str:
        """Writes path (.tex) and compiles it with pdflatex next to it; returns the path of the PDF."""
        tex = os.path.splitext(path)[0] + ".tex"
        picture = PDFPicture()
        picture.append(self.tikz(**kwargs))
        picture.print(tex)
        return os.path.splitext(tex)[0] + ".pdf"

    # --- SVG and HTML ---

    def svg(self, highlight: Sequence[int] = (), numbers: bool = None, ultraparallel: bool = None,
            unit: float = None, interactive: bool = False) -> str:
        """An SVG picture; unit: the length of a typical edge in pixels (by default 60, or more for a dense diagram,
        so that nearby nodes are at least 20 pixels apart). With interactive, the elements carry the indices used
        by the script of html() (and the dashed edges are always included, hidden or not)."""
        numbers = self.k <= 60 if numbers is None else numbers
        unit = max(60.0, 20.0 / self._spacing()) if unit is None else unit
        show_up = self._ultraparallel(ultraparallel)
        pts = self.xy * unit
        pts[:, 1] = -pts[:, 1]
        margin = 20.0
        lo = pts.min(axis=0) - margin if self.k else np.zeros(2)
        hi = pts.max(axis=0) + margin if self.k else np.ones(2)
        size = hi - lo
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{lo[0]:.1f} {lo[1]:.1f} {size[0]:.1f} '
               f'{size[1]:.1f}" width="{size[0]:.0f}" height="{size[1]:.0f}" font-family="sans-serif">',
               '<g id="scene">']
        for (i, j), lab in sorted(self.labels.items(), key=lambda e: isinstance(e[1], Fraction), reverse=True):
            dashed = isinstance(lab, Fraction)
            if dashed and not (show_up or interactive):
                continue
            (x1, y1), (x2, y2) = pts[i], pts[j]
            kind = "up" if dashed else ("par" if lab == 'inf' else "ang")
            attrs = f' class="e {kind}" data-i="{i}" data-j="{j}"' if interactive else ''
            if dashed and not show_up:
                attrs += ' style="display:none"'
            title = f"<title>{html.escape(self._edge_title(i, j, lab))}</title>"
            if dashed:
                out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#9a9a9a" '
                           f'stroke-width="0.8" stroke-dasharray="4 3"{attrs}>{title}</line>')
                continue
            if lab == 'inf':
                offsets, width = [0.0], 3.2
            else:
                offsets, width = {3: [0.0], 4: [-1.6, 1.6], 6: [-2.6, 0.0, 2.6]}[lab], 1.0
            length = math.hypot(x2 - x1, y2 - y1) or 1.0
            nx, ny = -(y2 - y1) / length, (x2 - x1) / length
            out.append(f'<g{attrs}>{title}')
            for o in offsets:
                out.append(f'<line x1="{x1 + o * nx:.1f}" y1="{y1 + o * ny:.1f}" x2="{x2 + o * nx:.1f}" '
                           f'y2="{y2 + o * ny:.1f}" stroke="#222" stroke-width="{width}"/>')
            out.append('</g>')
        for i in range(self.k):
            x, y = pts[i]
            attrs = f' class="v" data-i="{i}"' if interactive else ''
            ring = ' stroke="#d00" stroke-width="2.5"' if i in highlight else ' stroke="#fff" stroke-width="1"'
            out.append(f'<g{attrs}><title>{html.escape(self._node_title(i))}</title>'
                       f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{self.color[self.norms[i]]}"{ring}/>')
            if numbers or interactive:
                hidden = '' if numbers else ' style="display:none"'
                out.append(f'<text class="num" x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="9" fill="#333"{hidden}>'
                           f'{i}</text>')
            out.append('</g>')
        out.append('</g></svg>')
        return "\n".join(out)

    def _matrix(self) -> Dict:
        """The data of the matrix view: the order of the walls and the codes of the pairs (0: orthogonal, 3, 4, 6,
        1: parallel, 9: ultraparallel), with the values of cos^2 of the ultraparallel pairs."""
        order = _scaling_order(self.k, self.labels)
        pos = {w: p for p, w in enumerate(order)}
        codes = [[0] * self.k for _ in range(self.k)]
        cos2 = {}
        for (i, j), lab in self.labels.items():
            c = 1 if lab == 'inf' else (9 if isinstance(lab, Fraction) else lab)
            codes[pos[i]][pos[j]] = codes[pos[j]][pos[i]] = c
            if c == 9:
                cos2[f"{min(pos[i], pos[j])},{max(pos[i], pos[j])}"] = str(lab)
        return {"order": order, "codes": ["".join(str(c) for c in row) for row in codes], "cos2": cos2}

    def html(self, title: str = "Coxeter diagram", highlight: Sequence[int] = (), numbers: bool = None,
             ultraparallel: bool = None) -> str:
        """A self-contained HTML page: the diagram (zoom with the wheel, pan by dragging, click a wall to show its
        neighbours, a list of walls to highlight, toggles for the kinds of edges) and the matrix view."""
        numbers = self.k <= 60 if numbers is None else numbers
        show_up = self._ultraparallel(ultraparallel)
        counts = {"ang": sum(1 for lab in self.labels.values() if lab in _ANGLES),
                  "par": sum(1 for lab in self.labels.values() if lab == 'inf'),
                  "up": sum(1 for lab in self.labels.values() if isinstance(lab, Fraction))}
        legend = " ".join(f'<span class="key"><svg width="12" height="12"><circle cx="6" cy="6" r="5" '
                          f'fill="{c}"/></svg> (r, r) = {-n}</span>' for n, c in self.color.items())
        data = {"matrix": self._matrix(), "highlight": list(highlight),
                "colors": {str(n): c for n, c in self.color.items()}, "norms": self.norms}
        page = _HTML.replace("__TITLE__", html.escape(title))
        page = page.replace("__SUMMARY__", f"{self.k} walls; {counts['ang']} edges with an angle, {counts['par']} "
                                           f"parallel, {counts['up']} ultraparallel pairs")
        page = page.replace("__LEGEND__", legend)
        page = page.replace("__CHECK_UP__", "checked" if show_up else "")
        page = page.replace("__CHECK_NUM__", "checked" if numbers else "")
        page = page.replace("__SVG__", self.svg(highlight, numbers, show_up, interactive=True))
        return page.replace("__DATA__", json.dumps(data))

    def draw(self, path: str, **kwargs) -> str:
        """Writes the picture to path, by its extension: .html, .svg, .tex (the tikzpicture lines) or .pdf."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.pdf(path, **kwargs)
        if ext == ".html":
            text = self.html(**kwargs)
        elif ext == ".svg":
            text = self.svg(**kwargs)
        elif ext == ".tex":
            text = "".join(self.tikz(**kwargs))
        else:
            raise ValueError("The extension should be .html, .svg, .tex or .pdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


def draw_Coxeter_diagram(L: Lattice, roots: Sequence, path: str, **kwargs) -> CoxeterDiagram:
    """Computes the Coxeter diagram of the walls roots of L with its layout, and writes it to path (by the
    extension: .html, .svg, .tex, .pdf; see CoxeterDiagram.draw). Returns the diagram, to render it again."""
    diagram = CoxeterDiagram(L, roots)
    diagram.draw(path, **kwargs)
    return diagram


# --- the layout ---

def _distances(k: int, labels: Dict) -> np.ndarray:
    """The shortest-path distances of the diagram: 1 along an edge with an angle or a parallel pair, and 1 plus the
    hyperbolic distance arccosh |cos| between the mirrors along an ultraparallel pair (so that diagrams with few
    edges, e.g. compact polygons with orthogonal consecutive walls, are drawn by their geometry); disconnected pairs
    at the largest distance plus 1."""
    D = np.full((k, k), np.inf)
    np.fill_diagonal(D, 0.0)
    for (i, j), lab in labels.items():
        D[i, j] = D[j, i] = 1.0 + math.acosh(math.sqrt(lab)) if isinstance(lab, Fraction) else 1.0
    for m in range(k):  # Floyd-Warshall
        np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :], out=D)
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = (finite.max() if finite.size else 0.0) + 1.0
    return D


def _classical_scaling(D: np.ndarray, dim: int) -> np.ndarray:
    k = len(D)
    J = np.eye(k) - 1.0 / k
    B = -0.5 * J @ (D ** 2) @ J
    values, vectors = np.linalg.eigh(B)
    idx = np.argsort(values)[::-1][:dim]
    return vectors[:, idx] * np.sqrt(np.maximum(values[idx], 1e-9))


def diagram_layout(k: int, labels: Dict, iterations: int = 300, dim: int = 2, seed: int = 0) -> np.ndarray:
    """Positions in R^dim of the k nodes of a diagram with the given labels (see the module docstring), scaled so
    that the median length of the edges which are not ultraparallel is 1. The classical scaling which starts the
    stress majorization is perturbed slightly (deterministically, from seed): nodes exchanged by a symmetry of the
    diagram can have the same coordinates in it, and coincident points are a fixed point of the majorization."""
    if k <= 1:
        return np.zeros((k, dim))
    D = _distances(k, labels)
    X = _classical_scaling(D, dim)
    X = X + 1e-3 * np.random.default_rng(seed).standard_normal(X.shape) * max(float(np.abs(X).max()), 1.0)
    # SMACOF with the weights 1 / D^2 (the usual choice for graph drawing: local distances matter most)
    W = np.zeros_like(D)
    off = ~np.eye(k, dtype=bool)
    W[off] = 1.0 / D[off] ** 2
    V = np.diag(W.sum(axis=1)) - W
    Vp = np.linalg.pinv(V)
    for _ in range(iterations):
        dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        ratio = np.where(dist > 1e-12, D / np.maximum(dist, 1e-12), 0.0)
        Bm = -W * ratio
        np.fill_diagonal(Bm, 0.0)
        np.fill_diagonal(Bm, -Bm.sum(axis=1))
        X = Vp @ (Bm @ X)
    return _normalize(X, labels)


def _normalize(X: np.ndarray, labels: Dict) -> np.ndarray:
    """Centers X and scales it so that the median length of the edges which are not ultraparallel is 1."""
    lengths = [float(np.linalg.norm(X[i] - X[j])) for (i, j), lab in labels.items() if not isinstance(lab, Fraction)]
    scale = float(np.median(lengths)) if lengths else 1.0
    return (X - X.mean(axis=0)) / (scale if scale > 1e-9 else 1.0)


def _scaling_order(k: int, labels: Dict) -> List[int]:
    """An order of the walls for the matrix view: by the first coordinate of the classical scaling."""
    if k <= 1:
        return list(range(k))
    x = _classical_scaling(_distances(k, labels), 1)[:, 0]
    return [int(i) for i in np.argsort(x, kind="stable")]


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; color: #222; background: #fafafa; }
  header { padding: 8px 16px; background: #fff; border-bottom: 1px solid #ddd; display: flex; flex-wrap: wrap;
           gap: 6px 18px; align-items: center; }
  h1 { font-size: 16px; margin: 0; }
  .summary, .key { color: #555; font-size: 13px; }
  .key svg { vertical-align: middle; }
  label { white-space: nowrap; }
  input[type=text] { width: 14em; }
  #views { position: relative; height: calc(100vh - 90px); }
  #graph, #matrix { position: absolute; inset: 0; overflow: hidden; background: #fff; }
  #graph svg { width: 100%; height: 100%; cursor: grab; }
  #matrix { display: none; overflow: auto; }
  #tip { position: fixed; pointer-events: none; background: #222; color: #fff; padding: 3px 6px; border-radius: 3px;
         font-size: 12px; display: none; }
  .dim { opacity: 0.08; }
  .mark circle { stroke: #d00; stroke-width: 2.5; }
</style></head><body>
<header>
  <h1>__TITLE__</h1><span class="summary">__SUMMARY__</span><span>__LEGEND__</span>
</header>
<header>
  <label><input type="radio" name="view" value="graph" checked> diagram</label>
  <label><input type="radio" name="view" value="matrix"> matrix</label>
  <label><input type="checkbox" id="ang" checked> angles</label>
  <label><input type="checkbox" id="par" checked> parallel</label>
  <label><input type="checkbox" id="up" __CHECK_UP__> ultraparallel</label>
  <label><input type="checkbox" id="num" __CHECK_NUM__> numbers</label>
  <label>highlight walls <input type="text" id="hl" placeholder="e.g. 0 3 17"></label>
  <span class="summary">wheel: zoom, drag: pan, click a wall: its neighbours</span>
</header>
<div id="views"><div id="graph">__SVG__</div><div id="matrix"><canvas id="cv"></canvas></div></div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const svg = document.querySelector("#graph svg"), scene = document.getElementById("scene");
const edges = [...svg.querySelectorAll(".e")], nodes = [...svg.querySelectorAll(".v")];
// Zoom and pan on the viewBox
let vb = svg.getAttribute("viewBox").split(" ").map(Number);
svg.removeAttribute("width"); svg.removeAttribute("height");
const setVB = () => svg.setAttribute("viewBox", vb.join(" "));
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const r = svg.getBoundingClientRect(), f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const px = vb[0] + (e.clientX - r.left) / r.width * vb[2], py = vb[1] + (e.clientY - r.top) / r.height * vb[3];
  vb = [px - (px - vb[0]) * f, py - (py - vb[1]) * f, vb[2] * f, vb[3] * f]; setVB();
}, {passive: false});
let drag = null, moved = false;
svg.addEventListener("mousedown", e => { drag = [e.clientX, e.clientY]; moved = false; });
window.addEventListener("mouseup", () => { drag = null; });
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const r = svg.getBoundingClientRect(), s = Math.max(vb[2] / r.width, vb[3] / r.height);
  vb[0] -= (e.clientX - drag[0]) * s; vb[1] -= (e.clientY - drag[1]) * s; drag = [e.clientX, e.clientY];
  moved = true; setVB();
});
// Kinds of edges and numbers
const shown = {ang: true, par: true, up: document.getElementById("up").checked};
for (const kind of ["ang", "par", "up"]) {
  document.getElementById(kind).addEventListener("change", e => {
    shown[kind] = e.target.checked;
    for (const el of edges) if (el.classList.contains(kind)) el.style.display = shown[kind] ? "" : "none";
    drawMatrix();
  });
}
document.getElementById("num").addEventListener("change", e => {
  for (const t of svg.querySelectorAll(".num")) t.style.display = e.target.checked ? "" : "none";
});
// Neighbours of a wall
let focus = null;
function setFocus(i) {
  focus = (focus === i) ? null : i;
  const near = new Set([i]);
  if (focus !== null) for (const el of edges)
    if (el.style.display !== "none" && (+el.dataset.i === i || +el.dataset.j === i)) { near.add(+el.dataset.i); near.add(+el.dataset.j); }
  for (const el of edges) el.classList.toggle("dim", focus !== null && +el.dataset.i !== i && +el.dataset.j !== i);
  for (const el of nodes) el.classList.toggle("dim", focus !== null && !near.has(+el.dataset.i));
}
for (const el of nodes) el.addEventListener("click", () => { if (!moved) setFocus(+el.dataset.i); });
// Highlighted walls
function setHighlight(list) {
  const s = new Set(list);
  for (const el of nodes) el.classList.toggle("mark", s.has(+el.dataset.i));
  drawMatrix();
}
document.getElementById("hl").addEventListener("input", e => setHighlight(e.target.value.split(/[^0-9]+/).filter(x => x).map(Number)));
// Views
for (const r of document.querySelectorAll("input[name=view]")) r.addEventListener("change", e => {
  document.getElementById("graph").style.display = e.target.value === "graph" ? "" : "none";
  document.getElementById("matrix").style.display = e.target.value === "matrix" ? "block" : "none";
  drawMatrix();
});
// The matrix view: the pairs in the order of a one-dimensional scaling
const M = DATA.matrix, k = M.order.length;
const cell = Math.max(1, Math.min(14, Math.floor(900 / Math.max(k, 1))));
const CODE = {"3": "#222", "4": "#5b3", "6": "#a3c", "1": "#d22", "9": "#bbb"};
function drawMatrix() {
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d"), band = 6;
  cv.width = cv.height = k * cell + band;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  const kinds = {"3": "ang", "4": "ang", "6": "ang", "1": "par", "9": "up"};
  for (let a = 0; a < k; a++) {
    const row = M.codes[a];
    for (let b = 0; b < k; b++) {
      const c = row[b];
      if (c === "0" || !shown[kinds[c]]) continue;
      ctx.fillStyle = CODE[c]; ctx.fillRect(band + b * cell, band + a * cell, cell, cell);
    }
  }
  const hl = new Set(document.getElementById("hl").value.split(/[^0-9]+/).filter(x => x).map(Number));
  for (const w of DATA.highlight) hl.add(w);
  M.order.forEach((w, p) => {
    ctx.fillStyle = hl.has(w) ? "#d00" : DATA.colors[DATA.norms[w]];
    ctx.fillRect(0, band + p * cell, band - 1, cell); ctx.fillRect(band + p * cell, 0, cell, band - 1);
  });
}
const tip = document.getElementById("tip");
document.getElementById("cv").addEventListener("mousemove", e => {
  const r = e.target.getBoundingClientRect(), a = Math.floor((e.clientY - r.top - 6) / cell), b = Math.floor((e.clientX - r.left - 6) / cell);
  if (a < 0 || b < 0 || a >= k || b >= k) { tip.style.display = "none"; return; }
  const c = M.codes[a][b], i = M.order[a], j = M.order[b];
  const what = {"0": "orthogonal", "3": "angle pi/3", "4": "angle pi/4", "6": "angle pi/6", "1": "parallel",
                "9": "ultraparallel, cos^2 = " + M.cos2[Math.min(a, b) + "," + Math.max(a, b)]}[c];
  tip.textContent = a === b ? "wall " + i : i + " - " + j + ": " + what;
  tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; tip.style.display = "block";
});
document.getElementById("cv").addEventListener("mouseleave", () => { tip.style.display = "none"; });
setHighlight(DATA.highlight);
</script></body></html>
"""
