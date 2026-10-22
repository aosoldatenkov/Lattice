"""Tests of lattice.draw.diagram (pictures of Coxeter diagrams)."""
import json
import re
import shutil
from fractions import Fraction
import numpy as np
import pytest
from lattice.lattice import Lattice
from lattice.catalogue import E_lat, U_lat, I_lat
from lattice.draw import CoxeterDiagram, draw_Coxeter_diagram, diagram_layout
from lattice.vinberg.trace import trace_chamber


@pytest.fixture(scope="module")
def e10():
    L = U_lat() + E_lat(8)(-1)
    return L, trace_chamber(L, base=[3, 2, 1, 0, 0, 0, 0, 0, 0, 0])["walls"]


def test_e10(e10):
    """The E_10 diagram: a tree of 9 simple edges, drawn with one line each."""
    L, walls = e10
    D = CoxeterDiagram(L, walls)
    assert D.k == 10 and len(D.labels) == 9 and set(D.labels.values()) == {3}
    assert D.xy.shape == (10, 2) and np.isfinite(D.xy).all()
    svg = D.svg()
    assert svg.count("<circle") == 10 and svg.count("<line") == 9
    assert sum(1 for line in D.tikz() if line.startswith("\\draw ")) == 9


def test_edge_kinds():
    """Double edges (m = 4), and an ultraparallel pair drawn dashed, with cos^2 in the tooltip."""
    L = I_lat(1, 3)
    D = CoxeterDiagram(L, [[0, 0, 0, 1], [1, -1, -1, -1], [0, 1, -1, 0], [0, 0, 1, -1]])
    assert D.labels == {(0, 1): 4, (0, 3): 4, (2, 3): 3}
    assert D.svg().count("<line") == 2 + 2 + 1
    M = Lattice(2, [[-2, 3], [3, -2]])
    D = CoxeterDiagram(M, [[1, 0], [0, 1]])
    assert D.labels == {(0, 1): Fraction(9, 4)}
    assert 'stroke-dasharray' in D.svg() and "cos^2 = 9/4" in D.svg()
    assert 'stroke-dasharray' not in D.svg(ultraparallel=False)


def test_layout_deterministic(e10):
    L, walls = e10
    labels = CoxeterDiagram(L, walls).labels
    assert np.allclose(diagram_layout(10, labels), diagram_layout(10, labels))
    # Adjacent nodes are about one unit apart (the median edge length is 1)
    X = diagram_layout(10, labels)
    lengths = [np.hypot(*(X[i] - X[j])) for i, j in labels]
    assert np.median(lengths) == pytest.approx(1.0)


def test_html(e10, tmp_path):
    """The page embeds the SVG with the data of the script, and the matrix of the pairs."""
    L, walls = e10
    path = tmp_path / "e10.html"
    D = draw_Coxeter_diagram(L, walls, str(path), highlight=[2])
    text = path.read_text()
    assert text.count('class="v"') == 10 and text.count('class="e ang"') == 9
    data = json.loads(re.search(r"const DATA = (.*);\n", text).group(1))
    assert data["highlight"] == [2] and sorted(data["matrix"]["order"]) == list(range(10))
    codes = data["matrix"]["codes"]
    assert len(codes) == 10 and sum(row.count("3") for row in codes) == 2 * len(D.labels)


def test_files(e10, tmp_path):
    L, walls = e10
    D = CoxeterDiagram(L, walls)
    for ext in ("svg", "tex"):
        assert (tmp_path / f"d.{ext}").exists() is False
        D.draw(str(tmp_path / f"d.{ext}"))
        assert (tmp_path / f"d.{ext}").stat().st_size > 0
    with pytest.raises(ValueError):
        D.draw(str(tmp_path / "d.png"))
    if shutil.which("pdflatex"):
        pdf = D.draw(str(tmp_path / "d.pdf"))
        assert pdf.endswith(".pdf") and (tmp_path / "d.pdf").exists()
