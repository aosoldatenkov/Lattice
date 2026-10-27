"""Tests of lattice.vinberg.report (readable reports of the certificates, LaTeX and PDF)."""
import json
import shutil
import pytest
from lattice.lattice import Lattice
from lattice.catalogue import A_lat, U_lat
from lattice.vinberg.report import certificate_report, certificate_kind

pytest.importorskip("pynauty")

A3 = A_lat(3)(-1)


@pytest.fixture(scope="module")
def certificates(tmp_path_factory):
    """A certificate of each kind (small lattices), as files."""
    from lattice.vinberg.trace import trace_chamber
    from lattice.vinberg.orbits import trace_chamber_by_orbits
    from lattice.vinberg.nonreflective import search_nonreflective
    from lattice.vinberg.quotient import quotient_chamber
    d = tmp_path_factory.mktemp("certificates")
    path = lambda name: str(d / f"{name}.json")
    L = U_lat() + A_lat(2)(-1) * 3
    trace_chamber(L, walls_file=path("walls"))
    trace_chamber_by_orbits(L, path("walls"), certificate=path("orbits"))
    search_nonreflective(Lattice(1, [[11]]) + A3, check_interval=0, max_vertices=20000, certificate=path("symmetry"))
    search_nonreflective(Lattice(3, [[1, 0, 0], [0, -10, 0], [0, 0, -13]]), check_interval=0, max_vertices=20000,
                         certificate=path("geodesic"))
    quotient_chamber(L, certificate=path("finite"))
    quotient_chamber(Lattice(9, [[1 - 2 * int(i == j) for j in range(9)] for i in range(9)]),
                     certificate=path("parabolic"))
    quotient_chamber(Lattice(1, [[7]]) + A3, certificate=path("cyclic"))
    quotient_chamber(Lattice(1, [[19]]) + A3, certificate=path("nonelementary"))
    return path


@pytest.mark.parametrize("name, kind, phrases", [
    ("walls", "walls", ["not a certificate", "Coxeter diagram"]),
    ("orbits", "orbits", ["is \\textbf{reflective}", "Coxeter diagram", "G}_{2}", "\\tilde{E}_{6}"]),
    ("symmetry", "nonreflective", ["\\textbf{not reflective}", "\\textbf{hyperbolic}", "x^{2}-14x+1"]),
    ("geodesic", "nonreflective", ["irrational ends", "not a square"]),
    ("finite", "quotient", ["is \\textbf{reflective}", "finite of order 6"]),
    ("parabolic", "quotient", ["fixes the cusp", "generalized lattice Weyl vector", "$(c, r)$"]),
    ("cyclic", "quotient", ["virtually cyclic", "x^{4}-11x^{3}-8x^{2}-11x+1", "infinite dihedral"]),
    ("nonelementary", "quotient", ["non-elementary", "free group"]),
])
def test_report_content(certificates, name, kind, phrases, tmp_path):
    """The kind is recognized, and the LaTeX has the lattice (Gram matrix, discriminant form, genus) and the claim."""
    with open(certificates(name)) as f:
        assert certificate_kind(json.load(f)) == kind
    tex = certificate_report(certificates(name), path=str(tmp_path / "report.pdf"), compile=False)
    text = open(tex).read()
    for phrase in ["The lattice", "pmatrix", "Genus"] + phrases:
        assert phrase in text, phrase


def test_report_verify(certificates, tmp_path):
    """With verify, the outcome of the verification is in the report, also for a tampered certificate."""
    tex = certificate_report(certificates("orbits"), path=str(tmp_path / "ok.pdf"), verify=True, compile=False)
    assert "\\textbf{Verified}" in open(tex).read()
    with open(certificates("finite")) as f:
        bad = json.load(f)
    bad["structure"]["structure"] = "parabolic"
    bad["structure"]["cusp"] = [1, 0, 0, 0, 0, 0, 0, 0]
    tex = certificate_report(bad, path=str(tmp_path / "bad.pdf"), verify=True, compile=False)
    assert "verification failed" in open(tex).read()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
@pytest.mark.parametrize("name", ["orbits", "symmetry", "geodesic", "parabolic", "cyclic"])
def test_report_pdf(certificates, name, tmp_path):
    pdf = certificate_report(certificates(name), path=str(tmp_path / f"{name}.pdf"))
    assert open(pdf, "rb").read(4) == b"%PDF"
