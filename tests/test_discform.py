import pytest
from lattice.discform import DiscForm
from lattice.catalogue import D_lat


@pytest.mark.parametrize("n", [8, 16])
def test_d_n_overlattices(n):
    # D_n^*/D_n = (Z/2)^2. Gluing in the vector class gives Z^n (odd); for n = 0 mod 8
    # the two spinor classes give even unimodular lattices (E8 and D16+).
    D = DiscForm(D_lat(n))
    assert D.ord == [2, 2]
    assert len(D.list_iso()) == 3  # elements u with b(u, u) in Z
    overlattices = [D.overlattice(sorted(s)) for s in D.list_max_isospaces()]
    assert all(M.disc == 1 and M.rank == n for M in overlattices)
    assert sorted(bool(M.parity) for M in overlattices) == [False, False, True]
