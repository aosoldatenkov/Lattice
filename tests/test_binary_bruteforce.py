"""Brute-force checks of lattice.binary: results are compared with direct enumeration of the
vectors in a box, and with independent constructions.

Completeness of the listings: list_positive / list_negative are complete for definite lattices
and for indefinite lattices representing zero; for a periodic river they contain one vector per
orbit of <shift, -1>, and for degenerate lattices one vector per orbit of the transvections
f -> f + t e fixing the kernel vector e.
"""
import math
import random
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import pytest
from lattice._matrix import imat
from lattice.binary import BinLattice, list_bin_lattices
from lattice.lattice import Lattice

BOX = range(-4, 5)
FORMS = [(a, b, h) for a in BOX for b in BOX for h in BOX]
BOUND = 30


def is_square(n):
    return n >= 0 and math.isqrt(n) ** 2 == n


def kind(a, b, h):
    det = a * b - h * h
    if (a, b, h) == (0, 0, 0):
        return "zero"
    if det == 0:
        return "degenerate"
    if det > 0:
        return "definite"
    return "zero-representing" if is_square(-det) else "periodic"


def forms_of(*kinds):
    return [f for f in FORMS if kind(*f) in kinds]


def normalize(v):
    return v if v[0] > 0 or (v[0] == 0 and v[1] > 0) else (-v[0], -v[1])


def vectors(N):
    """Primitive vectors in [-N, N]^2, up to sign."""
    return [(x, y) for x in range(0, N + 1) for y in range(-N, N + 1)
            if math.gcd(x, y) == 1 and (x > 0 or y > 0)]


def brute_listing(L, N, bound=BOUND):
    """Primitive vectors (up to sign) in [-N, N]^2 with 0 < |norm| <= bound, by norm."""
    out = defaultdict(set)
    for v in vectors(N):
        n = L.sqr(v)
        if n != 0 and abs(n) <= bound:
            out[n].add(v)
    return out


def definite_radius(L, bound):
    """N such that all vectors with |norm| <= bound lie in [-N, N]^2: |v|^2 <= bound / lambda_min."""
    tr, det = abs(L.a + L.b), L.a * L.b - L.h * L.h
    lam = (tr - math.sqrt(tr * tr - 4 * det)) / 2
    return math.isqrt(int(bound / lam)) + 2


def act(v, S):
    """v S for a row vector v."""
    return (v[0] * S[0][0] + v[1] * S[1][0], v[0] * S[0][1] + v[1] * S[1][1])


def shift_orbit(v, S, K=12):
    orbit, w, u = {normalize(v)}, v, v
    S_inv = [[S[1][1], -S[0][1]], [-S[1][0], S[0][0]]]  # det S = 1
    for _ in range(K):
        w, u = act(w, S), act(u, S_inv)
        orbit |= {normalize(w), normalize(u)}
    return orbit


def merged(L, bound=BOUND):
    return {**L.list_positive(bound), **L.list_negative(bound)}


class TestListing:
    @pytest.mark.parametrize("abh", forms_of("definite"))
    def test_definite_complete(self, abh):
        L = BinLattice(*abh)
        assert merged(L) == dict(brute_listing(L, definite_radius(L, BOUND)))

    @pytest.mark.parametrize("abh", forms_of("zero-representing"))
    def test_zero_representing_complete(self, abh):
        """The list is finite and complete: it contains every vector found in a large box,
        and all its vectors are primitive with the right norms."""
        L = BinLattice(*abh)
        ours = merged(L)
        for n, vs in brute_listing(L, 40).items():
            assert vs <= ours.get(n, set()), (abh, n)
        for n, vs in ours.items():
            assert all(L.sqr(v) == n and math.gcd(*v) == 1 and v == normalize(v) for v in vs)

    @pytest.mark.parametrize("abh", forms_of("periodic"))
    def test_periodic_one_per_orbit(self, abh):
        """Every vector in the box is in the <shift, -1>-orbit of exactly one listed vector."""
        L = BinLattice(*abh)
        S = L.shift.tolist()
        ours = merged(L)
        orbits = {n: [shift_orbit(w, S) for w in vs] for n, vs in ours.items()}
        for n, vs in brute_listing(L, 15).items():
            for v in vs:
                assert sum(v in orb for orb in orbits.get(n, [])) == 1, (abh, n, v)
        for n, vs in ours.items():
            assert all(L.sqr(v) == n and math.gcd(*v) == 1 for v in vs)

    @pytest.mark.parametrize("abh", forms_of("degenerate"))
    def test_degenerate_one_per_orbit(self, abh):
        """v = k f + l e is equivalent to w = k f + l' e (up to sign) iff l = l' mod k, i.e.
        v - w (or v + w) is a multiple of k e."""
        L = BinLattice(*abh)
        e = L.can.e
        ours = merged(L)
        for n, vs in brute_listing(L, 20).items():
            k = math.isqrt(n // L.can.b)  # n = m k^2
            for v in vs:
                matches = [w for w in ours.get(n, ()) for s in (1, -1)
                           if (v[0] - s * w[0]) * e[1] == (v[1] - s * w[1]) * e[0]
                           and ((v[0] - s * w[0]) if e[0] else (v[1] - s * w[1])) % (k * (e[0] or e[1])) == 0]
                assert len(matches) == 1, (abh, n, v, matches)

    def test_zero_form(self):
        L = BinLattice(0, 0, 0)
        assert merged(L) == {} and L.list_roots() == {}


class TestZeroVectorsAndExtrema:
    @pytest.mark.parametrize("abh", [f for f in FORMS if kind(*f) not in ("zero",)])
    def test_isotropic_vectors(self, abh):
        L = BinLattice(*abh)
        brute = {v for v in vectors(25) if L.sqr(v) == 0}
        assert brute == {normalize(v) for v in L.zero}
        assert all(L.sqr(v) == 0 and math.gcd(*v) == 1 for v in L.zero)
        assert len(L.zero) == {"degenerate": 2, "zero-representing": 4}.get(kind(*abh), 0)

    @pytest.mark.parametrize("abh", forms_of("definite", "zero-representing", "periodic", "degenerate"))
    def test_min_positive_max_negative(self, abh):
        L = BinLattice(*abh)
        N = definite_radius(L, 4 * max(abs(L.a), abs(L.b), 1)) if kind(*abh) == "definite" else 20
        norms = [L.sqr(v) for v in vectors(N)]
        pos, neg = [n for n in norms if n > 0], [n for n in norms if n < 0]
        assert L.min_positive() == (min(pos) if pos else None)
        assert L.max_negative() == (max(neg) if neg else None)


class TestRoots:
    @pytest.mark.parametrize("abh", forms_of("definite", "zero-representing", "periodic"))
    def test_roots(self, abh):
        """list_roots against Lattice.is_root (an independent, gcd-based test) on a box."""
        L = BinLattice(*abh)
        M = Lattice(2, [[L.a, L.h], [L.h, L.b]])
        bound = 2 * L.disc
        N = definite_radius(L, bound) if kind(*abh) == "definite" else 20
        brute = defaultdict(set)
        for v in vectors(N):
            if M.is_root(imat(v)) and abs(L.sqr(v)) <= bound:
                brute[L.sqr(v)].add(v)
        ours = L.list_roots()
        assert all(L.is_root(r) and M.is_root(imat(r)) for rs in ours.values() for r in rs)
        if kind(*abh) == "periodic":
            S = L.shift.tolist()
            for n, rs in brute.items():
                assert all(any(r in shift_orbit(w, S) for w in ours.get(n, ())) for r in rs), (abh, n)
        else:
            assert ours == dict(brute)

    @pytest.mark.parametrize("abh", forms_of("degenerate"))
    def test_degenerate_roots(self, abh):
        """k f + l e is a root iff k divides 2."""
        L = BinLattice(*abh)
        m = L.can.b
        assert set(L.list_roots()) == {m, 4 * m}


def isotropic_lines(L):
    """The primitive isotropic vectors of an indefinite lattice representing zero, computed
    directly from the roots of a x^2 + 2h xy + b y^2 (independently of the topograph)."""
    a, b, h = L.a, L.b, L.h
    out = set()
    if a == 0:
        out.add((1, 0))
        g = math.gcd(b, 2 * h)
        out.add(normalize((b // g, -2 * h // g)))
    else:
        s = math.isqrt(h * h - a * b)
        for r in (-h + s, -h - s):  # x / y = r / a
            g = math.gcd(r, a)
            out.add(normalize((r // g, a // g)))
    return {w for v in out for w in (v, (-v[0], -v[1]))}


def complete_basis(e):
    """A vector g with det(e, g) = 1."""
    _, x, y = extended_gcd(e[0], e[1])
    return (-y, x)


def extended_gcd(a, b):
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)


def isometries_zero_representing(L, M):
    """All isometries L -> M (as determinants +-1 of the matrix), for lattices representing zero:
    an isometry maps an isotropic e of L to one of the four isotropic vectors e' of M, and
    f (with (e, f) = eta, (f, f) = beta) to f' = s g' + t e', where s and t are determined."""
    e = next(iter(isotropic_lines(L)))
    f = complete_basis(e)
    eta, beta = L.prod(e, f), L.sqr(f)
    dets = set()
    for e2 in isotropic_lines(M):
        g2 = complete_basis(e2)
        for s in (1, -1):
            if s * M.prod(e2, g2) != eta:
                continue
            num = beta - M.sqr(g2)
            if num % (2 * eta) == 0:
                t = num // (2 * eta)
                f2 = (s * g2[0] + t * e2[0], s * g2[1] + t * e2[1])
                d1 = e[0] * f[1] - e[1] * f[0]
                d2 = e2[0] * f2[1] - e2[1] * f2[0]
                dets.add(d1 * d2)
    return dets


def isometries_definite(L, M):
    """All isometries L -> M for definite lattices: images (u, v) of the standard basis with
    the norms a, b and product h."""
    N = definite_radius(M, max(abs(L.a), abs(L.b)))
    vs = [(x, y) for x in range(-N, N + 1) for y in range(-N, N + 1)]
    us = [u for u in vs if M.sqr(u) == L.a]
    ws = [w for w in vs if M.sqr(w) == L.b]
    return {u[0] * w[1] - u[1] * w[0] for u in us for w in ws
            if M.prod(u, w) == L.h and abs(u[0] * w[1] - u[1] * w[0]) == 1}


class TestIsomorphism:
    @pytest.mark.parametrize("k", ["definite", "zero-representing"])
    def test_against_brute_force(self, k):
        lats = [BinLattice(*f) for f in forms_of(k)]
        iso = isometries_definite if k == "definite" else isometries_zero_representing
        by_det = defaultdict(list)
        for L in lats:
            by_det[(L.disc, L.signature)].append(L)
        for group in by_det.values():
            for L, M in combinations(group, 2):
                dets = iso(L, M)
                assert L.is_isomorphic(M) == bool(dets), ((L.a, L.b, L.h), (M.a, M.b, M.h))
                assert L.is_isomorphic(M, oriented=True) == (1 in dets), ((L.a, L.b, L.h), (M.a, M.b, M.h))

    @pytest.mark.parametrize("abh", [f for f in FORMS if kind(*f) != "zero"])
    def test_invariance(self, abh):
        """Canonical data are invariant under random changes of basis; oriented isomorphism
        holds for det +1, and for det -1 iff there is an improper automorphism."""
        rng = random.Random(hash(abh))
        L = BinLattice(*abh)
        for det in (1, -1):
            g = imat([[1, 0], [0, det]])
            for _ in range(6):
                i, k = rng.randrange(2), rng.randint(-4, 4)
                E = imat([[1, 0], [0, 1]])
                E[i, 1 - i] = k
                g = E @ g
            G = g @ L.A @ g.T
            M = BinLattice(G[0, 0], G[1, 1], G[0, 1])
            assert (M.can.a, M.can.b, M.can.h, M.signature) == (L.can.a, L.can.b, L.can.h, L.signature)
            assert L.is_isomorphic(M) and M.is_isomorphic(L)
            assert L.is_isomorphic(M, oriented=True) == (det == 1 or L.improper_automorphism() is not None)
            assert (L.improper_automorphism() is None) == (M.improper_automorphism() is None)
            assert len(M.river) == len(L.river) and len(M.zero) == len(L.zero)
            assert {n: len(vs) for n, vs in merged(M).items()} == {n: len(vs) for n, vs in merged(L).items()}


class TestStructure:
    @pytest.mark.parametrize("abh", [f for f in FORMS if kind(*f) != "zero"])
    def test_canonical_basis(self, abh):
        L = BinLattice(*abh)
        c = L.can
        assert abs(c.e[0] * c.f[1] - c.e[1] * c.f[0]) == 1
        assert (L.sqr(c.e), L.sqr(c.f), L.prod(c.e, c.f)) == (c.a, c.b, c.h)

    @pytest.mark.parametrize("abh", forms_of("zero-representing", "periodic"))
    def test_river(self, abh):
        L = BinLattice(*abh)
        if (L.can.a, L.can.b) == (0, 0):
            assert L.river == []
            return
        assert L.river[0] == L.can or (L.river[0].a, L.river[0].b, L.river[0].h) == (L.can.a, L.can.b, L.can.h)
        for r in L.river:
            assert r.a > 0 > r.b and (L.sqr(r.e), L.sqr(r.f), L.prod(r.e, r.f)) == (r.a, r.b, r.h)
        for r1, r2 in zip(L.river, L.river[1:]):
            assert r1.e == r2.e or r1.f == r2.f  # one step of the flow
        if kind(*abh) == "periodic":
            S = L.shift
            assert (S @ L.A @ S.T == L.A).all() and S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0] == 1
            # The shift moves the river by one period: the edge after the period is can S
            nxt = L.flow(L.river[-1])[1]
            St = S.tolist()
            assert (nxt.e, nxt.f) == (act(L.river[0].e, St), act(L.river[0].f, St))
        else:
            assert L.shift is None
            ends = (L.river[0], L.river[-1])
            assert all(L.sqr(r.sum()) == 0 for r in (ends[1],))

    @pytest.mark.parametrize("abh", [f for f in FORMS if kind(*f) not in ("zero", "degenerate")])
    def test_seeds(self, abh):
        L = BinLattice(*abh)
        for s in L.pos:
            assert s.h >= 0 and s.a >= 0 and s.b >= 0
        for s in L.neg:
            assert s.h <= 0 and s.a <= 0 and s.b <= 0


class TestRationalScaling:
    @pytest.mark.parametrize("abh", [f for f in FORMS[::7] if kind(*f) != "zero"])
    @pytest.mark.parametrize("d", [2, 3, 6])
    def test_scaling(self, abh, d):
        """L(1/d) has the same vectors, roots, river and automorphisms as L, with norms / d."""
        L = BinLattice(*abh)
        M = BinLattice(*(Fraction(x, d) for x in abh))
        assert (M.can.a, M.can.b, M.can.h) == tuple(Fraction(x, d) for x in (L.can.a, L.can.b, L.can.h))
        assert (M.can.e, M.can.f, M.zero, M.signature) == (L.can.e, L.can.f, L.zero, L.signature)
        assert len(M.river) == len(L.river)
        assert (M.shift is None and L.shift is None) or (M.shift == L.shift).all()
        assert {n * d: vs for n, vs in merged(M, Fraction(BOUND, d)).items()} == merged(L)
        assert {n * d: vs for n, vs in M.list_roots().items()} == L.list_roots()
        assert (M.improper_automorphism() is None) == (L.improper_automorphism() is None)
        assert M.exp is None and M.parity is None or all(x % d == 0 for x in abh)


def test_list_bin_lattices_complete():
    """Every class with 0 < |det| <= 10, found among all Gram matrices in a box, is listed exactly
    once, with the right parity and signature."""
    max_d = 10
    brute = defaultdict(set)
    for a, b, h in product(range(-12, 13), repeat=3):
        det = abs(a * b - h * h)
        if 0 < det <= max_d:
            L = BinLattice(a, b, h)
            brute[(det, L.signature, L.parity)].add((L.can.a, L.can.b, L.can.h))
    for parity in (None, 0, 1):
        for signature in (None, (2, 0), (0, 2), (1, 1)):
            listed = defaultdict(list)
            for d, lats in list_bin_lattices(max_d, signature, parity).items():
                for L in lats:
                    assert L.disc == d
                    listed[(d, L.signature, L.parity)].append((L.can.a, L.can.b, L.can.h))
            expected = {k: v for k, v in brute.items()
                        if (signature is None or k[1] == signature) and (parity is None or k[2] == parity)}
            assert {k: sorted(v) for k, v in listed.items()} == {k: sorted(v) for k, v in expected.items()}
