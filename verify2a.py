"""Verification part 2: Bessel domain (Eq.12), PRL (14)-(16), counterexamples, lattice sums, Pizzetti, misc."""
import mpmath as mp
from math import factorial
mp.mp.dps = 25
pi = mp.pi
J = mp.besselj
G = mp.gamma

def sinc(x):
    return mp.mpf(1) if x == 0 else mp.sin(x) / x

def Lam(nu, z):
    """normalised Bessel: Gamma(nu+1)(2/z)^nu J_nu(z), Lam(nu,0)=1"""
    if z == 0:
        return mp.mpf(1)
    return G(nu + 1) * (2 / z) ** nu * J(nu, z)

def oscint(f, w, a=0, b=1):
    return mp.quad(f, mp.linspace(a, b, 5)) + mp.quadosc(f, [b, mp.inf], omega=w)

print("=== recheck V8 accurately ===")
for bb, pred in [(0.5, 1), (1.7, (1 + 1.7**2) / 3.4)]:
    f = lambda k: (1 - mp.cos(k) * sinc(bb * k)) / k ** 2 if k != 0 else (1 + bb**2 / 3) / 2
    v = mp.quad(f, mp.linspace(0, 60, 121)) + mp.quadosc(f, [60, mp.inf], omega=1)
    print(bb, mp.nstr(2 / pi * v, 15), pred)

print("\n=== K_8 deficit, exact corner formula ===")
from fractions import Fraction as Fr
a = [Fr(1, 2 * n - 1) for n in range(2, 9)]
m = 7
eps = sum(a) - 1
P = 1
for x in a: P *= x
den = 2 ** m * P
E1 = eps ** (m + 1) / (factorial(m + 1) * den)
E2 = 2 * eps ** (m + 2) / (factorial(m + 2) * den)
print("K_8 - pred = 2*2pi*(3/4)*(E2+2E1) =", float(2 * 2 * (Fr(3, 4)) * (E2 + 2 * E1)) * float(pi))

print("\n=== Eq.(12): Bessel identity incl. outside probabilistic range ===")
def rhs12(m1, b1, ms, bs):
    return 2 ** (m1 - 1 - sum(ms)) * G(m1) / b1 ** m1 * mp.fprod([b ** mj for mj, b in zip(ms, bs)]) / mp.fprod([G(1 + mj) for mj in ms])
def lhs12(m1, b1, ms, bs):
    f = lambda k: k ** (m1 - 1) * J(m1, b1 * k) * mp.fprod([J(mj, b * k) / k ** mj for mj, b in zip(ms, bs)]) if k != 0 else mp.mpf(0)
    return oscint(f, b1, 0, 2)
cases = [(1.5, 1, [0.0], [0.5]),        # d=3 with J0 block (not a density in 3D)
         (0.3, 1, [-0.4], [0.6]),       # mu2 < 0, mu1 non half-integer
         (1.2, 1, [0.25, 0.1], [0.3, 0.4]),
         (2.5, 1, [1.0, 0.5], [0.3, 0.2]),   # N=1 version would diverge (mu1>=3/2)
         (1.0, 1, [-0.45], [0.9])]
for c in cases:
    print(c, " lhs=", mp.nstr(lhs12(*c), 14), " rhs=", mp.nstr(rhs12(*c), 14))
print("sum b_j > b1 (should fail):", mp.nstr(lhs12(1.0, 1, [0.5], [1.3]), 12), mp.nstr(rhs12(1.0, 1, [0.5], [1.3]), 12))

print("\n=== PRL (14) check: int k^{nu-mu+1} J_nu(a1k)J_mu(a2k)cos(a3k)sinc(a4k) dk = 0 ===")
for nu, mu in [(0, 1), (0.5, 1.5), (1, 1.6)]:
    f = lambda k: k ** (nu - mu + 1) * J(nu, k) * J(mu, 0.3 * k) * mp.cos(0.2 * k) * sinc(0.1 * k)
    print(nu, mu, mp.nstr(oscint(f, 1, 0, 2), 10))

print("\n=== PRL (15) via radial reduction: 2pi int k J0(sqrt2 k)J0(bk)J0(ak) dk ===")
for aa, bb in [(0.8, 1.2), (0.5, 1.5)]:
    f = lambda k: k * J(0, mp.sqrt(2) * k) * J(0, bb * k) * J(0, aa * k)
    v = 2 * pi * oscint(f, 1, 0, 2)
    pred = 4 / mp.sqrt(((aa + bb) ** 2 - 2) * (2 - (aa - bb) ** 2))
    print(aa, bb, mp.nstr(v, 12), mp.nstr(pred, 12))

print("\n=== PRL (16) incl. a > b (PRL's 'wlog a<b' invalid) ===")
def lhs16(aa, bb):
    f = lambda k: J(0, mp.sqrt(2) * k) * J(0, bb * k) * J(1, aa * k)
    return 2 * pi * oscint(f, 1, 0, 2)
def rhs16(aa, bb):
    r = mp.sqrt(2)
    if aa <= abs(bb - r): return mp.mpf(0)
    if aa >= bb + r: return 2 * pi / aa
    return 2 / aa * mp.acos((2 + bb ** 2 - aa ** 2) / (2 * r * bb))
for aa, bb in [(0.6, 1.0), (1.0, 0.6), (3.0, 0.5), (0.2, 1.0)]:
    print(aa, bb, mp.nstr(lhs16(aa, bb), 12), mp.nstr(rhs16(aa, bb), 12))

print("\n=== strict vs non-strict ===")
f = lambda k: mp.cos(k) * sinc(k)
print("int cos(k)sinc(k) dk (a2=a1, N=2):", mp.nstr(2 * oscint(f, 1, 0, 1), 12), "(PRL Eq.7 would give 0)")
f = lambda k: mp.cos(k) * sinc(k / 2) ** 2
print("int cos(k)sinc(k/2)^2 dk (equality, N=3):", mp.nstr(2 * oscint(f, 1, 0, 1), 12))

print("\n=== footnote [15]: numerator at k=1 ===", mp.cos(1) - mp.sin(1))

print("\n=== counterexample: Gauss-regularised int_{R^3} sinc(|k|)cos(|k|/2) d^3k ===")
for e in [1e-2, 1e-3, 1e-4]:
    f = lambda k: 4 * pi * k ** 2 * sinc(k) * mp.cos(k / 2) * mp.exp(-e * k ** 2)
    print(e, mp.nstr(mp.quad(f, mp.linspace(0, 12 / mp.sqrt(e), 400)), 10))

print("\n=== Pizzetti d-dim polynomial locality (first step density ~ (a1^2-|x|^2)_+) ===")
def piz(d, a1, mus, bs):
    Sd = 2 * pi ** (mp.mpf(d) / 2) / G(mp.mpf(d) / 2)
    f = lambda k: k ** (d - 1) * Lam(mp.mpf(d) / 2 + 1, a1 * k) * mp.fprod([Lam(mu, b * k) for mu, b in zip(mus, bs)])
    lhs = oscint(f, a1, 0, 2)
    rhs = (2 * pi) ** d * d * (d + 2) / (2 * Sd ** 2 * a1 ** (d + 2)) * (a1 ** 2 - mp.mpf(d) / 2 * mp.fsum([b ** 2 / (mu + 1) for mu, b in zip(mus, bs)]))
    return lhs, rhs
for c in [(2, 1, [1.0], [0.5]), (3, 1, [1.5, 0.5], [0.3, 0.4]), (3, 1, [0.5], [0.6]), (4, 1, [2.0, 0.7], [0.4, 0.3])]:
    l, r = piz(*c)
    print(c, mp.nstr(l, 14), mp.nstr(r, 14))

