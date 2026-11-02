"""Verification part 1: exact deficits, polynomial locality, Cao Labora Thm 3.2, shell theorem 1D/2D/3D."""
from fractions import Fraction as Fr
from math import factorial
import itertools
import mpmath as mp
import sympy as sp

mp.mp.dps = 30
pi = mp.pi

def sinc(x):
    return mp.mpf(1) if x == 0 else mp.sin(x) / x

def osc_int(f, period, a=0):
    """integral of f over [a, inf) for oscillatory f"""
    return mp.quadosc(f, [a, mp.inf], period=period)

print("=== V1: Borwein I_8 exact via overshoot (simplex) formula ===")
a = [Fr(1, 2 * n - 1) for n in range(1, 9)]
m = 7
eps = sum(a[1:]) - a[0]
assert 0 < eps <= 2 * min(a[1:])
prod = 1
for x in a[1:]:
    prod *= x
tail = eps ** m / (factorial(m) * 2 ** m * prod)  # P(S > 1)
I8_over_pi = 1 - 2 * tail
print("I_8/pi =", I8_over_pi)
print("1 - I_8/pi =", float(1 - I8_over_pi))
# Borwein's published value: 467807924713440738696537864469/467807924720320453655260875000
print("matches Borwein rational:", I8_over_pi == Fr(467807924713440738696537864469, 467807924720320453655260875000))

# Direct high precision quadrature of I_8
f8 = lambda k: mp.fprod([sinc(k / (2 * n - 1)) for n in range(1, 9)])
import numpy as np
from scipy.integrate import quad as squad
def f8n(k):
    return np.prod([np.sinc(k/(2*n-1)/np.pi) for n in range(1,9)])
tot=0.0
for j in range(3000):
    tot+=squad(f8n,j,j+1,epsabs=1e-17,epsrel=1e-14)[0]
print("direct quadrature (pi - I_8):", np.pi-2*tot, " formula:", float(np.pi*(1-I8_over_pi)))

print("\n=== V2: J_57 deficit ===")
mp.mp.dps = 60
b = [mp.mpf(1) / (2 * n - 1) for n in range(2, 58)]
eps = mp.fsum(b) - 2
print("eps =", eps, " 2*min =", 2 * b[-1], " valid:", eps <= 2 * b[-1])
mm = len(b)
tailJ = eps ** mm / (mp.factorial(mm) * 2 ** mm * mp.fprod(b))
print("pi/2 - J_57 = pi*P(S>2) =", pi * tailJ)
eps56 = mp.fsum(b[:-1]) - 2
print("check N=56 overshoot negative:", eps56)
mp.mp.dps = 30

print("\n=== V3: polynomial locality family K_N ===")
def Lam32(k):
    return mp.mpf(1) if k == 0 else 3 * (mp.sin(k) - k * mp.cos(k)) / k ** 3

def K(N):
    f = lambda k: Lam32(k) * mp.fprod([sinc(k / (2 * n - 1)) for n in range(2, N + 1)])
    return 2 * osc_int(f, 2 * pi)

for N in [2, 3, 5, 7, 8]:
    pred = 3 * pi / 2 * (1 - mp.fsum([mp.mpf(1) / (3 * (2 * n - 1) ** 2) for n in range(2, N + 1)]))
    val = K(N)
    print(N, "K_N =", mp.nstr(val, 20), " pred =", mp.nstr(pred, 20), " diff =", mp.nstr(val - pred, 5))

print("\n=== V4: Cao Labora Thm 3.2 from polynomial locality (symbolic) ===")
x, a1, a2, a3 = sp.symbols('x a1 a2 a3', positive=True)
# density of U[-a1,a1]+U[-a2,a2]+U[-a3,a3]: f(x)=1/(2^3 2! a1a2a3) sum_g (prod g) (x + g.a)_+^2
# on central cell |x|<a2+a3-a1 (a1>=a2>=a3, a1<a2+a3) the active terms are those with x+g.a>0 near x=0
terms = 0
vals = {a1: sp.Rational(10, 10), a2: sp.Rational(9, 10), a3: sp.Rational(7, 10)}
for g in itertools.product([1, -1], repeat=3):
    s = g[0] * a1 + g[1] * a2 + g[2] * a3
    if s.subs(vals) > 0:  # positive offset -> active near 0
        terms += g[0] * g[1] * g[2] * (x + s) ** 2
P = sp.expand(terms / (8 * 2 * a1 * a2 * a3))
print("central polynomial P(x) =", sp.simplify(P))
# I_n = 2*pi*E P(-S), S = sum_{k>=4} U[-a_k,a_k]: E S^2 = sum a_k^2/3
ak = sp.symbols('s2', positive=True)  # s2 = sum_{k>=4} a_k^2
ours = sp.expand(2 * sp.pi * (P.subs(x, 0) + sp.diff(P, x, 2).subs(x, 0) / 2 * ak / 3))
theirs = sp.pi * (-(a1**2 + a2**2 + a3**2 + ak) - 2 * (a1**2 + a2**2 + a3**2) + 6 * (a1*a2 + a2*a3 + a1*a3)) / (12 * a1 * a2 * a3)
print("ours - CaoLabora =", sp.simplify(ours - theirs))
# numerical sanity: a=(1,0.9,0.7,0.3,0.2) : condition 0.9+0.7-1=0.6 > 0.5
vals5 = [mp.mpf(1), mp.mpf('0.9'), mp.mpf('0.7'), mp.mpf('0.3'), mp.mpf('0.2')]
f = lambda k: mp.fprod([sinc(c * k) for c in vals5])
num = 2 * osc_int(f, 2 * pi)
pred = float(theirs.subs({a1: 1, a2: sp.Rational(9, 10), a3: sp.Rational(7, 10), ak: sp.Rational(9 + 4, 100)}))
print("numeric:", mp.nstr(num, 15), " formula:", pred)
# beyond Cao Labora: replace sinc blocks by cos (Pearson) blocks: E S^2 = sum a_k^2
f = lambda k: sinc(k) * sinc(0.9 * k) * sinc(0.7 * k) * mp.cos(0.3 * k) * mp.cos(0.2 * k)
num = 2 * osc_int(f, 2 * pi)
pred = float(ours.subs({a1: 1, a2: sp.Rational(9, 10), a3: sp.Rational(7, 10), ak: 3 * sp.Rational(13, 100)}))
print("cos-blocks numeric:", mp.nstr(num, 15), " ours:", pred)

print("\n=== V8: 1D shell identity (corrected Eq.19 with exact deficit) ===")
def shell1d(a1, psi):
    f = lambda k: (1 - mp.cos(a1 * k) * psi(k)) / k ** 2 if k != 0 else mp.mpf(0)
    return (2 / pi) * (mp.quad(f, [0, 1e-6, 1]) + osc_int(f, 2 * pi / a1, 1))
for bb in [0.5, 1.0, 1.7, 3.0]:
    val = shell1d(1, lambda k: sinc(bb * k))
    pred = 1 if bb <= 1 else (1 + bb ** 2) / (2 * bb)
    print("uniform b=", bb, " E max(a1,|S|):", mp.nstr(val, 15), " pred:", pred)
for c in [0.4, 1.6]:  # deterministic shift: non-symmetric, non-centred step
    val = shell1d(1, lambda k: mp.cos(c * k))  # Re of e^{ick}
    print("shift c=", c, ":", mp.nstr(val, 15), " pred:", max(1, c))

print("\n=== V9: 2D Jensen identity int_0^inf J0(a1 k)(1-J0(bk))/k dk = log+(b/a1) ===")
for bb in [0.5, 0.99, 1.5, 4.0]:
    f = lambda k: mp.besselj(0, k) * (1 - mp.besselj(0, bb * k)) / k if k != 0 else mp.mpf(0)
    val = mp.quad(f, [0, 1]) + osc_int(f, 2 * pi / (1 + bb) if bb > 0 else 2 * pi, 1)
    print("b=", bb, ":", mp.nstr(val, 15), " pred:", mp.log(max(1, bb)))

print("\n=== V10: 3D Newton identity, non-radial S (segment) ===")
# int_{R^3} sinc(a1|k|) sinc(b k_x)/|k|^2 d^3k = 4 pi int_0^inf sinc(a1 k) Si(bk)/(bk) dk = 2 pi^2 E[1/max(a1,|S|)]
for bb in [0.5, 1.0, 2.0, 5.0]:
    f = lambda k: sinc(k) * mp.si(bb * k) / (bb * k) if k != 0 else mp.mpf(1)
    val = 4 * pi * (mp.quad(f, [0, 1]) + osc_int(f, 2 * pi, 1))
    E = 1 if bb <= 1 else (1 + mp.log(bb)) / bb
    print("b=", bb, ":", mp.nstr(val, 15), " pred:", mp.nstr(2 * pi ** 2 * E, 15))
