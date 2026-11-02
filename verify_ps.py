"""Checks for the Philcox--Slepian (PS21) extensions."""
import numpy as np
import mpmath as mp
from scipy.special import jv, j0, j1
from scipy.integrate import quad
mp.mp.dps = 20
pi = np.pi

def eps_sum(f, M, m0):
    m = np.arange(1, M + 1, dtype=float)
    return 0.5 * m0 + np.sum(f(m))

print("=== (a) defect beyond 2pi, N=1: f(t)=J1(8t)/t ===")
a = 8.0
f = lambda t: j1(a * t) / t
S = eps_sum(f, 4_000_000, a / 2)
I = 1.0  # int_0^inf J1(at)/t dt = 1
rho = lambda x: 2 / (pi * a) * np.sqrt(max(0.0, 1 - x * x / a ** 2))
C = a / 2
defect = 2 * pi * C * rho(2 * pi)
print("sum - integral =", S - I, "  predicted defect =", defect)

print("\n=== (b) defect beyond 2pi, N=2: f(t)=J1(5t)J1(3t)/t^2 ===")
a, b = 5.0, 3.0
f = lambda t: j1(a * t) * j1(b * t) / t ** 2
S = eps_sum(f, 2_000_000, a * b / 4)
g_int = lambda tt: float(f(tt))
Iv = mp.quad(lambda t: mp.besselj(1, a * t) * mp.besselj(1, b * t) / t ** 2, [0, 1, 2]) + \
     mp.quadosc(lambda t: mp.besselj(1, a * t) * mp.besselj(1, b * t) / t ** 2, [2, mp.inf], omega=a - b)
rho_a = lambda x: 2 / (pi * a) * np.sqrt(max(0.0, 1 - x * x / a ** 2))
rho_b = lambda x: 2 / (pi * b) * np.sqrt(max(0.0, 1 - x * x / b ** 2))
x0 = 2 * pi
lo, hi = max(-a, x0 - b), min(a, x0 + b)
gconv = quad(lambda y: rho_a(y) * rho_b(x0 - y), lo, hi, limit=200)[0]
C = (a / 2) * (b / 2)
print("sum - integral =", S - float(Iv), "  predicted =", 2 * pi * C * gconv, "  (4pi=", 4 * pi, "> a+b=8: one point)")

print("\n=== (c) boundary case a+b=2pi with beta=0: J0(at)J0(bt) ===")
a = 4.0; b = 2 * pi - 4.0
f = lambda t: j0(a * t) * j0(b * t)
# conditionally convergent: average partial sums over a window to kill the O(1/M) oscillation
M = 6_000_000
m = np.arange(1, M + 1, dtype=float)
terms = f(m)
cs = 0.5 + np.cumsum(terms)
S = cs[-200000:].mean()
Iv = mp.quad(lambda t: mp.besselj(0, a * t) * mp.besselj(0, b * t), [0, 1, 2]) + \
     mp.quadosc(lambda t: mp.besselj(0, a * t) * mp.besselj(0, b * t), [2, mp.inf], omega=a - b)
print("sum =", S, " integral =", float(Iv), " sum-integral =", S - float(Iv), " predicted 1/(2 sqrt(ab)) =", 1 / (2 * np.sqrt(a * b)))

print("\n=== (c') same with beta>0 at a+b=2pi: J1(at)J1(bt)/t^2 (Sum nu + N/2 - 1 - 2k = 2 > 0) ===")
f = lambda t: j1(a * t) * j1(b * t) / t ** 2
S = eps_sum(f, 2_000_000, a * b / 4)
Iv = mp.quad(lambda t: mp.besselj(1, a * t) * mp.besselj(1, b * t) / t ** 2, [0, 1, 2]) + \
     mp.quadosc(lambda t: mp.besselj(1, a * t) * mp.besselj(1, b * t) / t ** 2, [2, mp.inf], omega=a - b)
print("sum - integral =", S - float(Iv), " (predicted 0)")

print("\n=== (d) closed form for a PS sum via Theorem 7.5: sum eps_m m^{-1/2} J_{3/2}(2m) J_1(m/2) = sqrt(pi)/16 ===")
f = lambda t: t ** -0.5 * jv(1.5, 2 * t) * j1(0.5 * t)
S = eps_sum(f, 4_000_000, 0.0)
print("sum =", S, "  sqrt(pi)/16 =", np.sqrt(pi) / 16)
Iv = mp.quad(lambda t: t ** -0.5 * mp.besselj(1.5, 2 * t) * mp.besselj(1, 0.5 * t), [0, 1, 2]) + \
     mp.quadosc(lambda t: t ** -0.5 * mp.besselj(1.5, 2 * t) * mp.besselj(1, 0.5 * t), [2, mp.inf], omega=1.5)
print("integral =", float(Iv))

print("\n=== (e) sinc boundary example: sum eps_m sinc(2 pi m) = 1/2, integral = 1/4 ===")
print("defect predicted 2pi * (1/2) * (1/(4pi)) =", 2 * pi * 0.5 / (4 * pi))
