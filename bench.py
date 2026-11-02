"""Aliasing figure (three panels) and lattice-sum benchmark for section 5.
f(t) = t^{2k} prod_j t^{-nu_j} J_{nu_j}(a_j t);  sum_{m>=0} eps_m f(m) - int_0^inf f = 2 pi (-1)^k C sum_{1<=n<A/2pi} g^{(2k)}(2 pi n)."""
import time, json, numpy as np, mpmath as mp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1, jv
from scipy.integrate import quad, dblquad
import figs  # shared style
BLUE, ORANGE, VIOLET, INK2, GRID = figs.BLUE, figs.ORANGE, figs.VIOLET, figs.INK2, figs.GRID

# ---------- N=3 product, nu=(1,1,1), k=0, radii a=(5,3,1.5)*lam
R1 = np.array([5.0, 3.0, 1.5])
def rho(x, a):  # (F2) density for nu=1, d=1: semicircle
    return np.where(np.abs(x) < a, 2 / (np.pi * a) * np.sqrt(np.clip(1 - (x / a)**2, 0, None)), 0.0)
def g3(x, a):  # (rho1*rho2*rho3)(x): y=a1 sin th, z=a2 sin ph, inner quadrature split at the kinks of rho3
    def inner(th):
        pts = [p for p in [np.arcsin(np.clip((x - a[0] * np.sin(th) - s * a[2]) / a[1], -1, 1)) for s in (1, -1)] if -np.pi / 2 < p < np.pi / 2]
        return quad(lambda ph: (2 / np.pi)**2 * np.cos(th)**2 * np.cos(ph)**2 * rho(x - a[0] * np.sin(th) - a[1] * np.sin(ph), a[2]),
                    -np.pi / 2, np.pi / 2, limit=400, epsabs=1e-14, epsrel=1e-12, points=pts or None)[0]
    return quad(inner, -np.pi / 2, np.pi / 2, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
def defect3(lam):
    a = R1 * lam; A = a.sum(); C = np.prod(a / 2)
    return 2 * np.pi * C * sum(g3(2 * np.pi * n, a) for n in range(1, 10) if 2 * np.pi * n < A)
def f3(t, lam):
    a = R1 * lam
    return j1(a[0] * t) * j1(a[1] * t) * j1(a[2] * t) / t**3
def f3_0(lam):
    a = R1 * lam; return np.prod(a / 2)
def lattice3(lam, M):
    m = np.arange(1, M + 1, dtype=float)
    return 0.5 * f3_0(lam) + np.sum(f3(m, lam))
def integral3(lam, dps=25):
    mp.mp.dps = dps
    a = [mp.mpf(x) * lam for x in R1]
    f = lambda t: mp.besselj(1, a[0] * t) * mp.besselj(1, a[1] * t) * mp.besselj(1, a[2] * t) / t**3
    S = sorted({abs(a[0] + s1 * a[1] + s2 * a[2]) for s1 in (1, -1) for s2 in (1, -1)})
    return mp.quad(f, mp.linspace(0, 10, 21)) + mp.quadosc(f, [10, mp.inf], omega=float(min(S)))

# ---------- signed case N=1, k=1, nu=5/2: f(t)=t^{-1/2} J_{5/2}(a t)
def defect_signed(a):
    C = (a / 2)**2.5 / float(mp.gamma(3.5))
    s = 0.0
    for n in range(1, 10):
        x = 2 * np.pi * n
        if x < a: s += 15 / (4 * a**3) * (3 * x**2 / a**2 - 1)
    return -2 * np.pi * C * s
def series_signed(a, M=4_000_000):
    m = np.arange(1, M + 1, dtype=float); w = (1 - m / M)**3
    return np.sum(w * jv(2.5, a * m) / np.sqrt(m))  # f(0)=0
def integral_signed(a):  # int_0^inf t^{-1/2} J_{5/2}(a t) dt
    return np.sqrt(np.pi) / (2 * np.sqrt(2 * a))

if __name__ == "__main__":
    out = {}
    # ---------------- benchmark at lam=1 (A=9.5)
    lam = 1.0
    t0 = time.perf_counter(); Iref = integral3(lam, 30); t_ref = time.perf_counter() - t0
    Iref = float(Iref); out["Iref"] = Iref; out["t_ref"] = t_ref
    t0 = time.perf_counter(); D = defect3(lam); t_def = time.perf_counter() - t0
    out["defect"] = D; out["t_defect"] = t_def
    Ms = [10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    rows = []
    for M in Ms:
        t0 = time.perf_counter(); S = lattice3(lam, M); ts = time.perf_counter() - t0
        rows.append((M, abs(S - Iref) / abs(Iref), abs(S - D - Iref) / abs(Iref), ts))
    out["lattice_rows"] = rows
    # direct double-precision quadrature on [0,K] (scipy, oscillatory), error vs K
    qrows = []
    for K in [10, 30, 100, 300, 1000]:
        t0 = time.perf_counter()
        v, _ = quad(lambda t: f3(t, lam) if t > 0 else f3_0(lam), 0, K, limit=20000, epsabs=1e-14, epsrel=1e-13)
        tq = time.perf_counter() - t0
        qrows.append((K, abs(v - Iref) / abs(Iref), tq))
    out["quad_rows"] = qrows
    # ---------------- bandwidth scan for panel (b)
    lams = np.linspace(0.45, 1.6, 14)
    scan = []
    for l in lams:
        I = float(integral3(l, 15)); S = lattice3(l, 20_000); D = defect3(l)
        scan.append((float(l * R1.sum()), S - I, D))
    out["scan"] = scan
    # ---------------- signed case for panel (c)
    sig = []
    for a in [5.0, 6.8, 7.0, 8.0, 9.5, 11.0, 12.0, 13.5]:
        sig.append((a, float(series_signed(a) - integral_signed(a)), float(defect_signed(a))))
    out["signed"] = sig
    json.dump(out, open("bench.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("scan",)}, indent=1))
