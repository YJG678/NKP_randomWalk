import numpy as np
from scipy.special import j1
from scipy.integrate import quad
import mpmath as mp
mp.mp.dps = 20

def Lam1(z):
    z = np.asarray(z, float)
    out = np.ones_like(z)
    nz = z != 0
    out[nz] = 2 * j1(z[nz]) / z[nz]
    return out

print("=== lattice: sum_{k in Z^2} Lam1(a|k|)^3 (disk partial sums, smoothed) vs integral ===")
R = 1500
xs = np.arange(-R, R + 1, dtype=float)
X, Y = np.meshgrid(xs, xs, sparse=True)
r = np.sqrt(X ** 2 + Y ** 2)
for aa in [1.5, 2.0, 2.3, 2.8]:
    vals = Lam1(aa * r) ** 3
    # Riesz-type smoothing (1-r^2/R^2)^2 suppresses truncation oscillation; limit equals plain sum
    w = np.clip(1 - (r / R) ** 2, 0, None) ** 2
    s = np.sum(vals * w)
    f = lambda k: 2 * mp.pi * k * (2 * mp.besselj(1, aa * k) / (aa * k)) ** 3 if k else mp.mpf(0)
    integ = mp.quad(f, mp.linspace(0, 2, 5)) + mp.quadosc(f, [2, mp.inf], omega=aa)
    print(f"a={aa}: 3a vs 2pi: {3*aa:.3f} vs {2*np.pi:.3f}  sum={s:.8f}  integral={float(integ):.8f}  diff={s-float(integ):.2e}")

print("\n=== I_inf / pi ===")
n = np.arange(1, 400001, dtype=float)
def f(k):
    if k == 0: return 1.0
    return float(np.exp(np.sum(np.log(np.abs(np.sinc(k / (2 * n - 1) / np.pi)))))) * np.prod(np.sign(np.sinc(k / (2 * n[:50] - 1) / np.pi)))
tot = sum(quad(f, j, j + 1, epsabs=1e-15, epsrel=1e-12)[0] for j in range(0, 40))
print("I_inf/pi ~", 2 * tot / np.pi, " integrand at 40:", f(40.0))
