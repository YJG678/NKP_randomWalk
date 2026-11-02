"""Checks for the Lovat--Celozzi (LC25) connection: GR 6.578.8 via locality in dimension 2t+2."""
import mpmath as mp
mp.mp.dps = 25
pi = mp.pi
J = mp.besselj

def I_num(r, t, a, b, c):
    f = lambda x: x**(1 - r) * J(r, a * x) * J(t, b * x) * J(t, c * x) if x != 0 else (mp.mpf(0) if r < 1 else mp.mpf(0))
    fr = sorted(set([abs(s) for s in (a + b + c, a + b - c, a - b + c, -a + b + c) if abs(s) > 1e-12]))
    w = min(fr)
    return mp.quad(f, mp.linspace(0, 6, 13)) + mp.quadosc(f, [6, mp.inf], omega=w)

def I_formula(r, t, a, b, c):
    """2^{1-r} a^{r-2t-2} (bc)^t / (sqrt(pi) Gamma(t+1/2) Gamma(r-t)) * int (1-T^2)^{t-1/2} (1-(b^2+c^2+2bcT)/a^2)_+^{r-t-1} dT"""
    if mp.isint(r - t) and r - t <= 0:
        return mp.mpf(0)
    K = 2**(1 - r) * a**(r - 2 * t - 2) * (b * c)**t / (mp.sqrt(pi) * mp.gamma(t + 0.5) * mp.gamma(r - t))
    def g(T):
        base = 1 - (b**2 + c**2 + 2 * b * c * T) / a**2
        return mp.mpf(0) if base <= 0 else (1 - T**2)**(t - 0.5) * base**(r - t - 1)
    # breakpoint where the bracket vanishes
    Tst = (a**2 - b**2 - c**2) / (2 * b * c)
    pts = [-1, 1] if not (-1 < Tst < 1) else [-1, Tst, 1]
    return K * mp.quad(g, pts)

cases = [
    ("a>b+c, r-t=1 (plateau = Thm 7.5)", 2, 1, 3, 1, 1.5),
    ("a>b+c, r-t=2", 3, 1, 3, 1, 1.5),
    ("a>b+c, r-t=1/2", 1.5, 1, 3, 1, 1.5),
    ("a>b+c, r-t=0  -> 0", 1, 1, 3, 1, 1.5),
    ("a>b+c, r-t=-1 -> 0", 0.5, 1.5, 3, 1, 1.5),
    ("a>b+c, r-t=-2 -> 0", 0.5, 2.5, 4, 1, 1.5),
    ("a>b+c, r-t=-1/2 (nonzero)", 1, 1.5, 3, 1, 1.5),
    ("a<|b-c| -> 0", 1, 0.5, 0.4, 1, 1.5),
    ("intermediate, r-t=3/2", 2, 0.5, 2, 1, 1.5),
    ("intermediate, r-t=2", 3, 1, 2, 1, 1.5),
    ("non-half-integer t=0.3, a>b+c", 1.6, 0.3, 3, 1, 1.5),
    ("non-half-integer t=0.3, intermediate", 1.6, 0.3, 2, 1, 1.5),
]
for name, r, t, a, b, c in cases:
    r, t, a, b, c = map(mp.mpf, (r, t, a, b, c))
    print(f"{name:40s} num={mp.nstr(I_num(r,t,a,b,c),15):>22s}  formula={mp.nstr(I_formula(r,t,a,b,c),15)}")

print("\nLC benchmark I_ex2(2,1/2;3,1,2) =0.188628080701505 (LC);  16/(27 pi) =", mp.nstr(16 / (27 * pi), 16),
      "  formula:", mp.nstr(I_formula(2, mp.mpf(0.5), 3, 1, 2), 16))

print("\n=== new multi-factor benchmark (triangle plateau): int x^2 j0(x)j0(2x)j0(1.5x)j0(.2x)j0(.15x) dx = pi/12 ===")
j0 = lambda x: mp.mpf(1) if x == 0 else mp.sin(x) / x
f = lambda x: x**2 * j0(x) * j0(2 * x) * j0(1.5 * x) * j0(0.2 * x) * j0(0.15 * x)
v = mp.quad(f, mp.linspace(0, 6, 13)) + mp.quadosc(f, [6, mp.inf], omega=0.15)
print(mp.nstr(v, 15), mp.nstr(pi / 12, 15))
