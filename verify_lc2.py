"""Checks supporting the general-t proof of Proposition 7.9."""
import mpmath as mp
mp.mp.dps = 25
pi = mp.pi; J = mp.besselj; G = mp.gamma

def Lam(nu, z):
    return mp.mpf(1) if z == 0 else G(nu + 1) * (2 / z)**nu * J(nu, z)

def pdens(t):
    k = G(t + 1) / (mp.sqrt(pi) * G(t + 0.5))
    return lambda s: k * (1 - s * s)**(t - 0.5)

print("=== (1) product formula Lam_t(bx)Lam_t(cx) = E Lam_t(x R), R^2=b^2+c^2+2bcT, all real t ===")
for t in [mp.mpf('-0.3'), mp.mpf('0.3'), mp.mpf('2.7')]:
    p = pdens(t)
    for (b, c, x) in [(1, 1.5, 2.3), (0.7, 0.7, 5.1)]:
        lhs = Lam(t, b * x) * Lam(t, c * x)
        rhs = mp.quad(lambda T: p(T) * Lam(t, x * mp.sqrt(b * b + c * c + 2 * b * c * T)), [-1, 0, 1])
        print(f"t={mp.nstr(t,3)} b={b} c={c} x={x}: lhs={mp.nstr(lhs,15)} rhs={mp.nstr(rhs,15)}")

print("\n=== (2) H_r(z) = cosine transform of x^{2t-r+1}J_r(ax) on |z|<a (direct, r>2t+1/2) ===")
def H(r, t, a, z):
    return 2**(2 * t + 1 - r) * a**(r - 2 * t - 2) * G(t + 1) / G(r - t) * mp.hyp2f1(t + 1, t - r + 1, 0.5, (z / a)**2)
for (r, t, a, z) in [(2.2, 0.3, 1, 0.4), (3.5, 1.0, 2, 1.1)]:
    r, t, a, z = map(mp.mpf, (r, t, a, z))
    f = lambda x: x**(2 * t - r + 1) * J(r, a * x) * mp.cos(z * x)
    num = mp.quad(f, mp.linspace(0, 6, 13)) + mp.quadosc(f, [6, mp.inf], omega=a - z)
    print(f"r={r} t={t}: direct={mp.nstr(num,15)}  H={mp.nstr(H(r,t,a,z),15)}")

print("\n=== (3) identity (2^t Gamma(t+1))^{-1} E H_r(w S) = (a^2-w^2)^{r-t-1}/(2^{r-t-1} a^r Gamma(r-t)), incl. r<=t ===")
for (r, t, a, w) in [(0.2, 0.3, 1, 0.5), (0.8, 1.5, 1, 0.6), (-0.2, -0.3, 1, 0.7), (1.3, 0.3, 1, 0.5)]:
    r, t, a, w = map(mp.mpf, (r, t, a, w))
    p = pdens(t)
    lhs = mp.quad(lambda s: p(s) * H(r, t, a, w * s), [-1, 0, 1]) / (2**t * G(t + 1))
    rhs = (a * a - w * w)**(r - t - 1) / (2**(r - t - 1) * a**r * G(r - t))
    print(f"r={r} t={t} w={w}: lhs={mp.nstr(lhs,15)} rhs={mp.nstr(rhs,15)}")

print("\n=== (4) full proposition for negative / non-half-integer t ===")
def I_num(r, t, a, b, c):
    f = lambda x: x**(1 - r) * J(r, a * x) * J(t, b * x) * J(t, c * x)
    w = min(abs(s) for s in (a + b + c, a + b - c, a - b + c, -a + b + c) if abs(s) > 1e-12)
    return mp.quad(f, mp.linspace(0, 6, 13)) + mp.quadosc(f, [6, mp.inf], omega=w)
def I_formula(r, t, a, b, c):
    if mp.isint(r - t) and r - t <= 0:
        return mp.mpf(0)
    K = 2**(1 - r) * a**(r - 2 * t - 2) * (b * c)**t / (mp.sqrt(pi) * G(t + 0.5) * G(r - t))
    def g(T):
        base = 1 - (b**2 + c**2 + 2 * b * c * T) / a**2
        return mp.mpf(0) if base <= 0 else (1 - T**2)**(t - 0.5) * base**(r - t - 1)
    Tst = (a**2 - b**2 - c**2) / (2 * b * c)
    pts = [-1, Tst, 1] if -1 < Tst < 1 else [-1, 1]
    return K * mp.quad(g, pts)
for (name, r, t, a, b, c) in [("(ii) t=-0.3, r-t=0.9", 0.6, -0.3, 3, 1, 1.5),
                              ("(ii) t=-0.3, r-t=-0.1", -0.4, -0.3, 3, 1, 1.5),
                              ("(iii) t=-0.3, r-t=1.2", 0.9, -0.3, 2, 1, 1.5),
                              ("(i) t=-0.3", 0.6, -0.3, 0.4, 1, 1.5),
                              ("(ii) b=c, t=0.3, r-t=-0.5", -0.2, 0.3, 3, 1, 1)]:
    r, t, a, b, c = map(mp.mpf, (r, t, a, b, c))
    print(f"{name:32s} num={mp.nstr(I_num(r,t,a,b,c),15):>22s}  formula={mp.nstr(I_formula(r,t,a,b,c),15)}")
