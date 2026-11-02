"""Checks for the Chellino--Slepian (CS23) connection: triangle plateau and damping asymptotics."""
import mpmath as mp
mp.mp.dps = 25
pi = mp.pi
j0 = lambda x: mp.mpf(1) if x == 0 else mp.sin(x) / x
def j1(x):
    return mp.mpf(0) if x == 0 else mp.sin(x) / x**2 - mp.cos(x) / x
Lam32 = lambda x: mp.mpf(1) if x == 0 else 3 * j1(x) / x  # uniform ball in R^3

def osc(f, w):
    return mp.quad(f, mp.linspace(0, 4, 9)) + mp.quadosc(f, [4, mp.inf], omega=w)

def dT(r1, r2, r3):
    return min(r1 + r2 - r3, r1 + r3 - r2, r2 + r3 - r1)

print("=== triangle plateau: int k^2 j0(kr1)j0(kr2)j0(kr3) prod F = pi/(4 r1r2r3) prod F(0) if sum a < delta_T ===")
cases = [((1, 1, 1), [(j0, .3), (j0, .4)]),
         ((1, 1.5, 2), [(j0, .2), (j0, .25)]),
         ((1, 1.5, 2), [(Lam32, .45)]),
         ((1, 1, 1), [(j0, .6), (j0, .5)]),          # sum 1.1 > delta_T = 1 -> should fail
         ((1, 1.5, 2), [(j0, .3), (Lam32, .3)])]     # sum 0.6 > 0.5 -> should fail
for r, blocks in cases:
    f = lambda k: k**2 * j0(k * r[0]) * j0(k * r[1]) * j0(k * r[2]) * mp.fprod([F(a * k) for F, a in blocks])
    val = osc(f, min(r))
    pred = pi / (4 * r[0] * r[1] * r[2])
    s = sum(a for _, a in blocks)
    print(r, [a for _, a in blocks], "sum=", s, "delta_T=", dT(*r), " value=", mp.nstr(val, 15), " plateau=", mp.nstr(pred, 15))

print("\n=== damping asymptotics, l=0 ===")
# exact closed forms via int_0^inf e^{-tk} sin(ak)/k = atan(a/t);  int e^{-tk^2} sin(ak)/k = (pi/2) erf(a/(2 sqrt t))
for t in [mp.mpf('0.1'), mp.mpf('0.03'), mp.mpf('0.01')]:
    # outside triangle r=(1,1,3)
    Iexp = (mp.atan(3 / t) / 2 - mp.atan(5 / t) / 4 - mp.atan(1 / t) / 4) / 3
    Igau = (pi / 2) * (mp.erf(3 / (2 * mp.sqrt(t))) / 2 - mp.erf(5 / (2 * mp.sqrt(t))) / 4 - mp.erf(1 / (2 * mp.sqrt(t))) / 4) / 3
    # inside r=(1,1,1)
    Jexp = (3 * mp.atan(1 / t) - mp.atan(3 / t)) / 4
    Jgau = (pi / 2) * (3 * mp.erf(1 / (2 * mp.sqrt(t))) - mp.erf(3 / (2 * mp.sqrt(t)))) / 4
    print(f"t={mp.nstr(t,3)}: outside exp={mp.nstr(Iexp,10)} vs 2t/45={mp.nstr(2*t/45,10)} | outside gauss={mp.nstr(Igau,6)} (e^-1/4t={mp.nstr(mp.exp(-1/(4*t)),3)})"
          f" | inside exp-pi/4={mp.nstr(Jexp-pi/4,10)} vs -2t/3={mp.nstr(-2*t/3,10)} | inside gauss-pi/4={mp.nstr(Jgau-pi/4,4)}")

print("\n=== direct numerical check of the damped integrals (no closed forms) ===")
for t in [mp.mpf('0.05')]:
    f = lambda k: k**2 * mp.exp(-t * k) * j0(k) * j0(k) * j0(3 * k)
    print("exp-damped (1,1,3) quad:", mp.nstr(mp.quad(f, mp.linspace(0, 2000, 400)), 12),
          " closed:", mp.nstr((mp.atan(3 / t) / 2 - mp.atan(5 / t) / 4 - mp.atan(1 / t) / 4) / 3, 12))

print("\n=== higher multipoles l=(1,1,0), r=(1,1,3) outside triangle: Gaussian ~ exp small, exponential ~ linear in t ===")
for t in [mp.mpf('0.04'), mp.mpf('0.02'), mp.mpf('0.01')]:
    fe = lambda k: k**2 * mp.exp(-t * k) * j1(k) * j1(k) * j0(3 * k)
    fg = lambda k: k**2 * mp.exp(-t * k * k) * j1(k) * j1(k) * j0(3 * k)
    Ve = mp.quad(fe, mp.linspace(0, 40 / t, 800)); Vg = mp.quad(fg, mp.linspace(0, 12 / mp.sqrt(t), 400))
    print(f"t={mp.nstr(t,3)}: exp={mp.nstr(Ve,8)}  exp/t={mp.nstr(Ve/t,8)}   gauss={mp.nstr(Vg,4)}")

print("\n=== degenerate triangle r=(1,1,2): Gaussian limit = half plateau pi/16 ===")
for t in [mp.mpf('0.01'), mp.mpf('0.001')]:
    v = (pi / 2) * (mp.erf(2 / (2 * mp.sqrt(t))) / 2 - mp.erf(4 / (2 * mp.sqrt(t))) / 4) / 2
    print(mp.nstr(t,3), mp.nstr(v, 12), " pi/16=", mp.nstr(pi / 16, 12))
