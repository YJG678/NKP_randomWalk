"""Check of the general finite part in Proposition 4.2:
V_E(t) = u(0) + (t/pi^2) P_u(nu) + O(t^2) for nu = u dx with u smooth (not constant) near 0,
P_u(nu) = lim_{rho->0} [ int_{|y|>rho} |y|^-4 dnu - 4 pi u(0)/rho ].
Test density: u(y) = exp(-|y-a|^2), a = |a| e_3 (spherical mean known in closed form)."""
import mpmath as mp
mp.mp.dps = 60
for a in [mp.mpf(0), mp.mpf('0.7')]:
    M = (lambda r: 4*mp.pi*mp.e**(-(r*r+a*a))*(mp.sinh(2*r*a)/(2*r*a) if a > 0 else 1))  # spherical integral of u over |y|=r, per r^2
    u0 = mp.e**(-a*a)
    # finite part; the integrand is bounded near 0, so start at r0 = 1e-12 (avoids cancellation in M(r)-4 pi u0)
    r0 = mp.mpf('1e-12')
    P = mp.quad(lambda r: (M(r) - 4*mp.pi*u0)/r**2, [r0, 1, mp.inf]) + r0*(M(r0) - 4*mp.pi*u0)/r0**2
    for t in [mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')]:
        VE = mp.quad(lambda r: t*r*r*M(r)/(mp.pi**2*(t*t+r*r)**2), [0, t, 10*t, 1, mp.inf])
        pred = u0 + t/mp.pi**2*P
        print(f"|a|={float(a):.1f} t={float(t):.0e}  V_E={mp.nstr(VE,15)}  pred={mp.nstr(pred,15)}  (V_E-pred)/t^2={mp.nstr((VE-pred)/t**2,8)}")
