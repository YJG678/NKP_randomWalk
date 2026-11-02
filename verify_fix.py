import mpmath as mp
mp.mp.dps=25; pi=mp.pi; J=mp.besselj; G=mp.gamma
j0=lambda x: mp.sin(x)/x if x else mp.mpf(1)
j1=lambda x: (mp.sin(x)/x**2-mp.cos(x)/x) if x else mp.mpf(0)
# (1) multipole exponential-damping coefficient: predicted -X, X = int_{-1}^1 u (7-2u)^{-2} du
X=mp.quad(lambda u: u/(7-2*u)**2,[-1,1]); Xc=(mp.log(mp.mpf(9)/5)-mp.mpf(7)/5+mp.mpf(7)/9)/4
print("X numeric",mp.nstr(X,15)," closed (ln(9/5)-7/5+7/9)/4 =",mp.nstr(Xc,15))
for t in [mp.mpf('0.01'),mp.mpf('0.003'),mp.mpf('0.001')]:
    fe=lambda k: k**2*mp.exp(-t*k)*j1(k)*j1(k)*j0(3*k)
    Ve=mp.quad(fe,mp.linspace(0,40/t,int(4000)))
    print(f"t={t}: W^E/t = {mp.nstr(Ve/t,10)}   predicted -X = {mp.nstr(-X,10)}")
# (2) b=c in regime (iii) for non-half-integer t: formula vs quadrature
def I_num(r,t,a,b,c):
    f=lambda x: x**(1-r)*J(r,a*x)*J(t,b*x)*J(t,c*x)
    w=min(abs(s) for s in (a+b+c,a+b-c,a-b+c,-a+b+c) if abs(s)>1e-12)
    return mp.quad(f,mp.linspace(0,6,13))+mp.quadosc(f,[6,mp.inf],omega=w)
def KJ(r,t,a,b,c):
    K=2**(1-r)*a**(r-2*t-2)*(b*c)**t/(mp.sqrt(pi)*G(t+0.5)*G(r-t))
    def h(ph):
        T=mp.cos(ph); base=1-(b*b+c*c+2*b*c*T)/a**2
        return mp.mpf(0) if base<=0 else mp.sin(ph)**(2*t)*base**(r-t-1)
    Tst=(a*a-b*b-c*c)/(2*b*c); pts=[0,mp.mpf(10)**-8,0.01,0.3,pi/2,pi-0.3,pi-0.01,pi-mp.mpf(10)**-8,pi]
    if -1<Tst<1: pts=sorted(pts+[mp.acos(Tst)])
    return K*mp.quad(h,pts,maxdegree=10)
for (r,t,a,b,c) in [(1.0,0.3,1.2,1,1),(2.0,0.3,2.0,1,1),(1.5,-0.3,1.0,1,1)]:
    r,t,a,b,c=map(mp.mpf,(r,t,a,b,c))
    print(f"b=c: r={r} t={t} a={a}: num={mp.nstr(I_num(r,t,a,b,c),14)}  KJ={mp.nstr(mp.re(KJ(r,t,a,b,c)),14)}")
