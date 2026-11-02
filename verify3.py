import mpmath as mp
mp.mp.dps=25; pi=mp.pi
sinc=lambda x: mp.mpf(1) if x==0 else mp.sin(x)/x
def E_abs(phi, L=200, w=1):
    # int_0^inf (1-phi)/k^2 = int_0^L (1-phi)/k^2 + 1/L - int_L^inf phi/k^2  (phi oscillatory, mean 0)
    f=lambda k:(1-phi(k))/k**2 if k else mp.mpf(0)
    g=lambda k: phi(k)/k**2
    head=mp.quad(f,mp.linspace(mp.mpf(10)**-8,L,4*L))+mp.quad(f,[0,mp.mpf(10)**-8])
    return 2/pi*(head+1/mp.mpf(L)-mp.quadosc(g,[L,mp.inf],omega=w))
for b,pred in [(0.5,1),(1.7,(1+1.7**2)/3.4),(1.0,1)]:
    print('uniform b',b, mp.nstr(E_abs(lambda k: mp.cos(k)*sinc(b*k), w=abs(1-b) if b!=1 else 1),15), pred)
print('shift c=0.4', mp.nstr(E_abs(lambda k: mp.cos(k)*mp.cos(0.4*k), w=0.6),15))
print('shift c=1.6', mp.nstr(E_abs(lambda k: mp.cos(k)*mp.cos(1.6*k), w=0.6),15))
# equality case N=3: int cos k sinc(k/2)^2 = 2*int_0^inf, using closed-form decomposition check numerically with tail subtraction
f=lambda k: mp.cos(k)*sinc(k/2)**2
L=200
head=mp.quad(f,mp.linspace(0,L,4*L))
# tail: cos k * 4 sin^2(k/2)/k^2 = (2cos k -1 -cos2k)/k^2 -> nonosc part -1/k^2
tail=-1/mp.mpf(L)+mp.quadosc(lambda k:(2*mp.cos(k)-mp.cos(2*k))/k**2,[L,mp.inf],omega=1)
print('N=3 equality case:', mp.nstr(2*(head+tail),12))
