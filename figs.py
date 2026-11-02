"""Numerical figures for 'Random walks behind Borwein integrals'.
Run:  python3 figs.py   (writes fig_*.pdf into the current directory)
Every curve is either an exact formula from the paper or an independent
numerical evaluation (quadrature / Monte Carlo) used as a check."""
import numpy as np, mpmath as mp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc, j1
from scipy.signal import fftconvolve

# ---------- style (validated reference palette, light mode) ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W1, W2, W3 = 3.3, 6.6, 6.9

def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")

# =====================================================================
# Figure: the Borwein walk (plateau, erosion, deficit, messenger margin)
# =====================================================================
def fig_walk():
    dx = 2e-4
    x = np.arange(-2.4, 2.4 + dx / 2, dx)
    a = [1 / (2 * n - 1) for n in range(1, 17)]
    box = lambda an: np.where(np.abs(x) <= an, 1 / (2 * an), 0.0)
    dens = [box(a[0])]
    for n in range(1, 8):
        p = fftconvolve(dens[-1], box(a[n]), mode="same") * dx
        dens.append(p)
    rng = np.random.default_rng(1)
    S = sum(rng.uniform(-an, an, 4_000_000) for an in a[:8])
    fig, axs = plt.subplots(1, 3, figsize=(W3, 2.35), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    h, e = np.histogram(S, bins=240, range=(-2.4, 2.4), density=True)
    ax.bar(0.5 * (e[1:] + e[:-1]), h, width=e[1] - e[0], color="#d9d8d2", linewidth=0, label="Monte Carlo, $N=8$")
    for N in range(8):
        ax.plot(x, dens[N], color=RAMP[N], lw=1.4)
    ax.text(1.08, 0.47, r"$p_1$", color=RAMP[3], fontsize=8)
    ax.text(1.55, 0.18, r"$p_8$", color=RAMP[7], fontsize=8)
    ax.annotate("", xy=(-1, 0.56), xytext=(1, 0.56), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax.text(0, 0.58, "range of step 1", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(0, 0.75)
    ax.set_xlabel("position $x$"); ax.set_ylabel("density $p_N(x)$")
    ax.legend(loc="upper right", handlelength=1.0, fontsize=6.5)
    panel(ax, "(a)")
    # (b) margin
    ax = axs[1]
    Ns = np.arange(1, 17)
    margin = [1 - sum(a[1:N]) for N in Ns]
    cols = [BLUE if m >= 0 else ORANGE for m in margin]
    ax.bar(Ns, margin, color=cols, width=0.7)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("number of steps $N$"); ax.set_ylabel(r"$a_1-\sum_{n=2}^{N}a_n$")
    ax.text(10.5, 0.62, "$N\\leq7$: messengers\nnot yet arrived,\n$I_N=\\pi$", fontsize=7.5, color="#1c5cab", ha="center")
    ax.text(12.5, 0.1, "$N\\geq8$: $I_N<\\pi$", fontsize=7.5, color="#c4501f", ha="center")
    ax.set_ylim(-0.45, 1.05)
    ax.set_xticks([1, 4, 8, 12, 16])
    panel(ax, "(b)")
    # (c) deficit vs corner law
    ax = axs[2]
    ex = [-10.8324971297, -7.9237098410, -7.0297745453, -6.5573059955, -6.2547379986, -6.0405676880, -5.8795040365, -5.7533082805, -5.6514334899]
    co = [-10.8324971297, -7.9237098410, -7.0297741976, -6.5569767684, -6.2511996591, -6.0275104104, -5.8486551617, -5.6956430794, -5.5578167990]
    NN = np.arange(8, 17)
    ax.plot(NN, ex, color=BLUE, marker="o", ms=4, label="exact $\\mathbb{P}(|S_N|>1)$")
    ax.plot(NN, co, color=ORANGE, ls="--", lw=1.4, marker="s", ms=4, mfc="white", label="corner law")
    ax.plot(NN[:2], co[:2], ls="none", marker="s", ms=4, color=ORANGE)
    ax.set_xlabel("$N$"); ax.set_ylabel(r"$\log_{10}\,(\pi-I_N)/\pi$")
    ax.legend(loc="lower right", fontsize=7)
    ax.annotate("corner law exact\nfor $N=8,9$", xy=(8.15, -10.6), xytext=(9.6, -9.0), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    panel(ax, "(c)")
    fig.tight_layout(w_pad=1.2)
    fig.savefig("fig_walk.pdf"); plt.close(fig)

# =====================================================================
# Figure: value versus convergence
# =====================================================================
def fig_convergence():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    K = np.linspace(0.01, 60, 3000)
    ax = axs[0]
    part = 0.5 * (np.sin(1.5 * K) / 1.5 + np.sin(0.5 * K) / 0.5)
    ax.plot(K, part, color=ORANGE, lw=1.2, label=r"partial integral $\int_0^K$")
    eps = 1 / K**2
    gv = np.sqrt(np.pi) / (4 * np.sqrt(eps)) * (np.exp(-9 / (16 * eps)) + np.exp(-1 / (16 * eps)))
    ax.plot(K, gv, color=BLUE, label=r"Gaussian value, $\varepsilon=K^{-2}$")
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel(r"cut-off $K$  (resp. $\varepsilon^{-1/2}$)"); ax.set_ylabel(r"$\int\cos k\,\cos\frac{k}{2}\,\mathrm{d}k$")
    ax.set_ylim(-1.6, 2.2); ax.legend(loc="upper right", fontsize=7)
    ax.set_title(r"$d=1$: divergent integral, Gaussian value $0$", color=INK)
    panel(ax, "(a)")
    ax = axs[1]
    from scipy.special import sici
    part = 12 * np.pi * (sici(K)[0] - np.sin(K))
    gv = 12 * np.pi * (np.pi / 2 * (1 - erfc(K / 2)) - np.sqrt(np.pi) * K / 2 * np.exp(-K**2 / 4))
    ax.plot(K, part / np.pi**2, color=ORANGE, lw=1.2, label=r"spherical partial integral")
    ax.plot(K, gv / np.pi**2, color=BLUE, label="Gaussian value")
    ax.axhline(6, color=INK2, lw=0.8, ls=":")
    ax.text(58, 11.2, r"Gaussian value $6\pi^2=(2\pi)^3/V_3$", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_ylim(-0.5, 11.5)
    ax.set_xlabel(r"cut-off $K$  (resp. $\varepsilon^{-1/2}$)"); ax.set_ylabel(r"$\int_{|k|<K}\Lambda_{3/2}(|k|)\,\mathrm{d}^3k\;/\;\pi^2$")
    ax.legend(loc="lower right", fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax.set_title(r"$d=3$: Pinsky phenomenon at the centre of a ball", color=INK)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=2)
    fig.savefig("fig_convergence.pdf"); plt.close(fig)

# =====================================================================
# Figure: triangle plateau
# =====================================================================
def fig_triangle():
    rng = np.random.default_rng(7)
    n = 20_000_000
    def unit(n):
        v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
    X = unit(n) + unit(n) + unit(n)
    R = np.linalg.norm(X, axis=1)
    h, e = np.histogram(R, bins=150, range=(0, 3))
    rc = 0.5 * (e[1:] + e[:-1]); shell = 4 / 3 * np.pi * (e[1:]**3 - e[:-1]**3)
    px = h / n / shell
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.45))
    ax = axs[0]
    ax.axvspan(0, 1, color="#cde2fb", alpha=0.6, lw=0)
    ax.plot(rc, px * 8 * np.pi, color=BLUE, marker="o", ms=2.2, lw=0, label=r"Monte Carlo, $2\times10^7$ walks")
    ax.axhline(1, color=ORANGE, lw=1.4, ls="--", label=r"Lemma: $p_X=1/(8\pi r_1r_2r_3)$")
    ax.text(0.5, 0.25, "flat for\n$|x|<\\delta_T=1$", ha="center", fontsize=8, color="#1c5cab")
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.4)
    ax.set_xlabel(r"$|x|$"); ax.set_ylabel(r"$8\pi r_1r_2r_3\,p_X(x)$")
    ax.set_title(r"density of $X=\Theta_1+\Theta_2+\Theta_3$ in $\mathbb{R}^3$", color=INK)
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "(a)")
    ax = axs[1]
    aa = np.linspace(0.001, 4, 800)
    Iex = np.where(aa < 1, np.pi / 4, np.where(aa < 3, np.pi * (3 - aa) / (8 * aa), 0.0))
    ax.axvspan(0, 1, color="#cde2fb", alpha=0.6, lw=0); ax.axvspan(3, 4, color="#f0efec", lw=0)
    ax.plot(aa, Iex, color=BLUE, label="exact (locality)")
    mp.mp.dps = 20
    j0 = lambda z: mp.sin(z) / z if z != 0 else mp.mpf(1)
    pts = [0.3, 0.7, 1.3, 2.0, 2.6, 3.4]
    num = []
    for av in pts:
        f = lambda k: k**2 * j0(k)**3 * j0(av * k)
        num.append(float(mp.quad(f, mp.linspace(0, 40, 41)) + mp.quadosc(f, [40, mp.inf], omega=min(abs(3 - av), abs(1 - av), 1 + av) or 1)))
    ax.plot(pts, num, ls="none", marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.5, label="numerical quadrature")
    ax.text(0.5, 0.86, "plateau\n$\\pi/4$", ha="center", fontsize=8, color="#1c5cab")
    ax.text(3.5, 0.06, "no\ntriangle:\n$0$", ha="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 4); ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("radius $a$ of the extra Pearson step"); ax.set_ylabel(r"$\int_0^\infty k^2 j_0(k)^3 j_0(ak)\,\mathrm{d}k$")
    ax.set_title(r"triangle plateau, $\boldsymbol{r}=(1,1,1)$", color=INK)
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=2)
    fig.savefig("fig_triangle.pdf"); plt.close(fig)
    return list(zip(pts, num))

# =====================================================================
# Figure: Gaussian vs exponential damping
# =====================================================================
def W_err(r, p2):
    r1, r2, r3 = r; s = [r2 + r3 - r1, r3 + r1 - r2, r1 + r2 - r3]; S = sum(r)
    p = np.sqrt(p2); W0 = np.pi / (8 * r1 * r2 * r3) * (sum(np.sign(s)) - 1)
    from scipy.special import erf
    G = np.pi / (8 * r1 * r2 * r3) * (sum(erf(si / (2 * p)) for si in s) - erf(S / (2 * p)))
    Ee = (sum(np.arctan(si / p2) for si in s) - np.arctan(S / p2)) / (4 * r1 * r2 * r3)
    return G - W0, Ee - W0
def fig_damping():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.45))
    ax = axs[0]
    p2 = np.logspace(-3, 0, 200)
    g, e = W_err((1, 1, 1), p2)
    ax.loglog(p2, np.abs(e), color=ORANGE, label=r"exponential $e^{-p^2k}$")
    ax.loglog(p2, 2 / 3 * p2, color=ORANGE, lw=1, ls="--", label=r"prediction $\frac{2}{3}p^2$")
    m = np.abs(g) > 1e-17
    ax.loglog(p2[m], np.abs(g[m]), color=BLUE, label=r"Gaussian $e^{-p^2k^2}$")
    ax.set_ylim(1e-16, 2); ax.set_xlabel(r"damping strength $p^2$"); ax.set_ylabel(r"$|W^{\bullet}_{000}-\pi/4|$")
    ax.set_title(r"equilateral triangle $\boldsymbol{r}=(1,1,1)$", color=INK)
    ax.legend(loc="lower right", fontsize=7)
    panel(ax, "(a)")
    ax = axs[1]
    smin = np.logspace(-3, -0.05, 120)
    coef = []; gerr = []
    for s in smin:
        r = (1.0, 1.0, 2.0 - s)
        r1, r2, r3 = r; ss = [r2 + r3 - r1, r3 + r1 - r2, r1 + r2 - r3]
        coef.append(abs((sum(1 / x for x in ss) - 1 / sum(r)) / (4 * r1 * r2 * r3)))
        gerr.append(abs(W_err(r, 1e-4)[0]) / 1e-4)
    ax.loglog(smin, coef, color=ORANGE, label=r"exponential: $|W^{\mathrm{E}}-W|/p^2$")
    ax.loglog(smin, np.array(gerr) + 1e-30, color=BLUE, label=r"Gaussian: $|W^{\mathrm{G}}-W|/p^2$, $p^2=10^{-4}$")
    ax.loglog(smin, 1 / (4 * 2 * smin), color=INK2, lw=0.8, ls=":")
    ax.text(2e-3, 1 / (4 * 2 * 2e-3) * 1.6, r"$\propto 1/s_{\min}$", fontsize=8, color=INK2)
    ax.set_ylim(1e-8, 1e3)
    ax.set_xlabel(r"distance to degeneracy $s_{\min}$, $\boldsymbol{r}=(1,1,2-s_{\min})$"); ax.set_ylabel("error coefficient")
    ax.set_title("approaching a degenerate triangle", color=INK)
    ax.legend(loc="lower left", fontsize=7)
    panel(ax, "(b)")
    fig.tight_layout(w_pad=2)
    fig.savefig("fig_damping.pdf"); plt.close(fig)

# =====================================================================
# Figure: Gradshteyn-Ryzhik 6.578.8 at non-half-integer order
# =====================================================================
def I_formula(r, t, a, b, c):
    if mp.isint(r - t) and r - t <= 0: return mp.mpf(0)
    if a < abs(b - c): return mp.mpf(0)
    K = 2**(1 - r) * a**(r - 2 * t - 2) * (b * c)**t / (mp.sqrt(mp.pi) * mp.gamma(t + 0.5) * mp.gamma(r - t))
    # T = cos(phi): (1-T^2)^(t-1/2) dT = sin(phi)^(2t) dphi  (milder endpoint singularity)
    def g(ph):
        base = 1 - (b**2 + c**2 + 2 * b * c * mp.cos(ph)) / a**2
        return mp.mpf(0) if base <= 0 else mp.sin(ph)**(2 * t) * base**(r - t - 1)
    Tst = (a**2 - b**2 - c**2) / (2 * b * c)
    pts = [0, mp.acos(Tst), mp.pi] if -1 < Tst < 1 else [0, mp.pi/2, mp.pi]
    return K * mp.quad(g, pts)
def I_num(r, t, a, b, c):
    f = lambda x: x**(1 - r) * mp.besselj(r, a * x) * mp.besselj(t, b * x) * mp.besselj(t, c * x)
    w = min(abs(s) for s in (a + b + c, a + b - c, a - b + c, -a + b + c) if abs(s) > 1e-9)
    return mp.quad(f, mp.linspace(0, 8, 17)) + mp.quadosc(f, [8, mp.inf], omega=w)
def fig_gr():
    mp.mp.dps = 15
    t, b, c = mp.mpf(-0.3), mp.mpf(1), mp.mpf(1.5)
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.55))
    ax.axvspan(0, 0.5, color="#f0efec", lw=0); ax.axvspan(0.5, 2.5, color="#e8f1fc", lw=0)
    out = {}
    for r, col, lab in [(0.9, BLUE, r"$r=0.9$"), (1.7, VIOLET, r"$r=1.7$ ($r-t=2$)")]:
        r = mp.mpf(r)
        A = np.linspace(0.02, 5, 160)
        ax.plot(A, [float(I_formula(r, t, mp.mpf(x), b, c)) for x in A], color=col, label=lab + r": $K\mathcal{J}$")
        P = [0.3, 1.0, 2.0, 3.0, 4.2]
        V = [float(I_num(r, t, mp.mpf(x), b, c)) for x in P]
        ax.plot(P, V, ls="none", marker="o", ms=5, mfc="white", mec=col, mew=1.5)
        out[float(r)] = list(zip(P, V, [float(I_formula(r, t, mp.mpf(x), b, c)) for x in P]))
    ax.plot([], [], ls="none", marker="o", ms=5, mfc="white", mec=INK2, label="direct quadrature")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.text(0.25, 0.06, "(i)\nhole", ha="center", va="bottom", fontsize=7.5, color=INK2)
    for xx, s in [(1.5, "(iii) overlap"), (3.75, "(ii) shell inside ball")]:
        ax.text(xx, -0.075, s, ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 5); ax.set_ylim(-0.09, 0.58)
    ax.set_xlabel("$a$"); ax.set_ylabel(r"$\int_0^\infty x^{1-r}J_r(ax)J_t(x)J_t(1.5x)\,\mathrm{d}x$")
    ax.set_title(r"Gradshteyn--Ryzhik 6.578.8 at $t=-0.3$", color=INK)
    ax.legend(loc="upper left", fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("fig_gr.pdf"); plt.close(fig)
    return out

# =====================================================================
# Figure: aliasing
# =====================================================================
def fig_aliasing():
    M = 4_000_000
    m = np.arange(1, M + 1, dtype=np.float64); w = (1 - m / M)**3
    A = np.linspace(0.05, 20, 4000)
    D = 2 * sum(np.where(2 * np.pi * n < A, np.sqrt(np.clip(1 - (2 * np.pi * n / A)**2, 0, None)), 0) for n in (1, 2, 3))
    P = [3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 19.5]
    V = [A_ / 4 + np.sum(w * j1(A_ * m) / m) - 1 for A_ in P]
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.45))
    for n in (1, 2, 3):
        ax.axvline(2 * np.pi * n, color=GRID, lw=1)
    ax.plot(A, D, color=BLUE, label="exact aliasing defect")
    ax.plot(P, V, ls="none", marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.5, label=r"series $-$ integral ($4\times10^6$ terms)")
    ax.set_xticks([0, 2 * np.pi, 4 * np.pi, 6 * np.pi]); ax.set_xticklabels(["0", r"$2\pi$", r"$4\pi$", r"$6\pi$"])
    ax.text(3.1, 0.25, "sum = integral\n(walk cannot reach $2\\pi\\mathbb{Z}$)", fontsize=7.5, color=INK2, ha="center")
    ax.set_xlabel("bandwidth $a$"); ax.set_ylabel(r"$\sum_{m\geq0}\epsilon_m\frac{J_1(am)}{m}-\int_0^\infty\frac{J_1(at)}{t}\mathrm{d}t$")
    ax.set_xlim(0, 20); ax.set_ylim(-0.1, 4.4)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_aliasing.pdf"); plt.close(fig)
    return list(zip(P, V))

if __name__ == "__main__":
    # Figures used in the article: figure 2 (convergence) and figure 4 (damping).
    # The other functions draw panels from earlier drafts; run them with  python figs.py all
    import sys
    fig_convergence(); print("convergence done")
    fig_damping(); print("damping done")
    if "all" in sys.argv[1:]:
        fig_walk(); print("triangle", fig_triangle()); print("gr", fig_gr()); print("aliasing", fig_aliasing())
