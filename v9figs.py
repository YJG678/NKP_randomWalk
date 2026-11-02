"""Consolidated figure panels for the NJP version (uses figs.py style and bench.json data)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import mpmath as mp
import figs, bench
F = figs
def walk():
    dx = 2e-4; x = np.arange(-2.4, 2.4 + dx / 2, dx)
    a = [1 / (2 * n - 1) for n in range(1, 17)]
    box = lambda an: np.where(np.abs(x) <= an, 1 / (2 * an), 0.0)
    dens = [box(a[0])]
    for n in range(1, 8): dens.append(fftconvolve(dens[-1], box(a[n]), mode="same") * dx)
    rng = np.random.default_rng(1)
    S = sum(rng.uniform(-an, an, 4_000_000) for an in a[:8])
    fig, axs = plt.subplots(1, 2, figsize=(4.75, 2.3))
    ax = axs[0]
    h, e = np.histogram(S, bins=240, range=(-2.4, 2.4), density=True)
    ax.bar(0.5 * (e[1:] + e[:-1]), h, width=e[1] - e[0], color="#d9d8d2", linewidth=0, label="Monte Carlo, $N=8$")
    for N in range(8): ax.plot(x, dens[N], color=F.RAMP[N], lw=1.3)
    ax.text(1.08, 0.47, r"$p_1$", color=F.RAMP[3], fontsize=8); ax.text(1.5, 0.17, r"$p_8$", color=F.RAMP[7], fontsize=8)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(0, 0.72); ax.set_xlabel("position $x$"); ax.set_ylabel("density $p_N(x)$")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.0); F.panel(ax, "(b)")
    ax = axs[1]
    ex = [-10.8324971297, -7.9237098410, -7.0297745453, -6.5573059955, -6.2547379986, -6.0405676880, -5.8795040365, -5.7533082805, -5.6514334899]
    co = [-10.8324971297, -7.9237098410, -7.0297741976, -6.5569767684, -6.2511996591, -6.0275104104, -5.8486551617, -5.6956430794, -5.5578167990]
    NN = np.arange(8, 17)
    ax.plot(NN, ex, color=F.BLUE, marker="o", ms=3.5, label=r"exact $\mathbb{P}(|S_N|>1)$")
    ax.plot(NN, co, color=F.ORANGE, ls="--", lw=1.3, marker="s", ms=3.5, mfc="white", label="corner law")
    ax.plot(NN[:2], co[:2], ls="none", marker="s", ms=3.5, color=F.ORANGE)
    ax.set_xlabel("number of steps $N$"); ax.set_ylabel(r"$\log_{10}\,(\pi-I_N)/\pi$")
    ax.legend(loc="lower right", fontsize=6.5); F.panel(ax, "(c)")
    fig.tight_layout(w_pad=1.0); fig.savefig("fig1_walk.pdf"); plt.close(fig)
def triangle():
    rng = np.random.default_rng(7); n = 20_000_000
    def unit(n):
        v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1)[:, None]
    R = np.linalg.norm(unit(n) + unit(n) + unit(n), axis=1)
    h, e = np.histogram(R, bins=150, range=(0, 3)); rc = 0.5 * (e[1:] + e[:-1]); shell = 4 / 3 * np.pi * (e[1:]**3 - e[:-1]**3)
    fig, axs = plt.subplots(1, 2, figsize=(F.W2, 2.35))
    ax = axs[0]
    ax.axvspan(0, 1, color="#cde2fb", alpha=0.6, lw=0)
    ax.plot(rc, h / n / shell * 8 * np.pi, color=F.BLUE, marker="o", ms=2.2, lw=0, label=r"Monte Carlo, $2\times10^7$ walks")
    ax.axhline(1, color=F.ORANGE, lw=1.4, ls="--", label=r"Lemma: $p_X=1/(8\pi r_1r_2r_3)$")
    ax.text(0.5, 0.25, "flat for\n$|x|<\\delta_T=1$", ha="center", fontsize=8, color="#1c5cab")
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.4); ax.set_xlabel(r"$|x|$"); ax.set_ylabel(r"$8\pi r_1r_2r_3\,p_X(x)$")
    ax.set_title(r"spatial density of $X=\Theta_1+\Theta_2+\Theta_3$", color=F.INK); ax.legend(loc="upper right", fontsize=7); F.panel(ax, "(c)")
    ax = axs[1]
    aa = np.linspace(0.001, 4, 800)
    Iex = np.where(aa < 1, np.pi / 4, np.where(aa < 3, np.pi * (3 - aa) / (8 * aa), 0.0))
    ax.axvspan(0, 1, color="#cde2fb", alpha=0.6, lw=0); ax.axvspan(3, 4, color="#f0efec", lw=0)
    ax.plot(aa, Iex, color=F.BLUE, label="exact (locality)")
    pts = [0.3, 0.7, 1.3, 2.0, 2.6, 3.4]; num = [0.7853981633974483, 0.7853981633974483, 0.5135295683752547, 0.19634954084936207, 0.060415243338265236, 2.6e-25]
    ax.plot(pts, num, ls="none", marker="o", ms=5, mfc="white", mec=F.ORANGE, mew=1.5, label="numerical quadrature")
    ax.text(0.5, 0.86, "plateau\n$\\pi/4$", ha="center", fontsize=8, color="#1c5cab"); ax.text(3.5, 0.06, "no\ntriangle:\n$0$", ha="center", fontsize=8, color=F.INK2)
    ax.set_xlim(0, 4); ax.set_ylim(-0.03, 1.0); ax.set_xlabel("radius $a$ of an extra Pearson step"); ax.set_ylabel(r"$\int_0^\infty k^2 j_0(k)^3 j_0(ak)\,\mathrm{d}k$")
    ax.set_title(r"triangle plateau, $\boldsymbol{r}=(1,1,1)$", color=F.INK); ax.legend(loc="upper right", fontsize=7); F.panel(ax, "(d)")
    fig.tight_layout(w_pad=2); fig.savefig("fig3_triangle.pdf"); plt.close(fig)
def alias():
    d = json.load(open("bench.json"))
    fig, axs = plt.subplots(1, 2, figsize=(F.W2, 2.45))
    ax = axs[0]
    lams = np.linspace(0.42, 1.62, 90); Ad = lams * bench.R1.sum(); Dd = np.load("defect3_curve.npy")
    for n in (1, 2): ax.axvline(2 * np.pi * n, color=F.GRID, lw=1)
    ax.plot(Ad, Dd, color=F.BLUE, label="exact correction")
    sc = np.array(d["scan"])
    ax.plot(sc[:, 0], sc[:, 1], ls="none", marker="o", ms=4.5, mfc="white", mec=F.ORANGE, mew=1.4, label=r"sum $-$ integral (numerical)")
    ax.set_xticks([2 * np.pi, 4 * np.pi]); ax.set_xticklabels([r"$2\pi$", r"$4\pi$"])
    ax.set_xlabel(r"bandwidth $A=9.5\lambda$"); ax.set_ylabel("aliasing error")
    ax.set_title(r"$f(t)=J_1(5\lambda t)J_1(3\lambda t)J_1(\frac{3}{2}\lambda t)/t^3$", color=F.INK)
    ax.legend(loc="upper left", fontsize=7); F.panel(ax, "(b)")
    ax = axs[1]
    lr = np.array(d["lattice_rows"]); qr = np.array(d["quad_rows"])
    ax.loglog(lr[:, 0], lr[:, 1], color=F.ORANGE, marker="s", ms=3.5, label="integer-grid sum")
    ax.loglog(lr[:, 0], np.maximum(lr[:, 2], 1e-16), color=F.BLUE, marker="o", ms=3.5, label="sum $-$ correction")
    ax.loglog(qr[:, 0], np.maximum(qr[:, 1], 1e-16), color=F.VIOLET, marker="^", ms=3.5, ls="--", lw=1.3, label=r"quadrature on $[0,K]$")
    ax.set_xlabel(r"terms $M$ or cut-off $K$"); ax.set_ylabel("relative error"); ax.set_ylim(1e-16, 1)
    ax.set_title(r"$\lambda=1$ ($A=9.5$): benchmark", color=F.INK)
    ax.legend(loc="center right", fontsize=7, frameon=True, facecolor="white", edgecolor="none"); F.panel(ax, "(c)")
    fig.tight_layout(w_pad=2); fig.savefig("fig5_alias.pdf"); plt.close(fig)
if __name__ == "__main__":
    walk(); triangle(); alias(); print("ok")
