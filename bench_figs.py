import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figs, bench
from scipy.special import j1
B, O, V, INK2, GRID = figs.BLUE, figs.ORANGE, figs.VIOLET, figs.INK2, figs.GRID
d = json.load(open("bench.json"))
def panel(ax, s): figs.panel(ax, s)
fig, axs = plt.subplots(1, 3, figsize=(figs.W3, 2.45))
# (a) single block
ax = axs[0]
A = np.linspace(0.05, 20, 3000)
D = 2 * sum(np.where(2 * np.pi * n < A, np.sqrt(np.clip(1 - (2 * np.pi * n / A)**2, 0, None)), 0) for n in (1, 2, 3))
P = [3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 19.5]
Vv = [1.08e-07 * 0, 1.1e-07, 0.8816301038575616, 1.2379816234643823, 1.5559123242210209, 1.7039301700514748, 2.668895707453054, 3.0773157793525234, 3.934909327131069]
Vv[0] = -1.43e-07
for n in (1, 2, 3): ax.axvline(2 * np.pi * n, color=GRID, lw=1)
ax.plot(A, D, color=B, label="exact defect")
ax.plot(P, Vv, ls="none", marker="o", ms=4.5, mfc="white", mec=O, mew=1.4, label="numerical")
ax.set_xticks([0, 2 * np.pi, 4 * np.pi, 6 * np.pi]); ax.set_xticklabels(["0", r"$2\pi$", r"$4\pi$", r"$6\pi$"])
ax.set_xlim(0, 20); ax.set_xlabel("bandwidth $A=a$"); ax.set_ylabel("sum $-$ integral")
ax.set_title(r"$J_1(at)/t$", color=figs.INK); ax.legend(loc="upper left", fontsize=7); panel(ax, "(a)")
# (b) three blocks
ax = axs[1]
lams = np.linspace(0.42, 1.62, 90)
Ad = lams * bench.R1.sum()
import os
if os.path.exists("defect3_curve.npy"): Dd = np.load("defect3_curve.npy")
else:
    Dd = np.array([bench.defect3(l) for l in lams]); np.save("defect3_curve.npy", Dd)
for n in (1, 2): ax.axvline(2 * np.pi * n, color=GRID, lw=1)
ax.plot(Ad, Dd, color=B, label="exact defect")
sc = np.array(d["scan"])
ax.plot(sc[:, 0], sc[:, 1], ls="none", marker="o", ms=4.5, mfc="white", mec=O, mew=1.4, label="numerical")
ax.set_xticks([2 * np.pi, 4 * np.pi]); ax.set_xticklabels([r"$2\pi$", r"$4\pi$"])
ax.set_xlabel(r"bandwidth $A=a_1+a_2+a_3$"); ax.set_title(r"$J_1(5\lambda t)J_1(3\lambda t)J_1(\frac{3}{2}\lambda t)/t^3$", color=figs.INK)
ax.legend(loc="upper left", fontsize=7); panel(ax, "(b)")
# (c) signed case
ax = axs[2]
aa = np.linspace(4, 14, 2000)
ax.axhline(0, color=figs.AXIS, lw=0.8)
for n in (1, 2): ax.axvline(2 * np.pi * n, color=GRID, lw=1)
yy = np.array([bench.defect_signed(x) for x in aa])
for n in (1, 2):
    k = np.searchsorted(aa, 2 * np.pi * n); yy[k] = np.nan
ax.plot(aa, yy, color=V, label="exact defect")
sg = np.array(d["signed"])
ax.plot(sg[:, 0], sg[:, 1], ls="none", marker="o", ms=4.5, mfc="white", mec=O, mew=1.4, label="numerical")
ax.set_xticks([2 * np.pi, 4 * np.pi]); ax.set_xticklabels([r"$2\pi$", r"$4\pi$"])
ax.set_xlabel("bandwidth $a$"); ax.set_title(r"$t^{2}\cdot t^{-5/2}J_{5/2}(at)$, $k=1$: signed", color=figs.INK)
ax.legend(loc="lower right", fontsize=7); panel(ax, "(c)")
fig.tight_layout(w_pad=1.2); fig.savefig("fig_aliasing.pdf"); plt.close(fig)
# benchmark figure
fig, ax = plt.subplots(figsize=(figs.W1 * 1.3, 2.5))
lr = np.array(d["lattice_rows"]); qr = np.array(d["quad_rows"])
ax.loglog(lr[:, 0], lr[:, 1], color=O, marker="s", ms=4, label="integer-grid sum, uncorrected")
ax.loglog(lr[:, 0], np.maximum(lr[:, 2], 1e-16), color=B, marker="o", ms=4, label=r"integer-grid sum $-$ defect (5.4)")
ax.loglog(qr[:, 0], np.maximum(qr[:, 1], 1e-16), color=V, marker="^", ms=4, ls="--", lw=1.4, label=r"direct quadrature on $[0,K]$")
ax.set_xlabel(r"terms $M$ (sums) or cut-off $K$ (quadrature)"); ax.set_ylabel("relative error")
ax.set_ylim(1e-16, 1); ax.legend(loc="center right", fontsize=7, frameon=True, facecolor="white", edgecolor="none")
ax.set_title(r"$\int_0^\infty J_1(5t)J_1(3t)J_1(\frac{3}{2}t)\,t^{-3}\mathrm{d}t$, $A=9.5>2\pi$", color=figs.INK)
fig.tight_layout(); fig.savefig("fig_bench.pdf"); plt.close(fig)
print("ok")
