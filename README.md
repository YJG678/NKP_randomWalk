# Reproduction scripts

The scripts reproduce the figures, the table and the numerical checks in
"Finite-range locality in oscillatory Bessel integrals: random walks, damping and aliasing".
Requirements: Python 3, numpy, scipy, mpmath and matplotlib.

| Script | Output used in the paper |
|---|---|
| `figs.py` | `fig_convergence.pdf` (figure 2) and `fig_damping.pdf` (figure 4). The style settings are shared with `v9figs.py`. Other functions draw figures from earlier drafts. |
| `bench.py` | Benchmark for the three-Bessel example of section 5.3. Writes `bench.json`, which feeds table 1 and figure 5(c). |
| `bench_figs.py` | Defect curve for figure 5(b). Writes `defect3_curve.npy`. |
| `v9figs.py` | The Python panels of figures 1, 3 and 5 (`fig1_walk.pdf`, `fig3_triangle.pdf`, `fig5_alias.pdf`). Reads `bench.json` and `defect3_curve.npy`. |
| `verify*.py` | Independent checks of the numbers quoted in the text: deficits, shell identities, triangle plateau, damping coefficients, aliasing defects. `verify_finitepart.py` checks the general finite-part expansion of Proposition 4.2 for a non-constant density. |
| `run.log` | Output of a complete clean run of all scripts in the order below, with run times. |

Suggested order:

```
python bench.py
python bench_figs.py
python figs.py
python v9figs.py
python verify1.py
python verify2a.py
...
python verify_finitepart.py
```

Panel (a) of figures 1, 3 and 5 is drawn with TikZ inside the LaTeX source.
`verify_lc*.py` also check material on Gradshteyn–Ryzhik 6.578.8 that is not part of this article.
