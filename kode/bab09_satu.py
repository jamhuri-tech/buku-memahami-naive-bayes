"""Bab 9: satu fitur kontinu, dua kelas normal.

Fitur: titik cekung (terburuk). Rata-rata dan varians (pembagi N)
setiap kelas, lalu posterior di beberapa nilai fitur.
"""
import numpy as np
from scipy.stats import norm

from bab09_data import FITUR, muat

X, y = muat()
j = FITUR.index("titik cekung (terburuk)")
x = X[:, j]
prior = np.array([np.mean(y == 0), np.mean(y == 1)])
mu = np.array([x[y == k].mean() for k in (0, 1)])
var = np.array([x[y == k].var() for k in (0, 1)])
print(f"fitur: {FITUR[j]}")
print(f"prior    jinak {prior[0]:.4f}, ganas {prior[1]:.4f}")
print(f"rata-rata jinak {mu[0]:.4f}, ganas {mu[1]:.4f}")
print(f"sim. baku jinak {np.sqrt(var[0]):.4f}, ganas {np.sqrt(var[1]):.4f}")
print("   x     p(x|jinak)  p(x|ganas)  P(ganas|x)")
for t in (0.05, 0.10, 0.14, 0.18, 0.25):
    f = prior * norm.pdf(t, mu, np.sqrt(var))
    print(f"{t:5.2f}   {f[0] / prior[0]:9.3f}   {f[1] / prior[1]:9.3f}"
          f"     {f[1] / f.sum():.4f}")
