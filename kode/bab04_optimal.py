"""Bab 4: pengklasifikasi Bayes optimal dan galat Bayes.

Dua kelas satu fitur: x | y=0 ~ N(-1, 1), x | y=1 ~ N(+1, 1).
Aturan "kelas 1 jika x > t" untuk berbagai ambang t; galatnya dihitung
dengan rumus dan diperiksa dengan simulasi 200.000 pengamatan.
"""
import numpy as np
from scipy.stats import norm

from bab04_data import BENIH, dua_gaussian

MU, SD = (-1.0, 1.0), 1.0


def galat(t, prior1):
    """Peluang salah aturan 'kelas 1 jika x > t'."""
    salah0 = (1 - prior1) * norm.sf(t, MU[0], SD)   # 0 dikira 1
    salah1 = prior1 * norm.cdf(t, MU[1], SD)        # 1 dikira 0
    return salah0 + salah1


def ambang_bayes(prior1):
    """Titik tempat kedua posterior sama besar."""
    tengah = (MU[0] + MU[1]) / 2
    return tengah + SD**2 / (MU[1] - MU[0]) * np.log(
        (1 - prior1) / prior1)


rng = np.random.default_rng(BENIH)
for prior1 in (0.5, 0.2):
    x, y = dua_gaussian(200_000, prior1, MU, SD, rng)
    tb = ambang_bayes(prior1)
    print(f"prior kelas 1 = {prior1}: ambang Bayes t* = {tb:.4f}")
    print("   ambang t   galat rumus   galat simulasi")
    daftar = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    daftar = sorted([t for t in daftar if abs(t - tb) > 1e-9] + [tb])
    for t in daftar:
        sim = np.mean((x > t).astype(int) != y)
        tanda = "  <- t*" if t == tb else ""
        print(f"   {t:+8.4f}   {galat(t, prior1):.4f}"
              f"        {sim:.4f}{tanda}")
