"""Bab 4: kerugian tidak simetris.

Menandai pesan biasa sebagai penipuan (positif palsu) merugikan 9,
meloloskan penipuan (negatif palsu) merugikan 1. Dua kelas Gaussian
dengan prior penipuan 0,2. Ambang terbaik pada posterior adalah
9/(9 + 1) = 0,9; ambang itu kita cari juga dengan memindai x.
"""
import numpy as np
from scipy.stats import norm

BIAYA_FP, BIAYA_FN = 9.0, 1.0
PRIOR1, MU, SD = 0.2, (-1.0, 1.0), 1.0


def posterior1(x):
    a = PRIOR1 * norm.pdf(x, MU[1], SD)
    b = (1 - PRIOR1) * norm.pdf(x, MU[0], SD)
    return a / (a + b)


def kerugian(t):
    """Kerugian harapan aturan 'penipuan jika x > t'."""
    fp = (1 - PRIOR1) * norm.sf(t, MU[0], SD)
    fn = PRIOR1 * norm.cdf(t, MU[1], SD)
    return BIAYA_FP * fp + BIAYA_FN * fn


c = BIAYA_FP / (BIAYA_FP + BIAYA_FN)
print(f"ambang posterior c = {c:.2f}")
t = np.linspace(-2, 4, 60001)
k = kerugian(t)
terbaik = t[np.argmin(k)]
print(f"pindai x: kerugian minimum {k.min():.4f} di x = {terbaik:.4f}")
print(f"posterior di x itu = {posterior1(terbaik):.4f}")
print(f"rumus: x = ln 6 = {np.log(6):.4f}")
for t0, nama in ((np.log(4) / 2, "ambang galat terkecil"),
                 (np.log(6), "ambang kerugian terkecil")):
    print(f"{nama:25s}: x > {t0:.4f}, kerugian {kerugian(t0):.4f}")
