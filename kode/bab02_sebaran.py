"""Bab 2: memeriksa rumus sebaran dengan simulasi.

(1) Multinomial: dadu bersisi tiga dengan peluang (1/2, 1/3, 1/6)
    dilempar 4 kali; peluang hitungan (2, 1, 1).
(2) Binomial: 10 lemparan dengan peluang 0,3; rata-rata dan varians.
(3) Kepadatan normal baku: peluang jatuh di [-1, 1] sebagai luas.
"""
import numpy as np
from scipy.stats import binom, multinomial, norm

from bab04_data import BENIH

rng = np.random.default_rng(BENIH)
n = 200_000
H = rng.multinomial(4, [1 / 2, 1 / 3, 1 / 6], size=n)
sim = np.mean(np.all(H == [2, 1, 1], axis=1))
rumus = multinomial.pmf([2, 1, 1], 4, [1 / 2, 1 / 3, 1 / 6])
print(f"multinomial (2,1,1): rumus {rumus:.4f}, simulasi {sim:.4f}")
b = rng.binomial(10, 0.3, size=n)
print(f"binomial(10; 0,3): rata-rata {b.mean():.4f} (rumus 3)")
print(f"                   varians   {b.var():.4f} (rumus 2.1)")
z = rng.standard_normal(n)
print(f"normal baku di [-1, 1]: luas {norm.cdf(1) - norm.cdf(-1):.4f},"
      f" simulasi {np.mean(np.abs(z) <= 1):.4f}")
print(f"kepadatan normal baku di 0: {norm.pdf(0):.4f}")
print(f"P(binomial = 3) = {binom.pmf(3, 10, 0.3):.4f}")
