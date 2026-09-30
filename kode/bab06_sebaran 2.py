"""Bab 6: seberapa teliti frekuensi relatif sebagai taksiran.

Untuk kata dengan peluang kemunculan theta di sebuah kelas, taksiran
dari N pesan kelas itu adalah hitungan/N. Simulasi 100.000 kali
mengukur galat bakunya dan peluang hitungannya nol, lalu
membandingkannya dengan rumus sqrt(theta(1 - theta)/N) dan
(1 - theta)^N.
"""
import numpy as np

from bab04_data import BENIH

rng = np.random.default_rng(BENIH)
ulang = 100_000
# gb = galat baku taksiran; nol = peluang hitungannya nol
print(" theta    N  gb-sim  gb-rumus  nol-sim  nol-rumus")
for theta in (0.75, 0.05):
    for N in (4, 20, 100, 500):
        t = rng.binomial(N, theta, ulang) / N
        print(f"{theta:6.2f} {N:4d}  {t.std():.4f}  "
              f"{np.sqrt(theta * (1 - theta) / N):.4f}    "
              f"{np.mean(t == 0):.4f}   {(1 - theta) ** N:.4f}")
