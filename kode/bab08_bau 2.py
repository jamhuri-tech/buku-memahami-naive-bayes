"""Bab 8: satu fitur kategorik, sembilan nilai. Tabel hitungan fitur
bau per kelas, taksiran dengan alpha = 1, dan bobot buktinya."""
import numpy as np

from bab08_data import BAU, muat

huruf, y = muat()
bau = huruf[:, 4]
print(f"jamur: {len(y)}, beracun {y.sum()}, dapat dimakan "
      f"{len(y) - y.sum()}")
print("bau         racun  makan  P(.|racun)  P(.|makan)  bobot")
N1, N0 = y.sum(), len(y) - y.sum()
V = len(BAU)
for kode in sorted(BAU, key=lambda k: BAU[k]):
    n1 = np.sum((bau == kode) & (y == 1))
    n0 = np.sum((bau == kode) & (y == 0))
    t1 = (n1 + 1) / (N1 + V)
    t0 = (n0 + 1) / (N0 + V)
    print(f"{BAU[kode]:10s} {n1:6d} {n0:6d}     {t1:.5f}     "
          f"{t0:.5f}  {np.log(t1 / t0):+5.2f}")
