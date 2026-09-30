"""Bab 5: bentuk odds teorema Bayes naif.

Log-odds posterior = log-odds prior + jumlah log-rasio kemungkinan
setiap fitur. Dihitung untuk pesan berpola 1100 (memuat "hadiah" dan
"transfer", tidak memuat "segera" dan "rapat").
"""
import numpy as np

from bab04_data import KATA, PESAN_X, PESAN_Y
from bab05_naif import latih_naif

prior, theta = latih_naif(PESAN_X, PESAN_Y)
x = np.array([1, 1, 0, 0])


def sumbangan(x, prior, theta):
    """Log-odds prior dan log-rasio kemungkinan setiap fitur."""
    p1 = np.where(x == 1, theta[1], 1 - theta[1])
    p0 = np.where(x == 1, theta[0], 1 - theta[0])
    return np.log(prior[1] / prior[0]), np.log(p1 / p0)


if __name__ == "__main__":
    awal, s = sumbangan(x, prior, theta)
    print(f"log-odds prior           {awal:+.4f}")
    for kata, xj, sj in zip(KATA, x, s):
        ada = "ada  " if xj else "tidak"
        print(f"  {kata:9s} {ada}         {sj:+.4f}")
    total = awal + s.sum()
    print(f"log-odds posterior       {total:+.4f}")
    print(f"posterior = 1/(1 + exp(-{total:.4f})) = "
          f"{1 / (1 + np.exp(-total)):.4f}")
