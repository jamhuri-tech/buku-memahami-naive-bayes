"""Bab 4: menaksir P(x | y) dengan tabel pola utuh.

Bagian 1: sepuluh pesan rekaan, empat kata, 16 pola yang mungkin.
Bagian 2: simulasi. Untuk p fitur biner, berapa bagian pesan uji yang
polanya tidak pernah muncul di data latih kelasnya sendiri?
"""
from collections import Counter

import numpy as np

from bab04_data import BENIH, KATA, PESAN_X, PESAN_Y


def pola(baris):
    return "".join(str(v) for v in baris)


# ---- Bagian 2 ----
def bagian_tak_terlihat(p, n_latih, rng, n_uji=2000):
    theta = rng.uniform(0.05, 0.5, size=(2, p))   # P(x_j = 1 | k)
    def bangkit(n):
        y = rng.integers(0, 2, n)
        x = (rng.random((n, p)) < theta[y]).astype(np.uint8)
        return x, y
    xl, yl = bangkit(n_latih)
    xu, yu = bangkit(n_uji)
    terlihat = [set(map(bytes, xl[yl == k])) for k in (0, 1)]
    return np.mean([bytes(b) not in terlihat[k]
                    for b, k in zip(xu, yu)])


def bagian1():
    print("pola", " ".join(KATA), "-> hitungan penipuan / biasa")
    hit = {k: Counter(pola(b) for b in PESAN_X[PESAN_Y == k])
           for k in (0, 1)}
    semua = sorted(set(hit[0]) | set(hit[1]), reverse=True)
    for s in semua:
        print(f"  {s}   {hit[1][s]} / {hit[0][s]}")
    print(f"terisi {len(semua)} dari {2 ** len(KATA)} pola")
    for q in ("1100", "1001"):
        a, b = hit[1][q] / 4, hit[0][q] / 6
        pb = 0.4 * a + 0.6 * b
        teks = f"{0.4 * a / pb:.3f}" if pb > 0 else "0/0"
        print(f"pola {q}: P(x|pen) = {hit[1][q]}/4, "
              f"P(x|biasa) = {hit[0][q]}/6 -> posterior {teks}")


if __name__ == "__main__":
    bagian1()
    rng = np.random.default_rng(BENIH)
    print()
    print("bagian pesan uji yang polanya belum pernah terlihat")
    print("   p    n=1000  n=10000  n=100000")
    for p in (5, 10, 15, 20, 30):
        baris = [bagian_tak_terlihat(p, n, rng)
                 for n in (1000, 10_000, 100_000)]
        print(f"  {p:2d}    " + "   ".join(f"{b:.3f}" for b in baris))
