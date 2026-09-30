"""Bab 18: ukuran kalibrasi. ECE dengan 10 selang lebar sama, Brier
score, dan log-loss dari posterior validasi silang."""
import numpy as np

from bab18_data import kasus, posterior


def ece(q, y, selang=10):
    b = np.minimum((q * selang).astype(int), selang - 1)
    total = 0.0
    for i in range(selang):
        m = b == i
        if m.any():
            total += m.mean() * abs(q[m].mean() - y[m].mean())
    return total


def brier(q, y):
    return np.mean((q - y) ** 2)


def logloss(q, y):
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


if __name__ == "__main__":
    print("model                  akurasi   ECE     Brier   log-loss")
    for nama, X, y, m in kasus():
        q = posterior(m, X, y)
        print(f"{nama:21s}  {np.mean((q > 0.5) == y):.4f}   "
              f"{ece(q, y):.4f}  {brier(q, y):.4f}  {logloss(q, y):.4f}")
