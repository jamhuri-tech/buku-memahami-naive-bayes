"""Bab 5: fitur yang digandakan menghitung bukti dua kali.

Model sebenarnya: dua kelas berprior seimbang, sepuluh fitur biner
yang benar-benar bebas bersyarat kelas. Tiga fitur pertama lalu
disalin m kali (m = 1 berarti tanpa salinan). Naive Bayes dilatih pada
fitur yang sudah disalin; posterior sebenarnya hanya memakai sepuluh
fitur asli. Kita bandingkan akurasi dan log-loss keduanya.
"""
import numpy as np

from bab04_data import BENIH

P = 10


def bangkit(n, theta, rng):
    y = rng.integers(0, 2, n)
    x = (rng.random((n, P)) < theta[y]).astype(float)
    return x, y


def log_odds(x, prior1, theta):
    """Log-odds naive Bayes untuk fitur biner (Bab 5)."""
    a = x @ np.log(theta[1] / theta[0])
    b = (1 - x) @ np.log((1 - theta[1]) / (1 - theta[0]))
    return np.log(prior1 / (1 - prior1)) + a + b


def salin(x, m):
    return np.hstack([x] + [x[:, :3]] * (m - 1))


def nilai(lo, y):
    q = 1 / (1 + np.exp(-lo))
    q = np.clip(q, 1e-15, 1 - 1e-15)
    akurasi = np.mean((lo > 0) == y)
    logloss = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    ekstrem = np.mean((q < 0.01) | (q > 0.99))
    return akurasi, logloss, ekstrem


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    theta = np.vstack([rng.uniform(0.2, 0.5, P),
                       rng.uniform(0.5, 0.8, P)])
    xl, yl = bangkit(20_000, theta, rng)
    xu, yu = bangkit(200_000, theta, rng)

    lo_benar = log_odds(xu, 0.5, theta)
    a, l, e = nilai(lo_benar, yu)
    # ekstrem: bagian posterior yang < 0,01 atau > 0,99
    print("                     akurasi  log-loss  ekstrem")
    print(f"posterior benar       {a:.4f}   {l:.4f}    {e:.3f}")
    for m in (1, 2, 4, 8):
        xl_m, xu_m = salin(xl, m), salin(xu, m)
        th = np.vstack([xl_m[yl == k].mean(axis=0) for k in (0, 1)])
        lo = log_odds(xu_m, yl.mean(), th)
        a, l, e = nilai(lo, yu)
        print(f"naif, salinan m = {m}   {a:.4f}   {l:.4f}    {e:.3f}")
