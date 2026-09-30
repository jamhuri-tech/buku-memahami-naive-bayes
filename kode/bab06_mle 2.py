"""Bab 6: taksiran kemungkinan maksimum dicari secara numerik, lalu
dibandingkan dengan rumus tertutupnya (frekuensi relatif, rata-rata,
varians berpembagi N).
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

# ---- Bernoulli: kata "hadiah" di empat pesan penipuan ----
x = np.array([1, 1, 0, 1])


def nll_bern(t):
    return -np.sum(x * np.log(t) + (1 - x) * np.log(1 - t))


h = minimize_scalar(nll_bern, bounds=(1e-9, 1 - 1e-9), method="bounded",
                    options={"xatol": 1e-12})
print(f"Bernoulli : numerik {h.x:.6f}, rumus N1/N = {x.mean():.6f}")

# ---- kategorik: prior tiga kelas dari hitungan (3, 4, 5) ----
N = np.array([3, 4, 5])


def nll_kat(z):                  # parameter bebas: softmax dari z
    t = np.exp(z - z.max())
    t /= t.sum()
    return -np.sum(N * np.log(t))


h = minimize(nll_kat, np.zeros(3), method="BFGS",
             options={"gtol": 1e-10})
t = np.exp(h.x - h.x.max())
t /= t.sum()
print("kategorik : numerik " + " ".join(f"{v:.6f}" for v in t))
print("            rumus   " + " ".join(f"{v:.6f}" for v in N / N.sum()))

# ---- Gaussian: nilai satu fitur di satu kelas ----
g = np.array([5.0, 7.0, 9.0, 11.0])


def nll_gauss(par):
    mu, log_s2 = par
    s2 = np.exp(log_s2)
    return 0.5 * np.sum(np.log(2 * np.pi * s2) + (g - mu) ** 2 / s2)


h = minimize(nll_gauss, [0.0, 0.0], method="BFGS",
             options={"gtol": 1e-10})
print(f"Gaussian  : numerik mu = {h.x[0]:.6f}, "
      f"sigma^2 = {np.exp(h.x[1]):.6f}")
print(f"            rumus  mu = {g.mean():.6f}, "
      f"sigma^2 = {g.var():.6f} (bagi N)")
print(f"            np.var(ddof=1) = {g.var(ddof=1):.6f} "
      f"(bagi N - 1)")
