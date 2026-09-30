"""Bab 9: naive Bayes Gaussian tidak bergantung pada satuan fitur,
kecuali lewat var_smoothing.

Satu fitur dikali 1000 (misalnya satuan diubah). Tanpa var_smoothing,
log-odds setiap sampel tidak berubah. Dengan var_smoothing bawaan,
berubah, karena tambahan varians = 1e-9 x varians terbesar.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

from bab09_data import FITUR, muat

X, y = muat()
v = X.var(axis=0)
jb, jk = int(np.argmax(v)), int(np.argmin(v))
print(f"varians terbesar : {FITUR[jb]:28s} {v[jb]:.4g}")
print(f"varians terkecil : {FITUR[jk]:28s} {v[jk]:.4g}")
print(f"tambahan var_smoothing = 1e-9 x {v[jb]:.4g} = {1e-9 * v[jb]:.3g}")


def log_odds(m, X):
    L = m.predict_joint_log_proba(X)
    return L[:, 1] - L[:, 0]


X2 = X.copy()
X2[:, jk] *= 1000                       # satuan fitur terkecil diubah
for vs in (0.0, 1e-9):
    a = log_odds(GaussianNB(var_smoothing=vs).fit(X, y), X)
    b = log_odds(GaussianNB(var_smoothing=vs).fit(X2, y), X2)
    print(f"var_smoothing = {vs:g}: maks |selisih log-odds| = "
          f"{np.abs(a - b).max():.3g}")
