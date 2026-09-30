"""Pemeriksa Contoh Soal Bab 13."""
import math

import numpy as np
from sklearn.naive_bayes import GaussianNB

from bab04_data import PESAN_X, PESAN_Y
from bab13_nb import Bernoulli, Campuran, Gaussian, Multinomial

# ---- Contoh Soal 13.1: fitur campuran, prior dihitung sekali ----
lr_g = 0.5 * math.log(2 / 5) - 0.4 + 4          # Contoh Soal 9.2
lr_m = math.log(1331 / 960)                     # Contoh Soal 10.2
assert round(lr_g, 3) == 3.142 and round(lr_m, 3) == 0.327
prior = math.log(1 / 3)
assert round(prior, 3) == -1.099
lo = prior + lr_g + lr_m
assert round(lo, 3) == 2.370
assert round(1 / (1 + math.exp(-lo)), 3) == 0.915
lo2 = 2 * prior + lr_g + lr_m
assert round(lo2, 3) == 1.271 and round(1 / (1 + math.exp(-lo2)), 3) == 0.781
# periksa dengan kelas Campuran: 4 dokumen, dua bagian fitur
H = np.array([[2, 1, 0], [1, 2, 0], [1, 0, 0],
              [0, 1, 2], [1, 0, 1], [0, 1, 2]])
yh = np.array([1, 1, 1, 0, 0, 0])
G = np.array([[5.0], [7.0], [9.0], [0.0], [2.0], [4.0]])
# satu kelas penipuan tambahan di bagian G mengubah varians; cukup
# periksa bahwa log-gabungan = prior + jumlah log fitur kedua bagian
m = Campuran([Multinomial(1.0), Gaussian(0.0)]).fit((H, G), yh)
a = Multinomial(1.0).fit(H, yh)
b = Gaussian(0.0).fit(G, yh)
q = (np.array([[2, 0, 1]]), np.array([[6.0]]))
J = m.joint_log(q)
J2 = a.joint_log(q[0]) + b.joint_log(q[1]) - a.class_log_prior_
assert np.allclose(J, J2)

# ---- Contoh Soal 13.2: bentuk linear Bernoulli ----
nb = Bernoulli(1.0).fit(PESAN_X, PESAN_Y)
w = nb.log_theta_ - nb.log_1m_theta_
c = nb.log_1m_theta_.sum(axis=1)
assert np.allclose(np.round(c, 4), [-2.0262, -3.4782])
assert np.allclose(np.round(w[1], 4), [0.6931, 0.6931, 0.6931, -1.6094])
assert np.allclose(np.round(w[0], 4), [-1.0986, -0.5108, -1.0986, 0.5108])
J = nb.joint_log(np.array([[1, 1, 0, 0]]))[0]
assert np.allclose(np.round(J, 4), [-4.1465, -3.0082])
assert round(J[1] - J[0], 4) == 1.1383
assert round(1 / (1 + math.exp(-(J[1] - J[0]))), 4) == 0.7574

# ---- Contoh Soal 13.3: var_smoothing, fitur konstan di satu kelas ----
X = np.array([[5.0], [5.0], [4.0], [6.0]])
y = np.array([0, 0, 1, 1])
assert np.var(X) == 0.5
eps = 1e-9 * 0.5
g = GaussianNB().fit(X, y)
assert np.allclose(g.var_.ravel(), [eps, 1 + eps])
la = -0.5 * math.log(2 * math.pi * eps)
assert round(math.log(2 * math.pi * eps), 3) == -19.579
assert round(la, 3) == 9.789
lb = -0.5 * math.log(2 * math.pi * (1 + eps))
assert round(lb, 3) == -0.919
assert round(la - lb, 3) == 10.708
p5 = g.predict_proba([[5.0]])[0, 0]
assert round(p5, 5) == 0.99998
assert g.predict_proba([[5.5]])[0, 0] == 0.0
assert round(0.25 / (2 * eps) / 1e8, 3) == 2.5

print("Contoh Soal Bab 13: semua bilangan cocok")
