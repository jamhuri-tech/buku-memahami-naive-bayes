"""Pemeriksa Contoh Soal Bab 11."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# ---- Contoh Soal 11.1: bias kelas kecil pada multinomial ----
V, a = 10_000, 1
besar = F(5000 + a, 100_000 + a * V)
kecil = F(250 + a, 5000 + a * V)
assert besar == F(5001, 110_000) and kecil == F(251, 15_000)
assert round(float(besar), 4) == 0.0455 and round(float(kecil), 4) == 0.0167
r = besar / kecil
assert round(float(r), 3) == 2.717
assert round(math.log(r), 4) == 0.9995
assert F(a * V, 5000 + a * V) == F(2, 3)          # massa hitungan semu
assert F(a * V, 100_000 + a * V) == F(1, 11)

# ---- Contoh Soal 11.2: complement ----
kom_kecil = F(10_000 + 1, 200_000 + V)        # dua kelas besar
kom_besar = F(5000 + 250 + 1, 105_000 + V)    # satu besar + kecil
assert kom_kecil == F(10_001, 210_000) and kom_besar == F(5251, 115_000)
w_kecil = -math.log(kom_kecil)
w_besar = -math.log(kom_besar)
assert round(float(kom_kecil), 4) == 0.0476
assert round(float(kom_besar), 4) == 0.0457
assert round(w_kecil, 3) == 3.044 and round(w_besar, 3) == 3.087
assert round(w_besar - w_kecil, 3) == 0.042
assert round(math.log(r) / (w_besar - w_kecil)) == 24

# ---- Contoh Soal 11.3: dua kelas, complement = multinomial tanpa prior
rng = np.random.default_rng(1)
X = rng.integers(0, 4, size=(40, 6))
y = rng.integers(0, 2, 40)
Xu = rng.integers(0, 4, size=(200, 6))
mnb = MultinomialNB(alpha=1, fit_prior=False).fit(X, y).predict(Xu)
cnb = ComplementNB(alpha=1).fit(X, y).predict(Xu)
assert np.array_equal(mnb, cnb)

# ---- Contoh Soal 11.4: log-tf dan idf ----
for c, v in ((1, 0.693), (2, 1.099), (4, 1.609), (8, 2.197)):
    assert round(math.log(1 + c), 3) == v
assert round(math.log(5 / 2) + 1, 3) == 1.916
assert math.log(5 / 5) + 1 == 1
dok = ["kata umum langka", "kata umum", "kata umum", "kata umum"]
tv = TfidfVectorizer(norm=None).fit(dok)
idf = dict(zip(tv.get_feature_names_out(), tv.idf_))
assert round(idf["langka"], 3) == 1.916 and idf["kata"] == 1.0

print("Contoh Soal Bab 11: semua bilangan cocok")
