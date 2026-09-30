"""Pemeriksa Contoh Soal Bab 4. Setiap bilangan yang tercetak di
penyelesaian dihitung ulang di sini; assert gagal bila ada yang keliru.
Pecahan dihitung eksak dengan fractions.Fraction.
"""
import math
from collections import Counter
from fractions import Fraction as F

from bab04_data import PESAN_X, PESAN_Y

# ---- Contoh Soal 4.1: tes cepat dua kali ----
prev, sens, spes = F(5, 100), F(95, 100), F(95, 100)
pos_sakit = prev * sens                  # 0,0475
pos_sehat = (1 - prev) * (1 - spes)      # 0,0475
assert pos_sakit == pos_sehat == F(475, 10000)
p1 = pos_sakit / (pos_sakit + pos_sehat)
assert p1 == F(1, 2)
p2 = p1 * sens / (p1 * sens + (1 - p1) * (1 - spes))
assert p2 == F(19, 20)
# sekaligus dengan dua tes: prior 5% dan kemungkinan dikuadratkan
p2b = prev * sens**2 / (prev * sens**2 + (1 - prev) * (1 - spes)**2)
assert p2b == p2
# frekuensi alami 10.000 orang
assert 10000 * prev == 500 and 500 * sens == 475
assert 9500 * (1 - spes) == 475

# ---- Contoh Soal 4.2: tiga kelas, MAP lawan kemungkinan maksimum ----
prior = [F(1, 10), F(3, 10), F(6, 10)]
lik = [F(8, 10), F(5, 10), F(5, 100)]
skor = [a * b for a, b in zip(prior, lik)]
assert skor == [F(8, 100), F(15, 100), F(3, 100)]
tot = sum(skor)
assert tot == F(26, 100)
post = [s / tot for s in skor]
assert post == [F(4, 13), F(15, 26), F(3, 26)]
assert sum(post) == 1
assert round(float(post[0]), 3) == 0.308
assert round(float(post[1]), 3) == 0.577
assert round(float(post[2]), 3) == 0.115

# ---- Contoh Soal 4.3: galat Bayes pada tabel gabungan ----
gab = {"A": (F(30, 100), F(5, 100)), "B": (F(15, 100), F(15, 100)),
       "C": (F(5, 100), F(30, 100))}
assert sum(a + b for a, b in gab.values()) == 1
galat_bayes = sum(min(a, b) for a, b in gab.values())
assert galat_bayes == F(1, 4)
galat_selalu0 = sum(b for a, b in gab.values())
assert galat_selalu0 == F(1, 2)
# posterior kelas 1 di setiap nilai x
assert gab["A"][1] / sum(gab["A"]) == F(1, 7)
assert gab["B"][1] / sum(gab["B"]) == F(1, 2)
assert gab["C"][1] / sum(gab["C"]) == F(6, 7)
# galat bersyarat di A: 1/7; dikali P(A) = 0,35 -> 0,05
assert F(1, 7) * F(35, 100) == F(5, 100)

# ---- Contoh Soal 4.4: ambang dua kelas Gaussian ----
# mu0 = 0, mu1 = 2, sigma = 1, prior 0,8 / 0,2.
# logit(x) = ln(pi1/pi0) + 2x - 2
def logit(x):
    return math.log(0.2 / 0.8) + 2 * x - 2
x_setengah = 1 + math.log(2)
assert abs(logit(x_setengah)) < 1e-12
assert round(x_setengah, 3) == 1.693
x_09 = 1 + math.log(6)
assert abs(logit(x_09) - math.log(9)) < 1e-12
assert round(x_09, 3) == 2.792
post09 = 1 / (1 + math.exp(-logit(x_09)))
assert abs(post09 - 0.9) < 1e-12
# periksa langsung dengan kepadatan normal
def phi(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
a, b = 0.2 * phi(x_setengah - 2), 0.8 * phi(x_setengah)
assert abs(a - b) < 1e-15

# ---- Contoh Soal 4.5: kerugian tidak simetris ----
p = F(3, 4)          # P(penipuan | hadiah) dari Subbab 4.1
risiko_tandai = 9 * (1 - p)
risiko_loloskan = 1 * p
assert risiko_tandai == F(9, 4) and risiko_loloskan == F(3, 4)
assert F(9, 9 + 1) == F(9, 10)
# posterior berapa yang membuat kedua risiko sama
q = F(9, 10)
assert 9 * (1 - q) == 1 * q

# ---- Contoh Soal 4.6: tabel pola utuh dari sepuluh pesan ----
def pola(b):
    return "".join(map(str, b))
hit = {k: Counter(pola(b) for b in PESAN_X[PESAN_Y == k]) for k in (0, 1)}
assert (PESAN_Y == 1).sum() == 4 and (PESAN_Y == 0).sum() == 6
assert hit[1]["1100"] == 1 and hit[0]["1100"] == 0
assert hit[1]["1001"] == 0 and hit[0]["1001"] == 0
assert hit[1]["0001"] == 0 and hit[0]["0001"] == 2
terisi = set(hit[0]) | set(hit[1])
assert len(terisi) == 9
assert 2 * (2**4 - 1) == 30          # parameter tabel dua kelas
assert 2 * 4 == 8                    # parameter naive (Bab 5)
# posterior pola 1100: (2/5 * 1/4) / (2/5 * 1/4 + 3/5 * 0) = 1
s1, s0 = F(2, 5) * F(1, 4), F(3, 5) * F(0, 6)
assert s1 / (s1 + s0) == 1
# pola 0001: (2/5 * 0) / (0 + 3/5 * 2/6) = 0
s1, s0 = F(2, 5) * 0, F(3, 5) * F(2, 6)
assert s1 / (s1 + s0) == 0 and s0 == F(1, 5)

print("Contoh Soal Bab 4: semua bilangan cocok")
