"""Pemeriksa Contoh Soal Bab 7 (pecahan eksak)."""
import math
from fractions import Fraction as F

from scipy import integrate
from scipy.stats import beta

# ---- Contoh Soal 7.1: Laplace pada sepuluh pesan ----
def halus(N1, N, a=1):
    return F(N1 + a, N + 2 * a)
t1 = [halus(3, 4), halus(3, 4), halus(3, 4), halus(0, 4)]
t0 = [halus(1, 6), halus(2, 6), halus(1, 6), halus(4, 6)]
assert t1 == [F(2, 3), F(2, 3), F(2, 3), F(1, 6)]
assert t0 == [F(1, 4), F(3, 8), F(1, 4), F(5, 8)]


def skor(x, prior, t):
    s = prior
    for xj, tj in zip(x, t):
        s *= tj if xj else 1 - tj
    return s


s1 = skor([1, 0, 0, 1], F(2, 5), t1)
s0 = skor([1, 0, 0, 1], F(3, 5), t0)
assert F(2, 3) * F(1, 3) * F(1, 3) * F(1, 6) == F(2, 162) == F(1, 81)
assert s1 == F(2, 405)
assert F(1, 4) * F(5, 8) * F(3, 4) * F(5, 8) == F(75, 1024)
assert s0 == F(45, 1024)
post = s1 / (s1 + s0)
assert post == F(2048, 20273)
assert round(float(post), 3) == 0.101
# pola 1100
s1 = skor([1, 1, 0, 0], F(2, 5), t1)
s0 = skor([1, 1, 0, 0], F(3, 5), t0)
assert s1 == F(4, 81) and s0 == F(81, 5120)
assert round(float(s1 / (s1 + s0)), 3) == 0.757

# ---- Contoh Soal 7.2: aturan suksesi ----
assert F(4, 4) == 1
assert halus(4, 4) == F(5, 6)
assert 1 - halus(4, 4) == F(1, 6)
assert halus(0, 4) == F(1, 6)
# matahari terbit N hari berturut-turut: (N + 1)/(N + 2)
assert halus(10, 10) == F(11, 12)

# ---- Contoh Soal 7.3: prior Beta ----
# Beta(1,1) + 0 dari 4 -> Beta(1, 5)
a, b = 1 + 0, 1 + 4
assert (a, b) == (1, 5)
assert F(a, a + b) == F(1, 6)                     # rata-rata
m, _ = integrate.quad(lambda t: t * beta.pdf(t, 1, 5), 0, 1)
assert abs(m - 1 / 6) < 1e-12
# modus Beta(1,5) di 0; modus Beta(a,b) = (a-1)/(a+b-2)
assert F(a - 1, a + b - 2) == 0
# prior Beta(2,2) -> Beta(2, 6)
a, b = 2 + 0, 2 + 4
assert F(a - 1, a + b - 2) == F(1, 6)             # modus = Laplace
assert F(a, a + b) == F(1, 4)                     # rata-rata

# ---- Contoh Soal 7.4: multinomial, kosakata besar ----
tok, V = 1000, 9000
assert F(0 + 1, tok + V) == F(1, 10000)
assert F(50 + 1, tok + V) == F(51, 10000)
assert F(50, tok) == F(1, 20)
assert float(F(51, 10000)) == 0.0051
# alpha = 0,01
x = (50 + 0.01) / (tok + 0.01 * V)
assert round(x, 4) == 0.0459 and tok + 0.01 * V == 1090
assert round(0.01 / 1090 * 1e6, 2) == 9.17
# bagian massa peluang yang diberikan kepada hitungan semu
assert F(V, tok + V) == F(9, 10)

# ---- Contoh Soal 7.5: alpha besar, kelas tak seimbang ----
tp = F(10, 30 + 20)
tb = F(10, 70 + 20)
assert tp == F(1, 5) and tb == F(1, 9)
r = (1 - tp) / (1 - tb)
assert r == F(9, 10)
assert round(math.log(0.9), 4) == -0.1054
assert round(500 * math.log(0.9), 1) == -52.7
assert math.exp(500 * math.log(0.9)) < 1e-22
# dengan alpha = 1: 1/32 dan 1/72
tp1, tb1 = F(1, 32), F(1, 72)
r1 = (1 - tp1) / (1 - tb1)
assert r1 == F(31 * 72, 32 * 71) == F(279, 284)
assert round(500 * math.log(r1), 2) == -8.88

print("Contoh Soal Bab 7: semua bilangan cocok")
