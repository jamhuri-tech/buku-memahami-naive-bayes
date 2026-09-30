"""Pemeriksa Contoh Soal Bab 19: F1 makro dari matriks kebingungan."""
from fractions import Fraction as F

M = [[197, 2, 5], [35, 32, 21], [72, 19, 117]]    # neg, net, pos
f1 = []
for k in range(3):
    tp = M[k][k]
    presisi = F(tp, sum(M[i][k] for i in range(3)))
    recall = F(tp, sum(M[k]))
    f1.append(2 * presisi * recall / (presisi + recall))
assert [sum(M[i][k] for i in range(3)) for k in range(3)] == [304, 53, 143]
assert [sum(b) for b in M] == [204, 88, 208]
assert [round(float(v), 4) for v in f1] == [0.7756, 0.4539, 0.6667]
assert round(float(sum(f1) / 3), 4) == 0.6321
assert F(197 + 32 + 117, 500) == F(173, 250)
assert round(197 / 304, 3) == 0.648 and round(32 / 53, 3) == 0.604
assert round(117 / 143, 3) == 0.818
assert 2 * F(117, 143) * F(117, 208) / (F(117, 143) + F(117, 208)) \
    == F(2, 3)
print("Contoh Soal Bab 19: semua bilangan cocok")
