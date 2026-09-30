"""Pemeriksa Contoh Soal Bab 1."""
from fractions import Fraction as F

from bab04_data import KATA, PESAN_X, PESAN_Y

j = KATA.index("transfer")
ada = PESAN_X[:, j] == 1
assert ada.sum() == 5
assert F(int((ada & (PESAN_Y == 1)).sum()), int((PESAN_Y == 1).sum())) \
    == F(3, 4)
assert F(int((ada & (PESAN_Y == 1)).sum()), int(ada.sum())) == F(3, 5)
print("Contoh Soal Bab 1: semua bilangan cocok")
