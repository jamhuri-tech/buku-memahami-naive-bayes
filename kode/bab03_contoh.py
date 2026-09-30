"""Pemeriksa Contoh Soal Bab 3."""
import math

import numpy as np

from bab04_data import PESAN_X, PESAN_Y

Y = np.eye(2)[PESAN_Y]
H = (Y.T @ PESAN_X).astype(int)
assert H.tolist() == [[1, 2, 1, 4], [3, 3, 3, 0]]
assert H[:, 0].tolist() == [1, 3]
assert PESAN_X[:, 0].tolist() == [1, 1, 0, 1, 0, 0, 0, 1, 0, 0]
assert round(200 * math.log(0.001), 1) == -1381.6
assert round(200 * math.log10(0.001)) == -600
assert round(math.log(2) + math.log(3), 12) == round(math.log(6), 12)
print("Contoh Soal Bab 3: semua bilangan cocok")
