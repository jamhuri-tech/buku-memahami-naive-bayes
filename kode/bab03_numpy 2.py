"""Bab 3: NumPy dan SciPy secukupnya untuk naive Bayes."""
import numpy as np
from scipy import sparse

from bab04_data import PESAN_X, PESAN_Y

# hitungan kelas dan one-hot
print("bincount(y)       :", np.bincount(PESAN_Y))
Y = np.eye(2)[PESAN_Y]
print("one-hot 3 baris   :", Y[:3].tolist())
print("Y.T @ X (hitungan):", (Y.T @ PESAN_X).astype(int).tolist())

# broadcasting: membagi setiap baris dengan jumlahnya sendiri
N = Y.T @ PESAN_X + 1
theta = N / N.sum(axis=1, keepdims=True)
print("bentuk N, jumlah  :", N.shape, N.sum(axis=1, keepdims=True).shape)
print("jumlah baris theta:", theta.sum(axis=1))

# matriks jarang
S = sparse.csr_matrix(PESAN_X)
print(f"matriks jarang    : {S.nnz} unsur tak nol dari {S.shape[0]} x "
      f"{S.shape[1]}")
print("jumlah log theta  :", np.round(S @ np.log(theta).T, 3)[:2].tolist())
