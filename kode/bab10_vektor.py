"""Bab 10: dari pesan ke vektor hitungan kata (bag of words)."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from bab10_data import muat

teks, y = muat()
print(f"pesan: {len(y)}, spam {y.sum()} ({y.mean():.1%})")
vek = CountVectorizer()
X = vek.fit_transform(teks)
print(f"kosakata: {X.shape[1]} kata")
print(f"unsur tak nol: {X.nnz} dari {X.shape[0] * X.shape[1]} "
      f"({X.nnz / (X.shape[0] * X.shape[1]):.3%})")
panjang = np.asarray(X.sum(axis=1)).ravel()
print(f"token per pesan: rata-rata {panjang.mean():.1f}, "
      f"median {np.median(panjang):.0f}, maks {panjang.max()}")
i = 2
print("pesan ke-3:", teks[i][:44] + "...")
baris = X[i]
kata = vek.get_feature_names_out()
pasangan = sorted(zip(baris.indices, baris.data), key=lambda t: -t[1])
print("  hitungan:", ", ".join(f"{kata[j]} {c}" for j, c in pasangan[:7]))
