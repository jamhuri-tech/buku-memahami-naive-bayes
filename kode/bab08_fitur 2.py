"""Bab 8: lebih banyak fitur tidak selalu lebih baik.

Akurasi validasi silang naive Bayes kategorik (alpha = 1) dengan satu
fitur saja, dengan semua fitur, dan dengan fitur yang dipilih secara
serakah (setiap langkah menambah fitur yang paling menaikkan akurasi).
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import CategoricalNB

from bab04_data import BENIH
from bab08_data import FITUR, muat_kode

X, y, _ = muat_kode()
mc = X.max(axis=0) + 1
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)


def skor(kolom, alpha=1.0):
    m = CategoricalNB(alpha=alpha, min_categories=mc[kolom])
    return cross_val_score(m, X[:, kolom], y, cv=cv).mean()


if __name__ == "__main__":
    tunggal = np.array([skor([j]) for j in range(X.shape[1])])
    urut = np.argsort(-tunggal)
    print("satu fitur saja (lima terbaik):")
    for j in urut[:5]:
        print(f"  {FITUR[j]:22s} {tunggal[j]:.4f}")
    print(f"semua 22 fitur           {skor(list(range(22))):.4f}")
    print("pemilihan serakah:")
    pilih, sisa, kini = [], list(range(X.shape[1])), 0.0
    while sisa:
        nilai = [skor(pilih + [j]) for j in sisa]
        if max(nilai) < kini + 1e-4:
            print("  berhenti: tidak ada fitur yang menaikkan akurasi")
            break
        terbaik = sisa[int(np.argmax(nilai))]
        pilih.append(terbaik)
        sisa.remove(terbaik)
        kini = max(nilai)
        print(f"  + {FITUR[terbaik]:20s} {kini:.4f}")
