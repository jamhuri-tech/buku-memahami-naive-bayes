"""Bab 11: kelas tak seimbang. Kelas bernomor genap hanya disisakan
10% dokumen latihnya; data uji tetap seimbang. Recall rata-rata kelas
kecil dan kelas besar, akurasi, dan F1 makro. Hitungan kata, alpha 0,1.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from bab11_data import muat, timpang

tl, yl, tu, yu, _ = muat()
tl, yl = timpang(tl, yl)
kecil = np.arange(20) % 2 == 0
vek = CountVectorizer()
Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
if __name__ == "__main__":
    print(f"dokumen latih: kelas kecil rata-rata "
          f"{np.bincount(yl)[kecil].mean():.0f}, kelas besar "
          f"{np.bincount(yl)[~kecil].mean():.0f}")
    calon = [("multinomial", MultinomialNB(alpha=0.1)),
             ("multinomial, prior sama", MultinomialNB(alpha=0.1,
                                                       fit_prior=False)),
             ("complement", ComplementNB(alpha=0.1)),
             ("complement, norm=True", ComplementNB(alpha=0.1, norm=True))]
    print("model                   recall kecil besar  akurasi  F1 makro")
    for nama, M in calon:
        p = M.fit(Xl, yl).predict(Xu)
        r = recall_score(yu, p, average=None)
        print(f"{nama:24s}    {r[kecil].mean():.3f}   {r[~kecil].mean():.3f}"
              f"   {accuracy_score(yu, p):.4f}   "
              f"{f1_score(yu, p, average='macro'):.4f}")
