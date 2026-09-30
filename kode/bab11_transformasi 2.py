"""Bab 11: transformasi hitungan kata (Rennie dkk. 2003) untuk naive
Bayes multinomial dan complement, pada 20 Newsgroups yang sudah
dibersihkan. Akurasi data uji resmi, alpha = 0,1.

  hitungan : banyaknya kemunculan kata
  log-tf   : log(1 + hitungan)
  tf-idf   : log(1 + hitungan) x idf, tiap dokumen dinormalkan L2
"""
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from bab11_data import muat

tl, yl, tu, yu, _ = muat()
REPRESENTASI = [
    ("hitungan", CountVectorizer()),
    ("log-tf", TfidfVectorizer(use_idf=False, sublinear_tf=True,
                               norm=None)),
    ("tf-idf", TfidfVectorizer(sublinear_tf=True)),
]

if __name__ == "__main__":
    print("fitur       multinomial       complement")
    print("            akurasi  F1 makro  akurasi  F1 makro")
    for nama, vek in REPRESENTASI:
        Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
        baris = []
        for M in (MultinomialNB(alpha=0.1), ComplementNB(alpha=0.1)):
            p = M.fit(Xl, yl).predict(Xu)
            baris.append(f"{accuracy_score(yu, p):.4f}   "
                         f"{f1_score(yu, p, average='macro'):.4f}")
        print(f"{nama:10s}  " + "    ".join(baris))
