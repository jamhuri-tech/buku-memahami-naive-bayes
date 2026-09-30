"""Bab 14: melatih naive Bayes multinomial sambil membaca berkas.

Berkas SMS dibaca 500 baris sekali jalan; setiap potongan diubah
menjadi vektor dengan HashingVectorizer (tidak perlu kosakata) lalu
diberikan kepada partial_fit. Hasilnya dibandingkan dengan fit pada
seluruh data sekaligus.
"""
from itertools import islice

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

from bab10_data import BERKAS, muat

hv = HashingVectorizer(n_features=2**14, alternate_sign=False,
                       norm=None)


def potongan(berkas, ukuran):
    with open(berkas, encoding="latin-1") as f:
        while True:
            baris = list(islice(f, ukuran))
            if not baris:
                return
            label, teks = zip(*(b.rstrip("\n").split("\t", 1)
                                for b in baris))
            yield teks, (np.array(label) == "spam").astype(int)


if __name__ == "__main__":
    arus = MultinomialNB(alpha=0.1)
    banyak = 0
    for teks, y in potongan(BERKAS, 500):
        arus.partial_fit(hv.transform(teks), y, classes=[0, 1])
        banyak += 1
    semua_teks, semua_y = muat()
    X = hv.transform(semua_teks)
    utuh = MultinomialNB(alpha=0.1).fit(X, semua_y)
    print(f"potongan: {banyak}, pesan: {int(arus.class_count_.sum())}")
    print("hitungan kelas sama :",
          np.array_equal(arus.class_count_, utuh.class_count_))
    print("hitungan fitur sama :",
          np.array_equal(arus.feature_count_, utuh.feature_count_))
    d = np.abs(arus.predict_log_proba(X) - utuh.predict_log_proba(X))
    print(f"maks |selisih log-posterior| = {d.max():.1e}")
