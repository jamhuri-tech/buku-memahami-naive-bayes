"""Bab 14: hashing trick pada SMS spam.

HashingVectorizer memetakan setiap kata ke salah satu dari 2^b ember
tanpa menyimpan kosakata. Akurasi validasi silang naive Bayes
multinomial untuk beberapa ukuran ember dan alpha, dibandingkan dengan
CountVectorizer (kosakata 8.749 kata).
"""
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from bab04_data import BENIH
from bab10_data import muat

teks, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
BIT = [8, 12, 14, 16, 20]


def akurasi(vek, alpha):
    pipa = make_pipeline(vek, MultinomialNB(alpha=alpha))
    return cross_val_score(pipa, teks, y, cv=cv).mean()


def ember(b):
    return HashingVectorizer(n_features=2**b, alternate_sign=False,
                             norm=None)


if __name__ == "__main__":
    print("alpha  " + "  ".join(f"2^{b:<4d}" for b in BIT)
          + " kosakata")
    for a in (1.0, 0.1, 0.01):
        baris = [akurasi(ember(b), a) for b in BIT]
        baris.append(akurasi(CountVectorizer(), a))
        print(f"{a:<5g}  " + "  ".join(f"{v:.4f}" for v in baris))
