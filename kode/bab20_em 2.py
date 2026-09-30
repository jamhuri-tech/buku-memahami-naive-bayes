"""Bab 20: naive Bayes semi-terawasi dengan EM (Nigam dkk. 2000).

Dari data latih SMS spam hanya m pesan yang berlabel; sisanya dipakai
tanpa label. Langkah E: posterior setiap pesan tak berlabel. Langkah M:
naive Bayes multinomial dilatih ulang pada pesan berlabel (bobot 1) dan
pesan tak berlabel yang disalin untuk kedua kelas dengan bobot
lambda x posterior. Sepuluh iterasi; 20 ulangan pengambilan pesan
berlabel.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from bab04_data import BENIH
from bab10_data import muat

teks, y = muat()
tl, tu, yl, yu = train_test_split(teks, y, test_size=0.3, stratify=y,
                                  random_state=BENIH)
vek = CountVectorizer()
Xl, Xu = vek.fit_transform(tl), vek.transform(tu)


def em(Xlab, ylab, Xtak, lam, iterasi=10):
    nb = MultinomialNB(alpha=0.1).fit(Xlab, ylab)
    for _ in range(iterasi):
        P = nb.predict_proba(Xtak)
        X = sp.vstack([Xlab, Xtak, Xtak])
        yy = np.concatenate([ylab, np.zeros(Xtak.shape[0]),
                             np.ones(Xtak.shape[0])])
        w = np.concatenate([np.ones(len(ylab)), lam * P[:, 0],
                            lam * P[:, 1]])
        nb = MultinomialNB(alpha=0.1).fit(X, yy, sample_weight=w)
    return nb


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print("berlabel   hanya berlabel   EM, lambda=1   EM, lambda=0,1")
    for m in (20, 50, 200):
        a, b, c = [], [], []
        for _ in range(20):
            i = np.concatenate([
                rng.choice(np.where(yl == 1)[0], max(2, m // 7), False),
                rng.choice(np.where(yl == 0)[0], m - max(2, m // 7),
                           False)])
            j = np.setdiff1d(np.arange(len(yl)), i)
            a.append(MultinomialNB(alpha=0.1).fit(Xl[i], yl[i])
                     .score(Xu, yu))
            b.append(em(Xl[i], yl[i], Xl[j], 1.0).score(Xu, yu))
            c.append(em(Xl[i], yl[i], Xl[j], 0.1).score(Xu, yu))
        print(f"{m:6d}       {np.mean(a):.4f}          {np.mean(b):.4f}"
              f"          {np.mean(c):.4f}")
