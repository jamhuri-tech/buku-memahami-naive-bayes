"""Bab 20: gagasan NB-SVM (Wang & Manning 2012) dengan regresi logistik.

Fitur kehadiran kata (1-gram dan 2-gram) dikali log-rasio naive Bayes
r_j = log[(p_j/|p|_1) / (q_j/|q|_1)], dengan p dan q hitungan kehadiran
di kelas positif dan negatif ditambah alpha = 1, lalu regresi logistik
dilatih pada fitur yang sudah dikali. Untuk tiga kelas SmSA dipakai
satu-lawan-sisa, satu vektor r per kelas. Pembanding: naive Bayes
multinomial dan regresi logistik pada fitur kehadiran yang sama.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from bab04_data import BENIH
from bab10_data import muat as muat_sms
from bab19_data import muat as muat_smsa


def rasio(B, y, alpha=1.0):
    p = np.asarray(B[y == 1].sum(axis=0)).ravel() + alpha
    q = np.asarray(B[y == 0].sum(axis=0)).ravel() + alpha
    return np.log((p / p.sum()) / (q / q.sum()))


def nbsvm(Bl, yl, Bu, C=1.0):
    kelas = np.unique(yl)
    skor = []
    for k in kelas:
        yk = (yl == k).astype(int)
        r = rasio(Bl, yk)
        lr = LogisticRegression(C=C, max_iter=3000)
        lr.fit(Bl.multiply(r).tocsr(), yk)
        skor.append(lr.decision_function(Bu.multiply(r).tocsr()))
    if len(kelas) == 2:
        return kelas[(skor[1] > 0).astype(int)]
    return kelas[np.argmax(np.column_stack(skor), axis=1)]


def data():
    teks, y = muat_sms()
    a, b, c, d = train_test_split(teks, y, test_size=0.3, stratify=y,
                                  random_state=BENIH)
    yield "SMS spam", a, c, b, d
    tl, yl = muat_smsa("train")
    tu, yu = muat_smsa("test")
    yield "SmSA (uji)", tl, yl, tu, yu


if __name__ == "__main__":
    print("data        model              akurasi  F1 makro")
    for nama, tl, yl, tu, yu in data():
        vek = CountVectorizer(ngram_range=(1, 2), binary=True)
        Bl, Bu = vek.fit_transform(tl), vek.transform(tu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            hasil = [
                ("naive Bayes", MultinomialNB(alpha=0.1).fit(Bl, yl)
                 .predict(Bu)),
                ("regresi logistik", LogisticRegression(C=1.0,
                 max_iter=3000).fit(Bl, yl).predict(Bu)),
                ("NB + regresi log.", nbsvm(Bl, yl, Bu)),
            ]
        for model, p in hasil:
            print(f"{nama:10s}  {model:17s}  {accuracy_score(yu, p):.4f}"
                  f"   {f1_score(yu, p, average='macro'):.4f}")
