"""Bab 19: model pilihan (hitungan 1-2-gram, naive Bayes multinomial,
alpha = 0,1) dilatih pada seluruh data latih, lalu dinilai pada data
validasi dan data uji. Pembanding: tebakan kelas terbanyak dan regresi
logistik pada tf-idf 1-2-gram (C dipilih dengan validasi silang).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from bab19_data import muat

tl, yl = muat("train")
tv, yv = muat("valid")
tu, yu = muat("test")
KELAS = ["negatif", "netral", "positif"]


def model_nb():
    return make_pipeline(CountVectorizer(ngram_range=(1, 2)),
                         MultinomialNB(alpha=0.1)).fit(tl, yl)


if __name__ == "__main__":
    nb = model_nb()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegressionCV(Cs=[1, 10, 100], cv=5, max_iter=3000,
                                 scoring="f1_macro")).fit(tl, yl)
    print("model              data      akurasi  F1 makro  log-loss")
    for nama, X, Y in (("validasi", tv, yv), ("uji", tu, yu)):
        mayor = np.full(len(Y), "positif")
        print(f"kelas terbanyak    {nama:8s}  "
              f"{accuracy_score(Y, mayor):.4f}   "
              f"{f1_score(Y, mayor, average='macro'):.4f}      -")
        for label, m in (("naive Bayes", nb), ("regresi logistik", lr)):
            p = m.predict(X)
            print(f"{label:17s}  {nama:8s}  {accuracy_score(Y, p):.4f}"
                  f"   {f1_score(Y, p, average='macro'):.4f}    "
                  f"{log_loss(Y, m.predict_proba(X), labels=KELAS):.4f}")
    print(f"C regresi logistik: {lr[-1].C_[0]:g}")
    print("matriks kebingungan naive Bayes, data uji")
    print("(baris = sebenarnya, kolom = tebakan; neg, net, pos)")
    M = confusion_matrix(yu, nb.predict(tu), labels=KELAS)
    for k, baris in zip(KELAS, M):
        print(f"  {k:8s} " + " ".join(f"{v:5d}" for v in baris)
              + f"   recall {baris[KELAS.index(k)] / baris.sum():.3f}")
