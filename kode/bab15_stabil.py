"""Bab 15: kestabilan bobot. SMS spam diambil ulang (bootstrap) 50
kali; untuk setiap sampel ulang, bobot naive Bayes multinomial dan
regresi logistik dihitung ulang. Dicatat simpangan baku bobot setiap
kata (kata yang muncul >= 20 kali), dan urutan 20 kata paling spam.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from bab04_data import BENIH
from bab10_data import muat
from bab15_linear import linear_multinomial

teks, y = muat()
X = CountVectorizer().fit_transform(teks)
sering = np.where(np.asarray(X.sum(axis=0)).ravel() >= 20)[0]
rng = np.random.default_rng(BENIH)
W = {"naive Bayes": [], "regresi logistik": []}
for _ in range(50):
    i = rng.integers(0, len(y), len(y))
    W["naive Bayes"].append(
        linear_multinomial(MultinomialNB().fit(X[i], y[i]))[0][sering])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr = LogisticRegression(max_iter=2000).fit(X[i], y[i])
    W["regresi logistik"].append(lr.coef_[0][sering])

print("model              median sb   sb/|rata|   irisan 20 teratas")
for nama, w in W.items():
    w = np.array(w)
    sb = w.std(axis=0)
    rel = np.median(sb / np.maximum(np.abs(w.mean(axis=0)), 1e-12))
    atas = [set(np.argsort(-b)[:20]) for b in w]
    irisan = np.mean([len(atas[0] & a) for a in atas[1:]])
    print(f"{nama:17s}   {np.median(sb):8.3f}   {rel:9.3f}"
          f"        {irisan:5.1f}")
