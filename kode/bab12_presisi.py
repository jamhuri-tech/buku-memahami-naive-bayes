"""Bab 12: peluang yang dilaporkan jenuh di 0 dan 1.

Naive Bayes multinomial pada SMS spam (seluruh data). Untuk pesan
dengan log-odds besar, predict_proba mengembalikan tepat 1.0, sehingga
1 - P kehilangan seluruh informasi. Peluang kecil itu tetap dapat
dibaca dari predict_log_proba atau dari expit(-log-odds).
"""
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from bab10_data import muat

teks, y = muat()
X = CountVectorizer().fit_transform(teks)
nb = MultinomialNB().fit(X, y)
L = nb.predict_log_proba(X)
P = nb.predict_proba(X)
lo = L[:, 1] - L[:, 0]
print(f"pesan dengan P(spam) tepat 1.0: {np.sum(P[:, 1] == 1.0)}, "
      f"tepat 0.0: {np.sum(P[:, 1] == 0.0)}")
print("log-odds    P(spam)   1 - P(spam)   P(ham) dari log")
for t in (5, 20, 36, 38, 60):
    i = np.argmin(np.abs(lo - t))
    print(f"{lo[i]:7.2f}   {P[i, 1]:.6f}   {1 - P[i, 1]:.3e}"
          f"     {np.exp(L[i, 0]):.3e}")
print(f"expit(-60) = {expit(-60):.3e}, exp(-60) = {np.exp(-60):.3e}")
