"""Bab 14: berapa kali data latih dibaca.

Naive Bayes membaca setiap dokumen sekali. Regresi logistik (lbfgs)
mengulang hitungan gradien pada seluruh data setiap iterasi; n_iter_
mencatat banyaknya iterasi sampai konvergen. Data: SMS spam dan
20 Newsgroups (dibersihkan), fitur hitungan kata.
"""
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from bab10_data import muat as muat_sms
from bab11_data import muat as muat_20ng

teks, y = muat_sms()
tl, yl, tu, yu, _ = muat_20ng()
print("data            model              lintasan  akurasi uji")
for nama, a, b, c, d in (("SMS spam", teks[:4000], y[:4000],
                          teks[4000:], y[4000:]),
                         ("20 Newsgroups", tl, yl, tu, yu)):
    vek = CountVectorizer()
    Xl, Xu = vek.fit_transform(a), vek.transform(c)
    nb = MultinomialNB(alpha=0.1).fit(Xl, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000).fit(Xl, b)
    print(f"{nama:14s}  naive Bayes               1   "
          f"{accuracy_score(d, nb.predict(Xu)):.4f}")
    print(f"{'':14s}  regresi logistik  {lr.n_iter_.max():8d}   "
          f"{accuracy_score(d, lr.predict(Xu)):.4f}")
