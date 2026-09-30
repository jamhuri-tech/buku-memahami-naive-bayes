"""Bab 15: naive Bayes sebagai model linear.

Dari naive Bayes multinomial dan Bernoulli yang dilatih pada SMS spam,
dibentuk vektor bobot w dan intersep b sehingga log-odds = b + w . x.
Keduanya diperiksa terhadap predict_log_proba, lalu dimasukkan ke
LogisticRegression sebagai coef_ dan intercept_.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from bab10_data import muat


def linear_multinomial(nb):
    w = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    b = nb.class_log_prior_[1] - nb.class_log_prior_[0]
    return w, b


def linear_bernoulli(nb):
    lt = nb.feature_log_prob_                    # log theta
    l1 = np.log1p(-np.exp(lt))                   # log(1 - theta)
    w = (lt[1] - l1[1]) - (lt[0] - l1[0])
    b = (nb.class_log_prior_[1] - nb.class_log_prior_[0]
         + (l1[1] - l1[0]).sum())
    return w, b


if __name__ == "__main__":
    teks, y = muat()
    X = CountVectorizer().fit_transform(teks)
    B = (X > 0).astype(float)
    for nama, nb, fungsi, XX in (
            ("multinomial", MultinomialNB(), linear_multinomial, X),
            ("Bernoulli", BernoulliNB(), linear_bernoulli, B)):
        nb.fit(XX, y)
        w, b = fungsi(nb)
        L = nb.predict_log_proba(XX)
        lo = XX @ w + b
        lr = LogisticRegression()
        lr.classes_, lr.coef_, lr.intercept_ = np.array([0, 1]), \
            w[None, :], np.array([b])
        d = np.abs(lo - (L[:, 1] - L[:, 0])).max()
        sama = np.array_equal(lr.predict(XX), nb.predict(XX))
        print(f"{nama:11s}: b = {b:+8.4f}, selisih maks {d:.1e}, "
              f"sama: {sama}")
