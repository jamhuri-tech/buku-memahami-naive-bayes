"""Bab 12: posterior dari log-gabungan, dengan dan tanpa logsumexp.

Tiga cara menghitung posterior untuk dokumen 20 Newsgroups:
  (1) exp lalu dibagi jumlahnya (Listing 4.1), gagal bila semua nol;
  (2) logsumexp buatan sendiri (geser dengan maksimum);
  (3) predict_proba scikit-learn.
"""
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from bab11_data import muat


def posterior_langsung(J):
    e = np.exp(J)
    return e / e.sum(axis=1, keepdims=True)


def log_z(J):
    m = J.max(axis=1, keepdims=True)          # geser: maksimum jadi 0
    return m + np.log(np.exp(J - m).sum(axis=1, keepdims=True))


def posterior_log(J):
    return np.exp(J - log_z(J))


if __name__ == "__main__":
    tl, yl, tu, yu, _ = muat()
    vek = CountVectorizer()
    Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
    nb = MultinomialNB(alpha=0.1).fit(Xl, yl)
    J = nb.predict_joint_log_proba(Xu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P1 = posterior_langsung(J)
    P2 = posterior_log(J)
    P3 = nb.predict_proba(Xu)
    print(f"langsung : {np.isnan(P1).any(axis=1).mean():.3f} dokumen "
          f"berposterior nan")
    print(f"logsumexp: maks |selisih dengan predict_proba| = "
          f"{np.abs(P2 - P3).max():.1e}")
    print(f"scipy    : maks |log Z sendiri - logsumexp| = "
          f"{np.abs(log_z(J).ravel() - logsumexp(J, axis=1)).max():.1e}")
    tebak = np.argmax(J, axis=1)
    print(f"argmax log-gabungan = predict: "
          f"{np.array_equal(tebak, nb.predict(Xu))}")
