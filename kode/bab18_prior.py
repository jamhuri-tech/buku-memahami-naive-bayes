"""Bab 18: prior yang bergeser antara data latih dan data pakai.

SMS spam dibagi dua berstrata. Model dilatih pada separuh pertama (13%
spam). Separuh kedua diambil ulang sehingga bagian spamnya 2% atau 50%,
seperti di tempat pemakaian yang berbeda. Posterior dikoreksi dengan
menggeser log-odds sebesar log odds prior baru dikurangi log odds prior
latih; untuk naive Bayes koreksi itu sama dengan memasang class_prior.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from bab04_data import BENIH
from bab10_data import muat
from bab18_ukuran import logloss

teks, y = muat()
tl, tu, yl, yu = train_test_split(teks, y, test_size=0.5, stratify=y,
                                  random_state=BENIH)
vek = CountVectorizer()
Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
nb = MultinomialNB().fit(Xl, yl)
pi_latih = yl.mean()
rng = np.random.default_rng(BENIH)


def ambil_ulang(pi):
    """Indeks data uji dengan bagian spam pi (2.000 pesan)."""
    n1 = int(round(2000 * pi))
    s = rng.choice(np.where(yu == 1)[0], n1, replace=n1 > yu.sum())
    h = rng.choice(np.where(yu == 0)[0], 2000 - n1, replace=False)
    return np.concatenate([s, h])


if __name__ == "__main__":
    print(f"prior spam data latih: {pi_latih:.4f}")
    L = nb.predict_log_proba(Xu)
    lo = L[:, 1] - L[:, 0]
    print("prior pakai  posterior      akurasi  log-loss")
    for pi in (0.02, 0.5):
        i = ambil_ulang(pi)
        geser = np.log(pi / (1 - pi)) - np.log(pi_latih / (1 - pi_latih))
        for nama, g in (("tanpa koreksi", 0.0), ("dikoreksi", geser)):
            q = 1 / (1 + np.exp(-(lo[i] + g)))
            print(f"   {pi:4.2f}     {nama:13s}   "
                  f"{np.mean((q > 0.5) == yu[i]):.4f}   "
                  f"{logloss(q, yu[i]):.4f}")
        nb2 = MultinomialNB(class_prior=[1 - pi, pi]).fit(Xl, yl)
        q2 = nb2.predict_proba(Xu[i])[:, 1]
        q1 = 1 / (1 + np.exp(-(lo[i] + geser)))
        print(f"            class_prior = koreksi: "
              f"{np.allclose(q1, q2, atol=1e-12)}")
