"""Bab 12: hasil kali peluang kata menjadi nol di float64.

Untuk setiap dokumen uji, log-gabungan log(pi_k) + sum n_j log(theta_kj)
dihitung oleh scikit-learn. Bila hasil kali itu dihitung langsung,
nilainya exp(log-gabungan); kita hitung bagian dokumen yang hasil
kalinya nol di SEMUA kelas, sehingga posteriornya 0/0.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from bab10_data import muat as muat_sms
from bab11_data import muat as muat_20ng

print(f"float64: terkecil normal {np.finfo(float).tiny:.3e}, "
      f"terkecil {np.nextafter(0, 1):.3e}")
teks, y = muat_sms()
tl, yl, tu, yu, _ = muat_20ng()
print("data          token: median  maks  log-gab. min  nol semua")
for nama, (a, b, c) in (("SMS spam", (teks, y, teks)),
                        ("20 Newsgroups", (tl, yl, tu))):
    vek = CountVectorizer()
    Xl, Xu = vek.fit_transform(a), vek.transform(c)
    J = MultinomialNB(alpha=0.1).fit(Xl, b).predict_joint_log_proba(Xu)
    token = np.asarray(Xu.sum(axis=1)).ravel()
    nol = np.mean(np.exp(J).max(axis=1) == 0)
    print(f"{nama:13s}         {np.median(token):4.0f} {token.max():5d}"
          f"   {J.max(axis=1).min():11.1f}      {nol:.3f}")
