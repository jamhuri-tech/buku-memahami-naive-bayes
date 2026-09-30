"""Bab 10: kata dengan bobot bukti terbesar pada naive Bayes
multinomial untuk SMS spam (seluruh data, alpha = 1). Hanya kata yang
muncul sekurang-kurangnya 20 kali, dan hanya kata ASCII, yang
ditampilkan (berkas aslinya memuat beberapa huruf Latin-1)."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from bab10_data import muat

teks, y = muat()
vek = CountVectorizer()
X = vek.fit_transform(teks)
nb = MultinomialNB().fit(X, y)
kata = vek.get_feature_names_out()
bobot = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
sering = (np.asarray(X.sum(axis=0)).ravel() >= 20) & np.array(
    [k.isascii() for k in kata])
idx = np.where(sering)[0]
urut = idx[np.argsort(bobot[idx])]
print("paling spam         bobot   paling ham          bobot")
for a, b in zip(urut[::-1][:10], urut[:10]):
    print(f"{kata[a]:15s}   {bobot[a]:+5.2f}   {kata[b]:15s}   "
          f"{bobot[b]:+5.2f}")
