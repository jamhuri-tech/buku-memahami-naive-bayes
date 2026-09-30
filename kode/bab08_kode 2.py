"""Bab 8: tiga cara memberi fitur kategorik kepada naive Bayes.

(1) CategoricalNB pada kode 0, 1, 2, ...;
(2) BernoulliNB pada one-hot (setiap kategori jadi fitur biner);
(3) GaussianNB pada kode 0, 1, 2, ... (keliru: kode bukan ukuran).
Akurasi dan log-loss validasi silang lima lipatan.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder

from bab04_data import BENIH
from bab08_data import muat, muat_kode

X, y, _ = muat_kode()
oh = OneHotEncoder(sparse_output=False).fit_transform(muat()[0])
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
mc = X.max(axis=0) + 1
calon = [("CategoricalNB, kode", CategoricalNB(alpha=1,
                                               min_categories=mc), X),
         ("BernoulliNB, one-hot", BernoulliNB(alpha=1), oh),
         ("GaussianNB, kode", GaussianNB(), X)]
print(f"one-hot: {oh.shape[1]} kolom biner dari 22 fitur")
print("cara                   akurasi  log-loss")
for nama, m, XX in calon:
    h = cross_validate(m, XX, y, cv=cv,
                       scoring=("accuracy", "neg_log_loss"))
    print(f"{nama:21s}   {h['test_accuracy'].mean():.4f}"
          f"   {-h['test_neg_log_loss'].mean():.4f}")
