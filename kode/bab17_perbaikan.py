"""Bab 17: mengurangi pelanggaran asumsi naif pada data kanker.

Validasi silang lima lipatan (akurasi, log-loss) untuk naive Bayes
Gaussian pada: 30 fitur mentah; komponen utama dari fitur yang
dibakukan (PCA, semua 30 atau 10 pertama); dan fitur yang dipilih
serakah di dalam setiap lipatan (SequentialFeatureSelector, 5 fitur).
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bab04_data import BENIH
from bab09_data import muat

X, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
calon = [
    ("30 fitur mentah", GaussianNB()),
    ("PCA, 30 komponen", make_pipeline(StandardScaler(), PCA(30),
                                       GaussianNB())),
    ("PCA, 10 komponen", make_pipeline(StandardScaler(), PCA(10),
                                       GaussianNB())),
    ("5 fitur pilihan", make_pipeline(
        StandardScaler(),
        SequentialFeatureSelector(GaussianNB(), n_features_to_select=5,
                                  cv=3), GaussianNB())),
]
print("naive Bayes Gaussian     akurasi  log-loss")
for nama, m in calon:
    q = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    q = np.clip(q, 1e-15, 1 - 1e-15)
    ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    print(f"{nama:22s}   {np.mean((q > 0.5) == y):.4f}   {ll:.4f}")
