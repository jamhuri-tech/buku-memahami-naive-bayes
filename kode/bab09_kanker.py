"""Bab 9: naive Bayes Gaussian pada 30 fitur kanker payudara.

Validasi silang lima lipatan: akurasi, log-loss, dan bagian posterior
yang ekstrem (di bawah 0,001 atau di atas 0,999), dibandingkan dengan
regresi logistik yang fiturnya dibakukan.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from bab04_data import BENIH
from bab09_data import muat

X, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)


def nilai(model):
    P = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    q = np.clip(P[:, 1], 1e-15, 1 - 1e-15)
    ak = np.mean((q > 0.5) == y)
    ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    ek = np.mean((q < 1e-3) | (q > 1 - 1e-3))
    return ak, ll, ek


if __name__ == "__main__":
    log = FunctionTransformer(np.log1p)
    calon = [
        ("GaussianNB, data mentah", GaussianNB()),
        ("GaussianNB, dibakukan", make_pipeline(StandardScaler(),
                                                GaussianNB())),
        ("GaussianNB, log(1 + x)", make_pipeline(log, GaussianNB())),
        ("regresi logistik", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000))),
    ]
    print(f"{len(y)} sampel, {X.shape[1]} fitur, ganas {y.sum()}")
    print("model                      akurasi  log-loss  ekstrem")
    for nama, m in calon:
        ak, ll, ek = nilai(m)
        print(f"{nama:25s}  {ak:.4f}   {ll:.4f}    {ek:.3f}")
