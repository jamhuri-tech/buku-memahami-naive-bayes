"""Bab 9: fitur yang tidak bersebaran normal.

Empat cara memodelkan p(x_j | k) pada data kanker payudara, dibandingkan
dengan validasi silang lima lipatan:
  normal pada data mentah (GaussianNB),
  normal pada log(1 + x),
  diskretisasi 10 selang berkuantil sama + CategoricalNB,
  taksiran kepadatan kernel (KDE) per kelas per fitur.
"""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer

from bab04_data import BENIH
from bab09_data import muat


class KDENaiveBayes(ClassifierMixin, BaseEstimator):
    """Naive Bayes dengan kepadatan kernel Gaussian per fitur."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.log_prior_ = np.log([np.mean(y == k) for k in self.classes_])
        self.kde_ = [[gaussian_kde(X[y == k, j]) for j in range(X.shape[1])]
                     for k in self.classes_]
        return self

    def predict_proba(self, X):
        L = np.array([lp + sum(np.log(np.maximum(f(X[:, j]), 1e-300))
                               for j, f in enumerate(kde))
                      for lp, kde in zip(self.log_prior_, self.kde_)]).T
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


X, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
calon = [
    ("normal, data mentah", GaussianNB()),
    ("normal, log(1 + x)", make_pipeline(
        FunctionTransformer(np.log1p), GaussianNB())),
    ("10 selang + kategorik", make_pipeline(
        KBinsDiscretizer(10, encode="ordinal", strategy="quantile"),
        FunctionTransformer(lambda Z: Z.astype(int)),
        CategoricalNB(alpha=1, min_categories=10))),
    ("KDE per fitur", KDENaiveBayes()),
]
print("p(x_j | k)                akurasi  log-loss")
for nama, m in calon:
    P = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    q = np.clip(P, 1e-15, 1 - 1e-15)
    ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    print(f"{nama:24s}  {np.mean((q > 0.5) == y):.4f}   {ll:.4f}")
