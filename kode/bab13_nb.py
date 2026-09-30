"""Bab 13: naive Bayes dari nol dengan NumPy.

Satu kelas dasar menangani prior, logsumexp, dan prediksi; setiap
keluarga hanya menaksir parameternya dan menghitung jumlah log-peluang
fiturnya (_log_fitur). Konvensi mengikuti scikit-learn 1.6 supaya
hasilnya dapat dicocokkan sampai galat pembulatan.
"""
import numpy as np
from scipy.special import logsumexp


class NaiveBayes:
    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        self.Y_ = np.eye(K)[yi]              # one-hot, n x K
        Nk = self.Y_.sum(axis=0)
        self.class_log_prior_ = np.log(Nk / Nk.sum())
        self._taksir(X)
        return self

    def joint_log(self, X):
        return self.class_log_prior_ + self._log_fitur(X)

    def predict_log_proba(self, X):
        J = self.joint_log(X)
        return J - logsumexp(J, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        k = np.argmax(self.joint_log(X), axis=1)
        return self.classes_[k]


class Multinomial(NaiveBayes):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _taksir(self, X):
        N = np.asarray(self.Y_.T @ X) + self.alpha   # K x p
        N = N / N.sum(axis=1, keepdims=True)
        self.log_theta_ = np.log(N)

    def _log_fitur(self, X):
        return np.asarray(X @ self.log_theta_.T)


class Bernoulli(NaiveBayes):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _taksir(self, X):
        B = (X > 0).astype(float)
        Nk = self.Y_.sum(axis=0)[:, None]
        t = (np.asarray(self.Y_.T @ B) + self.alpha) / (
            Nk + 2 * self.alpha)
        self.log_theta_ = np.log(t)
        self.log_1m_theta_ = np.log1p(-t)

    def _log_fitur(self, X):
        B = (X > 0).astype(float)
        w = self.log_theta_ - self.log_1m_theta_  # kehadiran
        c = self.log_1m_theta_.sum(axis=1)       # semua absen
        return np.asarray(B @ w.T) + c


class Kategorik(NaiveBayes):
    def __init__(self, alpha=1.0, banyak_kategori=None):
        self.alpha = alpha
        self.banyak_kategori = banyak_kategori

    def _taksir(self, X):
        V = self.banyak_kategori
        if V is None:
            V = X.max(axis=0) + 1
        self.log_theta_ = []             # tabel per fitur
        for j in range(X.shape[1]):
            N = self.Y_.T @ np.eye(V[j])[X[:, j]] + self.alpha
            N = N / N.sum(axis=1, keepdims=True)
            self.log_theta_.append(np.log(N))

    def _log_fitur(self, X):
        return sum(t[:, X[:, j]].T
                   for j, t in enumerate(self.log_theta_))


class Gaussian(NaiveBayes):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def _taksir(self, X):
        Nk = self.Y_.sum(axis=0)[:, None]
        self.theta_ = (self.Y_.T @ X) / Nk
        v = [np.var(X[self.Y_[:, k] == 1], axis=0)
             for k in range(len(Nk))]
        eps = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ = np.array(v) + eps

    def _log_fitur(self, X):
        v, m = self.var_, self.theta_
        a = -0.5 * np.log(2 * np.pi * v).sum(axis=1)
        d = X[:, None, :] - m[None]      # n x K x p
        b = (d ** 2 / v[None]).sum(axis=2)
        return a - 0.5 * b


class Campuran(NaiveBayes):
    """Fitur dari beberapa keluarga. X adalah tuple matriks, satu
    untuk setiap bagian; prior hanya dihitung sekali."""

    def __init__(self, bagian):
        self.bagian = bagian         # daftar model keluarga

    def _taksir(self, X):
        for m, Xm in zip(self.bagian, X):
            m.Y_ = self.Y_
            m._taksir(Xm)

    def _log_fitur(self, X):
        return sum(m._log_fitur(Xm)
                   for m, Xm in zip(self.bagian, X))
