"""Bab 18: Platt scaling dan isotonic regression untuk naive Bayes.

Kalibrasi dilakukan di dalam setiap lipatan luar oleh
CalibratedClassifierCV (lima lipatan dalam), sehingga data uji tidak
pernah dipakai untuk kalibrasi. Juga dicetak parameter Platt a dan b
(P = sigmoid(a * log-odds + b)) yang ditaksir dari log-odds validasi
silang naive Bayes.
"""
import numpy as np
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from bab18_data import kasus, posterior
from bab18_ukuran import brier, ece, logloss

if __name__ == "__main__":
    print("data    kalibrasi   akurasi   ECE     Brier   log-loss")
    for nama, X, y, m in kasus():
        if "naive" not in nama:
            continue
        data = nama.split(",")[0]
        q0 = posterior(m, X, y)
        lo = logit(np.clip(q0, 1e-15, 1 - 1e-15))
        pl = LogisticRegression(penalty=None).fit(lo[:, None], y)
        for metode in ("tanpa", "sigmoid", "isotonic"):
            if metode == "tanpa":
                q = q0
            else:
                q = posterior(CalibratedClassifierCV(m, method=metode,
                                                     cv=5), X, y)
            teks = {"tanpa": "tanpa", "sigmoid": "Platt",
                    "isotonic": "isotonik"}[metode]
            print(f"{data:6s}  {teks:9s}   {np.mean((q > 0.5) == y):.4f}"
                  f"   {ece(q, y):.4f}  {brier(q, y):.4f}  "
                  f"{logloss(q, y):.4f}")
        print(f"        Platt: a = {pl.coef_[0, 0]:.3f}, "
              f"b = {pl.intercept_[0]:+.3f}")
