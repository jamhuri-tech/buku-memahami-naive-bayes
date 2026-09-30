"""Bab 18: ambang keputusan dengan biaya tidak simetris (Bab 4).

Menandai ham sebagai spam (positif palsu) berbiaya 9, meloloskan spam
berbiaya 1, sehingga ambang posterior 0,9. Posterior validasi silang
naive Bayes multinomial pada SMS spam, tanpa kalibrasi dan dengan Platt.
Dicetak banyaknya positif palsu, negatif palsu, dan biaya per 1.000
pesan untuk beberapa ambang.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from bab18_data import kasus, posterior

nama, X, y, m = kasus()[0]
q_mentah = posterior(m, X, y)
q_platt = posterior(CalibratedClassifierCV(m, method="sigmoid", cv=5),
                    X, y)
q_iso = posterior(CalibratedClassifierCV(m, method="isotonic", cv=5),
                  X, y)

if __name__ == "__main__":
    print("posterior   ambang     FP    FN   biaya/1000")
    for label, q in (("mentah", q_mentah), ("Platt", q_platt),
                     ("isotonik", q_iso)):
        for t in (0.5, 0.9, 0.99):
            fp = np.sum((q > t) & (y == 0))
            fn = np.sum((q <= t) & (y == 1))
            biaya = (9 * fp + fn) / len(y) * 1000
            print(f"{label:8s}    {t:5.2f}    {fp:4d}  {fn:4d}"
                  f"      {biaya:6.2f}")
    ts = np.linspace(0.01, 0.999, 999)
    for label, q in (("mentah", q_mentah), ("isotonik", q_iso)):
        b = [(9 * np.sum((q > t) & (y == 0))
              + np.sum((q <= t) & (y == 1))) for t in ts]
        print(f"ambang terbaik, posterior {label}: "
              f"{ts[np.argmin(b)]:.3f} (biaya/1000 "
              f"{min(b) / len(y) * 1000:.2f})")
