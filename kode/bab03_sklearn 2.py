"""Bab 3: antarmuka scikit-learn pada sepuluh pesan rekaan."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB

from bab04_data import BENIH, PESAN_X, PESAN_Y

model = BernoulliNB(alpha=1.0)
model.fit(PESAN_X, PESAN_Y)
print("classes_          :", model.classes_)
print("class_count_      :", model.class_count_)
x = np.array([[1, 1, 0, 0]])
print("predict           :", model.predict(x))
print("predict_proba     :", np.round(model.predict_proba(x), 4))
cv = StratifiedKFold(2, shuffle=True, random_state=BENIH)
for i, (latih, uji) in enumerate(cv.split(PESAN_X, PESAN_Y)):
    print(f"lipatan {i}: uji = pesan {sorted((uji + 1).tolist())}, "
          f"penipuan {PESAN_Y[uji].sum()} dari {len(uji)}")
akurasi = cross_val_score(model, PESAN_X, PESAN_Y, cv=cv)
print("akurasi per lipatan:", akurasi)
