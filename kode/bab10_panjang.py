"""Bab 10: pesan yang lebih panjang mendapat posterior yang lebih
ekstrem pada naive Bayes multinomial, karena setiap token menambah
satu suku bobot bukti. Log-odds validasi silang dikelompokkan menurut
banyaknya token pesan.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from bab04_data import BENIH
from bab10_data import muat

teks, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
pipa = make_pipeline(CountVectorizer(), MultinomialNB())
# selisih log-posterior = selisih log-gabungan (pembaginya sama)
L = cross_val_predict(pipa, teks, y, cv=cv, method="predict_log_proba")
log_odds = L[:, 1] - L[:, 0]
token = np.asarray(CountVectorizer().fit_transform(teks).sum(axis=1)).ravel()

if __name__ == "__main__":
    print("token     pesan  rata |log-odds|  ekstrem  akurasi")
    batas = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 200)]
    for a, b in batas:
        m = (token >= a) & (token <= b)
        lo = log_odds[m]
        ek = np.mean(np.abs(lo) > np.log(999))
        ak = np.mean((lo > 0) == y[m])
        print(f"{a:3d}-{b:<3d}  {m.sum():6d}      {np.abs(lo).mean():6.1f}"
              f"      {ek:.3f}    {ak:.4f}")
