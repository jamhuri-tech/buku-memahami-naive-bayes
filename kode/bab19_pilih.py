"""Bab 19: memilih representasi, keluarga, dan alpha dengan validasi
silang lima lipatan pada data latih SmSA (ukuran: F1 makro). Data
validasi dan data uji tidak disentuh di sini.
"""
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from bab04_data import BENIH
from bab19_data import muat

teks, y = muat("train")
pipa = Pipeline([("vek", CountVectorizer()), ("nb", MultinomialNB())])
kisi = {
    "vek": [CountVectorizer(), CountVectorizer(ngram_range=(1, 2)),
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)],
    "nb": [MultinomialNB(), ComplementNB()],
    "nb__alpha": [0.03, 0.1, 0.3, 1.0],
}
NAMA_VEK = ["hitungan 1-gram", "hitungan 1-2-gram", "tf-idf 1-2-gram"]
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)


def cari():
    return GridSearchCV(pipa, kisi, cv=cv, scoring="f1_macro",
                        n_jobs=1).fit(teks, y)


if __name__ == "__main__":
    gs = cari()
    r = gs.cv_results_
    print("fitur              keluarga      alpha  F1 makro (CV)")
    urut = np.argsort(-r["mean_test_score"])
    for i in urut[:6]:
        p = r["params"][i]
        v = NAMA_VEK[[str(k) for k in kisi["vek"]].index(str(p["vek"]))]
        k = type(p["nb"]).__name__.replace("NB", "")
        print(f"{v:17s}  {k:12s}  {p['nb__alpha']:5g}   "
              f"{r['mean_test_score'][i]:.4f}")
    print(f"terburuk: {r['mean_test_score'].min():.4f}")
