"""Bab 10: naive Bayes multinomial lawan Bernoulli pada SMS spam.

Kosakata dibatasi pada m kata yang paling sering (max_features),
dengan CountVectorizer di dalam pipeline sehingga kosakata hanya
dibentuk dari lipatan latih. Validasi silang lima lipatan, alpha = 1.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline

from bab04_data import BENIH
from bab10_data import muat

teks, y = muat()
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
UKURAN = [50, 200, 1000, 3000, None]


def nilai(model, m):
    pipa = make_pipeline(CountVectorizer(max_features=m), model)
    h = cross_validate(pipa, teks, y, cv=cv,
                       scoring=("accuracy", "neg_log_loss"))
    return h["test_accuracy"].mean(), -h["test_neg_log_loss"].mean()


if __name__ == "__main__":
    print("kosakata    multinomial          Bernoulli")
    print("            akurasi  log-loss    akurasi  log-loss")
    for m in UKURAN:
        a1, l1 = nilai(MultinomialNB(), m)
        a2, l2 = nilai(BernoulliNB(), m)
        teks_m = "semua" if m is None else str(m)
        print(f"{teks_m:>8s}    {a1:.4f}   {l1:.4f}      "
              f"{a2:.4f}   {l2:.4f}")
