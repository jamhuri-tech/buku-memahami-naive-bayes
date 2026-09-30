"""Bab 15: bobot naive Bayes multinomial lawan bobot regresi logistik
pada SMS spam (seluruh data, hitungan kata). Hanya kata yang muncul
sekurang-kurangnya 20 kali dibandingkan.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from bab10_data import muat
from bab15_linear import linear_multinomial

teks, y = muat()
vek = CountVectorizer()
X = vek.fit_transform(teks)
kata = vek.get_feature_names_out()
w_nb, b_nb = linear_multinomial(MultinomialNB().fit(X, y))
lr = LogisticRegression(max_iter=2000).fit(X, y)
w_lr = lr.coef_[0]
sering = np.asarray(X.sum(axis=0)).ravel() >= 20

if __name__ == "__main__":
    print(f"kata yang dibandingkan: {sering.sum()}")
    r = np.corrcoef(w_nb[sering], w_lr[sering])[0, 1]
    rho = spearmanr(w_nb[sering], w_lr[sering])[0]
    print(f"korelasi Pearson  : {r:.3f}")
    print(f"korelasi Spearman : {rho:.3f}")
    idx = np.where(sering)[0]
    a = set(idx[np.argsort(-w_nb[idx])[:20]])
    b = set(idx[np.argsort(-w_lr[idx])[:20]])
    print(f"20 kata paling spam yang sama: {len(a & b)}")
    print("paling spam, NB      paling spam, LR")
    for i, j in zip(idx[np.argsort(-w_nb[idx])[:8]],
                    idx[np.argsort(-w_lr[idx])[:8]]):
        print(f"{kata[i]:10s} {w_nb[i]:+5.2f}     {kata[j]:10s} "
              f"{w_lr[j]:+5.2f}")
    print("kata          bobot NB  bobot LR")
    for j in idx[np.argsort(-np.abs(w_nb[idx] - w_lr[idx]))[:6]]:
        print(f"{kata[j]:12s}  {w_nb[j]:+7.2f}  {w_lr[j]:+7.2f}")
