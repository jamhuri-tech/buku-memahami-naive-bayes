"""Bab 18: posterior validasi silang yang dipakai beberapa program.

SMS spam dengan naive Bayes multinomial dan regresi logistik (hitungan
kata), kanker payudara dengan naive Bayes Gaussian dan regresi
logistik (dibakukan). Posterior diperoleh dengan cross_val_predict lima
lipatan, benih buku.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bab04_data import BENIH
from bab09_data import muat as muat_kanker
from bab10_data import muat as muat_sms

CV = StratifiedKFold(5, shuffle=True, random_state=BENIH)


def kasus():
    teks, ys = muat_sms()
    Xk, yk = muat_kanker()
    return [
        ("SMS, naive Bayes", teks, ys,
         make_pipeline(CountVectorizer(), MultinomialNB())),
        ("SMS, regresi logistik", teks, ys,
         make_pipeline(CountVectorizer(),
                       LogisticRegression(max_iter=2000))),
        ("kanker, naive Bayes", Xk, yk, GaussianNB()),
        ("kanker, regresi log.", Xk, yk,
         make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000))),
    ]


def posterior(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_predict(model, X, y, cv=CV,
                                 method="predict_proba")[:, 1]
