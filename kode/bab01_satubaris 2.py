"""Bab 1: naive Bayes dalam beberapa baris scikit-learn.

Naive Bayes multinomial pada SMS Spam Collection, dinilai dengan
validasi silang lima lipatan, lalu dipakai untuk dua pesan rekaan.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from bab04_data import BENIH
from bab10_data import muat

teks, y = muat()
model = make_pipeline(CountVectorizer(), MultinomialNB())
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
akurasi = cross_val_score(model, teks, y, cv=cv)
print(f"akurasi validasi silang: {akurasi.mean():.4f}")

model.fit(teks, y)
baru = ["You won a free prize call now",
        "Are we still meeting for lunch tomorrow"]
for t, p in zip(baru, model.predict_proba(baru)[:, 1]):
    print(f"P(spam) = {p:.4f}  <- {t}")
