"""Bab 11: kepala pesan, tanda tangan, dan kutipan membocorkan kelas.
Naive Bayes multinomial pada hitungan kata, data latih dan uji resmi."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from bab11_data import muat

if __name__ == "__main__":
    print("data                       multinomial  complement")
    for bersih, nama in ((False, "pesan utuh"),
                         (True, "tanpa kepala/tanda/kutipan")):
        tl, yl, tu, yu, _ = muat(bersih)
        vek = CountVectorizer()
        Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
        a = accuracy_score(yu, MultinomialNB(alpha=0.1).fit(Xl, yl)
                           .predict(Xu))
        b = accuracy_score(yu, ComplementNB(alpha=0.1).fit(Xl, yl)
                           .predict(Xu))
        print(f"{nama:26s}   {a:.4f}      {b:.4f}")
    print(f"latih {len(yl)}, uji {len(yu)}, kosakata {Xl.shape[1]}")
