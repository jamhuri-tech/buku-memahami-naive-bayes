"""Bab 13: fitur campuran pada SMS spam. Kata (multinomial) digabung
dengan banyaknya angka di pesan, sebagai fitur Gaussian (log(1 + d))
atau kategorik (0, 1-2, 3-9, 10 atau lebih angka). Validasi silang
lima lipatan; kosakata dibentuk dari lipatan latih.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from bab04_data import BENIH
from bab10_data import muat
from bab13_nb import Campuran, Gaussian, Kategorik, Multinomial

teks, y = muat()
angka = np.array([sum(c.isdigit() for c in t) for t in teks])
kelompok = np.digitize(angka, [1, 3, 10])       # 0, 1, 2, 3
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)


def model(nama):
    if nama == "kata saja":
        return Multinomial(1.0)
    if nama == "kata + angka Gaussian":
        return Campuran([Multinomial(1.0), Gaussian()])
    return Campuran([Multinomial(1.0), Kategorik(1.0, [4])])


def fitur(nama, W, idx):
    if nama == "kata saja":
        return W
    if nama == "kata + angka Gaussian":
        return (W, np.log1p(angka[idx])[:, None])
    return (W, kelompok[idx][:, None])


if __name__ == "__main__":
    print(f"rata-rata angka per pesan: ham {angka[y == 0].mean():.2f}, "
          f"spam {angka[y == 1].mean():.2f}")
    print("model                    akurasi  log-loss")
    for nama in ("kata saja", "kata + angka Gaussian",
                 "kata + angka kategorik"):
        P = np.zeros(len(y))
        for latih, uji in cv.split(teks, y):
            vek = CountVectorizer()
            Wl = vek.fit_transform([teks[i] for i in latih])
            Wu = vek.transform([teks[i] for i in uji])
            m = model(nama).fit(fitur(nama, Wl, latih), y[latih])
            P[uji] = m.predict_proba(fitur(nama, Wu, uji))[:, 1]
        q = np.clip(P, 1e-15, 1 - 1e-15)
        print(f"{nama:23s}   {np.mean((P > 0.5) == y):.4f}   "
              f"{log_loss(y, q):.4f}")
