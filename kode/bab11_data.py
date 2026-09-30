"""20 Newsgroups (Lang 1995), lewat sklearn.datasets.fetch_20newsgroups.

18.846 pesan dari 20 kelompok diskusi Usenet, dibagi menurut tanggal
menjadi 11.314 pesan latih dan 7.532 pesan uji. Kepala pesan, tanda
tangan, dan kutipan dibuang (remove=...) kecuali diminta lain, karena
ketiganya membocorkan kelas (Subbab 11.1).
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups

BERSIH = ("headers", "footers", "quotes")


def muat(bersih=True):
    rm = BERSIH if bersih else ()
    latih = fetch_20newsgroups(subset="train", remove=rm)
    uji = fetch_20newsgroups(subset="test", remove=rm)
    return latih.data, latih.target, uji.data, uji.target, \
        latih.target_names


def timpang(teks, y, bagian=0.1, benih=20260928):
    """Kelas bernomor genap hanya disisakan sebagian kecil."""
    rng = np.random.default_rng(benih)
    kecil = y % 2 == 0
    simpan = np.where(kecil, rng.random(len(y)) < bagian, True)
    return [t for t, s in zip(teks, simpan) if s], y[simpan]
