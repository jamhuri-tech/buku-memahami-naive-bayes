"""Data kanker payudara Wisconsin (diagnostik), dari scikit-learn.

569 sampel jaringan, 30 fitur kontinu yang diukur dari citra inti sel.
Kelas 1 = ganas (malignant), kelas 0 = jinak (benign); scikit-learn
memakai kode kebalikannya, jadi labelnya kita balik.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

DASAR = ["radius", "tekstur", "keliling", "luas", "kehalusan",
         "kekompakan", "kecekungan", "titik cekung", "simetri",
         "dimensi fraktal"]
FITUR = ([f"{d} (rata-rata)" for d in DASAR]
         + [f"{d} (galat baku)" for d in DASAR]
         + [f"{d} (terburuk)" for d in DASAR])


def muat():
    d = load_breast_cancer()
    return d.data, 1 - d.target
