"""SMS Spam Collection (Almeida, Gomez Hidalgo & Yamakami 2011).

5.574 pesan singkat berbahasa Inggris, 747 di antaranya spam. Berkas
aslinya berenkode Latin-1: satu pesan per baris, label dan teks
dipisah tab. Kelas 1 = spam, kelas 0 = ham (pesan biasa).
"""
from pathlib import Path

import numpy as np

BERKAS = Path(__file__).resolve().parent.parent / "data" / \
    "SMSSpamCollection"


def muat():
    label, teks = [], []
    with open(BERKAS, encoding="latin-1") as f:
        for baris in f:
            a, b = baris.rstrip("\n").split("\t", 1)
            label.append(a)
            teks.append(b)
    return teks, (np.array(label) == "spam").astype(int)
