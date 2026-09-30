"""SmSA (IndoNLU): sentimen ulasan dan komentar berbahasa Indonesia.

Tiga berkas resmi: latih (11.000), validasi (1.260), dan uji (500).
Setiap baris: teks yang sudah dipraproses (huruf kecil, tanda baca
dipisah spasi), tab, lalu label positive, neutral, atau negative.
Label diterjemahkan menjadi positif, netral, negatif.
"""
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent.parent / "data" / "smsa"
LABEL = {"positive": "positif", "neutral": "netral",
         "negative": "negatif"}


def muat(bagian):
    teks, label = [], []
    with open(FOLDER / f"{bagian}_preprocess.tsv", encoding="utf-8") as f:
        for baris in f:
            t, l = baris.rstrip("\n").rsplit("\t", 1)
            teks.append(t)
            label.append(LABEL[l])
    return teks, np.array(label)
