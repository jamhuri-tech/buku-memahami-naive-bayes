"""Data jamur (UCI Mushroom): 8.124 jamur, 22 fitur kategorik.

Kelas 1 = beracun (p), kelas 0 = dapat dimakan (e). Nilai '?' pada
fitur akar_tangkai dibiarkan sebagai kategori tersendiri kecuali
diminta lain. Nama fitur diterjemahkan; kode huruf aslinya tetap.
"""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

BERKAS = Path(__file__).resolve().parent.parent / "data" / \
    "agaricus-lepiota.data"

FITUR = ["bentuk_tudung", "permukaan_tudung", "warna_tudung", "memar",
         "bau", "pelekatan_insang", "jarak_insang", "ukuran_insang",
         "warna_insang", "bentuk_tangkai", "akar_tangkai",
         "permukaan_atas_cincin", "permukaan_bawah_cincin",
         "warna_atas_cincin", "warna_bawah_cincin", "jenis_selubung",
         "warna_selubung", "jumlah_cincin", "jenis_cincin",
         "warna_spora", "populasi", "habitat"]

BAU = {"a": "almon", "l": "adas", "c": "kreosot", "y": "amis",
       "f": "busuk", "m": "apak", "n": "tanpa bau", "p": "menyengat",
       "s": "pedas"}


def muat():
    """Mengembalikan huruf mentah (n x 22) dan label (n,)."""
    d = np.loadtxt(BERKAS, dtype=str, delimiter=",")
    return d[:, 1:], (d[:, 0] == "p").astype(int)


def muat_kode():
    """Fitur dikodekan 0, 1, 2, ... per kolom (urutan abjad huruf)."""
    huruf, y = muat()
    enc = OrdinalEncoder(dtype=int)
    X = enc.fit_transform(huruf)
    return X, y, enc
