"""Bab 19: pemeriksaan data SmSA sebelum pemodelan."""
from collections import Counter

import numpy as np

from bab19_data import muat

bagian = {b: muat(b) for b in ("train", "valid", "test")}
nama = {"train": "latih", "valid": "validasi", "test": "uji"}
print("bagian     pesan  positif  netral  negatif  token median")
for b, (t, y) in bagian.items():
    c = Counter(y)
    n = len(y)
    tok = np.median([len(s.split()) for s in t])
    print(f"{nama[b]:9s} {n:6d}   {c['positif'] / n:.3f}   "
          f"{c['netral'] / n:.3f}   {c['negatif'] / n:.3f}      {tok:4.0f}")
tl = bagian["train"][0]
print(f"teks kembar di data latih: {len(tl) - len(set(tl))}")
for b in ("valid", "test"):
    sama = len(set(tl) & set(bagian[b][0]))
    print(f"teks {nama[b]} yang juga ada di data latih: {sama}")
