"""Bab 5: kebebasan bersyarat tidak sama dengan kebebasan.

Bagian 1 (Contoh Soal 5.1): demam dan batuk bebas bersyarat penyakit,
tetapi berkorelasi bila penyakitnya tidak diketahui.
Bagian 2 (Contoh Soal 5.2): dua lemparan koin yang saling bebas,
dengan kelas y = 1 bila keduanya sama. Bebas, tetapi tidak bebas
bersyarat kelas.
Korelasi dihitung dari 200.000 pengamatan bangkitan.
"""
import numpy as np

from bab04_data import BENIH


def korelasi(a, b):
    return np.corrcoef(a, b)[0, 1]


rng = np.random.default_rng(BENIH)
n = 200_000

flu = rng.random(n) < 0.2
demam = rng.random(n) < np.where(flu, 0.8, 0.1)
batuk = rng.random(n) < np.where(flu, 0.5, 0.2)
print("demam dan batuk")
for nama, m in (("semua orang", slice(None)), ("di antara flu", flu),
                ("di antara sehat", ~flu)):
    print(f"  korelasi {nama:16s}: {korelasi(demam[m], batuk[m]):+.3f}")
print(f"  P(batuk) = {batuk.mean():.3f}, "
      f"P(batuk | demam) = {batuk[demam].mean():.3f}")

x1 = rng.random(n) < 0.5
x2 = rng.random(n) < 0.5
y = x1 == x2
print("dua koin, kelas = keduanya sama")
for nama, m in (("semua lemparan", slice(None)), ("di kelas 1", y),
                ("di kelas 0", ~y)):
    print(f"  korelasi {nama:16s}: {korelasi(x1[m], x2[m]):+.3f}")
