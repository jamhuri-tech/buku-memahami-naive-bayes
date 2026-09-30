"""Bab 4: posterior kelas dari prior dan kemungkinan.

Bagian 1: seribu pesan dan kata "hadiah" (frekuensi alami).
Bagian 2: tiga kelas; aturan MAP lawan aturan kemungkinan maksimum.
"""
import numpy as np


def posterior(prior, kemungkinan):
    skor = prior * kemungkinan       # pembilang teorema Bayes
    return skor / skor.sum()         # dibagi peluang marginal


# ---- Bagian 1: seribu pesan ----
n = 1000
prior = np.array([0.8, 0.2])            # biasa, penipuan
p_hadiah = np.array([0.05, 0.6])        # P(hadiah | kelas)
banyak = n * prior
memuat = banyak * p_hadiah
print(f"dari {n} pesan: {banyak[1]:.0f} penipuan, "
      f"{banyak[0]:.0f} biasa")
print(f"memuat 'hadiah': {memuat[1]:.0f} penipuan, "
      f"{memuat[0]:.0f} biasa")
print(f"P(penipuan | hadiah) = {memuat[1]:.0f}/{memuat.sum():.0f}"
      f" = {memuat[1] / memuat.sum():.4f}")
print("fungsi posterior:", np.round(posterior(prior, p_hadiah), 4))

# ---- Bagian 2: tiga kelas ----
kelas = ["penipuan", "promosi", "pribadi"]
prior3 = np.array([0.1, 0.3, 0.6])
p_gratis = np.array([0.8, 0.5, 0.05])   # P(gratis | kelas)
post = posterior(prior3, p_gratis)
print()
print("kelas      prior  P(gratis|k)  skor    posterior")
for k in range(3):
    print(f"{kelas[k]:9s}  {prior3[k]:4.2f}   {p_gratis[k]:5.2f}"
          f"        {prior3[k] * p_gratis[k]:5.3f}   {post[k]:.4f}")
print("aturan MAP               :", kelas[np.argmax(post)])
print("aturan kemungkinan maks. :", kelas[np.argmax(p_gratis)])
