"""Pemeriksa Contoh Soal Bab 20: banyaknya parameter peluang fitur."""
K, p = 2, 10
nb = K * p
tan = K + 2 * K * (p - 1)
aode = p * ((2 * K - 1) + 2 * K * (p - 1))
assert (nb, tan, aode) == (20, 38, 390)
assert tan == K * (2 * p - 1)
# penuh (tabel pola utuh) untuk pembanding
assert K * (2 ** p - 1) == 2046
print("Contoh Soal Bab 20: semua bilangan cocok")
