"""Data berjalan buku: pesan singkat rekaan dan dua kelas Gaussian.

Pesan rekaan dipakai di Contoh Soal Bab 4-7, 10, dan 12. Setiap pesan
sudah diringkas menjadi empat fitur biner: apakah pesan itu memuat kata
"hadiah", "transfer", "segera", dan "rapat". Kelas 1 = penipuan,
kelas 0 = biasa. Datanya ditulis tangan, bukan dibangkitkan, supaya
setiap hitungan dapat dikerjakan dengan pensil.

Dua kelas Gaussian satu dimensi dipakai untuk pengklasifikasi Bayes
optimal (Bab 4) karena posterior dan galat Bayes-nya diketahui tepat.
"""
import numpy as np

BENIH = 20260928

KATA = ["hadiah", "transfer", "segera", "rapat"]
KELAS = ["biasa", "penipuan"]

# Sepuluh pesan latih. Baris = pesan, kolom = KATA.
PESAN_X = np.array([
    [1, 1, 1, 0],   # 1  penipuan
    [1, 0, 1, 0],   # 2  penipuan
    [0, 1, 1, 0],   # 3  penipuan
    [1, 1, 0, 0],   # 4  penipuan
    [0, 1, 0, 1],   # 5  biasa
    [0, 0, 1, 1],   # 6  biasa
    [0, 0, 0, 1],   # 7  biasa
    [1, 0, 0, 0],   # 8  biasa
    [0, 1, 0, 0],   # 9  biasa
    [0, 0, 0, 1],   # 10 biasa
])
PESAN_Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# Contoh isi pesan (untuk Bab 1 dan Bab 10); fitur di atas diambil
# dari kata-kata ini.
PESAN_TEKS = [
    "selamat nomor ini dapat hadiah segera transfer biaya admin",
    "hadiah undian menunggu hubungi segera",
    "segera transfer ke rekening ini sebelum jam lima",
    "klaim hadiah dengan transfer pulsa",
    "uang kas rapat sudah transfer ya",
    "segera ke ruang rapat dosen sudah datang",
    "rapat ditunda sampai besok",
    "selamat atas hadiah lomba kemarin",
    "sudah transfer untuk tiket kereta",
    "jangan lupa rapat jam tiga",
]


def dua_gaussian(n, prior1=0.5, mu=(-1.0, 1.0), sd=1.0, rng=None):
    """n pengamatan satu fitur dari dua kelas normal bervarians sama.

    Kelas 1 dipilih dengan peluang prior1, lalu x ~ N(mu[y], sd^2).
    """
    rng = np.random.default_rng(BENIH) if rng is None else rng
    y = (rng.random(n) < prior1).astype(int)
    x = rng.normal(np.asarray(mu)[y], sd)
    return x, y
