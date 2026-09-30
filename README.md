# Kode — Memahami Naive Bayes

Repositori pendamping buku **_Memahami Naive Bayes: Mengklasifikasi
dengan Peluang, Fitur demi Fitur_** (Edisi Pertama, 2026) oleh
Mohammad Jamhuri, seri *Memahami*.

Berisi seluruh kode Python yang dipakai buku, per bab, beserta data dan
pembangkit gambarnya. **Setiap angka keluaran yang tercetak di buku
dihasilkan oleh kode di sini**, dan `periksa.py` membuktikannya: skrip
itu menjalankan ulang kode setiap bab dan mencocokkan hasilnya dengan
blok keluaran yang tercetak di buku. Setiap hitungan tangan di kotak
*Contoh Soal* juga diperiksa, oleh `kode/babNN_contoh.py`.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)).

Data di `data/` berasal dari pihak lain dan **tidak** termasuk lisensi
0BSD: setiap data tetap mengikuti lisensi aslinya (CC BY 4.0 untuk data
UCI, Apache 2.0 untuk SmSA dari IndoNLU). Asal, sidik SHA-256, dan
lisensi setiap data dicatat di [`data/SUMBER.md`](data/SUMBER.md).
Data 20 Newsgroups tidak disimpan di sini; scikit-learn mengunduhnya
otomatis ketika kode Bab 11 pertama kali dijalankan.

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-naive-bayes.git
cd buku-memahami-naive-bayes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab04_posterior.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
selalu memakai benih tetap (20260928), sehingga keluarannya sama setiap
kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 04     # Bab 4 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Pemeriksaan yang sama berjalan otomatis di GitHub
Actions setiap kali kode berubah.

## Struktur

```
kode/
  bab04_*.py         kode Bab 4, dan seterusnya per bab
  babNN_contoh.py    pemeriksa hitungan tangan Contoh Soal Bab NN
data/SUMBER.md       asal, tanggal, sidik, dan lisensi setiap data
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

Nama berkas kode di buku sama dengan nama di sini: listing yang
merujuk `kode/bab04_posterior.py` berasal dari berkas itu.

`python gen_gambar.py bab04` membangkitkan gambar Bab 4 saja.

## Versi

Tag `edisi-1` menandai kode yang tepat dipakai untuk mencetak Edisi
Pertama. Cabang `main` dapat memuat perbaikan sesudahnya; setiap
perbaikan yang mengubah angka di buku dicatat di daftar errata.

`requirements.txt` mematok versi yang dipakai buku (Python 3.13.5,
NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, Matplotlib 3.10.0).
Versi lain biasanya berjalan, tetapi digit terakhir sebagian angka
dapat berbeda.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
