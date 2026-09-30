# Sumber data

Setiap data yang dipakai buku dicatat di sini ketika pertama kali
diambil: nama, alamat, tanggal pengambilan, sidik SHA-256, lisensi.
Data rekaan (pesan penipuan, tabel kategorik kecil, kelas Gaussian
bangkitan) ditulis atau dibangkitkan oleh kode bab dengan benih tetap
20260928 dan tidak disimpan di sini.

| Data | Asal | Diambil | SHA-256 | Lisensi | Bab |
|---|---|---|---|---|---|
| Jamur (UCI Mushroom), `agaricus-lepiota.data`, 8.124 x 23 | https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data; laman data https://archive.ics.uci.edu/dataset/73/mushroom (DOI 10.24432/C5959T). Asal: Schlimmer (1987) dari *The Audubon Society Field Guide to North American Mushrooms* (1981) | 28 Sep 2026 | e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e | CC BY 4.0 (UCI) | 8 |
| SMS Spam Collection v.1, `SMSSpamCollection` (5.574 pesan, 747 spam), dari `smsspamcollection.zip` (SHA-256 zip 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3) | https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip; laman data https://archive.ics.uci.edu/dataset/228/sms+spam+collection (DOI 10.24432/C5CC84). Rujukan: Almeida, Gómez Hidalgo & Yamakami (2011), ACM DocEng | 28 Sep 2026 | 7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d | CC BY 4.0 (UCI); readme: hak cipta Almeida & Gómez Hidalgo, bebas dipakai dengan menyebut sumber | 10, 12, 14, 15, 18 |
| SmSA (IndoNLU): `smsa/train_preprocess.tsv` (11.000), `valid_preprocess.tsv` (1.260), `test_preprocess.tsv` (500); ulasan dan komentar berbahasa Indonesia, label positive/neutral/negative | https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/ ; kartu data https://huggingface.co/datasets/indonlp/indonlu. Rujukan: Purwarianti & Crisdayanti (2019); Wilie dkk. (2020), IndoNLU, AACL | 29 Sep 2026 | latih 50f38ceed9b31521bf1581e126620532cc9b790712938159a2cdcf6906977a9b; valid 6ab41ddc9d58a35086f05ebd2e209c74cb03d87d4f51d6abdfba674eafbefa74; uji 4e8016daa8e4e1b193f2a28a79d3a7804dd4e3f926a3ab7e8a64264f5f416a6f | Apache 2.0, salinan lisensi di `smsa/LICENSE` (hak cipta 2020 HKUST, ITB, UI, UMN, Gojek, Prosa.ai). Repositori IndoNLU memakai MIT sejak 20 Sep 2020 dan berganti ke Apache 2.0 pada 14 Jun 2022 (commit `43e6133`, `b8644f0`); kartu Hugging Face dan lencana README IndoNLU masih menyebut MIT, belum diperbarui. Berkas yang kita ambil (29 Sep 2026) tunduk pada lisensi yang berlaku sekarang, Apache 2.0. Berkas disalin tanpa perubahan; IndoNLU tidak mempunyai berkas NOTICE | 19 |
| 20 Newsgroups, versi *bydate* (18.846 pesan: latih 11.314, uji 7.532) | Diunduh otomatis oleh `sklearn.datasets.fetch_20newsgroups` dari https://ndownloader.figshare.com/files/5975967 (`20news-bydate.tar.gz`) ke `~/scikit_learn_data`; **tidak disimpan di repositori**. Laman data https://archive.ics.uci.edu/dataset/113/twenty+newsgroups (DOI 10.24432/C5C323, pencipta T. Mitchell); asal: Lang (1995); versi *bydate* dari laman J. Rennie http://qwone.com/~jason/20Newsgroups/ | 28 Sep 2026 | arsip 8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610 (diperiksa scikit-learn saat mengunduh) | CC BY 4.0 (UCI) | 11, 12, 14, 15, 18 |

## Catatan lisensi

- Kode buku berlisensi 0BSD. Data di atas **bukan** bagian dari lisensi
  itu: setiap data tetap mengikuti lisensi aslinya di kolom *Lisensi*.
- CC BY 4.0 (jamur, SMS Spam, 20 Newsgroups) mewajibkan penyebutan
  sumber; tabel ini dan daftar pustaka buku memenuhinya.
- Apache 2.0 (SmSA) mewajibkan salinan lisensi dan pemberitahuan hak
  cipta ikut disebarkan bersama data; keduanya ada di `smsa/LICENSE`.
