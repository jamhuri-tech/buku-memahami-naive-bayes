# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20260928  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().


# ============================ Bab 4 ==================================

def bab04_frekuensi():
    """Diagram luas: lebar = prior, tinggi = P(hadiah | kelas)."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    lebar = {"biasa": 0.8, "penipuan": 0.2}
    tinggi = {"biasa": 0.05, "penipuan": 0.6}
    kiri = 0.0
    for nama, warna, muda in (("biasa", BIRU, BIRU_MUDA),
                              ("penipuan", JINGGA, JINGGA_MUDA)):
        w, h = lebar[nama], tinggi[nama]
        ax.add_patch(Rectangle((kiri, 0), w, 1, facecolor=muda,
                               edgecolor="white", lw=1.5))
        ax.add_patch(Rectangle((kiri, 0), w, h, facecolor=warna,
                               edgecolor="white", lw=1.5))
        n = int(round(1000 * w))
        m = int(round(1000 * w * h))
        ax.text(kiri + w / 2, 0.93, f"{nama}\n{n} pesan", ha="center",
                va="top", color=warna, fontsize=7.5)
        ax.text(kiri + w / 2, h + 0.03, f"{m} memuat\n“hadiah”",
                ha="center", va="bottom", color=warna, fontsize=7)
        kiri += w
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.8, 1.0])
    ax.set_xticklabels(["0", "0,8", "1"])
    ax.set_yticks([0, 0.05, 0.6, 1.0])
    ax.set_yticklabels(["0", "0,05", "0,6", "1"])
    kunci_label(ax, "x", "y")
    ax.set_xlabel("prior kelas (lebar)")
    ax.set_ylabel("P(hadiah | kelas) (tinggi)")
    ax.set_title("P(penipuan | hadiah) = 120 / (120 + 40) = 0,75")
    _rapikan(ax)
    simpan(fig, "bab04-frekuensi")


def bab04_optimal():
    from scipy.stats import norm
    x = np.linspace(-4, 4, 801)
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3.3), sharex=True,
                            gridspec_kw={"height_ratios": [1.3, 1]})
    for j, prior1 in enumerate((0.5, 0.2)):
        f0 = (1 - prior1) * norm.pdf(x, -1, 1)
        f1 = prior1 * norm.pdf(x, 1, 1)
        t = 0.5 * np.log((1 - prior1) / prior1)
        ax = axs[0, j]
        ax.plot(x, f0, color=BIRU, lw=1, label="$\\pi_0\\,p(x \\mid 0)$")
        ax.plot(x, f1, color=JINGGA, lw=1,
                label="$\\pi_1\\,p(x \\mid 1)$")
        ax.fill_between(x, 0, f1, where=x < t, color=JINGGA, alpha=0.35,
                        lw=0)
        ax.fill_between(x, 0, f0, where=x > t, color=BIRU, alpha=0.35,
                        lw=0)
        ax.axvline(t, color=HIJAU, lw=0.8, ls="--")
        ax.set_title(f"prior kelas 1 = {angka(prior1, 1)}")
        ax.set_ylim(0, 0.34)
        _rapikan(ax)
        ax = axs[1, j]
        ax.plot(x, f1 / (f0 + f1), color=HIJAU, lw=1.1)
        ax.axhline(0.5, color=ABU_GARIS, lw=0.6)
        ax.axvline(t, color=HIJAU, lw=0.8, ls="--")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("$x$")
        _rapikan(ax)
        galat = (1 - prior1) * norm.sf(t, -1, 1) + prior1 * norm.cdf(t, 1, 1)
        axs[0, j].text(0.98, 0.95 if j == 0 else 0.62,
                       f"$t^* = {angka_mat(t, 3)}$\n"
                       f"galat Bayes $= {angka_mat(galat, 4)}$",
                       transform=axs[0, j].transAxes, ha="right",
                       va="top", fontsize=6.5)
    axs[0, 0].legend(loc="upper left", fontsize=6.5)
    axs[1, 0].set_ylabel("$P(y = 1 \\mid x)$")
    fig.tight_layout()
    simpan(fig, "bab04-optimal")


def bab04_kosong():
    from bab04_tabel import bagian_tak_terlihat
    rng = np.random.default_rng(BENIH)
    ps = np.arange(2, 31, 2)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for n, warna in ((1000, JINGGA), (10_000, BIRU), (100_000, HIJAU)):
        b = [np.mean([bagian_tak_terlihat(p, n, rng, n_uji=1000)
                      for _ in range(5)]) for p in ps]
        ax.plot(ps, b, "o-", ms=2.5, lw=1, color=warna,
                label=f"n = {n:,}".replace(",", "."))
    ax.set_xlabel("banyaknya fitur biner $p$")
    ax.set_ylabel("bagian pola uji\nyang belum terlihat")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab04-kosong")


def bab04_kerugian():
    from scipy.stats import norm
    t = np.linspace(-2, 4, 601)
    fp = 0.8 * norm.sf(t, -1, 1)
    fn = 0.2 * norm.cdf(t, 1, 1)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for a, b, warna, nama in ((1, 1, BIRU, "biaya FP = FN = 1 (galat)"),
                              (9, 1, JINGGA, "biaya FP = 9, FN = 1")):
        k = a * fp + b * fn
        i = np.argmin(k)
        ax.plot(t, k, color=warna, lw=1.1, label=nama)
        ax.plot(t[i], k[i], "o", color=warna, ms=3.5)
        ax.annotate(f"$x = {angka_mat(t[i], 2)}$", (t[i], k[i]),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=6.5, color=warna)
    ax.set_xlabel("ambang $t$ (penipuan jika $x > t$)")
    ax.set_ylabel("kerugian harapan")
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right")
    _rapikan(ax)
    simpan(fig, "bab04-kerugian")


# ============================ Bab 5 ==================================

def _simpul(ax, x, y, teks, warna, r=0.13):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=warna,
                        lw=1.1, zorder=3))
    ax.text(x, y, teks, ha="center", va="center", color=warna,
            fontsize=7.5, zorder=4)


def _panah(ax, a, b, warna, r=0.13, **kw):
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    ax.annotate("", xy=b - d * r, xytext=a + d * r,
                arrowprops=dict(arrowstyle="-|>", color=warna, lw=0.8,
                                shrinkA=0, shrinkB=0, **kw))


def bab05_bintang():
    fig, axs = plt.subplots(1, 2, figsize=(4.7, 2.0))
    xs = [(-0.75, 0), (-0.25, 0), (0.25, 0), (0.75, 0)]
    nama = ["$x_1$", "$x_2$", "$x_3$", "$x_4$"]
    for ax, judul, penuh in ((axs[0], "tanpa asumsi", True),
                             (axs[1], "asumsi naif", False)):
        pusat = (0, 0.8)
        for (x, y), t in zip(xs, nama):
            _panah(ax, pusat, (x, y), BIRU)
        if penuh:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = xs[i], xs[j]
                    lengkung = 0.25 + 0.12 * (j - i)
                    ax.annotate("", xy=(b[0] - 0.1, b[1] - 0.08),
                                xytext=(a[0] + 0.1, a[1] - 0.08),
                                arrowprops=dict(
                                    arrowstyle="-|>", color=JINGGA,
                                    lw=0.6,
                                    connectionstyle=f"arc3,rad={lengkung}"))
        _simpul(ax, *pusat, "$y$", BIRU)
        for (x, y), t in zip(xs, nama):
            _simpul(ax, x, y, t, HIJAU if not penuh else ABU)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.85, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(judul)
    axs[0].text(0, -0.82, "$2^4 - 1 = 15$ peluang per kelas", ha="center",
                fontsize=6.8)
    axs[1].text(0, -0.82, "$4$ peluang per kelas", ha="center",
                fontsize=6.8)
    fig.tight_layout()
    simpan(fig, "bab05-bintang")


def bab05_bukti():
    from bab04_data import KATA, PESAN_X, PESAN_Y
    from bab05_naif import latih_naif
    from bab05_bukti import sumbangan
    prior, theta = latih_naif(PESAN_X, PESAN_Y)
    x = np.array([1, 1, 0, 0])
    awal, s = sumbangan(x, prior, theta)
    label = ["prior"] + [f"{k}\n({'ada' if v else 'tidak'})"
                         for k, v in zip(KATA, x)] + ["posterior"]
    fig, ax = plt.subplots(figsize=(4.5, 2.3))
    kini = 0.0
    nilai = [awal] + list(s)
    for i, v in enumerate(nilai):
        warna = JINGGA if v > 0 else BIRU
        ax.bar(i, v, bottom=kini, color=warna, width=0.6)
        ax.text(i, kini + v + (0.06 if v > 0 else -0.06),
                f"${angka_mat(v, 2)}$".replace("$-", "$-"),
                ha="center", va="bottom" if v > 0 else "top",
                fontsize=6.5)
        if i < len(nilai) - 1:
            ax.plot([i + 0.3, i + 0.7], [kini + v] * 2, color=ABU,
                    lw=0.6)
        kini += v
    ax.bar(len(nilai), kini, color=HIJAU, width=0.6)
    ax.text(len(nilai), kini + 0.06, f"${angka_mat(kini, 2)}$",
            ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color=ABU, lw=0.6)
    ax.set_xticks(range(len(label)))
    ax.set_xticklabels(label, fontsize=6.3)
    kunci_label(ax, "x")
    ax.set_ylabel("log-odds penipuan")
    ax.set_ylim(-1.0, 2.6)
    _rapikan(ax)
    simpan(fig, "bab05-bukti")


def bab05_ganda():
    from bab05_ganda import bangkit, log_odds, salin, P
    rng = np.random.default_rng(BENIH)
    theta = np.vstack([rng.uniform(0.2, 0.5, P),
                       rng.uniform(0.5, 0.8, P)])
    xl, yl = bangkit(20_000, theta, rng)
    xu, yu = bangkit(3000, theta, rng)
    q_benar = 1 / (1 + np.exp(-log_odds(xu, 0.5, theta)))
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.85), sharey=True)
    for ax, m in zip(axs, (1, 2, 4)):
        xl_m, xu_m = salin(xl, m), salin(xu, m)
        th = np.vstack([xl_m[yl == k].mean(axis=0) for k in (0, 1)])
        q = 1 / (1 + np.exp(-log_odds(xu_m, yl.mean(), th)))
        ax.scatter(q_benar, q, s=2, color=np.where(yu == 1, JINGGA, BIRU),
                   alpha=0.35, linewidths=0)
        ax.plot([0, 1], [0, 1], color=ABU, lw=0.6)
        ax.set_title(f"salinan $m = {m}$")
        ax.set_xlabel("posterior benar")
        ax.set_aspect("equal")
        _rapikan(ax)
    axs[0].set_ylabel("posterior naif")
    fig.tight_layout()
    simpan(fig, "bab05-ganda")

# ============================ Bab 6 ==================================

def bab06_kurva():
    t = np.linspace(0.3, 0.999, 700)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for N, warna in ((4, JINGGA), (40, BIRU), (400, HIJAU)):
        n1 = 3 * N // 4
        ll = n1 * np.log(t) + (N - n1) * np.log(1 - t)
        llm = n1 * np.log(0.75) + (N - n1) * np.log(0.25)
        ax.plot(t, np.exp(ll - llm), color=warna, lw=1.1,
                label=f"{n1} dari {N} pesan")
    ax.axvline(0.75, color=ABU, lw=0.6, ls="--")
    ax.set_xlabel("$\\theta = P(\\mathrm{hadiah} \\mid \\mathrm{penipuan})$")
    ax.set_ylabel("$L(\\theta) / L(\\hat\\theta)$")
    ax.legend(loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab06-kurva")


def bab06_sebaran():
    from scipy.stats import binom
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.9), sharey=True)
    for ax, (theta, N) in zip(axs, ((0.75, 20), (0.05, 20), (0.05, 100))):
        k = np.arange(N + 1)
        pk = binom.pmf(k, N, theta)
        tampil = pk > 1e-4
        warna = [MERAH if kk == 0 else BIRU for kk in k[tampil]]
        ax.bar(k[tampil] / N, pk[tampil], width=0.8 / N, color=warna)
        ax.axvline(theta, color=HIJAU, lw=0.8, ls="--")
        ax.set_title(f"$\\theta = {angka_mat(theta, 2)}$, $N = {N}$")
        ax.set_xlabel("$\\hat\\theta$")
        _rapikan(ax)
    axs[0].set_ylabel("peluang")
    axs[0].set_xlim(0.35, 1.02)
    axs[1].set_xlim(-0.02, 0.3)
    axs[2].set_xlim(-0.02, 0.15)
    fig.tight_layout()
    simpan(fig, "bab06-sebaran")


def bab06_nol():
    N = np.arange(1, 501)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for theta, warna in ((0.2, HIJAU), (0.05, BIRU), (0.01, JINGGA)):
        ax.plot(N, (1 - theta) ** N, color=warna, lw=1.1,
                label=f"$\\theta = {angka_mat(theta, 2)}$")
    ax.set_xscale("log")
    ax.axhline(0.01, color=ABU, lw=0.6, ls=":")
    ax.set_xlabel("banyaknya pengamatan kelas itu, $N$")
    ax.set_ylabel("P(hitungan nol)")
    ax.legend(loc="lower left")
    _rapikan(ax)
    simpan(fig, "bab06-nol")

# ============================ Bab 7 ==================================

def bab07_posterior():
    from bab04_data import PESAN_X, PESAN_Y
    from bab05_naif import posterior_naif
    from bab07_laplace import latih_halus
    alpha = np.logspace(-3, 2, 300)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for pola, warna in (("1100", JINGGA), ("1001", BIRU), ("0001", HIJAU)):
        x = np.array([int(c) for c in pola])
        q = [posterior_naif(x, *latih_halus(PESAN_X, PESAN_Y, a))[1]
             for a in alpha]
        ax.plot(alpha, q, color=warna, lw=1.1, label=f"pola {pola}")
    ax.axhline(0.4, color=ABU, lw=0.6, ls="--")
    ax.axvline(1, color=ABU, lw=0.6, ls=":")
    ax.text(1.2e-3, 0.43, "prior 0,4", ha="left", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("P(penipuan | pola)")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab07-posterior")


def bab07_beta():
    from scipy.stats import beta
    t = np.linspace(0.0005, 0.9995, 800)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.95), sharey=True)
    for ax, (n1, n, judul) in zip(axs, ((0, 4, "rapat: 0 dari 4"),
                                        (3, 4, "hadiah: 3 dari 4"))):
        ax.plot(t, beta.pdf(t, 1, 1), color=ABU, lw=0.9, ls="--",
                label="prior Beta(1, 1)")
        a, b = 1 + n1, 1 + n - n1
        ax.plot(t, beta.pdf(t, a, b), color=BIRU, lw=1.1,
                label=f"posterior Beta({a}, {b})")
        rata = a / (a + b)
        ax.axvline(n1 / n, color=MERAH, lw=0.8, label="MLE")
        ax.axvline(rata, color=HIJAU, lw=0.8, ls="-.",
                   label="rata-rata posterior")
        ax.set_title(judul)
        ax.set_xlabel("$\\theta$")
        ax.set_ylim(0, 5.2)
        _rapikan(ax)
    axs[0].set_ylabel("kepadatan")
    axs[1].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab07-beta")


def bab07_alpha():
    from sklearn.metrics import log_loss
    from sklearn.naive_bayes import BernoulliNB
    from bab07_alpha import bangkit, peluang_kata
    rng = np.random.default_rng(BENIH)
    q = peluang_kata(rng)
    Xu, yu = bangkit(20_000, q, rng)
    data = {n: bangkit(n, q, rng) for n in (50, 500, 5000)}
    alpha = np.logspace(-3, 1.3, 18)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    for (n, (X, y)), warna in zip(data.items(), (JINGGA, BIRU, HIJAU)):
        ll = [log_loss(yu, BernoulliNB(alpha=a).fit(X, y).predict_proba(Xu))
              for a in alpha]
        ax.plot(alpha, ll, "o-", ms=2.5, lw=1, color=warna,
                label=f"n = {n:,}".replace(",", "."))
        i = int(np.argmin(ll))
        ax.plot(alpha[i], ll[i], "o", ms=6, mfc="none", color=warna)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0,3", "0,5", "1", "2", "5"])
    ax.minorticks_off()
    kunci_label(ax, "y")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("log-loss uji")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
    _rapikan(ax)
    simpan(fig, "bab07-alpha")

# ============================ Bab 8 ==================================

def bab08_bau():
    from bab08_data import BAU, muat
    huruf, y = muat()
    bau = huruf[:, 4]
    N1, N0, V = y.sum(), len(y) - y.sum(), len(BAU)
    kode = sorted(BAU, key=lambda k: BAU[k])
    bobot, label = [], []
    for k in kode:
        n1 = np.sum((bau == k) & (y == 1))
        n0 = np.sum((bau == k) & (y == 0))
        bobot.append(np.log(((n1 + 1) / (N1 + V)) / ((n0 + 1) / (N0 + V))))
        label.append(f"{BAU[k]} ({n1}/{n0})")
    urut = np.argsort(bobot)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    b = np.array(bobot)[urut]
    ax.barh(range(V), b, color=[JINGGA if v > 0 else BIRU for v in b])
    ax.set_yticks(range(V))
    ax.set_yticklabels(np.array(label)[urut], fontsize=6.5)
    kunci_label(ax, "y")
    ax.axvline(0, color=ABU, lw=0.6)
    ax.set_xlabel("bobot bukti ln[P(bau | racun) / P(bau | dimakan)]")
    _rapikan(ax)
    simpan(fig, "bab08-bau")


def bab08_alpha():
    from sklearn.metrics import log_loss
    from sklearn.naive_bayes import CategoricalNB
    from bab08_data import muat_kode
    X, y, _ = muat_kode()
    mc = X.max(axis=0) + 1
    rng = np.random.default_rng(BENIH)
    alpha = np.logspace(-6, 1, 15)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    for n, warna in ((30, JINGGA), (100, BIRU), (1000, HIJAU)):
        ak, ll = [], []
        for a in alpha:
            u, v = [], []
            for _ in range(40):
                lt = rng.choice(len(y), n, replace=False)
                uj = np.setdiff1d(np.arange(len(y)), lt)
                P = CategoricalNB(alpha=a, min_categories=mc).fit(
                    X[lt], y[lt]).predict_proba(X[uj])
                u.append(np.mean(P.argmax(1) == y[uj]))
                v.append(log_loss(y[uj], P))
            ak.append(np.mean(u))
            ll.append(np.mean(v))
        axs[0].plot(alpha, ak, "o-", ms=2, lw=1, color=warna)
        axs[1].plot(alpha, ll, "o-", ms=2, lw=1, color=warna,
                    label=f"n = {n:,}".replace(",", "."))
    for ax, lab in zip(axs, ("akurasi uji", "log-loss uji")):
        ax.set_xscale("log")
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel(lab)
        _rapikan(ax)
    axs[1].set_yscale("log")
    axs[1].set_yticks([0.03, 0.1, 0.3, 1, 3])
    axs[1].set_yticklabels(["0,03", "0,1", "0,3", "1", "3"])
    axs[1].minorticks_off()
    kunci_label(axs[1], "y")
    axs[1].legend(loc="upper right", fontsize=6.3)
    fig.tight_layout()
    simpan(fig, "bab08-alpha")


def bab08_fitur():
    from bab08_fitur import FITUR, X, skor
    tunggal = np.array([skor([j]) for j in range(X.shape[1])])
    urut = np.argsort(tunggal)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.barh(range(len(urut)), tunggal[urut], color=BIRU, height=0.7)
    ax.set_yticks(range(len(urut)))
    ax.set_yticklabels([FITUR[j].replace("_", " ") for j in urut],
                       fontsize=6)
    kunci_label(ax, "y")
    semua = skor(list(range(X.shape[1])))
    ax.axvline(semua, color=JINGGA, lw=1, ls="--")
    ax.text(semua - 0.01, 1, f"semua fitur\n{angka(semua, 3)}",
            color=JINGGA, ha="right", fontsize=6.3)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("akurasi validasi silang (satu fitur, $\\alpha = 1$)")
    _rapikan(ax)
    simpan(fig, "bab08-fitur")


def bab08_hilang():
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import CategoricalNB
    from bab08_data import muat_kode
    from bab08_hilang import log_gabungan, posterior
    X, y, _ = muat_kode()
    mc = X.max(axis=0) + 1
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                      random_state=BENIH)
    m = CategoricalNB(alpha=0.01, min_categories=mc).fit(Xl, yl)
    modus = np.array([np.bincount(Xl[:, j]).argmax()
                      for j in range(X.shape[1])])
    rng = np.random.default_rng(BENIH)
    bagian = np.linspace(0, 0.9, 10)
    lewat, isi = [], []
    for b in bagian:
        u, v = [], []
        for _ in range(10):
            h = rng.random(Xu.shape) < b
            u.append(np.mean(posterior(log_gabungan(m, Xu, h)).argmax(1)
                             == yu))
            v.append(m.score(np.where(h, modus, Xu), yu))
        lewat.append(np.mean(u))
        isi.append(np.mean(v))
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.plot(bagian, lewat, "o-", ms=2.5, lw=1.1, color=HIJAU,
            label="faktor dilewati")
    ax.plot(bagian, isi, "o-", ms=2.5, lw=1.1, color=JINGGA,
            label="diisi modus")
    ax.axhline(max(np.mean(yu), 1 - np.mean(yu)), color=ABU, lw=0.6,
               ls=":")
    ax.set_xlabel("bagian nilai fitur yang hilang")
    ax.set_ylabel("akurasi uji")
    ax.legend(loc="lower left")
    _rapikan(ax)
    simpan(fig, "bab08-hilang")

# ============================ Bab 9 ==================================

def bab09_satu():
    from scipy.stats import norm
    from bab09_data import FITUR, muat
    X, y = muat()
    x = X[:, FITUR.index("titik cekung (terburuk)")]
    t = np.linspace(0, 0.32, 400)
    pri = [np.mean(y == 0), np.mean(y == 1)]
    fig, axs = plt.subplots(2, 1, figsize=(4.2, 3.1), sharex=True,
                            gridspec_kw={"height_ratios": [1.4, 1]})
    f = []
    for k, warna, nama in ((0, BIRU, "jinak"), (1, JINGGA, "ganas")):
        xs = x[y == k]
        axs[0].hist(xs, bins=np.linspace(0, 0.3, 31), density=True,
                    color=warna, alpha=0.3, lw=0)
        fk = norm.pdf(t, xs.mean(), xs.std())
        axs[0].plot(t, fk, color=warna, lw=1.1,
                    label=f"{nama}: $\\mathcal{{N}}({angka_mat(xs.mean(), 3)},"
                          f"\\,{angka_mat(xs.std(), 3)}^2)$")
        f.append(pri[k] * fk)
    axs[0].set_ylabel("kepadatan")
    axs[0].legend(loc="upper right", fontsize=6.3)
    q = f[1] / (f[0] + f[1])
    axs[1].plot(t, q, color=HIJAU, lw=1.1)
    axs[1].axhline(0.5, color=ABU_GARIS, lw=0.6)
    axs[1].set_ylabel("P(ganas | x)")
    axs[1].set_xlabel("titik cekung (terburuk)")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-satu")


def bab09_batas():
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from bab09_data import FITUR, muat
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.3))
    # kiri: rata-rata sama, varians berbeda
    x = np.linspace(-5, 5, 500)
    f0, f1 = 0.5 * norm.pdf(x, 0, 1), 0.5 * norm.pdf(x, 0, 2)
    ax = axs[0]
    ax.plot(x, f0, color=BIRU, lw=1.1, label="$\\sigma = 1$")
    ax.plot(x, f1, color=JINGGA, lw=1.1, label="$\\sigma = 2$")
    tt = np.sqrt(8 * np.log(2) / 3)
    for s in (-tt, tt):
        ax.axvline(s, color=HIJAU, lw=0.8, ls="--")
    ax.fill_between(x, 0, 0.21, where=np.abs(x) > tt, color=JINGGA,
                    alpha=0.08, lw=0)
    ax.set_title("rata-rata sama")
    ax.set_xlabel("$x$")
    ax.legend(loc="upper left", fontsize=6.3)
    ax.set_ylim(0, 0.21)
    _rapikan(ax)
    # kanan: dua fitur kanker, batas GaussianNB dan regresi logistik
    X, y = muat()
    j1 = FITUR.index("tekstur (rata-rata)")
    j2 = FITUR.index("titik cekung (terburuk)")
    Z = X[:, [j1, j2]]
    ax = axs[1]
    ax.scatter(Z[:, 0], Z[:, 1], s=3, c=np.where(y == 1, JINGGA, BIRU),
               alpha=0.5, linewidths=0)
    g1, g2 = np.meshgrid(np.linspace(8, 40, 300), np.linspace(0, 0.3, 300))
    G = np.column_stack([g1.ravel(), g2.ravel()])
    nb = GaussianNB().fit(Z, y)
    lr = LogisticRegression(max_iter=5000).fit(Z, y)
    ax.contour(g1, g2, nb.predict_proba(G)[:, 1].reshape(g1.shape),
               [0.5], colors=HIJAU, linewidths=1.1)
    ax.contour(g1, g2, lr.predict_proba(G)[:, 1].reshape(g1.shape),
               [0.5], colors=MERAH, linewidths=0.9, linestyles="--")
    from matplotlib.patches import Ellipse
    for k, warna in ((0, BIRU), (1, JINGGA)):
        for r in (1, 2):
            ax.add_patch(Ellipse(nb.theta_[k], 2 * r * np.sqrt(nb.var_[k, 0]),
                                 2 * r * np.sqrt(nb.var_[k, 1]),
                                 fill=False, color=warna, lw=0.6))
    ax.set_xlim(8, 40)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("tekstur (rata-rata)")
    ax.set_ylabel("titik cekung (terburuk)")
    ax.set_title("dua fitur kanker")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-batas")


def bab09_bentuk():
    from scipy.stats import gaussian_kde, norm
    from bab09_data import FITUR, muat
    X, y = muat()
    x = X[:, FITUR.index("luas (galat baku)")]
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    for ax, data, judul in ((axs[0], x, "luas (galat baku)"),
                            (axs[1], np.log1p(x), "log(1 + luas (galat baku))")):
        t = np.linspace(data.min(), data.max(), 400)
        for k, warna in ((0, BIRU), (1, JINGGA)):
            d = data[y == k]
            ax.hist(d, bins=30, density=True, color=warna, alpha=0.25, lw=0)
            ax.plot(t, norm.pdf(t, d.mean(), d.std()), color=warna, lw=1)
            ax.plot(t, gaussian_kde(d)(t), color=warna, lw=0.8, ls="--")
        ax.set_xlabel(judul)
        _rapikan(ax)
    axs[0].set_xlim(0, 200)
    axs[0].set_ylabel("kepadatan")
    fig.tight_layout()
    simpan(fig, "bab09-bentuk")


def bab09_yakin():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from bab09_data import muat
    X, y = muat()
    cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.9), sharey=True)
    for ax, (nama, m) in zip(axs, (
            ("naive Bayes Gaussian", GaussianNB()),
            ("regresi logistik", make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000))))):
        q = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        ax.hist([q[y == 0], q[y == 1]], bins=np.linspace(0, 1, 21),
                color=[BIRU, JINGGA], stacked=True, lw=0)
        ax.set_title(nama)
        ax.set_xlabel("P(ganas | x), validasi silang")
        _rapikan(ax)
    axs[0].set_ylabel("banyaknya sampel")
    fig.tight_layout()
    simpan(fig, "bab09-yakin")

# ============================ Bab 10 =================================

def bab10_model():
    from sklearn.feature_extraction.text import CountVectorizer
    from bab10_data import muat
    teks, y = muat()
    vek = CountVectorizer()
    X = vek.fit_transform(teks)
    kata = list(vek.get_feature_names_out())
    pilih = ["call", "free", "txt", "prize", "you", "ok", "come", "ll"]
    j = [kata.index(k) for k in pilih]
    B = (X[:, j] > 0).toarray()
    C = X[:, j].toarray()
    tot = np.asarray(X.sum(axis=1)).ravel()
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1), sharex=True)
    lebar = 0.38
    xs = np.arange(len(pilih))
    for k, warna, geser, nama in ((0, BIRU, -lebar / 2, "ham"),
                                  (1, JINGGA, lebar / 2, "spam")):
        m = y == k
        axs[0].bar(xs + geser, B[m].mean(axis=0), lebar, color=warna,
                   label=nama)
        axs[1].bar(xs + geser, C[m].sum(axis=0) / tot[m].sum(), lebar,
                   color=warna)
    axs[0].set_title("Bernoulli: P(kata hadir | kelas)")
    axs[1].set_title("multinomial: bagian token")
    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels(pilih, rotation=45, fontsize=6.3)
        kunci_label(ax, "x")
        _rapikan(ax)
    axs[0].legend(loc="upper right", fontsize=6.3)
    fig.tight_layout()
    simpan(fig, "bab10-model")


def bab10_ukuran():
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from bab10_dua import nilai
    ukuran = [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, None]
    xs = [8749 if m is None else m for m in ukuran]
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    for M, warna, nama in ((MultinomialNB, BIRU, "multinomial"),
                           (BernoulliNB, JINGGA, "Bernoulli")):
        h = [nilai(M(), m) for m in ukuran]
        axs[0].plot(xs, [a for a, _ in h], "o-", ms=2.5, lw=1,
                    color=warna, label=nama)
        axs[1].plot(xs, [b for _, b in h], "o-", ms=2.5, lw=1,
                    color=warna)
    for ax, lab in zip(axs, ("akurasi", "log-loss")):
        ax.set_xscale("log")
        ax.set_xlabel("besar kosakata")
        ax.set_ylabel(lab)
        _rapikan(ax)
    axs[0].legend(loc="lower right", fontsize=6.3)
    fig.tight_layout()
    simpan(fig, "bab10-ukuran")


def bab10_panjang():
    from bab10_panjang import log_odds, token, y
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    rng = np.random.default_rng(BENIH)
    ada = token > 0
    goyang = token[ada] * np.exp(rng.uniform(-0.08, 0.08, ada.sum()))
    ax.scatter(goyang, log_odds[ada], s=2,
               c=np.where(y[ada] == 1, JINGGA, BIRU), alpha=0.35,
               linewidths=0)
    ax.axhline(0, color=ABU, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("banyaknya token pesan")
    ax.set_ylabel("log-odds spam")
    _rapikan(ax)
    simpan(fig, "bab10-panjang")

# ============================ Bab 11 =================================

def bab11_alpha():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from bab11_data import muat
    tl, yl, tu, yu, _ = muat()
    vek = CountVectorizer()
    Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
    alpha = np.logspace(-3, 1, 13)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for M, warna, nama in ((MultinomialNB, BIRU, "multinomial"),
                           (ComplementNB, JINGGA, "complement")):
        ak = [accuracy_score(yu, M(alpha=a).fit(Xl, yl).predict(Xu))
              for a in alpha]
        ax.plot(alpha, ak, "o-", ms=2.5, lw=1.1, color=warna, label=nama)
    ax.set_xscale("log")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("akurasi uji")
    ax.legend(loc="lower left")
    _rapikan(ax)
    simpan(fig, "bab11-alpha")


def bab11_timpang():
    from sklearn.metrics import recall_score
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from bab11_timpang import Xl, Xu, yl, yu
    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    xs = np.arange(20)
    for M, warna, geser, nama in (
            (MultinomialNB(alpha=0.1), BIRU, -0.2, "multinomial"),
            (ComplementNB(alpha=0.1, norm=True), JINGGA, 0.2,
             "complement, norm=True")):
        r = recall_score(yu, M.fit(Xl, yl).predict(Xu), average=None)
        ax.bar(xs + geser, r, 0.4, color=warna, label=nama)
    for k in range(0, 20, 2):
        ax.axvspan(k - 0.5, k + 0.5, color=ABU_GARIS, alpha=0.3, lw=0,
                   zorder=0)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("kelas (arsir abu-abu: 10% data latih)")
    ax.set_ylabel("recall uji")
    ax.legend(loc="upper center", ncol=2, fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab11-timpang")


def bab11_tfidf():
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0))
    c = np.arange(0, 21)
    axs[0].plot(c, c, color=ABU, lw=0.8, ls="--", label="hitungan")
    axs[0].plot(c, np.log1p(c), "o-", ms=2, color=BIRU, lw=1.1,
                label="$\\log(1 + c)$")
    axs[0].set_xlabel("hitungan kata di dokumen, $c$")
    axs[0].set_ylim(0, 8)
    axs[0].legend(loc="upper left", fontsize=6.3)
    n = 1000
    df = np.arange(1, n + 1)
    axs[1].plot(df, np.log((1 + n) / (1 + df)) + 1, color=JINGGA, lw=1.1)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("banyaknya dokumen yang memuat kata")
    axs[1].set_ylabel("idf ($n = 1000$)")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-tfidf")

# ============================ Bab 12 =================================

def bab12_underflow():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from bab11_data import muat
    tl, yl, tu, yu, _ = muat()
    vek = CountVectorizer()
    Xl, Xu = vek.fit_transform(tl), vek.transform(tu)
    J = MultinomialNB(alpha=0.1).fit(Xl, yl).predict_joint_log_proba(Xu)
    token = np.asarray(Xu.sum(axis=1)).ravel()
    ada = token > 0
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    digit = -J.max(axis=1)[ada] / np.log(10)
    ax.scatter(token[ada], digit, s=2, color=BIRU, alpha=0.3,
               linewidths=0)
    ax.axhline(-np.log10(np.finfo(float).tiny), color=MERAH, lw=0.8,
               ls="--")
    ax.text(1.1, 420, "batas float64 (sekitar $10^{-308}$)",
            color=MERAH, fontsize=6.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("banyaknya token dokumen")
    ax.set_ylabel("$-\\log_{10}$(prior $\\times$ kemungkinan)")
    _rapikan(ax)
    simpan(fig, "bab12-underflow")


def bab12_geser():
    J = np.array([-1000.0, -1001.0, -1003.0])
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.8))
    lab = ["$k=1$", "$k=2$", "$k=3$"]
    axs[0].bar(lab, np.exp(J), color=ABU)
    axs[0].set_title("$e^{a_k}$ (semuanya 0)")
    axs[0].set_ylim(0, 1.1)
    e = np.exp(J - J.max())
    axs[1].bar(lab, e, color=BIRU)
    axs[1].set_title("$e^{a_k - m}$, $m = -1000$")
    axs[2].bar(lab, e / e.sum(), color=HIJAU)
    axs[2].set_title("posterior")
    for ax in axs:
        kunci_label(ax, "x")
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-geser")


def bab12_jenuh():
    from scipy.special import expit
    t = np.linspace(0, 60, 601)
    P = expit(t)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.semilogy(t, np.where(1 - P > 0, 1 - P, np.nan), color=MERAH,
                lw=1.2, label="$1 - P$ dihitung langsung")
    ax.semilogy(t, expit(-t), color=HIJAU, lw=1, ls="--",
                label="expit$(-t)$")
    ax.axvline(53 * np.log(2), color=ABU, lw=0.6, ls=":")
    ax.set_xlabel("log-odds $t$")
    ax.set_ylabel("peluang kelas lain")
    ax.legend(loc="lower left", fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab12-jenuh")

# ============================ Bab 13 =================================

def bab13_struktur():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def kotak(x, y, w, h, teks, warna, sub=""):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=warna,
                                    lw=1))
        ax.text(x + w / 2, y + h * 0.62, teks, ha="center", va="center",
                color=warna, fontsize=7, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                color=ABU, fontsize=5.6)

    kotak(2.6, 3.5, 4.8, 1.2, "NaiveBayes", BIRU,
          "fit, prior, logsumexp, predict")
    anak = [("Multinomial", "N + alpha"), ("Bernoulli", "log1p(-t)"),
            ("Kategorik", "tabel per fitur"), ("Gaussian", "var + eps"),
            ("Campuran", "jumlah bagian")]
    for i, (nama, sub) in enumerate(anak):
        x = 0.05 + i * 2.0
        kotak(x, 0.6, 1.8, 1.2, nama, HIJAU if i < 4 else JINGGA, sub)
        ax.annotate("", xy=(5.0, 3.45), xytext=(x + 0.9, 1.85),
                    arrowprops=dict(arrowstyle="-|>", color=ABU, lw=0.7))
    ax.text(5.0, 2.55, "_taksir(X) dan _log_fitur(X)", ha="center",
            fontsize=6.3, color=ABU, style="italic")
    simpan(fig, "bab13-struktur")


def bab13_selisih():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import (BernoulliNB, CategoricalNB,
                                     GaussianNB, MultinomialNB)
    from bab04_data import PESAN_X, PESAN_Y
    from bab08_data import muat_kode
    from bab09_data import muat as muat_kanker
    from bab10_data import muat as muat_sms
    from bab13_nb import Bernoulli, Gaussian, Kategorik, Multinomial
    Xj, yj, _ = muat_kode()
    teks, ys = muat_sms()
    Xs = CountVectorizer().fit_transform(teks)
    Xk, yk = muat_kanker()
    kasus = [("jamur", Kategorik(1.0), CategoricalNB(alpha=1.0), Xj, yj),
             ("SMS", Multinomial(1.0), MultinomialNB(alpha=1.0), Xs, ys),
             ("kanker", Gaussian(), GaussianNB(), Xk, yk)]
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.8), sharey=True)
    for ax, (nama, a, b, X, y) in zip(axs, kasus):
        d = np.abs(a.fit(X, y).predict_log_proba(X)
                   - b.fit(X, y).predict_log_proba(X)).ravel()
        d = np.log10(np.maximum(d, 1e-18))
        ax.hist(d, bins=np.arange(-18, -10.5, 0.5), color=BIRU, lw=0)
        ax.set_title(nama)
        ax.set_xlabel("$\\log_{10}$ |selisih|")
        _rapikan(ax)
    axs[0].set_ylabel("banyaknya nilai")
    fig.tight_layout()
    simpan(fig, "bab13-selisih")


def bab13_angka():
    from scipy.stats import norm
    from bab13_campuran import angka, y
    z = np.log1p(angka)
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    t = np.linspace(0, 4, 300)
    for k, warna, nama in ((0, BIRU, "ham"), (1, JINGGA, "spam")):
        ax.hist(z[y == k], bins=np.linspace(0, 4, 33), density=True,
                color=warna, alpha=0.35, lw=0, label=nama)
        ax.plot(t, norm.pdf(t, z[y == k].mean(), z[y == k].std()),
                color=warna, lw=1)
    for b in np.log1p([1, 3, 10]):
        ax.axvline(b - 1e-9, color=ABU, lw=0.6, ls=":")
    ax.set_xlabel("log(1 + banyaknya angka di pesan)")
    ax.set_ylabel("kepadatan")
    ax.legend(loc="upper right")
    _rapikan(ax)
    simpan(fig, "bab13-angka")

# ============================ Bab 14 =================================

def bab14_varians():
    from bab14_varians import gabung
    rng = np.random.default_rng(BENIH)
    pusat = 10.0 ** np.arange(0, 10)
    naif, chan = [], []
    z = rng.normal(0, 1, 100_000)
    for c in pusat:
        x = c + z
        s1 = s2 = 0.0
        g = (0, 0.0, 0.0)
        for p in np.split(x, 10):
            s1 += p.sum()
            s2 += (p * p).sum()
            g = gabung(g, (len(p), p.mean(), ((p - p.mean()) ** 2).sum()))
        benar = np.var(z)
        naif.append(abs(s2 / len(x) - (s1 / len(x)) ** 2 - benar) / benar)
        chan.append(abs(g[2] / g[0] - benar) / benar)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.loglog(pusat, np.maximum(naif, 1e-17), "o-", ms=2.5, color=MERAH,
              lw=1, label="jumlah kuadrat")
    ax.loglog(pusat, np.maximum(chan, 1e-17), "o-", ms=2.5, color=HIJAU,
              lw=1, label="Chan dkk.")
    ax.set_xlabel("rata-rata data (simpangan baku 1)")
    ax.set_ylabel("galat relatif varians")
    ax.legend(loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab14-varians")


def bab14_hash():
    from bab14_hash import akurasi, ember
    from sklearn.feature_extraction.text import CountVectorizer
    bit = np.arange(6, 21, 2)
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    for a, warna in ((1.0, BIRU), (0.1, JINGGA), (0.01, HIJAU)):
        ak = [akurasi(ember(b), a) for b in bit]
        ax.plot(2.0 ** bit, ak, "o-", ms=2.5, lw=1, color=warna,
                label=f"$\\alpha = {angka_mat(a, 2 if a < 0.1 else 1)}$")
    ax.axvline(8749, color=ABU, lw=0.6, ls=":")
    ax.text(8749 * 1.1, 0.93, "besar\nkosakata", fontsize=6.3, color=ABU)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("banyaknya ember")
    ax.set_ylabel("akurasi validasi silang")
    ax.legend(loc="lower right", fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab14-hash")


def bab14_tabrakan():
    from sklearn.feature_extraction.text import (CountVectorizer,
                                                 HashingVectorizer)
    from bab10_data import muat
    teks, _ = muat()
    kata = CountVectorizer().fit(teks).get_feature_names_out()
    V = len(kata)
    bit = np.arange(8, 23)
    amat = []
    for b in bit:
        H = HashingVectorizer(n_features=2**b, alternate_sign=False,
                              norm=None).transform(kata)
        e = H.indices
        _, c = np.unique(e, return_counts=True)
        amat.append(c[c > 1].sum() / V)
    m = 2.0 ** bit
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.plot(m, 1 - np.exp(-(V - 1) / m), color=BIRU, lw=1.1,
            label="$1 - e^{-(V-1)/m}$")
    ax.plot(m, amat, "o", ms=3, color=JINGGA, label="kosakata SMS")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("banyaknya ember $m$")
    ax.set_ylabel("bagian kata yang\nberbagi ember")
    ax.legend(loc="upper right")
    _rapikan(ax)
    simpan(fig, "bab14-tabrakan")

# ============================ Bab 15 =================================

def bab15_bidang():
    """Batas multinomial enam dokumen pada bidang (hadiah, rapat)."""
    w = np.log([11 / 4, 22 / 15, 11 / 60])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    g1, g2 = np.meshgrid(np.linspace(0, 6, 300), np.linspace(0, 6, 300))
    lo = w[0] * g1 + w[2] * g2
    ax.contourf(g1, g2, lo, levels=[-50, 0, 50],
                colors=[BIRU_MUDA, JINGGA_MUDA])
    ax.contour(g1, g2, lo, levels=[-2, -1, 1, 2], colors=ABU_GARIS,
               linewidths=0.5)
    ax.contour(g1, g2, lo, levels=[0], colors=HIJAU, linewidths=1.2)
    for n1 in range(7):
        for n3 in range(7):
            ax.plot(n1, n3, ".", color=JINGGA if w[0] * n1 + w[2] * n3 > 0
                    else BIRU, ms=3)
    ax.plot(2, 1, "o", mfc="none", color=MERAH, ms=7)
    ax.annotate("hadiah hadiah rapat", (2, 1), xytext=(2.6, 0.35),
                fontsize=6.3, color=MERAH)
    ax.set_xlabel("banyaknya token hadiah")
    ax.set_ylabel("banyaknya token rapat")
    ax.set_aspect("equal")
    _rapikan(ax)
    simpan(fig, "bab15-bidang")


def bab15_bobot():
    from bab15_lr import w_lr, w_nb, sering, kata
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.scatter(w_nb[sering], w_lr[sering], s=4, color=BIRU, alpha=0.5,
               linewidths=0)
    for k in ("claim", "prize", "txt", "call", "gt", "lt", "ok"):
        j = list(kata).index(k)
        ax.annotate(k, (w_nb[j], w_lr[j]), fontsize=6, color=JINGGA,
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.set_xlabel("bobot naive Bayes")
    ax.set_ylabel("bobot regresi logistik")
    _rapikan(ax)
    simpan(fig, "bab15-bobot")


def bab15_sumbangan():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from bab10_data import muat
    from bab15_linear import linear_multinomial
    teks, y = muat()
    vek = CountVectorizer()
    X = vek.fit_transform(teks)
    kata = vek.get_feature_names_out()
    w, b = linear_multinomial(MultinomialNB().fit(X, y))
    i = 2                                   # pesan ke-3 (spam)
    baris = X[i]
    j, n = baris.indices, baris.data
    s = n * w[j]
    urut = np.argsort(s)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    lab = [f"{kata[a]} (x{c})" if c > 1 else kata[a]
           for a, c in zip(j[urut], n[urut])] + ["intersep"]
    nilai = list(s[urut]) + [b]
    ax.barh(range(len(nilai)), nilai,
            color=[JINGGA if v > 0 else BIRU for v in nilai])
    ax.set_yticks(range(len(nilai)))
    ax.set_yticklabels(lab, fontsize=5.8)
    kunci_label(ax, "y")
    ax.axvline(0, color=ABU, lw=0.6)
    ax.set_xlabel(f"sumbangan ke log-odds spam (jumlah "
                  f"{angka(s.sum() + b, 1)})")
    _rapikan(ax)
    simpan(fig, "bab15-sumbangan")

# ============================ Bab 16 =================================

def _gambar_kurva(axs, daftar, judul):
    for ax, hasil, j in zip(axs, daftar, judul):
        n = [h[0] for h in hasil]
        ax.plot(n, [h[1] for h in hasil], "o-", ms=2.5, lw=1.1,
                color=BIRU, label="naive Bayes")
        ax.plot(n, [h[2] for h in hasil], "s-", ms=2.5, lw=1.1,
                color=JINGGA, label="regresi logistik")
        ax.set_xscale("log")
        ax.set_xlabel("banyaknya data latih $n$")
        ax.set_title(j)
        _rapikan(ax)
    axs[0].set_ylabel("galat uji")
    axs[0].legend(loc="upper right", fontsize=6.3)


def bab16_sim():
    from bab16_sim import kurva, model_a, model_b
    daftar = []
    for pembuat in (model_a, model_b):
        rng = np.random.default_rng(BENIH)
        daftar.append(kurva(pembuat(rng), rng))
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    _gambar_kurva(axs, daftar, ("A: asumsi naif benar",
                                "B: satu fitur disalin"))
    fig.tight_layout()
    simpan(fig, "bab16-sim")


def bab16_nyata():
    from bab16_nyata import KASUS, kurva
    daftar = [kurva(X, y, u, nb, lr) for _, X, y, u, nb, lr in KASUS]
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    _gambar_kurva(axs, daftar, ("SMS spam", "kanker payudara"))
    fig.tight_layout()
    simpan(fig, "bab16-nyata")


def bab16_dua():
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    from bab09_data import FITUR, muat
    X, y = muat()
    x = X[:, FITUR.index("titik cekung (terburuk)")]
    t = np.linspace(0, 0.3, 300)
    pri = [np.mean(y == 0), np.mean(y == 1)]
    f = [pri[k] * norm.pdf(t, x[y == k].mean(), x[y == k].std())
         for k in (0, 1)]
    lr = LogisticRegression(penalty=None).fit(x[:, None], y)
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    axs[0].plot(t, f[0], color=BIRU, lw=1.1, label="jinak")
    axs[0].plot(t, f[1], color=JINGGA, lw=1.1, label="ganas")
    axs[0].set_title("generatif: $p(y)\\,p(x \\mid y)$")
    axs[0].legend(loc="upper right", fontsize=6.3)
    axs[1].plot(t, f[1] / (f[0] + f[1]), color=HIJAU, lw=1.1,
                label="dari naive Bayes")
    axs[1].plot(t, lr.predict_proba(t[:, None])[:, 1], color=MERAH,
                lw=1, ls="--", label="regresi logistik")
    axs[1].scatter(x, y + np.random.default_rng(BENIH).uniform(
        -0.03, 0.03, len(y)), s=1.5, color=ABU, alpha=0.4, linewidths=0)
    axs[1].set_title("diskriminatif: $P(y = 1 \\mid x)$")
    axs[1].legend(loc="center right", fontsize=6)
    for ax in axs:
        ax.set_xlabel("titik cekung (terburuk)")
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-dua")

# ============================ Bab 17 =================================

def bab17_balik():
    a = np.linspace(-4, 4, 400)          # log-odds prior
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    b = np.linspace(0, 3, 400)           # log rasio kemungkinan
    A, B = np.meshgrid(a, b)
    satu = A + B > 0
    dua = A + 2 * B > 0
    ax.contourf(A, B, (satu != dua).astype(float), levels=[0.5, 1.5],
                colors=[JINGGA_MUDA])
    ax.plot(-b, b, color=BIRU, lw=1, label="batas, bukti sekali")
    ax.plot(-2 * b, b, color=JINGGA, lw=1, ls="--",
            label="batas, bukti dua kali")
    ax.plot(np.log(1 / 4), np.log(3), "o", color=MERAH, ms=4)
    ax.annotate("Contoh Soal 17.1", (np.log(1 / 4), np.log(3)),
                xytext=(-3.9, 2.75), fontsize=6.3, color=MERAH,
                arrowprops=dict(arrowstyle="-", color=MERAH, lw=0.5))
    ax.text(-2.9, 1.75, "keputusan\nberbalik", fontsize=6.5,
            color=JINGGA, ha="center")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 3)
    ax.set_xlabel("log-odds prior")
    ax.set_ylabel("log rasio kemungkinan fitur")
    ax.legend(loc="lower right", fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab17-balik")


def bab17_korelasi():
    from sklearn.naive_bayes import GaussianNB
    from scipy.stats import multivariate_normal as mvn
    rng = np.random.default_rng(BENIH)
    kasus = [((1, 1), 0.9, 0.9, "sama, simetris"),
             ((1, 0.3), 0.9, 0.9, "sama, tidak simetris"),
             ((1, 1), 0.9, -0.9, "berbeda arah")]
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.9))
    g1, g2 = np.meshgrid(np.linspace(-3, 4, 250), np.linspace(-3, 4, 250))
    G = np.column_stack([g1.ravel(), g2.ravel()])
    for ax, (mu1, r0, r1, judul) in zip(axs, kasus):
        S = [np.array([[1, r0], [r0, 1]]), np.array([[1, r1], [r1, 1]])]
        mu = [np.zeros(2), np.array(mu1, float)]
        X = np.vstack([rng.multivariate_normal(mu[k], S[k], 300)
                       for k in (0, 1)])
        y = np.repeat([0, 1], 300)
        ax.scatter(X[:, 0], X[:, 1], s=1.5, alpha=0.4, linewidths=0,
                   c=np.where(y == 1, JINGGA, BIRU))
        nb = GaussianNB().fit(X, y)
        ax.contour(g1, g2, nb.predict_proba(G)[:, 1].reshape(g1.shape),
                   [0.5], colors=HIJAU, linewidths=1)
        f = mvn(mu[1], S[1]).logpdf(G) - mvn(mu[0], S[0]).logpdf(G)
        ax.contour(g1, g2, f.reshape(g1.shape), [0], colors=MERAH,
                   linewidths=0.9, linestyles="--")
        ax.set_title(judul, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2, 4])
        ax.set_yticks([-2, 0, 2, 4])
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab17-korelasi")


def bab17_yakin():
    from sklearn.naive_bayes import GaussianNB
    from scipy.stats import multivariate_normal as mvn
    rng = np.random.default_rng(BENIH)
    S = np.array([[1, 0.9], [0.9, 1]])
    mu = [np.zeros(2), np.ones(2)]
    X = np.vstack([rng.multivariate_normal(m, S, 3000) for m in mu])
    y = np.repeat([0, 1], 3000)
    q_nb = GaussianNB().fit(X, y).predict_proba(X)[:, 1]
    f = mvn(mu[1], S).logpdf(X) - mvn(mu[0], S).logpdf(X)
    q = 1 / (1 + np.exp(-f))
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.scatter(q, q_nb, s=1.5, color=BIRU, alpha=0.3, linewidths=0)
    ax.plot([0, 1], [0, 1], color=ABU, lw=0.6)
    ax.axhline(0.5, color=ABU_GARIS, lw=0.5)
    ax.axvline(0.5, color=ABU_GARIS, lw=0.5)
    ax.set_xlabel("posterior sebenarnya")
    ax.set_ylabel("posterior naive Bayes")
    ax.set_aspect("equal")
    _rapikan(ax)
    simpan(fig, "bab17-yakin")

# ============================ Bab 18 =================================

def _keandalan(ax, q, y, warna, label, selang=10):
    b = np.minimum((q * selang).astype(int), selang - 1)
    xs, ys = [], []
    for i in range(selang):
        m = b == i
        if m.sum() >= 5:
            xs.append(q[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", ms=2.5, lw=1, color=warna, label=label)


def bab18_keandalan():
    from sklearn.calibration import CalibratedClassifierCV
    from bab18_data import kasus, posterior
    daftar = [k for k in kasus() if "naive" in k[0]]
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.4))
    for ax, (nama, X, y, m) in zip(axs, daftar):
        ax.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.8)
        _keandalan(ax, posterior(m, X, y), y, MERAH, "tanpa kalibrasi")
        _keandalan(ax, posterior(CalibratedClassifierCV(
            m, method="sigmoid", cv=5), X, y), y, BIRU, "Platt")
        _keandalan(ax, posterior(CalibratedClassifierCV(
            m, method="isotonic", cv=5), X, y), y, HIJAU, "isotonik")
        ax.set_title(nama.split(",")[0])
        ax.set_xlabel("posterior rata-rata selang")
        ax.set_aspect("equal")
        _rapikan(ax)
    axs[0].set_ylabel("bagian positif sebenarnya")
    axs[1].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab18-keandalan")


def bab18_platt():
    from scipy.special import expit, logit
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from bab18_data import kasus, posterior
    nama, X, y, m = kasus()[2]                   # kanker, naive Bayes
    q = np.clip(posterior(m, X, y), 1e-15, 1 - 1e-15)
    lo = logit(q)
    pl = LogisticRegression(penalty=None).fit(lo[:, None], y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(lo, y)
    t = np.linspace(-35, 35, 500)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.scatter(lo, y + np.random.default_rng(BENIH).uniform(-0.03, 0.03,
               len(y)), s=2, color=ABU, alpha=0.4, linewidths=0)
    ax.plot(t, expit(t), color=MERAH, lw=1, label="tanpa kalibrasi")
    ax.plot(t, pl.predict_proba(t[:, None])[:, 1], color=BIRU, lw=1.1,
            label=f"Platt ($a = {angka_mat(pl.coef_[0, 0], 2)}$)")
    ax.plot(t, iso.predict(t), color=HIJAU, lw=1, label="isotonik")
    ax.set_xlabel("log-odds naive Bayes")
    ax.set_ylabel("peluang ganas")
    ax.legend(loc="center right", fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab18-platt")


def bab18_prior():
    q = np.linspace(0.001, 0.999, 500)
    odds_latih = 0.1342 / (1 - 0.1342)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(q, q, color=ABU_GARIS, lw=0.8)
    for pi, warna in ((0.02, BIRU), (0.5, JINGGA)):
        o = q / (1 - q) * (pi / (1 - pi)) / odds_latih
        ax.plot(q, o / (1 + o), color=warna, lw=1.1,
                label=f"prior pakai {angka(pi, 2)}")
    ax.set_xlabel("posterior dengan prior latih 0,134")
    ax.set_ylabel("posterior terkoreksi")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab18-prior")

# ============================ Bab 19 =================================

def bab19_sebaran():
    from bab19_data import muat
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1))
    kelas = ["negatif", "netral", "positif"]
    lebar = 0.27
    for j, (b, warna, nama) in enumerate((("train", BIRU, "latih"),
                                           ("valid", HIJAU, "validasi"),
                                           ("test", JINGGA, "uji"))):
        t, y = muat(b)
        axs[0].bar(np.arange(3) + (j - 1) * lebar,
                   [np.mean(y == k) for k in kelas], lebar, color=warna,
                   label=nama)
        panjang = [len(s.split()) for s in t]
        axs[1].hist(panjang, bins=np.arange(0, 110, 5), density=True,
                    histtype="step", color=warna, lw=1.1, label=nama)
    axs[0].set_xticks(range(3))
    axs[0].set_xticklabels(kelas)
    kunci_label(axs[0], "x")
    axs[0].set_ylabel("bagian teks")
    axs[0].legend(loc="upper left", fontsize=6)
    axs[1].set_xlabel("banyaknya token")
    axs[1].set_ylabel("kepadatan")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab19-sebaran")


def bab19_pilih():
    from bab19_pilih import NAMA_VEK, cari, kisi
    gs = cari()
    r = gs.cv_results_
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1), sharey=True)
    for ax, keluarga in zip(axs, ("MultinomialNB", "ComplementNB")):
        for v, warna in zip(NAMA_VEK, (BIRU, HIJAU, JINGGA)):
            xs, ys = [], []
            for p, s in zip(r["params"], r["mean_test_score"]):
                nv = NAMA_VEK[[str(k) for k in kisi["vek"]].index(
                    str(p["vek"]))]
                if nv == v and type(p["nb"]).__name__ == keluarga:
                    xs.append(p["nb__alpha"])
                    ys.append(s)
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", ms=2.5,
                    lw=1, color=warna, label=v)
        ax.set_xscale("log")
        ax.set_xlabel("$\\alpha$")
        ax.set_title(keluarga.replace("NB", ""))
        _rapikan(ax)
    axs[0].set_ylabel("F1 makro (validasi silang)")
    axs[1].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    simpan(fig, "bab19-pilih")


def bab19_kata():
    from bab19_data import muat
    tl, yl = muat("train")
    tu, _ = muat("test")
    kata = ["gue", "banget", "sih", "tidak", "enak", "tempat"]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    lebar = 0.2
    xs = np.arange(len(kata))
    for j, (k, warna) in enumerate((("positif", HIJAU), ("netral", ABU),
                                    ("negatif", MERAH))):
        v = [np.mean([w in t.split() for t, l in zip(tl, yl) if l == k])
             for w in kata]
        ax.bar(xs + (j - 1.5) * lebar, v, lebar, color=warna,
               label=f"latih, {k}")
    v = [np.mean([w in t.split() for t in tu]) for w in kata]
    ax.bar(xs + 1.5 * lebar, v, lebar, color=JINGGA, label="uji, semua")
    ax.set_xticks(xs)
    ax.set_xticklabels(kata)
    kunci_label(ax, "x")
    ax.set_ylabel("bagian teks yang memuat")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper left", fontsize=5.8, ncol=2)
    _rapikan(ax)
    simpan(fig, "bab19-kata")

# ============================ Bab 20 =================================

def bab20_graf():
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.9))
    xs = [(-0.75, 0), (-0.25, 0), (0.25, 0), (0.75, 0)]
    nama = ["$x_1$", "$x_2$", "$x_3$", "$x_4$"]
    pusat = (0, 0.8)
    for ax, judul in zip(axs, ("naive Bayes", "TAN", "AODE (induk $x_2$)")):
        for p in xs:
            _panah(ax, pusat, p, BIRU, r=0.12)
        if judul == "TAN":
            for a, b in ((0, 1), (1, 2), (1, 3)):
                ax.annotate("", xy=(xs[b][0] - 0.1 * np.sign(xs[b][0] - xs[a][0]),
                                    -0.05),
                            xytext=(xs[a][0] + 0.1 * np.sign(xs[b][0] - xs[a][0]),
                                    -0.05),
                            arrowprops=dict(arrowstyle="-|>", color=JINGGA,
                                            lw=0.7,
                                            connectionstyle="arc3,rad=0.5"))
        if judul.startswith("AODE"):
            for b in (0, 2, 3):
                ax.annotate("", xy=(xs[b][0], -0.12),
                            xytext=(xs[1][0], -0.12),
                            arrowprops=dict(arrowstyle="-|>", color=JINGGA,
                                            lw=0.7,
                                            connectionstyle="arc3,rad=0.45"))
        _simpul(ax, *pusat, "$y$", BIRU, r=0.12)
        for (x, yy), t in zip(xs, nama):
            _simpul(ax, x, yy, t, HIJAU, r=0.12)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.75, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(judul, fontsize=7.5)
    fig.tight_layout()
    simpan(fig, "bab20-graf")


def bab20_em():
    from sklearn.naive_bayes import MultinomialNB
    from bab20_em import Xl, Xu, em, yl, yu
    rng = np.random.default_rng(BENIH)
    ukuran = [10, 20, 50, 100, 200, 500]
    a, b = [], []
    for m in ukuran:
        u, v = [], []
        for _ in range(20):
            i = np.concatenate([
                rng.choice(np.where(yl == 1)[0], max(2, m // 7), False),
                rng.choice(np.where(yl == 0)[0], m - max(2, m // 7),
                           False)])
            j = np.setdiff1d(np.arange(len(yl)), i)
            u.append(MultinomialNB(alpha=0.1).fit(Xl[i], yl[i])
                     .score(Xu, yu))
            v.append(em(Xl[i], yl[i], Xl[j], 0.1).score(Xu, yu))
        a.append(np.mean(u))
        b.append(np.mean(v))
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.plot(ukuran, a, "o-", ms=2.5, lw=1.1, color=BIRU,
            label="hanya pesan berlabel")
    ax.plot(ukuran, b, "s-", ms=2.5, lw=1.1, color=HIJAU,
            label="EM dengan pesan tak berlabel")
    ax.set_xscale("log")
    ax.set_xlabel("banyaknya pesan berlabel")
    ax.set_ylabel("akurasi uji")
    ax.legend(loc="lower right", fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab20-em")


def bab20_peta():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def kotak(x, y, teks, warna, w=2.6, h=0.8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=warna,
                                    lw=1))
        ax.text(x, y, teks, ha="center", va="center", fontsize=6.3,
                color=warna)

    kotak(5, 3, "naive Bayes", BIRU, w=2.2)
    tujuan = [(1.6, 5.2, "TAN, AODE,\njaringan Bayes", "melonggarkan asumsi"),
              (5, 5.3, "LDA, QDA", "kovarians penuh"),
              (8.4, 5.2, "regresi logistik,\nNB-SVM", "diskriminatif"),
              (1.6, 0.8, "EM semi-terawasi,\ncampuran Gaussian", "tanpa label"),
              (5, 0.7, "naive Bayes\nBayesian penuh", "prior parameter"),
              (8.4, 0.8, "model bahasa,\ntransformer", "urutan kata")]
    for x, yy, t, ket in tujuan:
        kotak(x, yy, t, HIJAU)
        ax.annotate("", xy=(x, yy - 0.45 if yy > 3 else yy + 0.45),
                    xytext=(5 + 0.6 * np.sign(x - 5), 3.4 if yy > 3 else 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=ABU, lw=0.6))
        ax.text((x + 5) / 2 + (0.35 if x >= 5 else -0.35),
                (yy + 3) / 2, ket, fontsize=5.4, color=ABU,
                ha="center", style="italic")
    simpan(fig, "bab20-peta")

# ============================ Bab 1 ==================================

def _kotak_teks(ax, x, y, w, h, teks, warna, ukuran=6.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04",
                                facecolor="white", edgecolor=warna, lw=1))
    ax.text(x, y, teks, ha="center", va="center", fontsize=ukuran,
            color=warna)


def bab01_alur():
    fig, ax = plt.subplots(figsize=(4.9, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    langkah = [("pesan", "“klaim hadiah\ndengan\ntransfer pulsa”"),
               ("fitur", "hadiah 1\ntransfer 1\nrapat 0 ..."),
               ("bukti", "log-odds prior\n+ bobot tiap\nfitur"),
               ("posterior", "peluang\npenipuan\n0,86"),
               ("keputusan", "tandai bila\nmelebihi\nambang")]
    for i, (judul, isi) in enumerate(langkah):
        x = 1 + i * 2
        ax.text(x, 2.65, judul, ha="center", fontsize=7, color=BIRU,
                weight="bold")
        _kotak_teks(ax, x, 1.4, 1.62, 1.5, isi,
                    HIJAU if i in (2, 3) else ABU, ukuran=5.6)
        if i < 4:
            ax.annotate("", xy=(x + 1.15, 1.4), xytext=(x + 0.85, 1.4),
                        arrowprops=dict(arrowstyle="-|>", color=ABU,
                                        lw=0.8))
    simpan(fig, "bab01-alur")


def bab01_sejarah():
    peristiwa = [(1763, "Bayes: membalik\narah peluang"),
                 (1814, "Laplace: aturan\nsuksesi"),
                 (1961, "Maron: klasifikasi\ndokumen otomatis"),
                 (1973, "Duda & Hart:\nbuku pengenalan pola"),
                 (1998, "Sahami dkk.:\npenyaring surel"),
                 (2002, "Ng & Jordan;\nGraham")]
    fig, ax = plt.subplots(figsize=(4.9, 1.6))
    ax.axhline(0, color=ABU_GARIS, lw=1)
    for i, (t, teks) in enumerate(peristiwa):
        x = i
        ax.plot(x, 0, "o", color=BIRU, ms=4)
        ax.text(x, 0.18 if i % 2 == 0 else -0.18, f"{t}", ha="center",
                va="bottom" if i % 2 == 0 else "top", fontsize=7,
                color=BIRU, weight="bold")
        ax.text(x, 0.45 if i % 2 == 0 else -0.45, teks, ha="center",
                va="bottom" if i % 2 == 0 else "top", fontsize=5.8,
                color=ABU)
    ax.set_xlim(-0.6, len(peristiwa) - 0.4)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    simpan(fig, "bab01-sejarah")


def bab01_peta():
    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    bagian = [(1.7, 4.6, "I  Fondasi\nBab 1-3", ABU),
              (5.0, 4.6, "II  Teorema Bayes\nke pengklasifikasi\nBab 4-7", BIRU),
              (8.3, 4.6, "III  Keluarga\nnaive Bayes\nBab 8-11", BIRU),
              (1.7, 1.4, "IV  Menghitung\ndengan benar\nBab 12-14", HIJAU),
              (5.0, 1.4, "V  Membaca\ndan menilai\nBab 15-18", HIJAU),
              (8.3, 1.4, "VI  Praktik\nBab 19-20", JINGGA)]
    for x, y, t, w in bagian:
        _kotak_teks(ax, x, y, 2.8, 1.8, t, w, ukuran=6.8)
    for (a, b) in (((3.1, 4.6), (3.6, 4.6)), ((6.4, 4.6), (6.9, 4.6)),
                   ((8.3, 3.7), (1.7, 2.3)), ((3.1, 1.4), (3.6, 1.4)),
                   ((6.4, 1.4), (6.9, 1.4))):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=ABU, lw=0.8))
    simpan(fig, "bab01-peta")

# ============================ Bab 2 ==================================

def bab02_bersyarat():
    from matplotlib.patches import Rectangle
    from bab04_data import PESAN_X, PESAN_Y
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.8))
    for ax, judul, syarat in ((axs[0], "ruang sampel: 10 pesan", None),
                              (axs[1], "syarat: memuat hadiah", 0)):
        for i in range(10):
            x = i % 5
            y = 1 - i // 5
            pen = PESAN_Y[i] == 1
            had = PESAN_X[i, 0] == 1
            aktif = syarat is None or had
            ax.add_patch(Rectangle((x, y), 0.9, 0.9,
                                   facecolor=(JINGGA if pen else BIRU)
                                   if aktif else "white",
                                   edgecolor=ABU_GARIS, alpha=0.85 if aktif
                                   else 1, lw=0.8))
            ax.text(x + 0.45, y + 0.45, "H" if had else "", ha="center",
                    va="center", fontsize=7, color="white" if aktif
                    else ABU)
        ax.set_xlim(-0.1, 5)
        ax.set_ylim(-0.1, 2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(judul, fontsize=7)
    simpan(fig, "bab02-bersyarat")


def bab02_sebaran():
    from scipy.stats import binom, norm
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.0))
    k = np.arange(0, 11)
    axs[0].bar(k, binom.pmf(k, 10, 0.3), color=BIRU, width=0.7)
    axs[0].bar(3, binom.pmf(3, 10, 0.3), color=JINGGA, width=0.7)
    axs[0].set_xticks(range(0, 11, 2))
    axs[0].set_title("binomial, $n = 10$, $\\theta = 0{,}3$")
    axs[0].set_xlabel("banyaknya sukses $k$")
    axs[0].set_ylabel("peluang")
    t = np.linspace(-3.5, 3.5, 400)
    axs[1].plot(t, norm.pdf(t), color=BIRU, lw=1.1)
    m = np.abs(t) <= 1
    axs[1].fill_between(t[m], 0, norm.pdf(t[m]), color=JINGGA, alpha=0.4,
                        lw=0)
    axs[1].text(0, 0.12, "luas\n0,683", ha="center", fontsize=6.5)
    axs[1].set_title("normal baku")
    axs[1].set_xlabel("$x$")
    axs[1].set_ylabel("kepadatan")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-sebaran")


def bab02_logit():
    p = np.linspace(0.005, 0.995, 400)
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.0))
    axs[0].plot(p, np.log(p / (1 - p)), color=BIRU, lw=1.1)
    axs[0].axhline(0, color=ABU_GARIS, lw=0.6)
    axs[0].axvline(0.5, color=ABU_GARIS, lw=0.6)
    axs[0].plot(0.75, np.log(3), "o", color=JINGGA, ms=3.5)
    axs[0].set_xlabel("peluang $p$")
    axs[0].set_ylabel("log-odds")
    t = np.linspace(-6, 6, 400)
    axs[1].plot(t, 1 / (1 + np.exp(-t)), color=HIJAU, lw=1.1)
    axs[1].axhline(0.5, color=ABU_GARIS, lw=0.6)
    axs[1].plot(np.log(3), 0.75, "o", color=JINGGA, ms=3.5)
    axs[1].set_xlabel("log-odds $t$")
    axs[1].set_ylabel("$\\sigma(t)$")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab02-logit")

# ============================ Bab 3 ==================================

def bab03_siaran():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.9, 1.7))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0.2, 2.3)
    ax.axis("off")

    def kisi(x0, bar, kol, warna, label, isi, w=0.5, warna_teks="white",
             ukuran=5.5):
        for i in range(bar):
            for j in range(kol):
                ax.add_patch(Rectangle((x0 + j * w, 1.0 + (bar - 1 - i)
                                        * 0.5), w, 0.5, facecolor=warna,
                                       edgecolor="white", lw=1))
                ax.text(x0 + j * w + w / 2, 1.0 + (bar - 1 - i) * 0.5
                        + 0.25, isi[i][j], ha="center", va="center",
                        fontsize=ukuran, color=warna_teks)
        ax.text(x0 + kol * w / 2, 0.65, label, ha="center",
                fontsize=6, color=ABU)

    kisi(0.2, 2, 4, BIRU, "N (2 x 4)",
         [["2", "3", "2", "5"], ["4", "4", "4", "1"]])
    ax.text(2.55, 1.5, "/", fontsize=12, color=ABU, ha="center",
            va="center")
    kisi(2.95, 2, 1, JINGGA, "jumlah\n(2 x 1)", [["12"], ["13"]])
    ax.text(3.95, 1.5, "\u2192", fontsize=12, color=ABU, ha="center",
            va="center")
    kisi(4.4, 2, 4, JINGGA_MUDA, "diperluas (2 x 4)",
         [["12"] * 4, ["13"] * 4], warna_teks=JINGGA)
    ax.text(6.8, 1.5, "=", fontsize=12, color=ABU, ha="center",
            va="center")
    kisi(7.3, 2, 4, HIJAU, "theta (2 x 4)",
         [["0,167", "0,250", "0,167", "0,417"],
          ["0,308", "0,308", "0,308", "0,077"]], w=1.4, ukuran=5.8)
    simpan(fig, "bab03-siaran")


def bab03_validasi():
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    for i in range(5):
        for j in range(5):
            warna = JINGGA if i == j else BIRU_MUDA
            ax.add_patch(Rectangle((j, 4 - i), 0.95, 0.8, facecolor=warna,
                                   edgecolor="white"))
        ax.text(-0.15, 4 - i + 0.4, f"putaran {i + 1}", ha="right",
                va="center", fontsize=6.3, color=ABU)
    ax.text(2.5, -0.35, "lima bagian data latih (jingga: bagian uji "
            "putaran itu)", ha="center", fontsize=6.3, color=ABU)
    ax.set_xlim(-1.6, 5.1)
    ax.set_ylim(-0.6, 5)
    ax.axis("off")
    simpan(fig, "bab03-validasi")


def bab03_jarang():
    from sklearn.feature_extraction.text import CountVectorizer
    from bab10_data import muat
    teks, _ = muat()
    X = CountVectorizer().fit_transform(teks[:200])
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    ax.spy(X, markersize=0.35, color=BIRU, aspect="auto")
    ax.set_xlabel(f"kata ({X.shape[1]} kolom)")
    ax.set_ylabel("pesan (200 baris)")
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(labelsize=6)
    simpan(fig, "bab03-jarang")

# ---------------------------------------------------------------------
#  Fungsi gambar bab baru disisipkan DI ATAS blok ini.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
