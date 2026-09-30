"""Bab 19: mengapa naive Bayes gagal pada data uji SmSA.

(1) Bagian teks yang memuat beberapa kata gaya bicara, per kelas di data
latih dan di seluruh data uji. (2) Sumbangan kata terhadap log-odds
positif lawan negatif untuk dua ulasan positif di data uji yang
ditebak negatif.
"""
import numpy as np

from bab19_evaluasi import model_nb, tl, tu, yl, yu

nb = model_nb()
vek, m = nb[0], nb[1]
kelas = list(m.classes_)

if __name__ == "__main__":
    print("kata     latih: positif  netral  negatif    uji")
    for w in ("gue", "banget", "tidak"):
        b = {k: np.mean([w in t.split() for t, l in zip(tl, yl) if l == k])
             for k in ("positif", "netral", "negatif")}
        u = np.mean([w in t.split() for t in tu])
        print(f"{w:8s}        {b['positif']:.3f}   {b['netral']:.3f}"
              f"    {b['negatif']:.3f}   {u:.3f}")
    w = (m.feature_log_prob_[kelas.index("positif")]
         - m.feature_log_prob_[kelas.index("negatif")])
    kata = vek.get_feature_names_out()
    for kunci in ("baterai nya kuat", "film horor indonesia patut"):
        i = next(j for j, t in enumerate(tu) if kunci in t)
        x = vek.transform([tu[i]])
        s = sorted(((kata[j], x[0, j] * w[j]) for j in x.indices),
                   key=lambda t: t[1])
        print(f"ulasan uji ke-{i} (label {yu[i]}), log-odds "
              f"pos-neg {sum(v for _, v in s):+.1f}")
        print("  paling negatif: " + ", ".join(
            f"{a} {v:+.1f}" for a, v in s[:2]))
        print("  paling positif: " + ", ".join(
            f"{a} {v:+.1f}" for a, v in s[-2:]))
