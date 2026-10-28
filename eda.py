# -*- coding: utf-8 -*-
"""
eda.py  --  Exploratory Data Analysis (EDA) dan Analisis Korelasi
Project Data Sains Kelompok 1: Pemetaan dan Prediksi Kemiskinan Daerah (Kemiskinan & Kesehatan)
Penanggung jawab : Christine Simbolon (Data Analyst & EDA Specialist)  -> BAB IV

Masukan  (hasil preprocessing.py, folder data/processed/):
    data_gabungan.csv            162 baris = 27 kab/kota x 6 tahun (2019-2024)
    garis_kemiskinan_tahunan.csv 6 baris  (2020-2025, provinsi)
    p3ke_kecamatan.csv           30 baris (kecamatan Kota Bandung, 2024)
    laporan_outlier.csv          ringkasan IQR dari tahap preprocessing

Keluaran:
    outputs/figures/gambarNN_*.png  12 gambar BAB IV (NN = nomor Gambar di laporan: 05-16)
    outputs/tables/*.csv         tabel statistik & korelasi
    outputs/tables/eda_ringkasan.json  angka-angka yang dikutip di laporan

Cara menjalankan:  python scripts/eda.py
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. KONFIGURASI (memakai config.py bila tersedia; jika tidak, nilai bawaan)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # type: ignore
    DATA_DIR = Path(getattr(config, "DATA_PROCESSED_DIR", ROOT / "data" / "processed"))
    FIG_DIR = Path(getattr(config, "FIGURES_DIR", ROOT / "outputs" / "figures"))
    TAB_DIR = Path(getattr(config, "TABLES_DIR", ROOT / "outputs" / "tables"))
except ImportError:
    DATA_DIR = ROOT / "data" / "processed"
    FIG_DIR = ROOT / "outputs" / "figures"
    TAB_DIR = ROOT / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Lokasi CSV: dicari berurutan agar skrip jalan baik dari root proyek maupun folder scripts/
for _cand in (DATA_DIR, ROOT / "data" / "processed", ROOT / "data", ROOT, Path.cwd()):
    if (Path(_cand) / "data_gabungan.csv").exists():
        DATA_DIR = Path(_cand)
        break
else:
    sys.exit("data_gabungan.csv tidak ditemukan. Jalankan preprocessing.py lebih dulu atau atur DATA_PROCESSED_DIR di config.py")

# Nama gambar yang dihasilkan skrip ini (hanya file-file ini yang dibersihkan sebelum run)
FIG_FILES = ["gambar05_top10_indeks_2024.png", "gambar06_tren_tahunan.png", "gambar07_arah_perubahan.png",
             "gambar08_scatter.png", "gambar09_distribusi.png", "gambar10_heatmap_korelasi.png",
             "gambar11_heatmap_wilayah_tahun.png", "gambar12_komposisi_dak.png", "gambar13_kabupaten_vs_kota.png",
             "gambar14_garis_kemiskinan.png", "gambar15_p3ke_kecamatan.png", "gambar16_korelasi_bar.png"]
for _f in FIG_FILES + [p.name for p in FIG_DIR.glob("fig*.png")]:   # fig*.png = nama versi lama
    (FIG_DIR / _f).unlink(missing_ok=True)
SAVED = []

DPI = 200
BLUE, ORANGE, GREEN, RED, GREY = "#1F4E79", "#D9822B", "#3B8B5A", "#B23A48", "#6B7280"
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 100, "savefig.dpi": DPI,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

SUMMARY = {}   # semua angka penting dikumpulkan di sini


def idn(x, nd=2):
    """Format angka gaya Indonesia: 1.234,56"""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def idn_axis(nd=2):
    return FuncFormatter(lambda v, _: idn(v, nd))


def idn_matrix(m, nd=2):
    """Matriks teks berformat Indonesia untuk anotasi heatmap"""
    return np.vectorize(lambda v: idn(v, nd))(np.asarray(m, dtype=float))


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    SAVED.append(name)
    print(f"  [gambar] {path.name}")


def short(name):
    name = name.title()
    return name.replace("Kabupaten ", "Kab. ")


def strength(r):
    a = abs(r)
    return "Lemah" if a < 0.30 else ("Sedang" if a < 0.50 else "Kuat")


# --------------------------------------------------------------------------
# 2. MEMUAT DATA
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_DIR / "data_gabungan.csv")
    gk = pd.read_csv(DATA_DIR / "garis_kemiskinan_tahunan.csv")
    p3 = pd.read_csv(DATA_DIR / "p3ke_kecamatan.csv")
    out = pd.read_csv(DATA_DIR / "laporan_outlier.csv")

    df["wilayah"] = df["nama_kabupaten_kota"].map(short)
    df["jenis"] = np.where(df["nama_kabupaten_kota"].str.startswith("KABUPATEN"), "Kabupaten", "Kota")
    df["is_kabupaten"] = (df["jenis"] == "Kabupaten").astype(int)
    df["total_miliar"] = df["total_alokasi"] / 1e9
    # indeks tahun sebelumnya dipulihkan dari delta_indeks (2019 -> nilai 2018)
    df["indeks_sebelumnya"] = (df["indeks_keparahan_kemiskinan"] - df["delta_indeks"]).round(2)
    df["alokasi_nol"] = (df["total_alokasi"] == 0).astype(int)
    return df, gk, p3, out


def banner(df, gk, p3):
    print("=" * 66)
    print("EDA - Pemetaan & Prediksi Kemiskinan Daerah Jawa Barat")
    print("=" * 66)
    n_nan = int(df[["kode_kabupaten_kota", "nama_kabupaten_kota", "tahun", "indeks_keparahan_kemiskinan",
                    "total_alokasi"]].isna().sum().sum())
    print(f"data_gabungan.csv            : {df.shape[0]} baris, nilai kosong = {n_nan}")
    print(f"                               {df.nama_kabupaten_kota.nunique()} kab/kota x "
          f"{df.tahun.nunique()} tahun ({df.tahun.min()}-{df.tahun.max()})")
    print(f"garis_kemiskinan_tahunan.csv : {gk.shape[0]} baris | p3ke_kecamatan.csv : {p3.shape[0]} baris")
    print("-" * 66)


def ringkasan_konsol():
    """Ringkasan hasil utama di terminal (dipakai sebagai bukti eksekusi di laporan)."""
    S = SUMMARY
    d = {r["variabel"]: r for r in S["distribusi"]}
    tr = {r["tahun"]: r for r in S["tren"]}
    kor = {r["variabel"]: r for r in S["korelasi"]}
    sen = {r["skenario"].split(" (")[0]: r for r in S["sensitivitas"]}
    print("-" * 66)
    print(f"[1] Distribusi : skewness indeks = {idn(S['indeks_stat']['skew'])} | "
          f"total DAK = {idn(d['Total alokasi DAK (Rp miliar)']['skewness'])}")
    print(f"[2] Tren       : rata-rata indeks {idn(tr[2019]['rata2'])} (2019) -> {idn(tr[2021]['rata2'])} (2021)"
          f" -> {idn(tr[2024]['rata2'])} (2024)")
    t1, b1 = S["top10_2024"][0], S["bottom5_2024"][0]
    print(f"[3] Peringkat  : tertinggi 2024 = {t1['wilayah']} ({idn(t1['indeks_keparahan_kemiskinan'])}),")
    print(f"                 terendah = {b1['wilayah']} ({idn(b1['indeks_keparahan_kemiskinan'])})")
    print(f"[4] Korelasi   : r(indeks sebelumnya) = {idn(kor['indeks_sebelumnya']['pearson_r'])} | "
          f"r(log DAK) = {idn(kor['log_total_alokasi']['pearson_r'])}")
    print(f"[5] Sensitivitas: tanpa DAK = 0 -> r(log DAK) = {idn(sen['Tanpa DAK = 0']['r_log'])}; "
          f"tanpa Cirebon 2023 -> r(total) = {idn(sen['Tanpa Kota Cirebon 2023']['r_total'])}")
    print(f"[6] Pendukung  : garis kemiskinan naik {idn(S['gk_total_naik_persen'], 1)}% (2020-2025); "
          f"P3KE {idn(S['p3ke']['total'], 0)} individu")
    print("=" * 66)
    print(f"Selesai: {len(SAVED)} gambar -> outputs/figures/")
    print(f"         {len(list(TAB_DIR.glob('*.csv')))} tabel CSV + eda_ringkasan.json -> outputs/tables/")
    print("=" * 66)


# --------------------------------------------------------------------------
# 3. DISTRIBUSI DATA (4.1.1)
# --------------------------------------------------------------------------
def distribusi(df):
    cols = {
        "indeks_keparahan_kemiskinan": "Indeks keparahan kemiskinan (poin)",
        "total_miliar": "Total alokasi DAK (Rp miliar)",
        "log_total_alokasi": "Log total alokasi DAK",
        "delta_indeks": "Perubahan indeks (delta_indeks)",
    }
    rows = []
    for c, label in cols.items():
        s = df[c]
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "variabel": label, "n": len(s), "rata2": s.mean(), "median": s.median(),
            "simpangan_baku": s.std(), "skewness": stats.skew(s, bias=False),
            "kurtosis": stats.kurtosis(s, bias=False),
            "koef_variasi_persen": s.std() / s.mean() * 100 if s.mean() else np.nan,
            "n_outlier_iqr": int(((s < lo) | (s > hi)).sum()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(TAB_DIR / "tabel_distribusi.csv", index=False)
    SUMMARY["distribusi"] = t.round(4).to_dict("records")

    s = df["indeks_keparahan_kemiskinan"]
    SUMMARY["indeks_stat"] = {"mean": s.mean(), "median": s.median(), "min": s.min(), "max": s.max(),
                              "skew": float(stats.skew(s, bias=False)), "sd": s.std()}
    # uji normalitas Shapiro-Wilk
    SUMMARY["shapiro_indeks"] = [float(v) for v in stats.shapiro(s)]
    SUMMARY["shapiro_log"] = [float(v) for v in stats.shapiro(df.loc[df.total_alokasi > 0, "log_total_alokasi"])]
    pos = df.loc[df.total_alokasi > 0, "log_total_alokasi"]
    SUMMARY["log_positif_skew"] = float(stats.skew(pos, bias=False))

    # ---- Gambar: histogram + boxplot
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 5.4), gridspec_kw={"height_ratios": [3, 1.6]})
    sns.histplot(df["indeks_keparahan_kemiskinan"], bins=20, kde=True, color=BLUE, ax=ax[0, 0])
    ax[0, 0].axvline(s.mean(), color=RED, ls="--", lw=1.2, label=f"Rata-rata {idn(s.mean())}")
    ax[0, 0].axvline(s.median(), color=GREEN, ls="-.", lw=1.2, label=f"Median {idn(s.median())}")
    ax[0, 0].set(title="(a) Histogram indeks keparahan", xlabel="Indeks keparahan (poin)", ylabel="Frekuensi")
    ax[0, 0].legend(frameon=False)
    sns.histplot(pos, bins=20, kde=True, color=ORANGE, ax=ax[0, 1])
    ax[0, 1].set(title="(b) Histogram log total DAK", xlabel="log(1 + total alokasi)", ylabel="")
    ax[0, 1].text(0.03, 0.93, f"{int(df.alokasi_nol.sum())} nilai 0 tidak ditampilkan", transform=ax[0, 1].transAxes,
                  fontsize=7.5, va="top", color=GREY)
    sns.boxplot(x=df["indeks_keparahan_kemiskinan"], color=BLUE, width=.5, ax=ax[1, 0],
                flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED, markersize=4))
    ax[1, 0].set(title="(c) Boxplot indeks keparahan", xlabel="Indeks keparahan (poin)")
    sns.boxplot(x=df["total_miliar"], color=ORANGE, width=.5, ax=ax[1, 1],
                flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED, markersize=4))
    ax[1, 1].set_xscale("symlog", linthresh=10)
    ax[1, 1].set_xlim(0, 4000)
    ax[1, 1].set_xticks([0, 10, 100, 1000])
    ax[1, 1].xaxis.set_major_formatter(idn_axis(0))
    ax[1, 1].set(title="(d) Boxplot total alokasi DAK", xlabel="Total alokasi (Rp miliar, skala symlog)")
    for a_ in (ax[0, 0], ax[1, 0]):
        a_.xaxis.set_major_formatter(idn_axis(1))
    fig.tight_layout()
    save(fig, "gambar09_distribusi.png")


# --------------------------------------------------------------------------
# 4. TREN TAHUNAN (4.1.2) + gambar tren, delta
# --------------------------------------------------------------------------
def tren(df):
    rows = []
    for th, g in df.groupby("tahun"):
        imax, imin = g.loc[g.indeks_keparahan_kemiskinan.idxmax()], g.loc[g.indeks_keparahan_kemiskinan.idxmin()]
        rows.append({
            "tahun": th, "rata2": g.indeks_keparahan_kemiskinan.mean(),
            "median": g.indeks_keparahan_kemiskinan.median(),
            "min": imin.indeks_keparahan_kemiskinan, "daerah_min": imin.wilayah,
            "maks": imax.indeks_keparahan_kemiskinan, "daerah_maks": imax.wilayah,
            "total_dak_miliar": g.total_miliar.sum(), "median_dak_miliar": g.total_miliar.median(),
            "n_alokasi_nol": int(g.alokasi_nol.sum()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(TAB_DIR / "tabel_tren_tahunan.csv", index=False)
    SUMMARY["tren"] = t.round(4).to_dict("records")

    # ---- Gambar tren
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.5))
    a = ax[0]
    a.fill_between(t.tahun, t["min"], t["maks"], color=BLUE, alpha=.12, label="Rentang min-maks")
    a.plot(t.tahun, t["rata2"], marker="o", color=BLUE, lw=2, label="Rata-rata")
    a.plot(t.tahun, t["median"], marker="s", color=ORANGE, lw=1.6, ls="--", label="Median")
    for x, y in zip(t.tahun, t["rata2"]):
        a.annotate(idn(y), (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=BLUE)
    a.set(title="(a) Indeks keparahan kemiskinan", xlabel="Tahun", ylabel="Indeks (poin)")
    a.yaxis.set_major_formatter(idn_axis(1))
    a.legend(frameon=False, loc="upper left")
    b = ax[1]
    b.bar(t.tahun, t.total_dak_miliar / 1000, color=BLUE, alpha=.85, label="Total DAK (kiri)")
    b.set(title="(b) Alokasi DAK kesehatan", xlabel="Tahun", ylabel="Total DAK (Rp triliun)", xticks=list(t.tahun))
    b.yaxis.set_major_formatter(idn_axis(1))
    b.set_ylim(0, 5.4)
    b2 = b.twinx()
    b2.plot(t.tahun, t.median_dak_miliar, color=ORANGE, marker="o", lw=2, label="Median per kab/kota (kanan)")
    b2.set_ylabel("Median (Rp miliar)")
    b2.set_ylim(0, 108)
    b2.spines["right"].set_visible(True)
    b2.grid(False)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "gambar06_tren_tahunan.png")

    # ---- Gambar dinamika delta_indeks: jumlah daerah naik/turun/tetap
    d = df.assign(arah=np.where(df.delta_indeks > 0, "Naik (memburuk)",
                         np.where(df.delta_indeks < 0, "Turun (membaik)", "Tetap")))
    ct = pd.crosstab(d.tahun, d.arah)[["Naik (memburuk)", "Tetap", "Turun (membaik)"]]
    ct.to_csv(TAB_DIR / "tabel_arah_perubahan.csv")
    SUMMARY["arah"] = ct.to_dict("index")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ct.plot(kind="bar", stacked=True, ax=ax[0], color=[RED, GREY, GREEN], width=.75, legend=False)
    ax[0].set(title="(a) Arah perubahan indeks", xlabel="Tahun", ylabel="Jumlah kab/kota")
    ax[0].tick_params(axis="x", rotation=0)
    ax[0].legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2))
    sns.boxplot(data=df, x="tahun", y="delta_indeks", color="#9DB7D5", ax=ax[1], width=.6,
                flierprops=dict(marker="o", markersize=3, markerfacecolor=RED, markeredgecolor=RED))
    ax[1].axhline(0, color="black", lw=.8)
    ax[1].set(title="(b) Sebaran delta_indeks", xlabel="Tahun", ylabel="Perubahan (poin)")
    ax[1].yaxis.set_major_formatter(idn_axis(1))
    fig.tight_layout()
    save(fig, "gambar07_arah_perubahan.png")


# --------------------------------------------------------------------------
# 5. VISUALISASI WILAYAH (4.2.x)
# --------------------------------------------------------------------------
def visual_wilayah(df):
    last = df[df.tahun == df.tahun.max()].copy()
    rata = df.groupby("wilayah").indeks_keparahan_kemiskinan.mean()
    last["rata2_19_24"] = last.wilayah.map(rata)
    top = last.sort_values("indeks_keparahan_kemiskinan", ascending=False)
    top10 = top.head(10)
    bot5 = top.tail(5).sort_values("indeks_keparahan_kemiskinan")
    top[["wilayah", "indeks_keparahan_kemiskinan", "rata2_19_24", "total_miliar"]].to_csv(
        TAB_DIR / "tabel_peringkat_2024.csv", index=False)
    SUMMARY["top10_2024"] = top10[["wilayah", "indeks_keparahan_kemiskinan", "rata2_19_24", "total_miliar"]].round(4).to_dict("records")
    SUMMARY["bottom5_2024"] = bot5[["wilayah", "indeks_keparahan_kemiskinan", "rata2_19_24", "total_miliar"]].round(4).to_dict("records")
    SUMMARY["rentang_2024"] = {"min": float(last.indeks_keparahan_kemiskinan.min()),
                               "min_daerah": last.loc[last.indeks_keparahan_kemiskinan.idxmin(), "wilayah"],
                               "max": float(last.indeks_keparahan_kemiskinan.max()),
                               "max_daerah": last.loc[last.indeks_keparahan_kemiskinan.idxmax(), "wilayah"]}
    SUMMARY["dak_2024"] = {"min": float(last.total_miliar.min()),
                           "min_daerah": last.loc[last.total_miliar.idxmin(), "wilayah"],
                           "max": float(last.total_miliar.max()),
                           "max_daerah": last.loc[last.total_miliar.idxmax(), "wilayah"]}

    # ---- Gambar 1: top 10 bar chart
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    pal = [RED if v >= top10.indeks_keparahan_kemiskinan.iloc[2] else BLUE for v in top10.indeks_keparahan_kemiskinan]
    bars = ax.barh(top10.wilayah[::-1], top10.indeks_keparahan_kemiskinan[::-1], color=pal[::-1], height=.7)
    ax.scatter(top10.rata2_19_24[::-1], top10.wilayah[::-1], color=ORANGE, zorder=3, s=28, marker="D",
               label="Rata-rata 2019-2024")
    for b, v, rr in zip(bars, top10.indeks_keparahan_kemiskinan[::-1], top10.rata2_19_24[::-1]):
        ax.text(max(v, rr) + .014, b.get_y() + b.get_height() / 2, idn(v), va="center", fontsize=8)
    ax.set(xlabel="Indeks keparahan kemiskinan 2024 (poin)", ylabel="")
    ax.set_xlim(0, top10.indeks_keparahan_kemiskinan.max() * 1.2)
    ax.xaxis.set_major_formatter(idn_axis(1))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", ls=":", lw=.6)
    save(fig, "gambar05_top10_indeks_2024.png")

    # ---- Gambar 6: heatmap wilayah x tahun
    pv = df.pivot_table(index="wilayah", columns="tahun", values="indeks_keparahan_kemiskinan")
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    pv.to_csv(TAB_DIR / "tabel_wilayah_tahun.csv")
    fig, ax = plt.subplots(figsize=(5.6, 7.4))
    hm = sns.heatmap(pv, annot=idn_matrix(pv), fmt="", cmap="YlOrRd", linewidths=.4, linecolor="white",
                     annot_kws={"fontsize": 6.5}, cbar_kws={"label": "Indeks keparahan (poin)", "shrink": .6}, ax=ax)
    hm.collections[0].colorbar.ax.yaxis.set_major_formatter(idn_axis(1))
    ax.grid(False)
    ax.set(xlabel="Tahun", ylabel="")
    ax.tick_params(axis="y", labelsize=7.5)
    save(fig, "gambar11_heatmap_wilayah_tahun.png")
    top5 = pv.mean(axis=1).head(5)
    SUMMARY["rata_wilayah_top5"] = top5.round(4).to_dict()
    SUMMARY["rata_wilayah_bottom5"] = pv.mean(axis=1).tail(5).round(4).to_dict()
    # konsistensi peringkat antar tahun (Spearman peringkat 2019 vs 2024, tahun berurutan)
    rk = {th: pv[th].rank() for th in pv.columns}
    SUMMARY["rank_corr_2019_2024"] = float(stats.spearmanr(rk[2019], rk[2024])[0])
    SUMMARY["rank_corr_berurutan"] = {f"{a}-{b}": float(stats.spearmanr(rk[a], rk[b])[0])
                                      for a, b in zip(pv.columns[:-1], pv.columns[1:])}

    # ---- Gambar 8: kabupaten vs kota
    g = df.groupby("jenis").agg(n_kabkota=("wilayah", "nunique"),
                                rata2_indeks=("indeks_keparahan_kemiskinan", "mean"),
                                median_indeks=("indeks_keparahan_kemiskinan", "median"),
                                sd_indeks=("indeks_keparahan_kemiskinan", "std"),
                                median_dak_miliar=("total_miliar", "median"),
                                rata2_dak_miliar=("total_miliar", "mean"))
    g.to_csv(TAB_DIR / "tabel_kabupaten_vs_kota.csv")
    SUMMARY["jenis"] = g.round(4).to_dict("index")
    u = stats.mannwhitneyu(df[df.jenis == "Kabupaten"].indeks_keparahan_kemiskinan,
                           df[df.jenis == "Kota"].indeks_keparahan_kemiskinan)
    SUMMARY["mannwhitney_jenis"] = {"U": float(u.statistic), "p": float(u.pvalue)}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw={"width_ratios": [1.6, 1]})
    sns.boxplot(data=df, x="tahun", y="indeks_keparahan_kemiskinan", hue="jenis", ax=ax[0],
                palette={"Kabupaten": BLUE, "Kota": ORANGE}, width=.7, linewidth=.9,
                flierprops=dict(marker="o", markersize=3))
    ax[0].set(title="(a) Per tahun", xlabel="Tahun", ylabel="Indeks keparahan (poin)", ylim=(0, .95))
    ax[0].yaxis.set_major_formatter(idn_axis(1))
    ax[0].legend(title="", frameon=False, loc="upper left", ncol=2)
    sns.violinplot(data=df, x="jenis", y="indeks_keparahan_kemiskinan", ax=ax[1],
                   palette={"Kabupaten": BLUE, "Kota": ORANGE}, inner="quartile", linewidth=.9, cut=0)
    ax[1].set(title="(b) Seluruh periode", xlabel="", ylabel="", ylim=(0, .95))
    ax[1].yaxis.set_major_formatter(idn_axis(1))
    fig.tight_layout()
    save(fig, "gambar13_kabupaten_vs_kota.png")

    # ---- Gambar 7: komposisi DAK per jenis dana
    jenis_dak = ["reguler", "penugasan", "bok", "jampersal", "akreditasi"]
    label = {"reguler": "DAK Reguler (fisik)", "penugasan": "DAK Penugasan (fisik)", "bok": "BOK (nonfisik)",
             "jampersal": "Jampersal (nonfisik)", "akreditasi": "Akreditasi (nonfisik)"}
    colors = [BLUE, "#6A9BD1", ORANGE, GREEN, "#B8A5D6"]
    agg = df.groupby("tahun")[jenis_dak].sum() / 1e12
    exc = df[~((df.wilayah == "Kota Cirebon") & (df.tahun == 2023))].groupby("tahun")[jenis_dak].sum()
    share = exc.div(exc.sum(axis=1), axis=0) * 100
    agg.to_csv(TAB_DIR / "tabel_komposisi_dak_triliun.csv")
    share.to_csv(TAB_DIR / "tabel_komposisi_dak_persen_tanpa_cirebon2023.csv")
    SUMMARY["komposisi_persen"] = share.round(2).to_dict("index")
    SUMMARY["komposisi_triliun"] = agg.round(4).to_dict("index")
    tot = df[jenis_dak].sum()
    SUMMARY["porsi_total_persen"] = (tot / tot.sum() * 100).round(2).to_dict()
    SUMMARY["nol_per_jenis"] = {c: int((df[c] == 0).sum()) for c in jenis_dak}
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.7))
    agg.plot(kind="bar", stacked=True, ax=ax[0], color=colors, width=.75, legend=False)
    ax[0].set(title="(a) Nilai absolut", xlabel="Tahun", ylabel="Rp triliun")
    ax[0].tick_params(axis="x", rotation=0)
    ax[0].yaxis.set_major_formatter(idn_axis(1))
    ax[0].grid(axis="x", visible=False); ax[1].grid(axis="x", visible=False)
    ax[0].annotate("Kota Cirebon\n(reguler 2023)", xy=(4, agg.loc[2023].sum() * .93), xytext=(.25, agg.loc[2023].sum() * .92),
                   fontsize=7, arrowprops=dict(arrowstyle="->", color=GREY), color=GREY)
    share.plot(kind="bar", stacked=True, ax=ax[1], color=colors, width=.75, legend=False)
    ax[1].set(title="(b) Proporsi tanpa Cirebon 2023", xlabel="Tahun", ylabel="Persen")
    ax[1].tick_params(axis="x", rotation=0)
    ax[1].set_ylim(0, 100)
    hs, _ = ax[0].get_legend_handles_labels()
    fig.legend(hs, [label[c] for c in jenis_dak], loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, "gambar12_komposisi_dak.png")


# --------------------------------------------------------------------------
# 6. SCATTER (4.2.3)
# --------------------------------------------------------------------------
def scatter(df):
    y = "indeks_keparahan_kemiskinan"
    pos = df[df.total_alokasi > 0]
    r_log = stats.pearsonr(pos.log_total_alokasi, pos[y])
    r_lag = stats.pearsonr(df.indeks_sebelumnya, df[y])
    SUMMARY["scatter"] = {"r_log_positif": [float(r_log[0]), float(r_log[1]), len(pos)],
                          "r_lag": [float(r_lag[0]), float(r_lag[1]), len(df)]}
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.9))
    a = ax[0]
    a.scatter(pos.total_miliar, pos[y], color=BLUE, s=22, alpha=.7, edgecolor="white", lw=.4, label="Observasi (DAK > 0)")
    z = np.polyfit(pos.log_total_alokasi, pos[y], 1)
    xs = np.linspace(pos.total_miliar.min(), pos.total_miliar.max(), 200)
    a.plot(xs, np.polyval(z, np.log1p(xs * 1e9)), color=RED, lw=1.6, label=f"Garis tren (r = {idn(r_log[0], 3)})")
    a.set_xscale("log")
    zero = df[df.total_alokasi == 0]
    a.scatter(np.full(len(zero), .8), zero[y], marker="x", color=GREY, s=24, label=f"DAK = 0 (n = {len(zero)})")
    cir = df[(df.wilayah == "Kota Cirebon") & (df.tahun == 2023)].iloc[0]
    a.annotate("Kota Cirebon 2023", (cir.total_miliar, cir[y]), xytext=(-78, 62), textcoords="offset points",
               fontsize=7, arrowprops=dict(arrowstyle="->", color=GREY))
    a.set(title="(a) Alokasi DAK vs indeks", xlabel="Total alokasi DAK (Rp miliar, skala log)",
          ylabel="Indeks keparahan (poin)", ylim=(0, .85))
    a.set_xticks([1, 10, 100, 1000]); a.set_xticklabels(["0*", "10", "100", "1.000"])
    a.yaxis.set_major_formatter(idn_axis(1))
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=7, ncol=1)
    b = ax[1]
    b.scatter(df.indeks_sebelumnya, df[y], c=BLUE, s=22, alpha=.7, edgecolor="white", lw=.4)
    m = np.polyfit(df.indeks_sebelumnya, df[y], 1)
    xs = np.linspace(df.indeks_sebelumnya.min(), df.indeks_sebelumnya.max(), 50)
    b.plot(xs, np.polyval(m, xs), color=RED, lw=1.6, label=f"Garis tren (r = {idn(r_lag[0])})")
    lim = [0, max(df.indeks_sebelumnya.max(), df[y].max()) + .03]
    b.plot(lim, lim, color=GREY, ls=":", lw=1, label="Garis y = x")
    b.set(title="(b) Persistensi indeks", xlabel="Indeks tahun sebelumnya (poin)", ylabel="Indeks tahun berjalan (poin)")
    b.xaxis.set_major_formatter(idn_axis(1)); b.yaxis.set_major_formatter(idn_axis(1))
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=7, ncol=1)
    b.set_ylim(0, .85)
    fig.tight_layout()
    save(fig, "gambar08_scatter.png")


# --------------------------------------------------------------------------
# 7. KORELASI (4.3)
# --------------------------------------------------------------------------
def korelasi(df):
    y = "indeks_keparahan_kemiskinan"
    xs = ["indeks_sebelumnya", "is_kabupaten", "tahun", "delta_indeks", "log_total_alokasi", "total_alokasi",
          "bok", "reguler", "penugasan", "jampersal", "akreditasi"]
    rows = []
    for x in xs:
        pr, pp = stats.pearsonr(df[x], df[y])
        sr, sp = stats.spearmanr(df[x], df[y])
        rows.append({"variabel": x, "pearson_r": pr, "p_pearson": pp, "spearman_rho": sr, "p_spearman": sp,
                     "arah": "Positif" if pr > 0 else "Negatif", "kekuatan": strength(pr)})
    t = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    t.to_csv(TAB_DIR / "tabel_korelasi_indeks.csv", index=False)
    SUMMARY["korelasi"] = t.round(5).to_dict("records")

    # --- sensitivitas terhadap Kota Cirebon 2023 dan nilai 0
    skenario = {
        "Seluruh data (n = 162)": df,
        "Tanpa Kota Cirebon 2023 (n = 161)": df[~((df.wilayah == "Kota Cirebon") & (df.tahun == 2023))],
        "Tanpa DAK = 0 (n = 149)": df[df.total_alokasi > 0],
        "Tanpa keduanya (n = 148)": df[(df.total_alokasi > 0) & ~((df.wilayah == "Kota Cirebon") & (df.tahun == 2023))],
    }
    rows = []
    for nama, d in skenario.items():
        a, pa = stats.pearsonr(d.total_alokasi, d[y]); b, pb = stats.pearsonr(d.log_total_alokasi, d[y])
        c, pc = stats.spearmanr(d.total_alokasi, d[y])
        rows.append({"skenario": nama, "n": len(d), "r_total": a, "p_total": pa, "r_log": b, "p_log": pb,
                     "rho_total": c, "p_rho": pc})
    s = pd.DataFrame(rows)
    s.to_csv(TAB_DIR / "tabel_sensitivitas_korelasi.csv", index=False)
    SUMMARY["sensitivitas"] = s.round(5).to_dict("records")

    # --- korelasi antar-wilayah (rata-rata 2019-2024 per kab/kota, n = 27)
    w = df.groupby("wilayah").agg(indeks=(y, "mean"), log_dak=("log_total_alokasi", "mean"),
                                  dak=("total_miliar", "median"))
    r1 = stats.pearsonr(w.log_dak, w.indeks); r2 = stats.spearmanr(w.dak, w.indeks)
    SUMMARY["antar_wilayah"] = {"pearson_log": [float(r1[0]), float(r1[1])], "spearman_median_dak": [float(r2[0]), float(r2[1])], "n": len(w)}
    # --- korelasi dalam tiap tahun
    per = {int(th): [float(v) for v in stats.spearmanr(g.total_alokasi, g[y])] for th, g in df.groupby("tahun")}
    SUMMARY["spearman_per_tahun"] = per
    # --- korelasi per jenis wilayah
    SUMMARY["korelasi_per_jenis"] = {
        j: [float(v) for v in stats.spearmanr(g.total_alokasi, g[y])] for j, g in df.groupby("jenis")}

    # --- Heatmap
    cols = [y, "indeks_sebelumnya", "delta_indeks", "log_total_alokasi", "total_alokasi",
            "bok", "reguler", "penugasan", "jampersal", "akreditasi", "tahun", "is_kabupaten"]
    nm = {y: "indeks_keparahan", "indeks_sebelumnya": "indeks_sebelumnya", "delta_indeks": "delta_indeks",
          "log_total_alokasi": "log_total_alokasi"}
    c = df[cols].corr(method="pearson").rename(index=nm, columns=nm)
    c.to_csv(TAB_DIR / "matriks_korelasi_pearson.csv")
    SUMMARY["antar_dak"] = {"reguler_total": float(c.loc["reguler", "total_alokasi"]),
                            "bok_total": float(c.loc["bok", "total_alokasi"]),
                            "bok_log": float(c.loc["bok", "log_total_alokasi"])}
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    hm = sns.heatmap(c, mask=mask, annot=idn_matrix(c), fmt="", cmap="RdBu_r", vmin=-1, vmax=1, center=0, linewidths=.5,
                     annot_kws={"fontsize": 7}, cbar_kws={"label": "Koefisien Pearson (r)", "shrink": .75}, ax=ax)
    hm.collections[0].colorbar.ax.yaxis.set_major_formatter(idn_axis(1))
    ax.grid(False)
    ax.tick_params(axis="x", rotation=45, labelsize=7.5); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    save(fig, "gambar10_heatmap_korelasi.png")

    # --- Bar korelasi Pearson vs Spearman
    tt = t.sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    yy = np.arange(len(tt)); h = .38
    ax.barh(yy + h / 2, tt.pearson_r, h, color=BLUE, label="Pearson (r)")
    ax.barh(yy - h / 2, tt.spearman_rho, h, color=ORANGE, label="Spearman (\u03c1)")
    ax.set_yticks(yy); ax.set_yticklabels(tt.variabel)
    for yi, v in zip(yy, tt.pearson_r):
        ax.text(v + (.015 if v >= 0 else -.015), yi + h / 2, idn(v), va="center", ha="left" if v >= 0 else "right", fontsize=7, color=BLUE)
    ax.axvline(0, color="black", lw=.8)
    for v in (-.5, -.3, .3, .5):
        ax.axvline(v, color=GREY, ls=":", lw=.8)
    ax.set(xlabel="Koefisien korelasi dengan indeks keparahan kemiskinan", xlim=(-.55, 1.0))
    ax.xaxis.set_major_formatter(idn_axis(1))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "gambar16_korelasi_bar.png")


# --------------------------------------------------------------------------
# 8. GARIS KEMISKINAN & P3KE (4.3.3)
# --------------------------------------------------------------------------
def pendukung(df, gk, p3):
    y = "indeks_keparahan_kemiskinan"
    gk = gk.copy()
    gk["selisih"] = gk.garis_kemiskinan_perkotaan - gk.garis_kemiskinan_perdesaan
    gk["selisih_persen"] = gk.selisih / gk.garis_kemiskinan_perdesaan * 100
    gk["naik_persen"] = gk.garis_kemiskinan_rata2.pct_change() * 100
    gk.to_csv(TAB_DIR / "tabel_garis_kemiskinan.csv", index=False)
    SUMMARY["gk"] = gk.round(3).to_dict("records")
    SUMMARY["gk_total_naik_persen"] = float((gk.garis_kemiskinan_rata2.iloc[-1] / gk.garis_kemiskinan_rata2.iloc[0] - 1) * 100)
    SUMMARY["gk_kota_naik_persen"] = float((gk.garis_kemiskinan_perkotaan.iloc[-1] / gk.garis_kemiskinan_perkotaan.iloc[0] - 1) * 100)
    SUMMARY["gk_desa_naik_persen"] = float((gk.garis_kemiskinan_perdesaan.iloc[-1] / gk.garis_kemiskinan_perdesaan.iloc[0] - 1) * 100)

    m = df.groupby("tahun")[y].mean().rename("indeks_rata2").reset_index().merge(gk[["tahun", "garis_kemiskinan_rata2"]], on="tahun")
    r = stats.pearsonr(m.garis_kemiskinan_rata2, m.indeks_rata2)
    SUMMARY["gk_vs_indeks_tahunan"] = {"n": len(m), "r": float(r[0]), "p": float(r[1])}

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw={"width_ratios": [1.4, 1]})
    a = ax[0]
    a.plot(gk.tahun, gk.garis_kemiskinan_perkotaan / 1e3, marker="o", color=BLUE, lw=2, label="Perkotaan")
    a.plot(gk.tahun, gk.garis_kemiskinan_perdesaan / 1e3, marker="s", color=ORANGE, lw=2, label="Perdesaan")
    for x, v in zip(gk.tahun, gk.garis_kemiskinan_perkotaan / 1e3):
        a.annotate(idn(v, 0), (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=BLUE)
    a.set(title="(a) Garis kemiskinan", xlabel="Tahun", ylabel="Rp ribu per kapita per bulan", xticks=list(gk.tahun))
    a.yaxis.set_major_formatter(idn_axis(0))
    a.legend(frameon=False, loc="lower right")
    b = ax[1]
    b.bar(gk.tahun, gk.selisih_persen, color=GREEN, width=.65)
    for x, v in zip(gk.tahun, gk.selisih_persen):
        b.text(x, v + .08, idn(v, 1), ha="center", fontsize=7)
    b.set(title="(b) Selisih kota-desa", xlabel="Tahun", ylabel="% terhadap perdesaan", xticks=list(gk.tahun))
    b.set_ylim(0, 3.2)
    b.yaxis.set_major_formatter(idn_axis(1))
    fig.tight_layout()
    save(fig, "gambar14_garis_kemiskinan.png")

    # ---- P3KE
    p = p3.sort_values("jumlah_individu", ascending=False).reset_index(drop=True)
    p["nama"] = p.bps_nama_kecamatan.str.title()
    r_p = stats.pearsonr(p.jumlah_kelurahan, p.jumlah_individu)
    r_p2 = stats.pearsonr(p.jumlah_kelurahan, p.rata2_individu_per_kelurahan)
    SUMMARY["p3ke"] = {"n_kecamatan": len(p), "total": int(p.jumlah_individu.sum()), "kelurahan": int(p.jumlah_kelurahan.sum()),
                       "max": [p.nama[0], int(p.jumlah_individu[0])], "min": [p.nama.iloc[-1], int(p.jumlah_individu.iloc[-1])],
                       "median": float(p.jumlah_individu.median()), "mean": float(p.jumlah_individu.mean()),
                       "r_kel_ind": [float(r_p[0]), float(r_p[1])], "r_kel_rata": [float(r_p2[0]), float(r_p2[1])],
                       "top3": p.head(3)[["nama", "jumlah_individu"]].values.tolist(),
                       "top5_share": float(p.head(5).jumlah_individu.sum() / p.jumlah_individu.sum() * 100),
                       "rata_max": [p.loc[p.rata2_individu_per_kelurahan.idxmax(), "nama"], float(p.rata2_individu_per_kelurahan.max())],
                       "rata_min": [p.loc[p.rata2_individu_per_kelurahan.idxmin(), "nama"], float(p.rata2_individu_per_kelurahan.min())],
                       "skew": float(stats.skew(p.jumlah_individu, bias=False))}
    p.drop(columns="nama").to_csv(TAB_DIR / "tabel_p3ke_urut.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.barh(p.nama[::-1], p.jumlah_individu[::-1], color=BLUE, height=.72)
    a.axvline(p.jumlah_individu.median(), color=RED, ls="--", lw=1, label=f"Median {idn(p.jumlah_individu.median(), 0)}")
    a.set(title="(a) Individu P3KE per kecamatan", xlabel="Jumlah individu (orang)")
    a.tick_params(axis="y", labelsize=6.5)
    a.xaxis.set_major_formatter(idn_axis(0))
    a.legend(frameon=False, loc="lower right")
    a.grid(axis="y", visible=False)
    b = ax[1]
    b.scatter(p.jumlah_kelurahan, p.jumlah_individu, color=ORANGE, s=34, edgecolor="white", lw=.5)
    mm = np.polyfit(p.jumlah_kelurahan, p.jumlah_individu, 1)
    xs = np.linspace(p.jumlah_kelurahan.min(), p.jumlah_kelurahan.max(), 20)
    b.plot(xs, np.polyval(mm, xs), color=RED, lw=1.5, label=f"r = {idn(r_p[0])}")
    b.set(title="(b) Kelurahan vs individu", xlabel="Jumlah kelurahan", ylabel="Jumlah individu (orang)")
    b.yaxis.set_major_formatter(idn_axis(0))
    b.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "gambar15_p3ke_kecamatan.png")


# --------------------------------------------------------------------------
# 9. PROGRAM UTAMA
# --------------------------------------------------------------------------
def main():
    df, gk, p3, out = load_data()
    banner(df, gk, p3)
    distribusi(df)
    tren(df)
    visual_wilayah(df)
    scatter(df)
    korelasi(df)
    pendukung(df, gk, p3)
    with open(TAB_DIR / "eda_ringkasan.json", "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=1, default=float)
    ringkasan_konsol()


if __name__ == "__main__":
    main()
