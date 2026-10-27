#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modeling.py  -  BAB V (Modeling) dan BAB VI (Evaluation)
Studi kasus: Pemetaan dan Prediksi Kemiskinan Daerah Berdasarkan Indikator Kesehatan

Membaca  : data_gabungan.csv (hasil preprocessing.py, 162 baris = 27 kab/kota x 2019-2024)
Menghasilkan:
  1. Clustering K-Means (27 kabupaten/kota) + Elbow, Silhouette, Davies-Bouldin
  2. Regresi: Linear Regression, Decision Tree, Random Forest (+ 3 baseline)
     - pembagian berdasarkan waktu: latih 2019-2023, uji 2024
     - tuning hiperparameter dengan validasi expanding-window (2021, 2022, 2023)
     - uji ketahanan rolling-origin (2021-2024) dan bootstrap antar-wilayah
  3. Tabel (CSV) ke  outputs/tables  dan gambar (PNG) ke  outputs/figures

Cara menjalankan (dari folder proyek):
    python scripts/modeling.py
atau  python scripts/modeling.py --data PATH_FOLDER_DATA --out PATH_FOLDER_OUTPUT
"""
import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_samples,
                             silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. PENGATURAN
# --------------------------------------------------------------------------
RS = 42                       # random_state tunggal
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEAR = 2024
VAL_YEARS = [2021, 2022, 2023]          # validasi expanding-window untuk tuning
ROLL_YEARS = [2021, 2022, 2023, 2024]   # uji ketahanan rolling-origin
N_BOOT = 2000
K_RANGE = list(range(2, 9))
JENIS = ["reguler", "penugasan", "bok", "jampersal", "akreditasi"]

GRID = {
    "DecisionTree": {"max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 3, 5, 10]},
    "RandomForest": {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5], "max_features": [1.0, 0.5]},
    "LinearRegression": {},
}
FIXED = {   # parameter tetap untuk uji ketahanan rolling-origin (tanpa tuning)
    "DecisionTree": {"max_depth": 4, "min_samples_leaf": 3},
    "RandomForest": {"max_depth": None, "min_samples_leaf": 3, "max_features": 1.0},
    "LinearRegression": {},
}
MODEL_LABEL = {"LinearRegression": "Linear Regression", "DecisionTree": "Decision Tree",
               "RandomForest": "Random Forest"}

DAK_FEATS = ["log_total_alokasi"] + ["log_" + j for j in JENIS] + ["alokasi_nol"]
SCENARIOS = {
    "S1": {"nama": "DAK saja", "fitur": DAK_FEATS},
    "S2": {"nama": "DAK + karakteristik wilayah", "fitur": DAK_FEATS + ["is_kabupaten", "tahun"]},
    "S3": {"nama": "Riwayat indeks (tanpa DAK)", "fitur": ["indeks_sebelumnya", "is_kabupaten", "tahun"]},
    "S4": {"nama": "Riwayat indeks + DAK", "fitur": ["indeks_sebelumnya", "is_kabupaten", "tahun"] + DAK_FEATS},
}
CL_FEATS = ["ik_rata", "ik_sd", "ik_perubahan", "log_dak"]
CL_LABEL = {"ik_rata": "Rata-rata indeks", "ik_sd": "Volatilitas (SD)", "ik_perubahan": "Perubahan 2019–2024",
            "log_dak": "log DAK (median)"}

BLUE, ORANGE, GREEN, RED, GREY = "#1F4E79", "#D9822B", "#3B8B5A", "#B23A48", "#6B7280"
CL_COL = ["#3B8B5A", "#D9822B", "#B23A48", "#1F4E79", "#7B5EA7", "#6B7280"]
DPI = 200
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 100,
    "savefig.dpi": DPI, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def idn(x, nd=2):
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def idn_axis(nd=2):
    return FuncFormatter(lambda v, _: idn(v, nd))


# --------------------------------------------------------------------------
# 2. PATH
# --------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="folder berisi data_gabungan.csv")
ap.add_argument("--out", default=None, help="folder keluaran (akan dibuat: figures/ dan tables/)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
cands = [Path(args.data)] if args.data else []
cands += [ROOT / "data" / "processed", ROOT / "data", ROOT, Path.cwd()]
DATA_DIR = next((c for c in cands if (Path(c) / "data_gabungan.csv").exists()), None)
if DATA_DIR is None:
    sys.exit("data_gabungan.csv tidak ditemukan. Jalankan preprocessing.py lebih dulu atau gunakan --data")
OUT = Path(args.out) if args.out else ROOT / "outputs"
FIG_DIR, TAB_DIR = OUT / "figures", OUT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = {}


def savefig(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  [gambar] {name}")


def savetab(df, name, index=False):
    df.to_csv(TAB_DIR / name, index=index)
    print(f"  [tabel ] {name}")


def short(name):
    return name.title().replace("Kabupaten ", "Kab. ")


# --------------------------------------------------------------------------
# 3. DATA
# --------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA_DIR / "data_gabungan.csv")
    df["wilayah"] = df["nama_kabupaten_kota"].map(short)
    df["is_kabupaten"] = df["nama_kabupaten_kota"].str.startswith("KABUPATEN").astype(int)
    y = "indeks_keparahan_kemiskinan"
    df["indeks_sebelumnya"] = (df[y] - df["delta_indeks"]).round(2)   # 2019 -> nilai 2018
    for j in JENIS:
        df["log_" + j] = np.log1p(df[j])
    df["alokasi_nol"] = (df["total_alokasi"] == 0).astype(int)
    return df.sort_values(["kode_kabupaten_kota", "tahun"]).reset_index(drop=True)


df = load()
Y = "indeks_keparahan_kemiskinan"
print(f"Data: {df.shape[0]} baris, {df['kode_kabupaten_kota'].nunique()} kab/kota, tahun {df.tahun.min()}-{df.tahun.max()}")
assert len(df) == 162 and df.isna().sum().sum() == 0

# ==========================================================================
# BAGIAN A. CLUSTERING K-MEANS
# ==========================================================================
print("\n[A] Clustering K-Means")


def region_table(d):
    g = d.groupby(["kode_kabupaten_kota", "wilayah"])
    r = g.agg(ik_rata=(Y, "mean"), ik_sd=(Y, "std"), is_kab=("is_kabupaten", "first")).reset_index()
    ik19 = d[d.tahun == 2019].set_index("kode_kabupaten_kota")[Y]
    ik24 = d[d.tahun == 2024].set_index("kode_kabupaten_kota")[Y]
    r["ik_perubahan"] = r["kode_kabupaten_kota"].map(ik24) - r["kode_kabupaten_kota"].map(ik19)
    med = d[d.total_alokasi > 0].groupby("kode_kabupaten_kota")["total_alokasi"].median() / 1e9
    r["dak_median_miliar"] = r["kode_kabupaten_kota"].map(med)
    r["log_dak"] = np.log1p(r["dak_median_miliar"])
    return r


reg = region_table(df)
assert reg[CL_FEATS].notna().all().all() and len(reg) == 27
Z = StandardScaler().fit_transform(reg[CL_FEATS])

rows = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=50, random_state=RS).fit(Z)
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_),
                 "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                 "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
                 "ukuran_klaster_terkecil": int(np.bincount(km.labels_).min())})
ksel = pd.DataFrame(rows)
K_BEST = int(ksel.loc[ksel["silhouette"].idxmax(), "k"])
savetab(ksel, "tabel_pemilihan_k.csv")
print(ksel.round(3).to_string(index=False))
print("k terbaik (silhouette maksimum):", K_BEST)

km = KMeans(n_clusters=K_BEST, n_init=50, random_state=RS).fit(Z)
reg["klaster_raw"] = km.labels_
order = reg.groupby("klaster_raw")["ik_rata"].mean().sort_values().index.tolist()
reg["klaster"] = reg["klaster_raw"].map({o: i + 1 for i, o in enumerate(order)})
sil_s = silhouette_samples(Z, km.labels_)
reg["silhouette"] = sil_s
ward = AgglomerativeClustering(n_clusters=K_BEST, linkage="ward").fit(Z)
ari_ward = adjusted_rand_score(km.labels_, ward.labels_)

# stabilitas: bootstrap 27 wilayah, ARI terhadap hasil utama
rng = np.random.default_rng(RS)
aris = []
for _ in range(300):
    idx = rng.choice(len(Z), len(Z), replace=True)
    kb = KMeans(n_clusters=K_BEST, n_init=10, random_state=RS).fit(Z[idx])
    aris.append(adjusted_rand_score(km.labels_[idx], kb.predict(Z[idx]) if False else kb.labels_))
# ARI pada titik yang terambil: bandingkan label bootstrap dengan label utama pada titik yang sama
ari_boot = float(np.mean(aris))

prof = reg.groupby("klaster").agg(
    jumlah=("wilayah", "size"), ik_rata=("ik_rata", "mean"), ik_sd=("ik_sd", "mean"),
    ik_perubahan=("ik_perubahan", "mean"), dak_median_miliar=("dak_median_miliar", "mean"),
    pangsa_kabupaten=("is_kab", "mean"), silhouette=("silhouette", "mean")).reset_index()
prof["anggota"] = prof["klaster"].map(
    lambda c: "; ".join(reg.loc[reg.klaster == c].sort_values("ik_rata")["wilayah"]))
savetab(prof, "tabel_profil_klaster.csv")
savetab(reg[["kode_kabupaten_kota", "wilayah", "klaster", "ik_rata", "ik_sd", "ik_perubahan",
             "dak_median_miliar", "is_kab", "silhouette"]].sort_values(["klaster", "ik_rata"]),
        "tabel_anggota_klaster.csv")
print(prof.drop(columns="anggota").round(3).to_string(index=False))
print("ARI KMeans vs Ward:", round(ari_ward, 3), "| ARI bootstrap rata-rata:", round(ari_boot, 3))

SUMMARY["klaster"] = {
    "k_terbaik": K_BEST, "silhouette": float(silhouette_score(Z, km.labels_)),
    "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)), "ari_ward": float(ari_ward),
    "ari_bootstrap": ari_boot, "inertia": float(km.inertia_),
    "silhouette_k3": float(ksel.loc[ksel.k == 3, "silhouette"].iloc[0]),
}
# k = 3 sebagai pembanding (contoh rendah-sedang-tinggi pada panduan dosen)
km3 = KMeans(n_clusters=3, n_init=50, random_state=RS).fit(Z)
reg["klaster_k3_raw"] = km3.labels_
o3 = reg.groupby("klaster_k3_raw")["ik_rata"].mean().sort_values().index.tolist()
reg["klaster_k3"] = reg["klaster_k3_raw"].map({o: i + 1 for i, o in enumerate(o3)})
prof3 = reg.groupby("klaster_k3").agg(jumlah=("wilayah", "size"), ik_rata=("ik_rata", "mean"),
                                       ik_sd=("ik_sd", "mean"), ik_perubahan=("ik_perubahan", "mean"),
                                       dak_median_miliar=("dak_median_miliar", "mean"),
                                       pangsa_kabupaten=("is_kab", "mean")).reset_index()
prof3["anggota"] = prof3["klaster_k3"].map(
    lambda c: "; ".join(reg.loc[reg.klaster_k3 == c].sort_values("ik_rata")["wilayah"]))
savetab(prof3, "tabel_profil_klaster_k3.csv")
SUMMARY["klaster"]["ari_k_best_vs_k3"] = float(adjusted_rand_score(reg["klaster"], reg["klaster_k3"]))

# --- gambar: elbow + silhouette
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2))
ax[0].plot(ksel.k, ksel.inertia, "-o", color=BLUE, ms=4)
ax[0].set_xlabel("Jumlah klaster (k)"); ax[0].set_ylabel("Inertia (WCSS)")
ax[0].set_title("(a) Metode Elbow"); ax[0].yaxis.set_major_formatter(idn_axis(0))
ax[1].plot(ksel.k, ksel.silhouette, "-s", color=ORANGE, ms=4)
ax[1].axvline(K_BEST, color=GREY, ls="--", lw=1)
ax[1].annotate(f"maks. k = {K_BEST}\n({idn(ksel.silhouette.max(), 3)})",
               (K_BEST, ksel.silhouette.max()), xytext=(K_BEST + 1.5, ksel.silhouette.max() - 0.005),
               fontsize=8, color=GREY)
ax[1].set_xlabel("Jumlah klaster (k)"); ax[1].set_ylabel("Silhouette score")
ax[1].set_title("(b) Silhouette Score"); ax[1].yaxis.set_major_formatter(idn_axis(2))
for a in ax:
    a.set_xticks(K_RANGE)
fig.tight_layout()
savefig(fig, "gambar5_2_elbow_silhouette.png")

# --- gambar: peta klaster (PCA) + profil
pca = PCA(n_components=2, random_state=RS).fit(Z)
P = pca.transform(Z)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
for c in sorted(reg.klaster.unique()):
    m = (reg.klaster == c).values
    ax[0].scatter(P[m, 0], P[m, 1], s=34, color=CL_COL[c - 1], edgecolor="white", lw=.5, label=f"Klaster {c}", zorder=3)
for i, r in reg.iterrows():
    ax[0].annotate(r.wilayah.replace("Kab. ", "").replace("Kota ", "K. "), (P[i, 0], P[i, 1]),
                   xytext=(3, 3), textcoords="offset points", fontsize=5, color="#333333")
ax[0].set_xlabel(f"PC1 ({idn(pca.explained_variance_ratio_[0] * 100, 1)}%)")
ax[0].set_ylabel(f"PC2 ({idn(pca.explained_variance_ratio_[1] * 100, 1)}%)")
ax[0].set_title("(a) Sebaran 27 kab/kota pada bidang PCA"); ax[0].legend(loc="best", frameon=True)
zdf = pd.DataFrame(Z, columns=CL_FEATS); zdf["klaster"] = reg.klaster.values
hm = zdf.groupby("klaster")[CL_FEATS].mean()
hm.columns = [CL_LABEL[c] for c in CL_FEATS]
sns.heatmap(hm, annot=np.vectorize(lambda v: idn(v, 2))(hm.values), fmt="", cmap="RdBu_r", center=0, vmin=-2.2, vmax=2.2,
            linewidths=.5, linecolor="white", cbar=False, ax=ax[1], annot_kws={"fontsize": 8})
ax[1].set_title("(b) Profil klaster (rata-rata z-score)"); ax[1].set_ylabel("Klaster"); ax[1].set_xlabel("")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=35, ha="right", fontsize=7)
fig.tight_layout()
savefig(fig, "gambar5_3_peta_klaster.png")

# --- gambar: silhouette per sampel
fig, ax = plt.subplots(figsize=(5.4, 4.6))
y0 = 0
for c in sorted(reg.klaster.unique()):
    s = reg.loc[reg.klaster == c, "silhouette"].sort_values().values
    ax.barh(np.arange(y0, y0 + len(s)), s, color=CL_COL[c - 1], edgecolor="white", height=.9, label=f"Klaster {c}")
    ax.text(-0.04, y0 + len(s) / 2, str(c), ha="right", va="center", fontsize=8, fontweight="bold")
    y0 += len(s) + 1
ax.axvline(SUMMARY["klaster"]["silhouette"], color=GREY, ls="--", lw=1)
ax.text(SUMMARY["klaster"]["silhouette"] + .01, y0 - 1.5, f"rata-rata {idn(SUMMARY['klaster']['silhouette'], 3)}", fontsize=8, color=GREY)
ax.set_yticks([]); ax.set_xlabel("Koefisien silhouette"); ax.set_title("Silhouette per kabupaten/kota")
ax.legend(loc="lower right"); ax.xaxis.set_major_formatter(idn_axis(1))
fig.tight_layout()
savefig(fig, "gambar6_5_silhouette_sampel.png")

# ==========================================================================
# BAGIAN B. REGRESI
# ==========================================================================
print("\n[B] Regresi")


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), "n": int(len(y))}


def make_model(name, params):
    if name == "LinearRegression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "DecisionTree":
        return DecisionTreeRegressor(random_state=RS, **params)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=1, **params)
    raise ValueError(name)


def combos(name):
    g = GRID[name]
    if not g:
        return [{}]
    keys = list(g)
    return [dict(zip(keys, v)) for v in product(*[g[k] for k in keys])]


def fit_pred(name, params, feats, tr, te):
    m = make_model(name, params)
    m.fit(tr[feats], tr[Y])
    return m, m.predict(te[feats])


# --- baseline
hist = {}   # riwayat per wilayah: nilai 2018 (dari indeks_sebelumnya 2019) + 2019..2024
for kode, g in df.groupby("kode_kabupaten_kota"):
    g = g.sort_values("tahun")
    hist[kode] = {2018: float(g.iloc[0]["indeks_sebelumnya"]), **{int(t): float(v) for t, v in zip(g.tahun, g[Y])}}


def baseline_pred(kind, te, tr):
    if kind == "Rata-rata latih":
        return np.full(len(te), tr[Y].mean())
    if kind == "Persistensi (indeks tahun lalu)":
        return te["indeks_sebelumnya"].values
    if kind == "Rata-rata historis wilayah":
        out = []
        for _, r in te.iterrows():
            vals = [v for t, v in hist[r.kode_kabupaten_kota].items() if t < r.tahun]
            out.append(np.mean(vals))
        return np.array(out)
    raise ValueError(kind)


BASELINES = ["Rata-rata latih", "Persistensi (indeks tahun lalu)", "Rata-rata historis wilayah"]

# --- tuning (expanding window) dan evaluasi holdout 2024
train = df[df.tahun.isin(TRAIN_YEARS)].copy()
test = df[df.tahun == TEST_YEAR].copy()
print(f"Latih {len(train)} baris (2019-2023); uji {len(test)} baris (2024)")


def tune(name, feats):
    best = None
    for p in combos(name):
        errs = []
        for v in VAL_YEARS:
            tr = train[train.tahun < v]; va = train[train.tahun == v]
            _, pr = fit_pred(name, p, feats, tr, va)
            errs.append(np.sqrt(mean_squared_error(va[Y], pr)))
        score = float(np.mean(errs))
        if best is None or score < best[0] - 1e-12:
            best = (score, p)
    return best


hold_rows, hold_pred, final_models = [], {}, {}
for sc, spec in SCENARIOS.items():
    for name in ["LinearRegression", "DecisionTree", "RandomForest"]:
        cv_rmse, params = tune(name, spec["fitur"])
        m, pr = fit_pred(name, params, spec["fitur"], train, test)
        met = metrics(test[Y], pr)
        hold_rows.append({"skenario": sc, "nama_skenario": spec["nama"], "model": MODEL_LABEL[name], "model_key": name,
                          "cv_rmse": cv_rmse, "param": json.dumps(params, default=str), **met})
        hold_pred[(sc, name)] = pr
        final_models[(sc, name)] = m
        print(f"  {sc} {name:17s} cvRMSE={cv_rmse:.4f}  uji: MAE={met['MAE']:.4f} RMSE={met['RMSE']:.4f} R2={met['R2']:+.3f}  {params}")
hold = pd.DataFrame(hold_rows)
base_hold = []
for b in BASELINES:
    pr = baseline_pred(b, test, train)
    hold_pred[("BASE", b)] = pr
    base_hold.append({"skenario": "-", "nama_skenario": "Baseline", "model": b, "model_key": b, "cv_rmse": np.nan,
                      "param": "", **metrics(test[Y], pr)})
base_hold = pd.DataFrame(base_hold)
savetab(pd.concat([hold, base_hold], ignore_index=True), "tabel_hasil_uji_2024.csv")

# pemilihan model akhir: RMSE validasi terkecil (tanpa melihat data uji)
best_row = hold.loc[hold["cv_rmse"].idxmin()]
B_SC, B_NAME = best_row["skenario"], best_row["model_key"]
print("Model terpilih (validasi):", B_SC, B_NAME, "cv_rmse=", round(best_row.cv_rmse, 4))
tune_rows = []
for sc, spec in SCENARIOS.items():
    for name in ["DecisionTree", "RandomForest"]:
        pass
savetab(hold[["skenario", "nama_skenario", "model", "param", "cv_rmse"]], "tabel_hasil_tuning.csv")

# --- rolling-origin (parameter tetap)
roll_pred = {}
roll_rows = []
roll_idx = df[df.tahun.isin(ROLL_YEARS)].index
base_df = df.loc[roll_idx].copy()
for sc, spec in SCENARIOS.items():
    for name in ["LinearRegression", "DecisionTree", "RandomForest"]:
        pr_all = pd.Series(index=roll_idx, dtype=float)
        for v in ROLL_YEARS:
            tr = df[df.tahun < v]; te = df[df.tahun == v]
            _, pr = fit_pred(name, FIXED[name], spec["fitur"], tr, te)
            pr_all.loc[te.index] = pr
        roll_pred[(sc, name)] = pr_all.values
        met = metrics(base_df[Y], pr_all.values)
        per = {f"RMSE_{v}": float(np.sqrt(mean_squared_error(df[df.tahun == v][Y], pr_all.loc[df[df.tahun == v].index]))) for v in ROLL_YEARS}
        roll_rows.append({"skenario": sc, "nama_skenario": spec["nama"], "model": MODEL_LABEL[name], "model_key": name, **met, **per})
for b in BASELINES:
    pr_all = pd.Series(index=roll_idx, dtype=float)
    for v in ROLL_YEARS:
        tr = df[df.tahun < v]; te = df[df.tahun == v]
        pr_all.loc[te.index] = baseline_pred(b, te, tr)
    roll_pred[("BASE", b)] = pr_all.values
    met = metrics(base_df[Y], pr_all.values)
    per = {f"RMSE_{v}": float(np.sqrt(mean_squared_error(df[df.tahun == v][Y], pr_all.loc[df[df.tahun == v].index]))) for v in ROLL_YEARS}
    roll_rows.append({"skenario": "-", "nama_skenario": "Baseline", "model": b, "model_key": b, **met, **per})
roll = pd.DataFrame(roll_rows)
savetab(roll, "tabel_rolling_origin.csv")
print(roll[["skenario", "model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))

# --- bootstrap antar-wilayah (pada prediksi rolling-origin)
y_roll = base_df[Y].values
codes = base_df["kode_kabupaten_kota"].values
region_idx = {c: np.where(codes == c)[0] for c in np.unique(codes)}
reg_list = list(region_idx)
rng = np.random.default_rng(RS)
boot_sets = [np.concatenate([region_idx[c] for c in rng.choice(reg_list, len(reg_list), replace=True)]) for _ in range(N_BOOT)]


def rmse_of(idx, pr):
    return float(np.sqrt(np.mean((y_roll[idx] - pr[idx]) ** 2)))


def boot_diff(pa, pb):
    d = np.array([rmse_of(i, pa) - rmse_of(i, pb) for i in boot_sets])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ci(pr):
    v = np.array([rmse_of(i, pr) for i in boot_sets])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


comp_rows = []
def add_comp(label, a, b):
    pa, pb = roll_pred[a], roll_pred[b]
    ra, rb = rmse_of(np.arange(len(y_roll)), pa), rmse_of(np.arange(len(y_roll)), pb)
    mean, lo, hi = boot_diff(pa, pb)
    comp_rows.append({"perbandingan": label, "rmse_A": ra, "rmse_B": rb, "selisih_A_minus_B": ra - rb,
                      "ci_low": lo, "ci_high": hi, "A_lebih_baik": bool(hi < 0), "B_lebih_baik": bool(lo > 0)})


for name in ["LinearRegression", "DecisionTree", "RandomForest"]:
    add_comp(f"{MODEL_LABEL[name]}: S4 (riwayat+DAK) vs S3 (riwayat saja)", ("S4", name), ("S3", name))
for name in ["LinearRegression", "DecisionTree", "RandomForest"]:
    add_comp(f"{MODEL_LABEL[name]}: S1 (DAK saja) vs rata-rata latih", ("S1", name), ("BASE", "Rata-rata latih"))
best_s3_name = roll[(roll.skenario == "S3")].sort_values("RMSE").iloc[0]["model_key"]
add_comp(f"{MODEL_LABEL[best_s3_name]} S3 vs persistensi", ("S3", best_s3_name), ("BASE", "Persistensi (indeks tahun lalu)"))
add_comp(f"{MODEL_LABEL[best_s3_name]} S3 vs rata-rata historis wilayah", ("S3", best_s3_name), ("BASE", "Rata-rata historis wilayah"))
add_comp("Persistensi vs rata-rata historis wilayah", ("BASE", "Persistensi (indeks tahun lalu)"), ("BASE", "Rata-rata historis wilayah"))
add_comp(f"Model terpilih ({B_SC} {MODEL_LABEL[B_NAME]}) vs persistensi", (B_SC, B_NAME), ("BASE", "Persistensi (indeks tahun lalu)"))
add_comp(f"Model terpilih ({B_SC} {MODEL_LABEL[B_NAME]}) vs rata-rata historis wilayah", (B_SC, B_NAME), ("BASE", "Rata-rata historis wilayah"))
add_comp(f"Model terpilih ({B_SC} {MODEL_LABEL[B_NAME]}) vs rata-rata latih", (B_SC, B_NAME), ("BASE", "Rata-rata latih"))
comp = pd.DataFrame(comp_rows)
savetab(comp, "tabel_perbandingan_bootstrap.csv")
print(comp.round(4).to_string(index=False))
ci_rows = []
for k, pr in roll_pred.items():
    lo, hi = boot_ci(pr)
    ci_rows.append({"skenario": k[0], "model": k[1], "rmse": rmse_of(np.arange(len(y_roll)), pr), "ci_low": lo, "ci_high": hi})
savetab(pd.DataFrame(ci_rows), "tabel_rmse_ci.csv")

# --- permutation importance berkelompok (rolling-origin, model S4)
def group_importance(name, sc, reps=30):
    feats = SCENARIOS[sc]["fitur"]
    groups = {"Indeks tahun sebelumnya": ["indeks_sebelumnya"], "Tahun": ["tahun"], "Jenis wilayah (kab/kota)": ["is_kabupaten"],
              "Alokasi DAK (semua fitur DAK)": DAK_FEATS}
    groups = {g: [c for c in cols if c in feats] for g, cols in groups.items()}
    groups = {g: cols for g, cols in groups.items() if cols}
    rng = np.random.default_rng(RS)
    res = {g: [] for g in groups}
    for v in ROLL_YEARS:
        tr = df[df.tahun < v]; te = df[df.tahun == v]
        m = make_model(name, FIXED[name]).fit(tr[feats], tr[Y])
        base = np.sqrt(mean_squared_error(te[Y], m.predict(te[feats])))
        for g, cols in groups.items():
            inc = []
            for _ in range(reps):
                tx = te[feats].copy()
                perm = rng.permutation(len(tx))
                tx[cols] = tx[cols].values[perm]
                inc.append(np.sqrt(mean_squared_error(te[Y], m.predict(tx))) - base)
            res[g].append(np.mean(inc))
    return pd.DataFrame({"kelompok_fitur": list(res), "kenaikan_rmse": [np.mean(v) for v in res.values()],
                         "model": MODEL_LABEL[name], "skenario": sc})


imp = pd.concat([group_importance("RandomForest", "S4"), group_importance("LinearRegression", "S4")], ignore_index=True)
savetab(imp, "tabel_importance_kelompok.csv")
print(imp.round(4).to_string(index=False))

# koefisien Linear Regression (terstandar) skenario S4 pada data latih 2019-2023
lr = final_models[("S4", "LinearRegression")]
coef = pd.DataFrame({"fitur": SCENARIOS["S4"]["fitur"], "koef_terstandar": lr[-1].coef_})
savetab(coef, "tabel_koefisien_lr_s4.csv")
rf = final_models[("S4", "RandomForest")]
rfi = pd.DataFrame({"fitur": SCENARIOS["S4"]["fitur"], "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
savetab(rfi, "tabel_importance_rf_s4.csv")

# --- residual & anomali (model terpilih)
sel = (B_SC, B_NAME)
res_df = base_df[["kode_kabupaten_kota", "wilayah", "tahun", Y]].copy()
res_df["prediksi"] = roll_pred[sel]
res_df["residual"] = res_df[Y] - res_df["prediksi"]
sd = res_df["residual"].std(ddof=1)
res_df["z_residual"] = res_df["residual"] / sd
res_df["abs_res"] = res_df["residual"].abs()
anom = res_df.sort_values("abs_res", ascending=False).head(10)
savetab(anom.drop(columns="abs_res"), "tabel_anomali_residual.csv")
reg_res = res_df.groupby("wilayah").agg(mae=("abs_res", "mean"), bias=("residual", "mean"), n=("residual", "size")).reset_index().sort_values("mae", ascending=False)
savetab(reg_res, "tabel_residual_per_wilayah.csv")
print(anom[["wilayah", "tahun", Y, "prediksi", "residual", "z_residual"]].round(3).to_string(index=False))
print(reg_res.head(6).round(3).to_string(index=False))

hold_df = test[["kode_kabupaten_kota", "wilayah", "tahun", Y]].copy()
hold_df["prediksi"] = hold_pred[sel]; hold_df["residual"] = hold_df[Y] - hold_df["prediksi"]
savetab(hold_df, "tabel_prediksi_2024_model_terpilih.csv")

# ==========================================================================
# GAMBAR REGRESI
# ==========================================================================
# alur pemodelan
fig, ax = plt.subplots(figsize=(7.0, 4.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=color, ec="#333333", lw=.8))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=6.6, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=6.6, linespacing=1.25)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color="#333333", lw=.9))
ax.text(0.05, 5.82, "A. Jalur regresi (memprediksi indeks keparahan)", fontsize=8, fontweight="bold", color=BLUE, va="top")
xs = [0.05, 2.05, 4.05, 6.05, 8.05]; w, h = 1.8, 1.7
B1 = [("Data gabungan", "162 observasi\n27 kab/kota\n2019–2024"), ("Rekayasa fitur", "indeks tahun lalu,\nDAK (log),\njenis wilayah, tahun"),
      ("Split waktu", "latih 2019–2023\n(135 obs.)\nuji 2024 (27 obs.)"), ("Latih + tuning", "LR · DT · RF\nvalidasi expanding\nwindow 2021–2023"),
      ("Evaluasi", "MAE · RMSE · R²\nbaseline, rolling-\norigin, bootstrap")]
for x, (t, b) in zip(xs, B1):
    box(x, 3.55, w, h, t, b, "#EAF1F8")
for i in range(4):
    arrow(xs[i] + w + 0.03, 4.4, xs[i + 1] - 0.03, 4.4)
ax.text(0.05, 3.0, "B. Jalur clustering (mengelompokkan wilayah)", fontsize=8, fontweight="bold", color=ORANGE, va="top")
B2 = [("Agregasi wilayah", "27 baris: rata-rata,\nSD, perubahan,\nlog DAK"), ("Standardisasi", "z-score\n(4 fitur)"),
      ("K-Means", "k = 2 sampai 8\nn_init = 50\nrandom_state = 42"), ("Pemilihan k", "Elbow, Silhouette,\nDavies–Bouldin"),
      ("Profil klaster", "rata-rata fitur,\nanggota, validasi\nWard dan bootstrap")]
for x, (t, b) in zip(xs, B2):
    box(x, 0.6, w, h, t, b, "#FCF1E4")
for i in range(4):
    arrow(xs[i] + w + 0.03, 1.45, xs[i + 1] - 0.03, 1.45)
savefig(fig, "gambar5_1_alur_pemodelan.png")

# perbandingan RMSE uji 2024 dan rolling
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
palet = {"Linear Regression": BLUE, "Decision Tree": GREEN, "Random Forest": ORANGE}
for a, (tab, ttl) in zip(ax, [(hold, "(a) Data uji 2024 (n = 27)"), (roll[roll.skenario != "-"], "(b) Rolling-origin 2021–2024 (n = 108)")]):
    sub = tab.copy()
    sns.barplot(data=sub, x="skenario", y="RMSE", hue="model", palette=palet, order=["S1", "S2", "S3", "S4"], ax=a,
                edgecolor="white", linewidth=.6)
    bsrc = base_hold if a is ax[0] else roll[roll.skenario == "-"]
    for _, b in bsrc.iterrows():
        if b["model"] == "Persistensi (indeks tahun lalu)":
            a.axhline(b["RMSE"], color=RED, ls="--", lw=1); a.text(3.45, b["RMSE"] + .002, "persistensi", color=RED, fontsize=7, ha="right")
        if b["model"] == "Rata-rata latih":
            a.axhline(b["RMSE"], color=GREY, ls=":", lw=1.2); a.text(3.45, b["RMSE"] + .002, "rata-rata latih", color=GREY, fontsize=7, ha="right")
    a.set_title(ttl); a.set_xlabel("Skenario fitur"); a.set_ylabel("RMSE (poin indeks)" if a is ax[0] else "")
    a.yaxis.set_major_formatter(idn_axis(3)); a.legend_.remove() if a.legend_ else None
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
savefig(fig, "gambar6_1_perbandingan_rmse.png")

# aktual vs prediksi
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6))
lim = [0, max(res_df[Y].max(), res_df.prediksi.max()) + .04]
for a, d, ttl in [(ax[0], hold_df, "(a) Data uji 2024"), (ax[1], res_df, "(b) Rolling-origin 2021–2024")]:
    a.plot(lim, lim, color=GREY, ls="--", lw=1, label="prediksi sempurna")
    a.scatter(d.prediksi, d[Y], s=24, color=BLUE, alpha=.75, edgecolor="white", lw=.4)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("Indeks prediksi"); a.set_ylabel("Indeks aktual"); a.set_title(ttl)
    a.xaxis.set_major_formatter(idn_axis(1)); a.yaxis.set_major_formatter(idn_axis(1))
    mm = metrics(d[Y], d.prediksi)
    a.text(.04, .93, f"R² = {idn(mm['R2'], 2)}\nRMSE = {idn(mm['RMSE'], 3)}", transform=a.transAxes, va="top", fontsize=8,
           bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC"))
ax[0].legend(loc="lower right")
fig.tight_layout()
savefig(fig, "gambar6_2_aktual_vs_prediksi.png")

# residual + anomali
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
ax[0].axhline(0, color=GREY, lw=1)
ax[0].scatter(res_df.prediksi, res_df.residual, s=22, color=BLUE, alpha=.7, edgecolor="white", lw=.4)
for _, r in anom.head(5).iterrows():
    ax[0].annotate(f"{r.wilayah.replace('Kab. ', '').replace('Kota ', 'K. ')} {int(r.tahun)}", (r.prediksi, r.residual),
                   xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=RED)
ax[0].set_xlabel("Indeks prediksi"); ax[0].set_ylabel("Residual (aktual − prediksi)"); ax[0].set_title("(a) Residual vs prediksi")
ax[0].xaxis.set_major_formatter(idn_axis(1)); ax[0].yaxis.set_major_formatter(idn_axis(2))
tp = reg_res.head(8).iloc[::-1]
ax[1].barh(tp.wilayah.str.replace("Kab. ", "").str.replace("Kota ", "K. "), tp.mae, color=ORANGE, edgecolor="white")
for i, v in enumerate(tp.mae):
    ax[1].text(v + .002, i, idn(v, 3), va="center", fontsize=7)
ax[1].set_xlabel("MAE residual 2021–2024 (poin indeks)"); ax[1].set_title("(b) Wilayah dengan galat terbesar")
ax[1].xaxis.set_major_formatter(idn_axis(2)); ax[1].set_xlim(0, tp.mae.max() * 1.18)
fig.tight_layout()
savefig(fig, "gambar6_4_residual_anomali.png")

# importance berkelompok
fig, ax = plt.subplots(figsize=(6.2, 3.0))
imp_p = imp.copy()
order_g = imp_p[imp_p.model == "Random Forest"].sort_values("kenaikan_rmse")["kelompok_fitur"].tolist()
sns.barplot(data=imp_p, y="kelompok_fitur", x="kenaikan_rmse", hue="model", order=order_g[::-1],
            palette={"Random Forest": ORANGE, "Linear Regression": BLUE}, ax=ax, edgecolor="white")
ax.axvline(0, color=GREY, lw=1)
ax.set_xlabel("Kenaikan RMSE saat fitur diacak (rata-rata rolling-origin)"); ax.set_ylabel("")
ax.set_title("Kepentingan kelompok fitur (permutation importance, skenario S4)")
ax.xaxis.set_major_formatter(idn_axis(3)); ax.legend(title="", loc="lower right")
fig.tight_layout()
savefig(fig, "gambar6_3_importance.png")

# --------------------------------------------------------------------------
# RINGKASAN
# --------------------------------------------------------------------------
SUMMARY["regresi"] = {
    "n_latih": int(len(train)), "n_uji": int(len(test)), "n_rolling": int(len(roll_idx)),
    "model_terpilih": {"skenario": B_SC, "model": MODEL_LABEL[B_NAME], "cv_rmse": float(best_row.cv_rmse),
                       "uji_MAE": float(best_row.MAE), "uji_RMSE": float(best_row.RMSE), "uji_R2": float(best_row.R2),
                       "param": best_row.param},
    "std_target_uji": float(test[Y].std(ddof=1)), "mean_target_uji": float(test[Y].mean()),
    "std_residual_rolling": float(sd),
}
json.dump(SUMMARY, open(TAB_DIR / "modeling_ringkasan.json", "w"), ensure_ascii=False, indent=2, default=float)
print("\nSelesai. Tabel:", TAB_DIR, "| Gambar:", FIG_DIR)
