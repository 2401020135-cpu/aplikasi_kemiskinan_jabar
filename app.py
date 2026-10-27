# -*- coding: utf-8 -*-
"""
Aplikasi Pemetaan dan Prediksi Kemiskinan Daerah Berdasarkan Indikator Kesehatan
Project 2 - Kelompok 3 (Project Data Sains, UMRAH)

Jalankan:  streamlit run app.py
Struktur : app.py, data/*.csv, outputs/tables/*.csv + modeling_ringkasan.json
           (dua folder terakhir dihasilkan oleh scripts/preprocessing.py dan scripts/modeling.py)

Catatan konsistensi dengan modeling.py:
- Tahun target 2021-2023 (rolling-origin) memakai parameter TETAP (FIXED), karena parameter
  hasil tuning dipilih dari validasi 2021-2023 sehingga akan terlalu optimistis bila dipakai lagi.
- Tahun target 2024 (holdout) dan 2025 (proyeksi) memakai parameter hasil tuning (tabel_hasil_uji_2024.csv).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).resolve().parent
DATA, TAB = BASE / "data", BASE / "outputs" / "tables"
RS = 42
TEST_YEAR = 2024
Y = "indeks_keparahan_kemiskinan"
JENIS = ["reguler", "penugasan", "bok", "jampersal", "akreditasi"]
DAK_FEATS = ["log_total_alokasi"] + ["log_" + j for j in JENIS] + ["alokasi_nol"]
S4 = ["indeks_sebelumnya", "is_kabupaten", "tahun"] + DAK_FEATS
CL_FEATS = ["ik_rata", "ik_sd", "ik_perubahan", "log_dak"]
CL_LABEL = {"ik_rata": "Rata-rata indeks", "ik_sd": "Volatilitas (SD)",
            "ik_perubahan": "Perubahan 2019–2024", "log_dak": "log DAK (median)"}
CL_COL = ["#3B8B5A", "#D9822B", "#B23A48", "#1F4E79", "#7B5EA7", "#6B7280"]  # sama dengan gambar laporan

# Parameter tetap untuk uji rolling-origin (sama dengan FIXED di scripts/modeling.py, Subbab 5.4 laporan)
FIXED = {
    "DecisionTree": {"max_depth": 4, "min_samples_leaf": 3},
    "RandomForest": {"max_depth": None, "min_samples_leaf": 3, "max_features": 1.0},
    "LinearRegression": {},
}
MODEL_KEY = {"Random Forest": "RandomForest", "Decision Tree": "DecisionTree",
             "Linear Regression": "LinearRegression"}
KORELASI = {"indeks_sebelumnya": "Indeks tahun sebelumnya", "delta_indeks": "Perubahan indeks (delta)",
            "is_kabupaten": "Status kabupaten", "tahun": "Tahun", "log_total_alokasi": "Log total alokasi DAK",
            "total_alokasi": "Total alokasi DAK", "reguler": "DAK Reguler", "penugasan": "DAK Penugasan",
            "bok": "DAK BOK", "jampersal": "DAK Jampersal", "akreditasi": "DAK Akreditasi"}

st.set_page_config(page_title="Kemiskinan Jawa Barat", page_icon="📊", layout="wide")

# ------------------------------------------------------------------ cek file
WAJIB = [DATA / "data_gabungan.csv", DATA / "laporan_outlier.csv", DATA / "garis_kemiskinan_tahunan.csv",
         DATA / "p3ke_kecamatan.csv", TAB / "modeling_ringkasan.json", TAB / "tabel_pemilihan_k.csv",
         TAB / "tabel_hasil_uji_2024.csv", TAB / "tabel_rolling_origin.csv",
         TAB / "tabel_importance_kelompok.csv", TAB / "tabel_anomali_residual.csv"]
hilang = [str(p.relative_to(BASE)) for p in WAJIB if not p.exists()]
if hilang:
    st.error("File berikut tidak ditemukan. Jalankan scripts/preprocessing.py dan scripts/modeling.py terlebih dahulu.")
    st.code("\n".join(hilang))
    st.stop()


# ------------------------------------------------------------------ fungsi bantu
def idn(x, nd=3):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_p(p):
    return "< 0,001" if p < 0.001 else idn(p, 3)


@st.cache_data
def baca_csv(path):
    return pd.read_csv(path)


def tabel(nama):
    return baca_csv(str(TAB / nama))


def data_csv(nama):
    return baca_csv(str(DATA / nama))


@st.cache_data
def load():
    df = pd.read_csv(DATA / "data_gabungan.csv")
    df["wilayah"] = df["nama_kabupaten_kota"].str.title().str.replace("Kabupaten ", "Kab. ")
    df["is_kabupaten"] = df["nama_kabupaten_kota"].str.startswith("KABUPATEN").astype(int)
    df["indeks_sebelumnya"] = (df[Y] - df["delta_indeks"]).round(2)
    for j in JENIS:
        df["log_" + j] = np.log1p(df[j])
    df["alokasi_nol"] = (df["total_alokasi"] == 0).astype(int)
    return df.sort_values(["kode_kabupaten_kota", "tahun"]).reset_index(drop=True)


@st.cache_data
def klaster(df, k):
    g = df.groupby(["kode_kabupaten_kota", "wilayah"])
    r = g.agg(ik_rata=(Y, "mean"), ik_sd=(Y, "std"), is_kab=("is_kabupaten", "first")).reset_index()
    a19 = df[df.tahun == 2019].set_index("kode_kabupaten_kota")[Y]
    a24 = df[df.tahun == 2024].set_index("kode_kabupaten_kota")[Y]
    r["ik_perubahan"] = r.kode_kabupaten_kota.map(a24) - r.kode_kabupaten_kota.map(a19)
    med = df[df.total_alokasi > 0].groupby("kode_kabupaten_kota")["total_alokasi"].median() / 1e9
    r["dak_median_miliar"] = r.kode_kabupaten_kota.map(med)
    r["log_dak"] = np.log1p(r["dak_median_miliar"])
    Z = StandardScaler().fit_transform(r[CL_FEATS])
    km = KMeans(n_clusters=k, n_init=50, random_state=RS).fit(Z)
    urut = r.assign(c=km.labels_).groupby("c")["ik_rata"].mean().sort_values().index.tolist()
    r["klaster"] = pd.Series(km.labels_).map({o: i + 1 for i, o in enumerate(urut)}).values
    P = PCA(n_components=2, random_state=RS).fit_transform(Z)
    r["PC1"], r["PC2"] = P[:, 0], P[:, 1]
    return r


@st.cache_data
def korelasi(d):
    """Korelasi Pearson dan Spearman tiap variabel dengan indeks keparahan (162 observasi)."""
    baris = []
    for v, lab in KORELASI.items():
        r, p = stats.pearsonr(d[v], d[Y])
        s, ps = stats.spearmanr(d[v], d[Y])
        baris.append({"Variabel": lab, "Pearson r": r, "p Pearson": p, "Spearman ρ": s, "p Spearman": ps})
    return pd.DataFrame(baris).sort_values("Pearson r", key=abs, ascending=False).reset_index(drop=True)


def fitur_dak(miliar):
    """miliar: dict jenis -> nilai (Rp miliar) -> fitur log DAK untuk model."""
    rp = {j: miliar[j] * 1e9 for j in JENIS}
    tot = sum(rp.values())
    f = {"log_total_alokasi": np.log1p(tot), "alokasi_nol": int(tot == 0)}
    f.update({"log_" + j: np.log1p(rp[j]) for j in JENIS})
    return f


def buat_model(key, param):
    if key == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=1, **param)
    if key == "DecisionTree":
        return DecisionTreeRegressor(random_state=RS, **param)
    return make_pipeline(StandardScaler(), LinearRegression())


@st.cache_resource
def latih(tahun_target, nama):
    """Latih model pada seluruh data sebelum tahun_target (rolling-origin), skenario S4.
    Parameter: FIXED untuk tahun < 2024 (rolling-origin), hasil tuning untuk 2024 dan seterusnya."""
    key = MODEL_KEY[nama]
    df = load()
    tr = df[df.tahun < tahun_target]
    param = TUNED[key] if tahun_target >= TEST_YEAR else FIXED[key]
    return buat_model(key, param).fit(tr[S4], tr[Y])


# ------------------------------------------------------------------ data & hasil pemodelan
RING = json.loads((TAB / "modeling_ringkasan.json").read_text(encoding="utf-8"))
HOLD = tabel("tabel_hasil_uji_2024.csv")
ROLL = tabel("tabel_rolling_origin.csv")
TUNED = {r.model_key: json.loads(r.param) for r in HOLD[HOLD.skenario == "S4"].itertuples()}
K = RING["klaster"]["k_terbaik"]
df = load()
reg = klaster(df, K)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Kemiskinan Jawa Barat")
st.sidebar.caption("Project 2 · Kelompok 3 · Project Data Sains\nUniversitas Maritim Raja Ali Haji")
HAL = ["🏠 Beranda", "🧹 Data & Preprocessing", "🔍 Eksplorasi Data (EDA)",
       "🧩 Clustering K-Means", "🎯 Prediksi & Evaluasi Regresi"]
hal = st.sidebar.radio("Menu", HAL)

# ================================================================== 1 BERANDA
if hal == HAL[0]:
    st.title("Pemetaan dan Prediksi Kemiskinan Daerah Berdasarkan Indikator Kesehatan")
    st.write("Aplikasi ini memuat hasil analisis 27 kabupaten/kota Jawa Barat (2019–2024): "
             "pengelompokan wilayah dengan **K-Means** dan prediksi indeks keparahan kemiskinan "
             "dengan **Linear Regression, Decision Tree, dan Random Forest**. Indikator kesehatan "
             "yang dipakai adalah alokasi DAK bidang kesehatan (ukuran anggaran, bukan luaran kesehatan).")
    mt = RING["regresi"]["model_terpilih"]
    c = st.columns(4)
    c[0].metric("Observasi", f"{len(df)}", "27 wilayah × 6 tahun")
    c[1].metric("Jumlah klaster (k)", K, f"Silhouette {idn(RING['klaster']['silhouette'])}")
    c[2].metric("Model terpilih", f"{mt['model']} ({mt['skenario']})")
    c[3].metric("RMSE / R² uji 2024", f"{idn(mt['uji_RMSE'])} / {idn(mt['uji_R2'])}")
    st.subheader("Pertanyaan yang dijawab")
    st.markdown("1. Bagaimana pengelompokan kabupaten/kota dan profil tiap kelompok?\n"
                "2. Apakah alokasi DAK kesehatan berhubungan dengan indeks keparahan kemiskinan?\n"
                "3. Seberapa akurat indeks dapat diprediksi, dan wilayah mana yang menyimpang dari pola umum?")
    st.info("Temuan utama: indeks tahun sebelumnya adalah prediktor terkuat, sedangkan besar alokasi DAK "
            "hampir tidak menambah kemampuan prediksi. Ini **bukan** bukti DAK tidak efektif, karena data "
            "hanya memuat alokasi, bukan realisasi atau luaran kesehatan.")

# ================================================================== 2 DATA
elif hal == HAL[1]:
    st.title("Data dan Preprocessing")
    t1, t2, t3 = st.tabs(["Data gabungan", "Cleaning & outlier", "Normalisasi"])
    with t1:
        st.write("Hasil integrasi dataset indeks keparahan dan alokasi DAK (kunci: kode kab/kota + tahun).")
        f = st.multiselect("Filter wilayah", sorted(df.wilayah.unique()))
        d = df[df.wilayah.isin(f)] if f else df
        st.dataframe(d[["wilayah", "tahun", Y, "delta_indeks", "total_alokasi"] + JENIS],
                     width="stretch", hide_index=True)
        st.caption(f"{len(d)} baris · {df.isna().sum().sum()} missing value pada data gabungan")
        st.dataframe(df[[Y, "total_alokasi"]].describe().round(3), width="stretch")
    with t2:
        st.write("Deteksi outlier dengan metode IQR (batas bawah/atas = Q1 − 1,5·IQR / Q3 + 1,5·IQR).")
        st.dataframe(data_csv("laporan_outlier.csv"), width="stretch", hide_index=True)
        nol = int((df.total_alokasi == 0).sum())
        st.write(f"Nilai alokasi 0 ({nol} observasi) ditandai dengan fitur `alokasi_nol` dan tidak dihapus. "
                 "Belum dapat dipastikan apakah nilai 0 berarti daerah tidak menerima dana atau data yang "
                 "belum dilaporkan, karena sumber data tidak menjelaskannya; pengaruhnya diperiksa lewat "
                 "analisis sensitivitas di laporan.")
        st.write("Anomali alokasi reguler Kota Cirebon 2023 (Rp2,66 triliun) dipertahankan "
                 "dan ditangani lewat transformasi logaritma.")
    with t3:
        st.write("Dua transformasi dipakai: **log(1 + x)** pada nilai rupiah yang sangat condong, "
                 "dan **StandardScaler (z-score)** sebelum K-Means dan Linear Regression.")
        pilih = st.selectbox("Bandingkan sebelum dan sesudah log", ["total_alokasi"] + JENIS)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(df, x=pilih, nbins=30, title="Sebelum (Rp)"), width="stretch")
        c2.plotly_chart(px.histogram(df.assign(v=np.log1p(df[pilih])), x="v", nbins=30,
                                     title="Sesudah log(1 + x)"), width="stretch")

# ================================================================== 3 EDA
elif hal == HAL[2]:
    st.title("Eksplorasi Data (EDA)")
    thn = st.sidebar.select_slider("Tahun", options=sorted(df.tahun.unique()), value=2024)
    c1, c2 = st.columns(2)
    top = df[df.tahun == thn].nlargest(10, Y).sort_values(Y)
    c1.plotly_chart(px.bar(top, x=Y, y="wilayah", orientation="h",
                           title=f"10 wilayah dengan indeks tertinggi, {thn}"), width="stretch")
    tren = df.groupby("tahun")[Y].mean().reset_index()
    fig = px.line(tren, x="tahun", y=Y, markers=True, title="Rata-rata indeks keparahan per tahun")
    c2.plotly_chart(fig, width="stretch")
    st.caption("Lonjakan 2021 terjadi serentak: 24 dari 27 daerah naik, bertepatan dengan pandemi COVID-19.")
    c3, c4 = st.columns(2)
    hm = df.pivot(index="wilayah", columns="tahun", values=Y)
    c3.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd",
                              title="Heat Map indeks 27 kab/kota"), width="stretch")
    pos = df[df.total_alokasi > 0]
    r = pos[Y].corr(pos["log_total_alokasi"])
    c4.plotly_chart(px.scatter(pos, x="log_total_alokasi", y=Y, color=pos.is_kabupaten.map({1: "Kabupaten", 0: "Kota"}),
                               title=f"DAK (log) vs indeks, r = {idn(r)} (n = {len(pos)}, tanpa DAK = 0)"),
                    width="stretch")

    st.subheader("Korelasi dengan indeks keparahan")
    kor = korelasi(df)
    a, b = st.columns([3, 2])
    lg = kor.melt(id_vars="Variabel", value_vars=["Pearson r", "Spearman ρ"], var_name="Koefisien", value_name="Nilai")
    a.plotly_chart(px.bar(lg, x="Nilai", y="Variabel", color="Koefisien", barmode="group", orientation="h",
                          title=f"Pearson dan Spearman terhadap indeks (n = {len(df)})",
                          category_orders={"Variabel": kor["Variabel"].tolist()[::-1]}), width="stretch")
    tampil = pd.DataFrame({"Variabel": kor["Variabel"],
                           "Pearson r": kor["Pearson r"].map(lambda v: idn(v, 2)), "p": kor["p Pearson"].map(fmt_p),
                           "Spearman ρ": kor["Spearman ρ"].map(lambda v: idn(v, 2)), "p ": kor["p Spearman"].map(fmt_p)})
    b.dataframe(tampil, width="stretch", hide_index=True)
    st.caption("Korelasi `delta_indeks` bersifat mekanis karena dihitung dari indeks tahun berjalan (target), "
               "sehingga tidak dipakai sebagai prediktor. Variabel DAK umumnya berkorelasi lemah dan tidak signifikan.")

    st.subheader("Data pendukung")
    a, b = st.columns(2)
    a.plotly_chart(px.line(data_csv("garis_kemiskinan_tahunan.csv"), x="tahun",
                           y=["garis_kemiskinan_perdesaan", "garis_kemiskinan_perkotaan"], markers=True,
                           title="Garis kemiskinan Jawa Barat (Rp/kapita/bulan)"), width="stretch")
    p3 = data_csv("p3ke_kecamatan.csv").nlargest(10, "jumlah_individu").sort_values("jumlah_individu")
    b.plotly_chart(px.bar(p3, x="jumlah_individu", y="bps_nama_kecamatan", orientation="h",
                          title="10 kecamatan dengan individu P3KE terbanyak, Kota Bandung"), width="stretch")

# ================================================================== 4 CLUSTER
elif hal == HAL[3]:
    st.title("Clustering K-Means")
    ks = tabel("tabel_pemilihan_k.csv")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(ks, x="k", y="inertia", markers=True, title="(a) Metode Elbow"), width="stretch")
    f = px.line(ks, x="k", y="silhouette", markers=True, title="(b) Silhouette Score")
    f.add_vline(x=K, line_dash="dash")
    c2.plotly_chart(f, width="stretch")
    sil = RING["klaster"]["silhouette"]
    kualitas = "lemah hingga sedang" if sil < 0.5 else "cukup kuat"
    st.success(f"k terpilih = {K} (Silhouette maksimum {idn(sil)}; "
               f"Davies–Bouldin {idn(RING['klaster']['davies_bouldin'])}). Struktur klaster tergolong {kualitas}, "
               "sehingga batas antarkelompok bersifat landai.")
    kl = reg.assign(Klaster="Klaster " + reg.klaster.astype(str))
    a, b = st.columns([3, 2])
    a.plotly_chart(px.scatter(kl, x="PC1", y="PC2", color="Klaster", text="wilayah",
                              color_discrete_map={f"Klaster {i}": CL_COL[i - 1] for i in range(1, K + 1)},
                              title="Sebaran 27 kab/kota pada bidang PCA").update_traces(textposition="top center",
                              textfont_size=9), width="stretch")
    prof = kl.groupby("Klaster")[CL_FEATS].mean().rename(columns=CL_LABEL).round(3)
    prof.insert(0, "Jumlah", kl.groupby("Klaster").size())
    b.subheader("Profil klaster"); b.dataframe(prof, width="stretch")
    st.subheader("Cari klaster suatu wilayah")
    w = st.selectbox("Kabupaten/Kota", sorted(reg.wilayah))
    row = reg[reg.wilayah == w].iloc[0]
    m = st.columns(4)
    m[0].metric("Klaster", int(row.klaster)); m[1].metric("Rata-rata indeks", idn(row.ik_rata))
    m[2].metric("Volatilitas (SD)", idn(row.ik_sd)); m[3].metric("Perubahan 2019–2024", idn(row.ik_perubahan))
    with st.expander("Daftar anggota seluruh klaster"):
        st.dataframe(kl.sort_values(["klaster", "ik_rata"])[["Klaster", "wilayah", "ik_rata", "ik_sd",
                                                              "ik_perubahan", "dak_median_miliar"]].round(3),
                     width="stretch", hide_index=True)

# ================================================================== 5 REGRESI
else:
    st.title("Prediksi dan Evaluasi Regresi")
    t1, t2 = st.tabs(["🎯 Prediksi", "📈 Evaluasi model"])
    with t1:
        st.write("Model dilatih ulang pada seluruh data **sebelum** tahun yang dipilih "
                 "(seperti uji rolling-origin), memakai skenario S4 (riwayat indeks + DAK).")
        tahun_akhir = int(df.tahun.max())
        c1, c2 = st.columns(2)
        w = c1.selectbox("Kabupaten/Kota", sorted(df.wilayah.unique()), index=0)
        thn = c2.selectbox("Tahun yang diprediksi", [2021, 2022, 2023, 2024, tahun_akhir + 1], index=4)
        dw = df[df.wilayah == w].set_index("tahun")
        prev = float(dw.loc[thn - 1, Y]) if (thn - 1) in dw.index else float(dw[Y].iloc[-1])
        ref = dw.loc[thn] if thn in dw.index else dw.iloc[-1]
        st.markdown("**Masukan model**")
        i1, i2 = st.columns([1, 2])
        prev = i1.number_input("Indeks tahun sebelumnya", 0.0, 2.0, prev, 0.01)
        cols = i2.columns(5)
        dak = {j: cols[k].number_input(f"{j} (Rp miliar)", 0.0, 3000.0, float(round(ref[j] / 1e9, 2)), 1.0)
               for k, j in enumerate(JENIS)}
        if thn > tahun_akhir:
            st.info(f"Tahun {thn} adalah **proyeksi**: data gabungan hanya sampai {tahun_akhir} dan alokasi DAK "
                    f"{thn} belum tersedia, sehingga kolom DAK diisi nilai {tahun_akhir} sebagai asumsi (dapat diubah). "
                    "Random Forest dan Decision Tree tidak dapat mengekstrapolasi fitur tahun, jadi tahun "
                    f"{thn} diperlakukan seperti {tahun_akhir} oleh pohon keputusan.")
        if st.button("Prediksi", type="primary"):
            x = {"indeks_sebelumnya": prev, "is_kabupaten": int(dw.is_kabupaten.iloc[0]), "tahun": thn,
                 **fitur_dak(dak)}
            X = pd.DataFrame([x])[S4]
            pred = {n: float(latih(thn, n).predict(X)[0]) for n in MODEL_KEY}
            p_rf = pred["Random Forest"]
            m = st.columns(5)
            m[0].metric("Random Forest (terpilih)", idn(p_rf))
            m[1].metric("Decision Tree", idn(pred["Decision Tree"]))
            m[2].metric("Linear Regression", idn(pred["Linear Regression"]))
            m[3].metric("Persistensi (indeks tahun lalu)", idn(prev))
            if thn in dw.index:
                akt = float(dw.loc[thn, Y])
                m[4].metric("Nilai aktual", idn(akt), f"galat RF {idn(akt - p_rf)}", delta_color="off")
            else:
                m[4].metric("Nilai aktual", "tidak ada di data")
            rmse_roll = float(ROLL[(ROLL.skenario == "S4") & (ROLL.model_key == "RandomForest")].RMSE.iloc[0])
            rmse_uji = RING["regresi"]["model_terpilih"]["uji_RMSE"]
            if thn < TEST_YEAR:
                st.caption("Parameter model: tetap, sama dengan uji rolling-origin (Subbab 5.4), karena parameter hasil "
                           "tuning dipilih memakai validasi 2021–2023. "
                           f"Galat tipikal Random Forest pada rolling-origin: RMSE ≈ {idn(rmse_roll)} poin.")
            else:
                st.caption("Parameter model: hasil tuning (dipilih pada data 2019–2023). "
                           f"Galat tipikal Random Forest: RMSE ≈ {idn(rmse_uji)} poin (uji 2024) dan "
                           f"{idn(rmse_roll)} poin (rolling-origin).")
            st.caption("Wilayah berindeks tinggi dan bergejolak cenderung kurang akurat; "
                       "model tidak dapat mengantisipasi guncangan mendadak seperti 2021.")
            riw = dw[Y].reset_index()
            fg = px.line(riw, x="tahun", y=Y, markers=True, title=f"Riwayat indeks {w}")
            fg.add_trace(go.Scatter(x=[thn], y=[p_rf], mode="markers", name="Prediksi RF",
                                    marker=dict(size=14, symbol="star", color="#D9822B")))
            st.plotly_chart(fg, width="stretch")
    with t2:
        h, ro = HOLD, ROLL
        kolom = ["skenario", "model", "MAE", "RMSE", "R2"]
        a, b = st.columns(2)
        a.subheader("Data uji 2024 (n = 27)"); a.dataframe(h[kolom].round(3), width="stretch", hide_index=True)
        b.subheader("Rolling-origin 2021–2024 (n = 108)")
        b.dataframe(ro[kolom].round(3), width="stretch", hide_index=True)
        pl = pd.concat([h.assign(uji="Uji 2024"), ro.assign(uji="Rolling-origin")])
        pl = pl[pl.skenario != "-"]
        st.plotly_chart(px.bar(pl, x="skenario", y="RMSE", color="model", barmode="group", facet_col="uji",
                               title="Perbandingan RMSE (makin kecil makin baik)"), width="stretch")
        BASE_HIST = "Rata-rata historis wilayah"
        b24 = float(h.loc[h.model == BASE_HIST, "RMSE"].iloc[0])
        brol = float(ro.loc[ro.model == BASE_HIST, "RMSE"].iloc[0])
        if b24 < h[h.skenario != "-"].RMSE.min() and brol < ro[ro.skenario != "-"].RMSE.min():
            st.warning(f"Baseline rata-rata historis wilayah (RMSE {idn(b24)} pada uji 2024; {idn(brol)} pada "
                       "rolling-origin) masih mengungguli semua model machine learning. Satu tahun uji tidak "
                       "cukup untuk menilai model, sehingga hasil rolling-origin dan selang bootstrap perlu "
                       "dibaca bersama.")
        else:
            st.info("Bandingkan setiap model dengan baseline pada tabel di atas; satu tahun uji tidak cukup "
                    "untuk menilai model.")
        st.subheader("Kepentingan fitur dan anomali")
        c1, c2 = st.columns(2)
        im = tabel("tabel_importance_kelompok.csv")
        c1.plotly_chart(px.bar(im, x="kenaikan_rmse", y="kelompok_fitur", color="model", barmode="group",
                               orientation="h", title="Permutation importance (S4)"), width="stretch")
        c2.markdown("**10 residual terbesar (rolling-origin)**")
        c2.dataframe(tabel("tabel_anomali_residual.csv").drop(columns=["kode_kabupaten_kota"]).round(3),
                     width="stretch", hide_index=True)
