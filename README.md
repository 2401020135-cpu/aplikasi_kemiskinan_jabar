# Pemetaan dan Prediksi Kemiskinan Daerah Berdasarkan Indikator Kesehatan

**Project Data Sains — Kelompok 3**
Program Studi Teknik Informatika, Universitas Maritim Raja Ali Haji
Tahun Akademik 2026/2027 · Dosen Pengampu: Nurfalinda, S.T., M.Cs.

Aplikasi web interaktif (Streamlit) untuk menyajikan hasil **pemetaan klaster wilayah** dan **prediksi indeks keparahan kemiskinan** di 27 kabupaten/kota Jawa Barat. Aplikasi tidak melakukan analisis baru; seluruh angka dibaca dari berkas hasil `scripts/modeling.py` sehingga konsisten dengan laporan.

---

## Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1 | Nabiilah Rifa Musyifa | 2401020162 |
| 2 | Andrian Yuza Swanda | 2401020157 |
| 3 | Fitri Arfiana | 2401020135 |
| 4 | Aisyah Nazwa Ramadhani | 2401020136 |
| 5 | Christine Simbolon | 2401020158 |

---

## Ringkasan Hasil

| Aspek | Hasil |
|-------|-------|
| Jumlah observasi | 162 (27 kabupaten/kota × 6 tahun, 2019–2024) |
| Clustering | K-Means **k = 2**, Silhouette **0,341** (struktur lemah–sedang) |
| Klaster 1 | 15 wilayah, rata-rata indeks 0,214, relatif stabil |
| Klaster 2 | 12 wilayah, rata-rata indeks 0,376, bergejolak, didominasi kabupaten |
| Model terpilih | **Random Forest skenario S4** (riwayat indeks + DAK) |
| Kinerja uji 2024 | MAE 0,080 · RMSE 0,099 · R² 0,363 |
| Kinerja rolling-origin 2021–2024 | RMSE 0,139 · R² 0,128 |
| Pembanding terkuat | Baseline rata-rata historis wilayah: RMSE 0,084 (2024) dan 0,114 (rolling-origin) |

**Catatan penting:** model machine learning **belum mengungguli** baseline rata-rata historis wilayah. Alokasi DAK kesehatan tidak menambah kemampuan prediksi (korelasi hampir nol), tetapi ini bukan bukti DAK tidak efektif karena data hanya memuat alokasi, bukan realisasi maupun luaran kesehatan.

### Perbandingan model pada data uji 2024 (skenario S4)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0,078 | 0,100 | 0,345 |
| Decision Tree | 0,084 | 0,107 | 0,256 |
| **Random Forest (terpilih)** | **0,080** | **0,099** | **0,363** |

### Hiperparameter terpilih (hasil grid search)

| Model | Parameter |
|-------|-----------|
| Random Forest (S4) | `max_depth=5`, `min_samples_leaf=1`, `max_features=0.5`, `n_estimators=300` |
| Decision Tree (S4) | `max_depth=3`, `min_samples_leaf=10` |
| Linear Regression | Tanpa hiperparameter (`StandardScaler` + OLS) |
| K-Means | `k=2` (dicari k=2–8), `n_init=50`, `random_state=42` |

---

## Menu Aplikasi

| Menu | Isi |
|------|-----|
| Beranda | Ringkasan proyek, empat indikator utama, dan pertanyaan data science |
| Data & Preprocessing | Data gabungan dengan filter wilayah, outlier IQR, perbandingan distribusi sebelum/sesudah log |
| Eksplorasi Data (EDA) | Wilayah tertinggi per tahun, tren, peta panas indeks, hubungan DAK–indeks, garis kemiskinan, sebaran P3KE |
| Clustering K-Means | Elbow & Silhouette, sebaran PCA, profil dan anggota klaster, pencarian klaster per wilayah |
| Prediksi & Evaluasi Regresi | Prediksi interaktif (RF, Decision Tree, Linear Regression) + baseline persistensi, tabel/grafik evaluasi, permutation importance, 10 residual terbesar |

---

## Struktur Repositori

```
.
├── app.py                    # Aplikasi Streamlit
├── requirements.txt          # Dependensi Python
├── README.md
├── .gitignore
├── data/
│   ├── data_gabungan.csv
│   ├── garis_kemiskinan_tahunan.csv
│   ├── p3ke_kecamatan.csv
│   └── laporan_outlier.csv
├── outputs/
│   ├── figures/              # 8 gambar hasil modeling (PNG)
│   └── tables/               # 16 tabel hasil (CSV/JSON)
├── scripts/
│   └── modeling.py           # Regenerasi model, tabel, dan gambar
├── screenshots/              # Tangkapan layar aplikasi (6 halaman)
└── laporan/
    ├── LAPORAN PROJECT 2 KELOMPOK 3.docx
    └── LAPORAN PROJECT 2 KELOMPOK 3.pdf
```

> Nama berkas di dalam `data/` dan `outputs/` sengaja memakai garis bawah karena dipanggil langsung oleh `app.py` dan `scripts/modeling.py`. Mengubahnya akan membuat aplikasi gagal membaca data.

---

## Instalasi

```bash
git clone https://github.com/2401020135-cpu/aplikasi_kemiskinan_jabar.git
cd aplikasi_kemiskinan_jabar
pip install -r requirements.txt
```

Disarankan memakai virtual environment:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

## Regenerasi Model (opsional)

```bash
python scripts/modeling.py --data data --out outputs
```

---

## Metodologi Singkat

1. **Data preparation** — penggabungan 4 dataset BPS/DINKES/DPPKB, pembersihan nilai 0 yang tidak valid, transformasi log alokasi DAK, dan penandaan outlier IQR tanpa menghapus baris.
2. **EDA** — distribusi, tren tahunan, peta panas indeks per wilayah, dan uji korelasi Pearson/Spearman.
3. **Clustering** — empat fitur per wilayah (rata-rata indeks, volatilitas, perubahan 2019–2024, log median DAK) distandardisasi, lalu K-Means dengan pemilihan k melalui Elbow dan Silhouette.
4. **Regresi** — empat skenario fitur (S1 DAK saja, S2 DAK + karakteristik wilayah, S3 riwayat indeks + tahun, S4 riwayat + DAK), tuning dengan validasi expanding-window 2021–2023, evaluasi holdout 2024 dan rolling-origin, serta bootstrap antarwilayah untuk selang kepercayaan RMSE.
5. **Evaluasi clustering** — Silhouette, Davies–Bouldin, Calinski–Harabasz, ARI terhadap Ward, dan stabilitas bootstrap.

---

## Lisensi dan Sumber Data

Dibuat untuk keperluan akademik Mata Kuliah Project Data Sains. Data bersumber dari portal data terbuka Pemerintah Provinsi Jawa Barat (galura.jabarprov.go.id), BPS, DINKES Jawa Barat, dan Open Data Kota Bandung.
