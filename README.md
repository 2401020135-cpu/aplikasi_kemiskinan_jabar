# 📊 Aplikasi Kemiskinan Jawa Barat - Project 2 Data Science

**Mata Kuliah:** Project Data Science  
**Kelompok:** 3  
**Tahun Akademik:** 2026  
**Deadline Pengumpulan:** 29 Oktober 2026

---

## 📋 Deskripsi Project

Aplikasi interaktif berbasis **Streamlit** untuk analisis dan prediksi **Indeks Keparahan Kemiskinan** di 27 kabupaten/kota Jawa Barat menggunakan data BPS (2019–2024).

### Tujuan
1. Memahami pola kemiskinan di Jawa Barat melalui EDA
2. Mengidentifikasi klaster wilayah dengan K-Means Clustering
3. Membandingkan 3 model prediksi: Random Forest, Decision Tree, Linear Regression
4. Menyajikan hasil dalam aplikasi interaktif yang user-friendly

---

## 🎯 Fitur Aplikasi

| Menu | Deskripsi |
|------|-----------|
| **Beranda** | Overview project, metrik model, dan visualisasi utama |
| **Data** | Eksplorasi dataset, preprocessing, dan statistik deskriptif |
| **EDA** | Analisis exploratory dengan peta panas, distribusi, korelasi |
| **Clustering** | Hasil K-Means dengan 3 klaster dan profil per wilayah |
| **Prediksi** | Input manual untuk prediksi indeks keparahan dengan 3 model |
| **Evaluasi** | Perbandingan metrik (RMSE, MAE, R²) dan validasi rolling-origin |

---

## 📊 Dataset

**Sumber:** Badan Pusat Statistik (BPS)

| Variabel | Tipe | Keterangan |
|----------|------|-----------|
| Indeks Keparahan Kemiskinan | Target | Variabel dependen (y) |
| Indeks Kedalaman Kemiskinan | Numerik | Kedalaman kemiskinan |
| Total DAK Kesehatan | Numerik | Dana Alokasi Khusus Kesehatan (Rp) |
| Garis Kemiskinan | Numerik | Garis kemiskinan per tahun |
| Persentase P3KE | Numerik | Program Penanggulangan Kemiskinan |
| ... | ... | (lihat laporan untuk detail lengkap) |

**Periode:** 2019–2024  
**Observasi:** 27 kabupaten/kota × 6 tahun = 162 obs (sebelum outlier handling)

---

## 🏗️ Struktur Folder

```
aplikasi_kemiskinan_jabar/
├── app.py                          # File utama aplikasi Streamlit
├── requirements.txt                # Dependency Python
├── .gitignore                      # Git ignore rules
├── README.md                       # Dokumentasi (file ini)
│
├── data/                           # Dataset
│   ├── data_gabungan.csv          # Data utama (gabungan BPS)
│   ├── garis_kemiskinan_tahunan.csv
│   ├── p3ke_kecamatan.csv         # Data program P3KE
│   └── laporan_outlier.csv        # Hasil deteksi outlier
│
├── outputs/                        # Hasil modeling & visualisasi
│   ├── figures/                   # Gambar (.png)
│   │   ├── gambar5_1_alur_pemodelan.png
│   │   ├── gambar5_2_elbow_silhouette.png
│   │   ├── gambar5_3_peta_klaster.png
│   │   ├── gambar6_1_perbandingan_rmse.png
│   │   ├── gambar6_2_aktual_vs_prediksi.png
│   │   ├── gambar6_3_importance.png
│   │   ├── gambar6_4_residual_anomali.png
│   │   └── gambar6_5_silhouette_sampel.png
│   │
│   └── tables/                    # Tabel hasil (.csv, .json)
│       ├── modeling_ringkasan.json           # Ringkasan model (R², MAE, RMSE)
│       ├── tabel_anggota_klaster.csv        # Anggota setiap klaster
│       ├── tabel_hasil_uji_2024.csv         # Hasil evaluasi data uji 2024
│       ├── tabel_importance_*.csv           # Permutation importance
│       ├── tabel_koefisien_lr_s4.csv        # Koefisien Linear Regression
│       ├── tabel_rolling_origin.csv         # Hasil rolling-origin validation
│       └── ... (lihat laporan untuk detail)
│
└── scripts/                        # Script utilitas
    └── modeling.py               # Script untuk regenerasi model & tabel
```

---

## ⚙️ Instalasi & Setup

### Prerequisites
- Python 3.9+
- pip atau conda

### Langkah Instalasi

1. **Clone atau download repository:**
   ```bash
   git clone https://github.com/2401020135-cpu/aplikasi_kemiskinan_jabar.git
   cd aplikasi_kemiskinan_jabar
   ```

2. **Buat virtual environment (opsional tapi recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Cara Menjalankan

### Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Regenerasi Model (Opsional)

Jika ingin melatih ulang model atau mengubah data:

```bash
python scripts/modeling.py --data data --out outputs
```

**Parameter:**
- `--data`: Direktori folder data (default: `data`)
- `--out`: Direktori output hasil modeling (default: `outputs`)

---

## 📈 Model & Metrik

### Model yang Digunakan

| Model | Parameter | R² (Data Uji 2024) | RMSE | MAE |
|-------|-----------|-------------------|------|-----|
| **Random Forest** | n_estimators=100, max_depth=10 | 0.399 | 0.063 | 0.052 |
| **Decision Tree** | max_depth=5, min_samples_leaf=2 | 0.422 | 0.056 | 0.044 |
| **Linear Regression** | default | 0.489 | 0.052 | 0.039 |

**Model Terpilih:** Linear Regression (R² tertinggi: 0.489)

### Validasi Rolling-Origin

Menggunakan rolling-origin validation dengan window 1 tahun (2021–2024):
- Perbandingan RMSE antar periode
- Selang kepercayaan 95% via bootstrap
- Uji stabilitas prediksi

---

## 📁 Laporan Lengkap

Lihat file **`Laporan_Project_2_Kelompok_3_revisi.docx`** untuk:
- Analisis exploratory data (EDA) lengkap
- Metodologi preprocessing & feature engineering
- Hasil clustering dan profil klaster
- Evaluasi model detail (RMSE, MAE, R², residual analysis)
- Insight dan kesimpulan

**Total:** 83 halaman | **Format:** DOCX | **Tanggal:** 28 Oktober 2026

---

## 📸 Screenshot Aplikasi

Lihat folder `screenshots_aplikasi/` untuk 6 screenshot:
1. **beranda.png** - Halaman utama dengan metrik model
2. **data.png** - Halaman data & preprocessing
3. **eda.png** - Halaman eksplorasi data
4. **clustering.png** - Hasil K-Means clustering
5. **prediksi.png** - Fitur prediksi manual
6. **evaluasi.png** - Tab evaluasi model

---

## 🔧 Dependencies

Lihat `requirements.txt` untuk versi package. Utama:
- **streamlit** - Framework aplikasi web
- **pandas** - Data manipulation
- **numpy** - Numerik computation
- **scikit-learn** - Machine learning models
- **matplotlib, seaborn** - Visualisasi
- **plotly** - Interactive charts
- **joblib** - Menyimpan model

---

## 👥 Anggota Kelompok 3

- Anggota 1: NIM 2401020135
- Anggota 2: ...
- Anggota 3: ...

*(Lihat laporan untuk detail lengkap)*

---

## 📝 Lisensi & Penggunaan

Project ini dibuat untuk keperluan akademik Mata Kuliah Project Data Science.

**Data Source:** Badan Pusat Statistik (BPS) - Public Domain

---

## 📞 Kontak & Support

Untuk pertanyaan atau issue terkait project, silakan buat **Issue** di repository ini atau hubungi anggota kelompok.

---

**Last Updated:** 28 Oktober 2026  
**Status:** ✅ Final Submission
