"""
scripts/preprocessing.py
Pembersihan & penggabungan data - Kelompok 1 (Pemetaan dan Prediksi Kemiskinan Daerah)

Cara pakai:
    python preprocessing.py                      # pakai folder default
    python preprocessing.py --raw data/raw --out data/processed

Output (di folder --out):
    data_gabungan.csv         -> indeks keparahan + DAK kesehatan, 27 kab/kota x 6 tahun (162 baris)
    garis_kemiskinan_tahunan.csv -> garis kemiskinan Jawa Barat per tahun (konteks, level provinsi)
    p3ke_kecamatan.csv        -> P3KE Kota Bandung per kecamatan (analisis terpisah)
    laporan_outlier.csv       -> ringkasan outlier (metode IQR) untuk BAB III
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Nama file (sesuaikan kalau nama file di config.py beda) ----
F_GARIS = "od_17256_garis_kemiskinan_berdasarkan_daerah_tempat_tinggal_v6.csv"
F_DAK = "od_17472_alokasi_anggaran_kesehatan_smbr_apbd_kabupatenkota_v3.csv"
F_INDEKS = "od_19998_indeks_keparahan_kemiskinan_brdsrkn_kabupatenkota_v1.csv"
F_P3KE = "p3ke_kota_bandung_berdasarkan_jumlah_individu.csv"

TAHUN_MIN, TAHUN_MAX = 2019, 2024  # irisan tahun indeks keparahan & anggaran kesehatan


def info(nama, df):
    """Cek cepat: ukuran, missing, duplikat."""
    print(f"[{nama}] baris={len(df)}, kolom={df.shape[1]}, "
          f"missing={int(df.isna().sum().sum())}, duplikat={int(df.duplicated().sum())}")


def baca(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    # rapikan teks (spasi berlebih, huruf besar)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip().str.upper()
    return df


# ---------------------------------------------------------------- 1. INDEKS
def siapkan_indeks(raw):
    df = baca(raw / F_INDEKS)
    info("indeks keparahan (asli)", df)
    df = df.drop_duplicates()
    df = df[["kode_kabupaten_kota", "nama_kabupaten_kota", "tahun", "indeks_keparahan_kemiskinan"]]
    df = df.sort_values(["kode_kabupaten_kota", "tahun"])
    # fitur tambahan: perubahan dari tahun sebelumnya (dihitung sebelum difilter)
    df["delta_indeks"] = df.groupby("kode_kabupaten_kota")["indeks_keparahan_kemiskinan"].diff().round(2)
    df = df[df["tahun"].between(TAHUN_MIN, TAHUN_MAX)]
    return df


# ---------------------------------------------------------- 2. ANGGARAN KESEHATAN
def siapkan_anggaran(raw):
    df = baca(raw / F_DAK)
    info("anggaran kesehatan (asli)", df)
    df = df.drop_duplicates()
    df["jumlah_alokasi"] = pd.to_numeric(df["jumlah_alokasi"], errors="coerce")
    # nilai 0 = daerah tidak menerima dana itu, BUKAN missing -> tidak diisi/dibuang
    print(f"   nilai 0 pada jumlah_alokasi: {(df['jumlah_alokasi'] == 0).mean():.0%}")

    # 5 baris per kab/kota per tahun (5 jenis DAK) -> 1 baris, tiap jenis jadi kolom
    out = (df.pivot_table(index=["kode_kabupaten_kota", "tahun"],
                          columns="jenis_dana_alokasi_khusus", values="jumlah_alokasi", aggfunc="sum")
             .reset_index())
    out.columns = ["kode_kabupaten_kota", "tahun"] + [c.lower() for c in out.columns[2:]]
    jenis = [c for c in out.columns[2:]]
    out["total_alokasi"] = out[jenis].sum(axis=1)                     # variabel turunan (Tabel 2)
    out["log_total_alokasi"] = np.log1p(out["total_alokasi"]).round(3)  # distribusi sangat miring kanan
    return out


# ---------------------------------------------------------- 3. GARIS KEMISKINAN
def siapkan_garis(raw):
    df = baca(raw / F_GARIS)
    info("garis kemiskinan (asli)", df)
    df = df.drop_duplicates()
    df = df.rename(columns={"periode": "tahun"})
    # nilai 0 (September 2023) tidak mungkin sebagai garis kemiskinan -> dianggap tidak valid (NaN)
    print(f"   nilai 0 pada garis_kemiskinan: {int((df['garis_kemiskinan'] == 0).sum())} baris -> NaN")
    df["garis_kemiskinan"] = df["garis_kemiskinan"].replace(0, np.nan)
    # level provinsi -> rata-rata per tahun, dipisah perkotaan/perdesaan
    pv = df.pivot_table(index="tahun", columns="daerah_tempat_tinggal",
                        values="garis_kemiskinan", aggfunc="mean")
    pv.columns = ["garis_kemiskinan_" + c.lower() for c in pv.columns]
    pv["garis_kemiskinan_rata2"] = pv.mean(axis=1)
    return pv.reset_index()


# ---------------------------------------------------------------- 4. P3KE
def siapkan_p3ke(raw):
    df = baca(raw / F_P3KE)
    info("P3KE (asli)", df)
    df = df.drop_duplicates()
    # kolom BPS & Kemendagri isinya dobel -> pakai BPS saja
    df = df[["bps_kode_kecamatan", "bps_nama_kecamatan", "bps_nama_desa_kelurahan"
             if "bps_nama_desa_kelurahan" in df.columns else "bps_desa_kelurahan",
             "jumlah_individu", "tahun"]]
    df = df.rename(columns={"bps_nama_desa_kelurahan": "kelurahan", "bps_desa_kelurahan": "kelurahan"})
    kec = (df.groupby(["bps_kode_kecamatan", "bps_nama_kecamatan"], as_index=False)
             .agg(jumlah_kelurahan=("kelurahan", "nunique"), jumlah_individu=("jumlah_individu", "sum")))
    kec["rata2_individu_per_kelurahan"] = (kec["jumlah_individu"] / kec["jumlah_kelurahan"]).round(1)
    return kec


# ---------------------------------------------------------------- OUTLIER
def laporan_outlier(df, kolom):
    """Hitung outlier metode IQR. Hanya dilaporkan, TIDAK dihapus."""
    baris = []
    for c in kolom:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        baris.append({"kolom": c, "batas_bawah": round(lo, 3), "batas_atas": round(hi, 3),
                      "jumlah_outlier": n, "persen": round(n / len(df) * 100, 1)})
    return pd.DataFrame(baris)


# ---------------------------------------------------------------- MAIN
def main(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    indeks = siapkan_indeks(raw)
    dak = siapkan_anggaran(raw)
    garis = siapkan_garis(raw)

    # tabel analisis utama: indeks + anggaran (kode kab/kota + tahun)
    gab = indeks.merge(dak, on=["kode_kabupaten_kota", "tahun"], how="inner", validate="one_to_one")
    # tandai outlier anggaran (IQR atas) - data TIDAK dihapus
    q1, q3 = gab["total_alokasi"].quantile([0.25, 0.75])
    gab["outlier_alokasi"] = (gab["total_alokasi"] > q3 + 1.5 * (q3 - q1)).astype(int)
    print(f"\nHasil gabungan: {gab.shape[0]} baris, {gab.shape[1]} kolom "
          f"(harapan: 27 kab/kota x 6 tahun = 162 baris)")
    print("Missing setelah merge:", gab.isna().sum()[gab.isna().sum() > 0].to_dict() or "tidak ada")

    gab.to_csv(out / "data_gabungan.csv", index=False)
    garis.to_csv(out / "garis_kemiskinan_tahunan.csv", index=False)  # konteks, level provinsi
    laporan_outlier(gab, ["indeks_keparahan_kemiskinan", "total_alokasi",
                          "log_total_alokasi"]).to_csv(out / "laporan_outlier.csv", index=False)
    siapkan_p3ke(raw).to_csv(out / "p3ke_kecamatan.csv", index=False)

    print("\nSelesai. File tersimpan di:", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="folder CSV mentah")
    ap.add_argument("--out", default="data/processed", help="folder output")
    a = ap.parse_args()
    main(a.raw, a.out)
