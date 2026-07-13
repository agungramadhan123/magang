"""
=============================================================================
EKSTRAKSI FITUR HRV (Heart Rate Variability) DARI DATABASE AFDB
=============================================================================

Tujuan  : Mengubah data rekaman EKG mentah menjadi dataset CSV yang bersih
          dan siap dipakai oleh model Machine Learning / Deep Learning.

Alur    :
  1. Baca file .hea   → konfirmasi sampling rate = 250 Hz
  2. Baca file .qrsc/.qrs → ambil lokasi tiap detak jantung (QRS)
  3. Hitung IBI (Inter-Beat Interval) = jarak waktu antar-detak (ms)
  4. Potong IBI menjadi jendela 10 detik
  5. Hitung 10 fitur HRV di setiap jendela
  6. Baca file .atr    → tentukan label kelas ritme di jendela itu
  7. Simpan hasilnya ke file CSV

Kelas   :
  0 = Normal (N)
  1 = Atrial Fibrillation (AFIB)
  2 = Atrial Flutter (AFL)
  Selain itu → jendela diabaikan

Output  : dataset_hrv.csv
=============================================================================
"""

import os
import numpy as np
import wfdb
import csv

FOLDER       = "./files"           # Lokasi file-file rekaman
RECORDS_FILE = "RECORDS"           # File daftar nama rekaman
FS           = 250                 # Sampling rate yang diharapkan (Hz)
WINDOW_SEC   = 10                  # Durasi jendela dalam detik
OUTPUT_FILE  = "dataset_hrv.csv"   # Nama file output

# Pemetaan label ritme ke angka kelas
LABEL_MAP = {
    "(N":    0,   # Normal
    "(AFIB": 1,   # Atrial Fibrillation
    "(AFL":  2,   # Atrial Flutter
}


# ─────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI PEMBANTU
# ─────────────────────────────────────────────────────────────

def baca_daftar_rekaman(folder):
    """
    Membaca file RECORDS untuk mendapatkan daftar nama rekaman.

    PENJELASAN:
    File RECORDS berisi satu nama rekaman per baris, misalnya:
        04015
        04043
        ...
    Fungsi ini membaca file tersebut dan mengembalikan list berisi
    nama-nama rekaman tanpa spasi/newline.
    """
    path = os.path.join(folder, RECORDS_FILE)
    with open(path, "r") as f:
        return [baris.strip() for baris in f if baris.strip()]


def konfirmasi_sampling_rate(folder, nama_rekaman):
    """
    Membaca file .hea dan memastikan sampling rate = 250 Hz.

    PENJELASAN:
    File .hea (header) menyimpan metadata rekaman, termasuk:
    - Jumlah sinyal (2 lead ECG)
    - Sampling rate (fs)
    - Panjang sinyal dalam sampel

    Contoh baris pertama file 04015.hea:
        04015 2 250 9205760  9:00:00
              │ │   │        └── waktu mulai
              │ │   └── panjang sinyal (sampel)
              │ └── sampling rate
              └── jumlah sinyal

    Kita mengecek apakah fs == 250. Jika tidak, rekaman dilewati.
    """
    header = wfdb.rdheader(os.path.join(folder, nama_rekaman))
    return header.fs == FS


def baca_lokasi_detak(folder, nama_rekaman):
    """
    Membaca file .qrsc atau .qrs untuk mendapatkan lokasi (indeks sampel)
    setiap detak jantung.

    PENJELASAN:
    File .qrs berisi hasil deteksi otomatis posisi QRS complex
    (puncak detak jantung). File .qrsc adalah versi yang sudah
    dikoreksi oleh ahli (lebih akurat).

    Fungsi ini mengembalikan array berisi indeks sampel, misalnya:
        [61, 246, 432, 614, 798, ...]
    Artinya detak pertama terjadi di sampel ke-61, kedua di 246, dst.

    Prioritas: .qrsc (koreksi) > .qrs (otomatis)
    """
    path = os.path.join(folder, nama_rekaman)

    # Coba baca .qrsc dulu (versi terkoreksi, lebih akurat)
    if os.path.exists(path + ".qrsc"):
        ann = wfdb.rdann(path, "qrsc")
    else:
        ann = wfdb.rdann(path, "qrs")

    return ann.sample


def hitung_ibi(lokasi_detak, fs):
    """
    Menghitung IBI (Inter-Beat Interval) dalam milidetik.

    PENJELASAN:
    IBI = jarak waktu antara satu detak jantung ke detak berikutnya.

    Cara hitung:
    1. Ambil selisih antara lokasi detak berturut-turut (dalam sampel)
       Contoh: lokasi = [61, 246, 432]
               selisih = [185, 186]  (dalam satuan sampel)

    2. Konversi ke milidetik:
       IBI (ms) = selisih_sampel / fs * 1000
       Contoh: 185 / 250 * 1000 = 740 ms

    3. Simpan juga waktu kapan IBI itu terjadi (pakai titik tengah
       antara dua detak sebagai acuan waktu).

    Return:
      waktu_ibi : array waktu terjadinya IBI (dalam detik)
      nilai_ibi : array nilai IBI (dalam milidetik)
    """
    # Selisih antar-detak berturut-turut (dalam satuan sampel)
    selisih_sampel = np.diff(lokasi_detak)

    # Konversi ke milidetik
    nilai_ibi = selisih_sampel / fs * 1000.0

    # Waktu terjadinya setiap IBI (dalam detik), pakai titik detak kedua
    # dari pasangan sebagai acuan waktu
    waktu_ibi = lokasi_detak[1:] / fs

    return waktu_ibi, nilai_ibi


def baca_label_ritme(folder, nama_rekaman):
    """
    Membaca file .atr untuk mendapatkan anotasi ritme jantung.

    PENJELASAN:
    File .atr berisi penanda pergantian ritme jantung.
    Setiap entri memiliki:
      - sample  : di titik sampel ke-berapa ritme berubah
      - symbol  : selalu '+' (penanda ritme)
      - aux_note: nama ritme, misal '(N', '(AFIB', '(AFL'

    Contoh isi .atr untuk rekaman 04015:
      Sampel      Label
      30          (N        → Dari sini, ritme = Normal
      102584      (AFIB     → Dari sini, ritme = Atrial Fibrillation
      119604      (N        → Dari sini, ritme kembali Normal
      ...

    Fungsi ini mengembalikan dua array:
      - posisi_perubahan: kapan ritme berubah (dalam sampel)
      - label_ritme     : ritme apa yang dimulai di titik itu
    """
    path = os.path.join(folder, nama_rekaman)
    ann = wfdb.rdann(path, "atr")

    posisi_perubahan = ann.sample
    label_ritme = ann.aux_note

    return posisi_perubahan, label_ritme


def tentukan_label_jendela(awal_detik, akhir_detik, posisi_perubahan, label_ritme, fs):
    """
    Menentukan label kelas untuk satu jendela waktu tertentu.

    PENJELASAN:
    Satu jendela (misal detik 10-20) bisa tumpang tindih dengan
    beberapa zona ritme yang berbeda. Kita perlu mencari tahu
    ritme mana yang DOMINAN (paling lama durasinya) di jendela itu.

    Caranya:
    1. Konversi batas jendela ke satuan sampel
    2. Untuk setiap zona ritme yang overlap dengan jendela,
       hitung berapa lama (sampel) zona itu ada di dalam jendela
    3. Pilih ritme dengan durasi terbanyak → itu label kelasnya

    Contoh:
      Jendela detik 10-20 (sampel 2500-5000)
      Zona ritme:  (N    dari sampel 0 - 3000
                   (AFIB dari sampel 3000 - 8000

      Durasi (N    di jendela = 3000 - 2500 = 500 sampel
      Durasi (AFIB di jendela = 5000 - 3000 = 2000 sampel
      → Dominan = (AFIB → Label = 1

    Return:
      Angka kelas (0, 1, atau 2), atau None jika label tidak dikenali
    """
    awal_sampel = int(awal_detik * fs)
    akhir_sampel = int(akhir_detik * fs)

    # Hitung durasi setiap ritme di dalam jendela ini
    durasi_per_label = {}

    for i in range(len(posisi_perubahan)):
        label = label_ritme[i]

        # Awal zona = posisi perubahan ritme ini
        zona_awal = posisi_perubahan[i]

        # Akhir zona = posisi perubahan berikutnya, atau tak terhingga
        if i + 1 < len(posisi_perubahan):
            zona_akhir = posisi_perubahan[i + 1]
        else:
            zona_akhir = float("inf")

        # Hitung overlap antara zona ritme dan jendela
        overlap_awal = max(zona_awal, awal_sampel)
        overlap_akhir = min(zona_akhir, akhir_sampel)

        if overlap_awal < overlap_akhir:
            durasi = overlap_akhir - overlap_awal
            durasi_per_label[label] = durasi_per_label.get(label, 0) + durasi

    # Tidak ada ritme yang tercakup di jendela ini
    if not durasi_per_label:
        return None

    # Ambil label dengan durasi terbanyak (dominan)
    label_dominan = max(durasi_per_label, key=durasi_per_label.get)

    # Petakan ke angka kelas, atau None jika tidak dikenali
    return LABEL_MAP.get(label_dominan, None)


def hitung_fitur_hrv(nilai_ibi):
    """
    Menghitung 10 fitur HRV dari sekumpulan nilai IBI dalam satu jendela.

    PENJELASAN SETIAP FITUR:

    1. Mean IBI
       = Rata-rata jarak antar-detak (ms)
       → Menggambarkan kecepatan detak jantung rata-rata.
         Mean IBI tinggi → jantung berdetak lambat
         Mean IBI rendah → jantung berdetak cepat

    2. Median IBI
       = Nilai tengah jarak antar-detak (ms)
       → Mirip Mean, tapi lebih tahan terhadap nilai ekstrem (outlier).

    3. SDNN (Standard Deviation of NN intervals)
       = Simpangan baku dari semua IBI
       → Mengukur TOTAL variabilitas detak jantung.
         SDNN tinggi → variabilitas tinggi (biasanya sehat)
         SDNN rendah → variabilitas rendah (bisa tanda masalah)

    4. RMSSD (Root Mean Square of Successive Differences)
       = Akar kuadrat dari rata-rata kuadrat selisih IBI berturut-turut
       → Mengukur variabilitas JANGKA PENDEK (detak ke detak).
         Sangat berguna untuk mendeteksi aritmia seperti AF.

    5. Coefficient of Variation (CV)
       = SDNN / Mean IBI × 100 (%)
       → SDNN yang dinormalisasi. Berguna untuk membandingkan
         variabilitas antar-pasien yang detak jantungnya berbeda.

    6. Mean Successive Difference (MeanSD)
       = Rata-rata nilai absolut selisih IBI berturut-turut
       → Mirip RMSSD tapi tanpa kuadrat, lebih intuitif.

    7. Std Successive Difference (StdSD)
       = Simpangan baku dari selisih IBI berturut-turut
       → Mengukur seberapa "tidak teratur" perubahan detak.

    8. Turning Point Ratio (TPR)
       = Proporsi titik balik dalam deretan IBI
       → Titik balik = IBI yang lebih besar atau lebih kecil dari
         kedua tetangganya. Pada AF, pola detak sangat acak
         sehingga TPR cenderung tinggi.

    9. Poincaré SD1
       = Simpangan baku tegak lurus garis identitas pada plot Poincaré
       → Mengukur variabilitas JANGKA PENDEK.
         Rumus: SD1 = SDSD / √2
         (SDSD = std dari selisih IBI berturut-turut)

    10. Poincaré SD2
        = Simpangan baku sepanjang garis identitas pada plot Poincaré
        → Mengukur variabilitas JANGKA PANJANG.
          Rumus: SD2 = √(2×SDNN² - SD1²)

    11. SD1/SD2 Ratio
        = Rasio variabilitas jangka pendek terhadap jangka panjang
        → Pada AF, rasio ini cenderung berbeda dari normal.

    Return: dictionary berisi 10 fitur, atau None jika data tidak cukup
    """
    n = len(nilai_ibi)

    # Minimal butuh 3 IBI untuk menghitung semua fitur
    if n < 3:
        return None

    # ---- Fitur Dasar ----
    mean_ibi   = np.mean(nilai_ibi)
    median_ibi = np.median(nilai_ibi)
    sdnn       = np.std(nilai_ibi, ddof=1)  # ddof=1 → sampel std

    # ---- Selisih berturut-turut (successive differences) ----
    selisih = np.diff(nilai_ibi)  # IBI[i+1] - IBI[i]

    rmssd   = np.sqrt(np.mean(selisih ** 2))
    mean_sd = np.mean(np.abs(selisih))
    std_sd  = np.std(selisih, ddof=1)

    # ---- Coefficient of Variation ----
    cv = (sdnn / mean_ibi) * 100 if mean_ibi != 0 else 0

    # ---- Turning Point Ratio ----
    # Titik balik: IBI[i] > kedua tetangga ATAU IBI[i] < kedua tetangga
    jumlah_titik_balik = 0
    for i in range(1, n - 1):
        if ((nilai_ibi[i] > nilai_ibi[i-1] and nilai_ibi[i] > nilai_ibi[i+1]) or
            (nilai_ibi[i] < nilai_ibi[i-1] and nilai_ibi[i] < nilai_ibi[i+1])):
            jumlah_titik_balik += 1

    # TPR = jumlah titik balik / jumlah kemungkinan titik balik
    tpr = jumlah_titik_balik / (n - 2) if n > 2 else 0

    # ---- Poincaré ----
    sd1 = std_sd / np.sqrt(2)
    sd2_kuadrat = 2 * (sdnn ** 2) - (sd1 ** 2)
    sd2 = np.sqrt(sd2_kuadrat) if sd2_kuadrat > 0 else 0

    sd1_sd2 = sd1 / sd2 if sd2 != 0 else 0

    return {
        "mean_ibi":   round(mean_ibi, 4),
        "median_ibi": round(median_ibi, 4),
        "sdnn":       round(sdnn, 4),
        "rmssd":      round(rmssd, 4),
        "cv":         round(cv, 4),
        "mean_sd":    round(mean_sd, 4),
        "std_sd":     round(std_sd, 4),
        "tpr":        round(tpr, 4),
        "sd1":        round(sd1, 4),
        "sd2":        round(sd2, 4),
        "sd1_sd2":    round(sd1_sd2, 4),
    }


# ─────────────────────────────────────────────────────────────
# PROGRAM UTAMA
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EKSTRAKSI FITUR HRV - DATASET AFDB")
    print("=" * 60)

    # Langkah 0: Baca daftar rekaman
    daftar_rekaman = baca_daftar_rekaman(FOLDER)
    print(f"\nDitemukan {len(daftar_rekaman)} rekaman di file RECORDS.")

    # Siapkan wadah untuk menampung semua baris dataset
    semua_baris = []

    # Header kolom CSV
    kolom = [
        "record", "window_start_sec", "window_end_sec",
        "mean_ibi", "median_ibi", "sdnn", "rmssd", "cv",
        "mean_sd", "std_sd", "tpr", "sd1", "sd2", "sd1_sd2",
        "label"
    ]

    rekaman_diproses = 0
    rekaman_dilewati = 0
    total_jendela    = 0
    jendela_diabaikan = 0

    for nama in daftar_rekaman:
        print(f"\n── Memproses rekaman: {nama} ", end="")

        # ── LANGKAH 1: Konfirmasi sampling rate ──
        if not konfirmasi_sampling_rate(FOLDER, nama):
            print("→ DILEWATI (sampling rate ≠ 250 Hz)")
            rekaman_dilewati += 1
            continue

        # ── LANGKAH 2: Baca lokasi detak jantung ──
        try:
            lokasi_detak = baca_lokasi_detak(FOLDER, nama)
        except Exception as e:
            print(f"→ DILEWATI (gagal baca QRS: {e})")
            rekaman_dilewati += 1
            continue

        # ── LANGKAH 3: Hitung IBI ──
        waktu_ibi, nilai_ibi = hitung_ibi(lokasi_detak, FS)

        if len(nilai_ibi) < 3:
            print("→ DILEWATI (IBI terlalu sedikit)")
            rekaman_dilewati += 1
            continue

        # ── LANGKAH 4: Baca label ritme dari .atr ──
        try:
            posisi_perubahan, label_ritme = baca_label_ritme(FOLDER, nama)
        except Exception as e:
            print(f"→ DILEWATI (gagal baca ATR: {e})")
            rekaman_dilewati += 1
            continue

        # ── LANGKAH 5: Potong menjadi jendela 10 detik ──
        waktu_akhir_sinyal = waktu_ibi[-1]
        jumlah_jendela_rekaman = 0
        jumlah_diabaikan_rekaman = 0

        awal = 0.0
        while awal + WINDOW_SEC <= waktu_akhir_sinyal:
            akhir = awal + WINDOW_SEC

            # Ambil IBI yang jatuh di jendela ini
            mask = (waktu_ibi >= awal) & (waktu_ibi < akhir)
            ibi_jendela = nilai_ibi[mask]

            # ── LANGKAH 6: Hitung fitur HRV ──
            fitur = hitung_fitur_hrv(ibi_jendela)

            if fitur is None:
                awal = akhir
                jumlah_diabaikan_rekaman += 1
                continue

            # ── LANGKAH 7: Tentukan label kelas ──
            label = tentukan_label_jendela(
                awal, akhir, posisi_perubahan, label_ritme, FS
            )

            # Abaikan jendela jika label bukan N, AFIB, atau AFL
            if label is None:
                awal = akhir
                jumlah_diabaikan_rekaman += 1
                continue

            # Simpan baris data
            baris = [nama, awal, akhir]
            baris += list(fitur.values())
            baris.append(label)
            semua_baris.append(baris)

            jumlah_jendela_rekaman += 1
            awal = akhir

        rekaman_diproses += 1
        total_jendela += jumlah_jendela_rekaman
        jendela_diabaikan += jumlah_diabaikan_rekaman
        print(f"→ {jumlah_jendela_rekaman} jendela "
              f"({jumlah_diabaikan_rekaman} diabaikan)")

    # ── Simpan ke CSV ──
    output_path = os.path.join(FOLDER, "..", OUTPUT_FILE)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(kolom)
        writer.writerows(semua_baris)

    # ── Ringkasan ──
    print("\n" + "=" * 60)
    print("  RINGKASAN")
    print("=" * 60)
    print(f"  Rekaman diproses    : {rekaman_diproses}")
    print(f"  Rekaman dilewati    : {rekaman_dilewati}")
    print(f"  Total jendela valid : {total_jendela}")
    print(f"  Jendela diabaikan   : {jendela_diabaikan}")
    print(f"  File output         : {os.path.abspath(output_path)}")

    # Hitung distribusi kelas
    if semua_baris:
        labels = [b[-1] for b in semua_baris]
        nama_kelas = {0: "Normal (N)", 1: "AFIB", 2: "AFL"}
        print(f"\n  Distribusi Kelas:")
        for kode, nama_k in nama_kelas.items():
            jumlah = labels.count(kode)
            print(f"    Kelas {kode} ({nama_k:10s}) : {jumlah:>6} jendela")

    print("\nSelesai! ✓")


if __name__ == "__main__":
    main()
