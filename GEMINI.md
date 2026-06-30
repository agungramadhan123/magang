# PROMPT SPESIFIKASI PEKERJAAN: AI ENGINEER UNTUK DETEKSI ATRIAL FIBRILLATION (SIHEDAF 2.0)

## 1. KONTEKS & TUJUAN
Saya sedang magang sebagai AI Engineer dalam proyek bernama SIHEDAF. Proyek ini bertujuan untuk membuat sistem deteksi penyakit Atrial Fibrillation (AFIB) berbasis smartwatch (menggunakan sensor PPG). Model klasifikasi akan ditransmisikan dan dijalankan di sisi backend server.
Karena keterbatasan data PPG berlabel klinis, strategi saya adalah melatih model menggunakan data ECG dari MIT-BIH Atrial Fibrillation Database (AFDB) yang sudah memiliki anotasi klinis lengkap. Saya akan mengekstrak fitur RR-Interval dari ECG, melatih model klasifikasi ritme jantung (Normal vs AFIB), yang nantinya model tersebut akan diumpankan data PP-Interval dari sensor PPG smartwatch.

Tolong buatkan skrip Python terstruktur (atau kode notebook) untuk menyelesaikan pipeline pemodelan AI ini langkah demi langkah:

---

## 2. PIPELINE PEKERJAAN YANG HARUS DIBUAT

### TAHAP 1: PREPROCESSING DATA & FILTER OUTLIERS
1. Baca file ECG (.dat, .hea), anotasi puncak R (.qrs), dan anotasi ritme (.atr) menggunakan library `wfdb`.
2. Hitung deret waktu RR-Interval awal dari selisih indeks sampel puncak R dibagi frekuensi sampling (fs = 250 Hz).
3. Buat fungsi filtering "Rolling Median / 20% Rule" untuk mendeteksi dan membersihkan outliers (noise artifact/ektopik) pada RR-interval untuk mengubahnya menjadi NN-interval (Normal-to-Normal) yang bersih.

### TAHAP 2: SEGMENTASI & LABELING HOMOGEN
1. Lakukan segmentasi/windowing pada deret NN-interval yang sudah bersih (misal menggunakan window berisi 100 detak jantung).
2. Lakukan pelabelan (labeling) yang homogen pada setiap window berdasarkan anotasi di file `.atr`:
   - Label `0 (Normal)` jika window tersebut 100% berisi ritme normal `(N`.
   - Label `1 (AFIB)` jika window tersebut 100% berisi ritme atrial fibrillation `(AFIB`.
   - Buang/abaikan window transisi yang bercampur antara Normal dan AFIB untuk mencegah bias pada model.

### TAHAP 3: EKSTRAKSI FITUR (HRV & NON-LINEAR)
Untuk setiap window segmentasi yang valid, ekstrak fitur-fitur Heart Rate Variability (HRV) berikut:
1. **Domain Waktu (Time-Domain)**: Mean RR, SDNN, RMSSD, pNN50, dan Coefficient of Variation (CV).
2. **Geometris & Non-Linear**:
   - Nilai $SD_1$ dan $SD_2$ dari Poincaré Plot, serta rasio $SD_1/SD_2$.
   - Sample Entropy (SampEn) untuk mengukur keacakan ritme jantung.
3. Outputkan hasil ekstraksi ini ke dalam bentuk DataFrame Pandas siap latih.

### TAHAP 4: PEMODELAN ML / DL (PATIENT-INDEPENDENT)
1. Lakukan pembagian data Train/Test secara *Patient-Independent* (record dari pasien yang ada di Test Set tidak boleh muncul sama sekali di Train Set) untuk mencegah kebocoran data (*data leakage*).
2. Latih beberapa model pengklasifikasi ringan (seperti XGBoost, Random Forest, SVM, atau MLP 3-layer).
3. Evaluasi performa model menggunakan metrik: Akurasi, Sensitivitas (Recall), Spesifisitas (Specificity), F1-Score, dan kurva ROC-AUC.

### TAHAP 5: FUNGSI INFERENSI UNTUK BACKEND
Buat sebuah fungsi inferensi Python mandiri (clean code) yang nantinya akan diserahkan ke tim backend dengan struktur input-output seperti ini:
- **Input**: Array 1D berisi deret waktu interval detak jantung (PP-Interval/RR-Interval mentah dari smartwatch).
- **Proses**: Filter outliers -> Ekstraksi fitur HRV -> Load model terlatih -> Prediksi kelas.
- **Output**: Label prediksi (Normal/AFIB) dan probabilitas keyakinan model.

---

## 3. FORMAT DAN STANDAR KODE
* Tulis kode dalam bahasa Python menggunakan library standar: `numpy`, `pandas`, `scipy`, `wfdb`, `scikit-learn`, dan `matplotlib`.
* Berikan penjelasan singkat berupa komentar di baris-baris kode yang krusial (terutama bagian rumus HRV dan filter outliers).