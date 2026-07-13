# Penjelasan: Parameter IQR Clipping pada Data HRV (Aritmia)

Dalam pemodelan data medis seperti Heart Rate Variability (HRV) untuk deteksi Atrial Fibrillation (AFIB) dan Atrial Flutter (AFL), penanganan *outlier* tidak bisa disamakan dengan data umum.

## 1. Mengapa 1% (0.01) dan 99% (0.99) alih-alih 25% dan 75%?
* **Penyakit Aritmia Sangat Fluktuatif**: Jantung penderita AFIB/AFL secara alami menghasilkan jarak antar detak (IBI) yang sangat ekstrem tak beraturan.
* **Mencegah Over-clipping**: Jika menggunakan rentang kuartil standar (25% dan 75%), terlalu banyak "sinyal penyakit" yang terpotong karena dianggap *error*. Penggunaan rentang 1% - 99% difungsikan untuk memperlebar batas "kewajaran", sehingga kita hanya membuang *noise* murni (misalnya goyangan kabel sensor EKG).

## 2. Peran Faktor 1.5
Secara matematis, faktor 1.5 dikalikan dengan selisih rentang persentil. Karena rentang yang kita gunakan sudah teramat lebar (persentil 1 hingga 99), hasil perkaliannya menciptakan batas potong (*clipping bounds*) yang **luar biasa jauh**.

**Fungsi Utama**:
Kombinasi ini tidak lagi berfungsi sebagai detektor anomali statistik biasa, melainkan bertransformasi menjadi **Jaring Pengaman (Safety Net)** absolut. Jaring ini hanya akan memangkas angka yang mustahil secara biologis (seperti error sensor yang melonjak hingga jutaan), sehingga algoritma model terlindungi dari kerusakan komputasi. Di saat yang sama, data ekstrem asli bawaan penyakit jantung akan terjamin 100% utuh dan lolos ke dalam pemodelan.
