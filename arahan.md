bantu saya membuat dataset berdasarkan database di folder files.
ouput dari program ini dapat membuat sebuah data yang bersih dan siap pakai oleh model machine learning maupun deeplearning.
yang perlu anda lakukan ekstraksi agar mendapatkan fitur RR dengan tahapan:
1. Baca file .hea untuk mengonfirmasi bahwa sampling rate ($f_s$) adalah 250 Hz.
2. Baca file .qrsc Hitung selisih antar-sampel detak berturut-turut, lalu konversikan ke satuan waktu (milidetik) menggunakan angka 250 tadi. Sekarang kamu punya array panjang berisi semua nilai IBI. 
3. Potong data menjadi jendela-jendela waktu berdurasi 10 detik. Caranya, kelompokkan nilai-nilai IBI yang waktu kejadiannya jatuh di dalam rentang detik ke 0–10, 10–20, 20–30, dan seterusnya.
4. Di setiap jendela 10 detik, hitung fitur-fitur HRV seperti:
Mean IBI
Median IBI
SDNN
RMSSD
Coefficient of Variation
Mean/Std Successive Difference
Turning Point Ratio
Poincaré SD1
Poincaré SD2
SD1/SD2
5. Buka file .atr. Cari tahu ritme apa yang mendominasi rentang waktu 10 detik tersebut (misal detik 10–20 dominan (AFIB), lalu petakan menjadi angka indeks kelas (0 untuk N, 1 untuk AFIB, 2 untuk AFL). Jika ada label selain ketiga itu abaikan jendela tersebut.
6. Preprocessing & Penanganan Outlier:
Gunakan metode IQR Clipping dengan parameter `lower_percentile=0.01`, `upper_percentile=0.99`, dan `factor=1.5`. Pendekatan ini bukan untuk membuang anomali statistik biasa, melainkan bertindak sebagai "Safety Net" untuk memotong murni error/glitch dari sensor tanpa memotong fluktuasi alami sinyal jantung ekstrem dari penyakit Aritmia.