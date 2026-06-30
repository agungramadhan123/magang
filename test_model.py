from modul import predict_rhythm

print("=== PENGUJIAN MODEL SIHEDAF 2.0 ===")

# Skenario 1: Data Pasien Normal (Irama Jantung Teratur)
# Selisih detak jantung rata-rata 0.8 detik (~75 bpm) dengan variasi sangat kecil
data_pasien_normal = [0.81, 0.82, 0.80, 0.81, 0.83, 0.81, 0.82, 0.80, 0.81, 0.82, 0.81, 0.83]

try:
    hasil_1 = predict_rhythm(data_pasien_normal)
    print("\n[Tes 1] Data Ritme Teratur:")
    print(f"Prediksi     : {hasil_1['prediction']}")
    print(f"Probabilitas : {hasil_1['probability'] * 100:.2f}%")
except Exception as e:
    print(f"Error: {e}")

# Skenario 2: Data Pasien AFIB (Irama Jantung Acak & Cepat)
# Selisih detak jantung bervariasi drastis antara cepat (0.3s) dan lambat (0.8s)
data_pasien_afib = [0.41, 0.75, 0.35, 0.55, 0.82, 0.44, 0.38, 0.70, 0.42, 0.65, 0.33, 0.52]

try:
    hasil_2 = predict_rhythm(data_pasien_afib)
    print("\n[Tes 2] Data Ritme Acak (Gejala AFIB):")
    print(f"Prediksi     : {hasil_2['prediction']}")
    print(f"Probabilitas : {hasil_2['probability'] * 100:.2f}%")
except Exception as e:
    print(f"Error: {e}")
