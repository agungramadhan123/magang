"""
modul.py — Pipeline Lengkap Deteksi Atrial Fibrillation (SIHEDAF 2.0)
======================================================================
Modul ini mengimplementasikan 5 tahap pipeline sesuai spesifikasi GEMINI.md:

  TAHAP 1: Preprocessing Data & Filter Outliers (Rolling Median / 20% Rule)
  TAHAP 2: Segmentasi & Labeling Homogen (window 100 beat)
  TAHAP 3: Ekstraksi Fitur HRV (Time-Domain + Non-Linear)
  TAHAP 4: Pemodelan ML Patient-Independent (RF, SVM, MLP)
  TAHAP 5: Fungsi Inferensi untuk Backend

Library: numpy, pandas, scipy, wfdb, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as scipy_signal

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib

warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# TAHAP 1: PREPROCESSING DATA & FILTER OUTLIERS
# ════════════════════════════════════════════════════════════════════════════

def load_annotations(record_path: str):
    """
    Membaca anotasi QRS (puncak R) dan ATR (label ritme) dari file WFDB.

    Parameters
    ----------
    record_path : str
        Path ke record tanpa ekstensi, misal 'files/04015'.

    Returns
    -------
    beat_ann : wfdb.Annotation
        Anotasi lokasi puncak R (QRS).
    rhythm_ann : wfdb.Annotation
        Anotasi label ritme (Normal vs AFIB).
    fs : int/float
        Sampling frequency (Hz).
    """
    beat_ann = wfdb.rdann(record_path, 'qrs')
    rhythm_ann = wfdb.rdann(record_path, 'atr')
    fs = beat_ann.fs
    return beat_ann, rhythm_ann, fs


def compute_rr_intervals(beat_ann, fs):
    """
    Menghitung RR-Interval dalam detik dari selisih indeks sampel puncak R
    dibagi frekuensi sampling.

    Returns
    -------
    rr_intervals : np.ndarray  — durasi RR-interval (detik)
    rr_start_samples : np.ndarray — indeks sampel awal tiap RR
    """
    r_peaks = beat_ann.sample
    rr_intervals = np.diff(r_peaks) / fs
    rr_start_samples = r_peaks[:-1]
    return rr_intervals, rr_start_samples


def filter_outliers_rolling_median(rr_intervals, threshold_ratio=0.2):
    """
    Filter outlier RR-interval menggunakan Rolling Median / 20% Rule.

    Algoritma:
    - Hitung rolling median (window=11) dari deret RR-interval.
    - Tandai RR-interval sebagai outlier jika deviasi dari median lokal
      melebihi 20% (threshold_ratio) dari nilai median tersebut.
    - RR-interval outlier diganti dengan nilai median lokal (menghasilkan
      NN-interval / Normal-to-Normal interval yang bersih).

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array RR-interval mentah dalam detik.
    threshold_ratio : float
        Toleransi deviasi dari median lokal (default: 0.2 = 20%).

    Returns
    -------
    nn_intervals : np.ndarray
        Array NN-interval yang sudah dibersihkan dari outlier.
    is_valid : np.ndarray (bool)
        Mask boolean — True jika RR asli valid (bukan outlier).
    """
    n = len(rr_intervals)
    nn_intervals = rr_intervals.copy()
    is_valid = np.ones(n, dtype=bool)

    # Rolling median dengan window 11 (5 tetangga kiri + 5 kanan)
    window = 11
    half_w = window // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        local_median = np.median(rr_intervals[start:end])

        # 20% Rule: jika |RR - median| > 20% * median → outlier
        if abs(rr_intervals[i] - local_median) > threshold_ratio * local_median:
            nn_intervals[i] = local_median  # Ganti dengan median lokal
            is_valid[i] = False

    return nn_intervals, is_valid


def label_rr_intervals(rhythm_ann, rr_start_samples):
    """
    Labeling dinamis — memetakan setiap RR-interval ke ritme yang sedang aktif
    menggunakan binary search (np.searchsorted).

    Returns
    -------
    rr_labels : np.ndarray — label ritme per RR (misal '(N', '(AFIB')
    """
    rhythm_change_samples = rhythm_ann.sample
    rhythm_labels = rhythm_ann.aux_note

    # Binary search O(log n) untuk mencari ritme aktif
    rhythm_indices = np.searchsorted(rhythm_change_samples, rr_start_samples) - 1
    rhythm_indices = np.clip(rhythm_indices, 0, len(rhythm_labels) - 1)

    rr_labels = np.array([rhythm_labels[idx] for idx in rhythm_indices])
    return rr_labels


def preprocess_single_record(record_path):
    """
    Pipeline preprocessing lengkap untuk satu record:
    Load → Hitung RR → Label → Filter Outlier (Rolling Median 20% Rule)

    Returns
    -------
    nn_intervals : np.ndarray — NN-interval bersih (detik)
    labels : np.ndarray — label ritme bersih ('N', 'AFIB', dll)
    """
    beat_ann, rhythm_ann, fs = load_annotations(record_path)
    rr_intervals, rr_start_samples = compute_rr_intervals(beat_ann, fs)
    rr_labels = label_rr_intervals(rhythm_ann, rr_start_samples)

    # Bersihkan label: hapus karakter '('
    clean_labels = np.array([lbl.replace('(', '') for lbl in rr_labels])

    # Filter outlier menggunakan Rolling Median / 20% Rule
    nn_intervals, is_valid = filter_outliers_rolling_median(rr_intervals)

    return nn_intervals, clean_labels, is_valid


# ════════════════════════════════════════════════════════════════════════════
# TAHAP 2: SEGMENTASI & LABELING HOMOGEN
# ════════════════════════════════════════════════════════════════════════════

def segment_and_label(nn_intervals, labels, window_size=100):
    """
    Segmentasi deret NN-interval menjadi window berisi `window_size` detak,
    lalu labeling homogen:
      - Label 0 (Normal) jika 100% window berisi ritme 'N'
      - Label 1 (AFIB)   jika 100% window berisi ritme 'AFIB'
      - Buang window transisi (campuran N dan AFIB) untuk mencegah bias

    Parameters
    ----------
    nn_intervals : np.ndarray — deret NN-interval bersih
    labels : np.ndarray — label ritme per interval
    window_size : int — jumlah detak per window (default: 100)

    Returns
    -------
    segments : list[np.ndarray] — list segment NN-interval
    segment_labels : list[int] — label per segment (0=Normal, 1=AFIB)
    """
    n = len(nn_intervals)
    segments = []
    segment_labels = []

    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        window_nn = nn_intervals[start:end]
        window_labels = labels[start:end]

        unique_labels = set(window_labels)

        # Labeling homogen — hanya terima window murni N atau murni AFIB
        if unique_labels == {'N'}:
            segments.append(window_nn)
            segment_labels.append(0)  # Normal
        elif unique_labels == {'AFIB'}:
            segments.append(window_nn)
            segment_labels.append(1)  # AFIB
        # Window campuran (transisi) → dibuang untuk mencegah bias

    return segments, segment_labels


# ════════════════════════════════════════════════════════════════════════════
# TAHAP 3: EKSTRAKSI FITUR (HRV & NON-LINEAR)
# ════════════════════════════════════════════════════════════════════════════

def _sample_entropy(rr, m=2, r_ratio=0.2):
    """
    Menghitung Sample Entropy (SampEn) untuk mengukur keacakan/irregularitas
    ritme jantung.

    SampEn = -ln(A/B) dimana:
      - B = jumlah pasangan template match panjang m
      - A = jumlah pasangan template match panjang m+1
      - r = toleransi = r_ratio * std(rr)

    Nilai tinggi → ritme lebih acak/irregular (ciri AFIB)
    Nilai rendah → ritme lebih teratur (ciri Normal)

    Parameters
    ----------
    rr : np.ndarray — segment RR/NN-interval
    m : int — embedding dimension (default: 2)
    r_ratio : float — toleransi relatif terhadap std (default: 0.2)
    """
    N = len(rr)
    r = r_ratio * np.std(rr)

    if r == 0 or N < m + 2:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([rr[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0:
        return 0.0

    return -np.log(A / B) if A > 0 else 0.0


def extract_hrv_features(rr_segment):
    """
    Ekstraksi fitur Heart Rate Variability (HRV) dari satu segment NN-interval.

    Fitur yang diekstrak:
    ─── Domain Waktu (Time-Domain) ───
      1. Mean RR      : rata-rata interval (detik)
      2. SDNN         : standar deviasi interval
      3. RMSSD        : root mean square of successive differences
      4. pNN50        : persentase selisih berturutan > 50ms
      5. CV           : coefficient of variation (SDNN / Mean RR)

    ─── Geometris & Non-Linear ───
      6. SD1          : dispersi jangka pendek (Poincaré)
      7. SD2          : dispersi jangka panjang (Poincaré)
      8. SD1_SD2_ratio: rasio SD1/SD2
      9. SampEn       : Sample Entropy (keacakan ritme)

    Parameters
    ----------
    rr_segment : np.ndarray — array NN-interval dalam detik

    Returns
    -------
    features : dict — dictionary fitur HRV
    """
    rr = np.array(rr_segment, dtype=float)
    diff_rr = np.diff(rr)  # Selisih berturutan

    # ── Domain Waktu ──
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)  # Standar deviasi (ddof=1 untuk sample)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))  # Root mean square successive diff
    nn50 = np.sum(np.abs(diff_rr) > 0.05)   # Selisih > 50ms
    pnn50 = (nn50 / len(diff_rr)) * 100     # Persentase
    cv = sdnn / mean_rr if mean_rr > 0 else 0  # Coefficient of Variation

    # ── Poincaré Plot: SD1 dan SD2 ──
    # SD1 = std dari selisih berturutan / sqrt(2) → variabilitas jangka pendek
    # SD2 = std dari penjumlahan berturutan / sqrt(2) → variabilitas jangka panjang
    rr_n = rr[:-1]   # RR(n)
    rr_n1 = rr[1:]   # RR(n+1)
    sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)
    sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)
    sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else 0

    # ── Sample Entropy ──
    sampen = _sample_entropy(rr, m=2, r_ratio=0.2)

    return {
        'Mean_RR': mean_rr,
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'CV': cv,
        'SD1': sd1,
        'SD2': sd2,
        'SD1_SD2_ratio': sd1_sd2_ratio,
        'SampEn': sampen,
    }


def build_feature_matrix(segments, segment_labels):
    """
    Membangun matriks fitur X dan vektor label y dari seluruh segmen.

    Returns
    -------
    df_features : pd.DataFrame — DataFrame fitur HRV siap latih (termasuk kolom 'Label')
    """
    rows = []
    for seg, lbl in zip(segments, segment_labels):
        feats = extract_hrv_features(seg)
        feats['Label'] = lbl
        rows.append(feats)

    df_features = pd.DataFrame(rows)
    return df_features


# ════════════════════════════════════════════════════════════════════════════
# MULTI-RECORD PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def get_valid_record_ids(files_dir='files'):
    """
    Membaca file RECORDS dan mengembalikan ID record yang valid
    (memiliki file .dat, .qrs, dan .atr).
    """
    records_file = os.path.join(files_dir, 'RECORDS')
    with open(records_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    valid_ids = []
    for rec_id in all_ids:
        rec_path = os.path.join(files_dir, rec_id)
        has_dat = os.path.exists(f'{rec_path}.dat')
        has_qrs = os.path.exists(f'{rec_path}.qrs')
        has_atr = os.path.exists(f'{rec_path}.atr')
        if has_dat and has_qrs and has_atr:
            valid_ids.append(rec_id)

    return valid_ids


def multi_record_pipeline(files_dir='files', window_size=100):
    """
    Pipeline lengkap Tahap 1-3 untuk SELURUH record valid.

    Untuk setiap record:
      1. Preprocessing & filter outlier (Rolling Median 20% Rule)
      2. Segmentasi window (100 beat) dengan labeling homogen
      3. Ekstraksi fitur HRV

    Returns
    -------
    df_all : pd.DataFrame — DataFrame gabungan fitur HRV dari semua record
    patient_ids : np.ndarray — ID pasien untuk tiap baris (untuk patient-split)
    """
    valid_ids = get_valid_record_ids(files_dir)
    print(f"Record valid ditemukan: {len(valid_ids)}")

    all_rows = []

    for rec_id in valid_ids:
        rec_path = os.path.join(files_dir, rec_id)
        print(f"  Memproses {rec_id}...", end=' ')

        try:
            # Tahap 1: Preprocessing
            nn_intervals, labels, is_valid = preprocess_single_record(rec_path)

            # Tahap 2: Segmentasi & labeling homogen
            segments, segment_labels = segment_and_label(
                nn_intervals, labels, window_size=window_size
            )

            if len(segments) == 0:
                print(f"0 segmen valid — skip")
                continue

            # Tahap 3: Ekstraksi fitur HRV
            for seg, lbl in zip(segments, segment_labels):
                feats = extract_hrv_features(seg)
                feats['Label'] = lbl
                feats['Patient'] = rec_id
                all_rows.append(feats)

            n_normal = sum(1 for l in segment_labels if l == 0)
            n_afib = sum(1 for l in segment_labels if l == 1)
            print(f"{len(segments)} segmen (N={n_normal}, AFIB={n_afib})")

        except Exception as e:
            print(f"ERROR — {e}")
            continue

    df_all = pd.DataFrame(all_rows)
    patient_ids = df_all['Patient'].values if len(df_all) > 0 else np.array([])

    print(f"\n═══ TOTAL ═══")
    print(f"Total segmen : {len(df_all)}")
    if len(df_all) > 0:
        print(f"Normal (0)   : {(df_all['Label'] == 0).sum()}")
        print(f"AFIB (1)     : {(df_all['Label'] == 1).sum()}")
        print(f"Pasien unik  : {df_all['Patient'].nunique()}")

    return df_all, patient_ids


# ════════════════════════════════════════════════════════════════════════════
# TAHAP 4: PEMODELAN ML / DL (PATIENT-INDEPENDENT)
# ════════════════════════════════════════════════════════════════════════════

def train_evaluate_models(df_features, model_save_dir='models'):
    """
    Tahap 4: Latih dan evaluasi model klasifikasi dengan pembagian data
    Patient-Independent menggunakan GroupKFold.

    Mencegah data leakage: record dari pasien yang sama TIDAK akan muncul
    di train set dan test set secara bersamaan.

    Model yang dilatih:
      1. Random Forest
      2. SVM (RBF kernel)
      3. MLP 3-layer (64-32-16 neuron)

    Metrik evaluasi: Akurasi, Sensitivitas, Spesifisitas, F1-Score, ROC-AUC.

    Parameters
    ----------
    df_features : pd.DataFrame — DataFrame fitur HRV (dengan kolom 'Label' dan 'Patient')
    model_save_dir : str — direktori untuk menyimpan model terlatih

    Returns
    -------
    results : dict — hasil evaluasi per model
    best_model : object — model dengan F1-Score tertinggi
    best_scaler : StandardScaler — scaler yang digunakan untuk model terbaik
    feature_names : list — nama fitur yang digunakan
    """
    os.makedirs(model_save_dir, exist_ok=True)

    feature_cols = [c for c in df_features.columns if c not in ('Label', 'Patient')]
    X = df_features[feature_cols].values
    y = df_features['Label'].values
    groups = df_features['Patient'].values

    feature_names = feature_cols

    # Definisi model
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale', probability=True, random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),  # 3-layer
            activation='relu', solver='adam',
            max_iter=500, random_state=42
        ),
    }

    # Patient-Independent split menggunakan GroupKFold (5 fold)
    gkf = GroupKFold(n_splits=5)
    results = {}

    print("\n════ TAHAP 4: PEMODELAN ML (PATIENT-INDEPENDENT) ════\n")

    best_f1 = -1
    best_model = None
    best_scaler = None

    for name, model in models.items():
        print(f"── {name} ──")

        # Kumpulkan prediksi dari semua fold
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standarisasi fitur (fit hanya pada train, transform keduanya)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train
            model_clone = _clone_model(model)
            model_clone.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model_clone.predict(X_test_scaled)
            y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)

        # Metrik evaluasi
        acc = accuracy_score(all_y_true, all_y_pred)
        sens = recall_score(all_y_true, all_y_pred, pos_label=1)      # Sensitivitas
        spec = recall_score(all_y_true, all_y_pred, pos_label=0)      # Spesifisitas
        f1 = f1_score(all_y_true, all_y_pred, pos_label=1)
        roc_auc = roc_auc_score(all_y_true, all_y_prob)
        cm = confusion_matrix(all_y_true, all_y_pred)

        results[name] = {
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'y_prob': all_y_prob,
        }

        print(f"  Akurasi      : {acc:.4f}")
        print(f"  Sensitivitas : {sens:.4f}")
        print(f"  Spesifisitas : {spec:.4f}")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  ROC-AUC      : {roc_auc:.4f}")
        print(f"  Confusion Mx : {cm.tolist()}")
        print()

        # Track model terbaik berdasarkan F1
        if f1 > best_f1:
            best_f1 = f1
            # Re-train model terbaik pada SELURUH data untuk deployment
            final_scaler = StandardScaler()
            X_scaled_all = final_scaler.fit_transform(X)
            final_model = _clone_model(model)
            final_model.fit(X_scaled_all, y)
            best_model = final_model
            best_scaler = final_scaler
            best_name = name

    # Simpan model terbaik
    model_path = os.path.join(model_save_dir, 'best_model.joblib')
    scaler_path = os.path.join(model_save_dir, 'scaler.joblib')
    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)
    print(f"✓ Model terbaik ({best_name}, F1={best_f1:.4f}) disimpan ke {model_path}")
    print(f"✓ Scaler disimpan ke {scaler_path}")

    return results, best_model, best_scaler, feature_names


def _clone_model(model):
    """Clone model sklearn dengan parameter yang sama."""
    from sklearn.base import clone
    return clone(model)


# ════════════════════════════════════════════════════════════════════════════
# TAHAP 5: FUNGSI INFERENSI UNTUK BACKEND
# ════════════════════════════════════════════════════════════════════════════

def predict_rhythm(raw_intervals, model_path='models/best_model.joblib',
                   scaler_path='models/scaler.joblib'):
    """
    Fungsi inferensi mandiri untuk tim backend.

    INPUT:
        raw_intervals : array 1D berisi deret waktu interval detak jantung
                        (PP-Interval / RR-Interval mentah dari smartwatch),
                        dalam satuan DETIK.

    PROSES:
        1. Filter outliers (Rolling Median / 20% Rule) → NN-interval bersih
        2. Ekstraksi fitur HRV (9 fitur time-domain + non-linear)
        3. Load model terlatih & scaler
        4. Prediksi kelas

    OUTPUT:
        dict dengan:
          - 'prediction': str — 'Normal' atau 'AFIB'
          - 'probability': float — probabilitas keyakinan model (0.0 - 1.0)
          - 'features': dict — fitur HRV yang diekstrak (untuk debugging)

    Contoh penggunaan:
    >>> result = predict_rhythm([0.82, 0.85, 0.79, 0.83, ...])
    >>> print(result['prediction'], result['probability'])
    'Normal' 0.97
    """
    raw_intervals = np.array(raw_intervals, dtype=float)

    if len(raw_intervals) < 10:
        raise ValueError(
            f"Minimal 10 interval diperlukan, diterima {len(raw_intervals)}"
        )

    # Langkah 1: Filter outliers → NN-interval bersih
    nn_intervals, _ = filter_outliers_rolling_median(raw_intervals, threshold_ratio=0.2)

    # Langkah 2: Ekstraksi fitur HRV
    features = extract_hrv_features(nn_intervals)

    # Langkah 3: Load model & scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Langkah 4: Prediksi
    feature_names = ['Mean_RR', 'SDNN', 'RMSSD', 'pNN50', 'CV',
                     'SD1', 'SD2', 'SD1_SD2_ratio', 'SampEn']
    X = np.array([[features[f] for f in feature_names]])
    X_scaled = scaler.transform(X)

    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    label = 'Normal' if pred_class == 0 else 'AFIB'
    confidence = pred_proba[pred_class]

    return {
        'prediction': label,
        'probability': float(confidence),
        'features': features,
    }


# ════════════════════════════════════════════════════════════════════════════
# SELF-TEST (saat dijalankan langsung: python modul.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("  SIHEDAF 2.0 — Pipeline Deteksi Atrial Fibrillation")
    print("=" * 70)

    # ── Tahap 1-3: Multi-record preprocessing + feature extraction ──
    df_all, patient_ids = multi_record_pipeline(files_dir='files', window_size=100)

    if len(df_all) == 0:
        print("ERROR: Tidak ada segmen valid. Pipeline berhenti.")
        exit(1)

    # Simpan dataset fitur
    df_all.to_csv('hrv_features_dataset.csv', index=False)
    print(f"\n✓ Dataset fitur disimpan ke hrv_features_dataset.csv")

    # ── Tahap 4: Pemodelan ML ──
    results, best_model, best_scaler, feature_names = train_evaluate_models(df_all)

    # ── Tahap 5: Test inferensi ──
    print("\n════ TAHAP 5: TEST FUNGSI INFERENSI ════\n")
    # Ambil sampel NN-interval Normal dari record 04015
    nn_sample, labels_sample, _ = preprocess_single_record('files/04015')
    # Ambil 100 interval pertama yang berlabel N
    normal_mask = labels_sample == 'N'
    normal_nn = nn_sample[normal_mask][:100]

    test_result = predict_rhythm(
        normal_nn,
        model_path='models/best_model.joblib',
        scaler_path='models/scaler.joblib'
    )
    print(f"  Input       : {len(normal_nn)} interval (dari record 04015, ritme Normal)")
    print(f"  Prediksi    : {test_result['prediction']}")
    print(f"  Probabilitas: {test_result['probability']:.4f}")

    print("\n" + "=" * 70)
    print("  ✓ Pipeline lengkap selesai tanpa error.")
    print("=" * 70)
