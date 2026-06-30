import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# Style setting
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

out_dir = '/home/gung/Dokumen/magang'
os.makedirs(out_dir, exist_ok=True)

df_all = pd.read_csv('hrv_features_dataset.csv')

# 1. Distribusi Fitur HRV per Kelas
key_features = ['Mean_RR', 'SDNN', 'RMSSD', 'pNN50', 'SD1', 'SampEn']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    normal_vals = df_all[df_all['Label'] == 0][feat]
    afib_vals = df_all[df_all['Label'] == 1][feat]

    sns.histplot(normal_vals, kde=True, color='teal', label='Normal', stat='density', alpha=0.5, bins=30, edgecolor='none', ax=ax)
    sns.histplot(afib_vals, kde=True, color='crimson', label='AFIB', stat='density', alpha=0.5, bins=30, edgecolor='none', ax=ax)

    ax.set_title(feat, fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

fig.suptitle('Distribusi Fitur HRV: Normal vs AFIB (Seluruh Record)', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(f'{out_dir}/hrv_distribution.png')
plt.close()

# 2. Confusion Matrix untuk SVM (Model Terbaik)
# Kita memuat model terbaik dan memprediksi keseluruhan data untuk ilustrasi
best_model = joblib.load('models/best_model.joblib')
best_scaler = joblib.load('models/scaler.joblib')

feature_cols = [c for c in df_all.columns if c not in ('Label', 'Patient')]
X = df_all[feature_cols].values
y_true = df_all['Label'].values
X_scaled = best_scaler.transform(X)

y_pred = best_model.predict(X_scaled)
y_prob = best_model.predict_proba(X_scaled)[:, 1]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'AFIB'], yticklabels=['Normal', 'AFIB'])
plt.title('Confusion Matrix — SVM (All Data)', fontsize=13, weight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{out_dir}/confusion_matrix.png')
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"SVM (AUC = {roc_auc_val:.4f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')
plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Klasifikasi Normal vs AFIB', fontsize=14, weight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{out_dir}/roc_curve.png')
plt.close()

print("Images generated.")
