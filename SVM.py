# =========================================================
# ECG MIT-BIH - SVM WITH MORPHOLOGICAL FEATURE EXTRACTION
# Performance Evaluation per N Subsets with Custom Grid Search
# =========================================================

import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# =========================================================
# CONFIGURATION & PATHS
# =========================================================
dataset_path = r"C:\Users\amine\OneDrive\Desktop\PFE-CODE\mitbih"

records = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230','100','103','105','111','113','117','121','123',
    '200','202','210','212','213','214','219','221','222','228',
    '231','232','233','234'
]

CLASSES = {'N':['N'], 'L':['L'], 'R':['R'], 'V':['V'], 'A':['A']}
CLASSES_LIST = ['N', 'L', 'R', 'V', 'A']
label_map = {cls: idx for idx, cls in enumerate(CLASSES_LIST)}
WINDOW = 150  
FS = 360      

# Dernaha khawia hna bech n'initialisouha dynamique l-taht hda N_values
tables_data_store = {}

# =========================================================
# SIGNAL PROCESSING & FEATURE EXTRACTION ENGINE
# =========================================================

def butter_bandpass(sig, fs=360, low=0.5, high=40):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def map_class(sym):
    for k, v in CLASSES.items():
        if sym in v: return k
    return None

def extract_morphological_features(beat, fs=360):
    r_idx = len(beat) // 2
    r_amp = beat[r_idx]
    
    q_zone = beat[max(0, r_idx-30):r_idx]
    s_zone = beat[r_idx:min(len(beat), r_idx+30)]
    p_zone = beat[max(0, r_idx-90):max(0, r_idx-30)]
    t_zone = beat[min(len(beat), r_idx+30):min(len(beat), r_idx+130)]
    
    q_idx = (r_idx - 30) + np.argmin(q_zone) if len(q_zone) > 0 else r_idx - 15
    q_amp = beat[q_idx]
    
    s_idx = r_idx + np.argmin(s_zone) if len(s_zone) > 0 else r_idx + 15
    s_amp = beat[s_idx]
    
    p_idx = (r_idx - 90) + np.argmax(p_zone) if len(p_zone) > 0 else r_idx - 60
    p_amp = beat[p_idx]
    
    t_idx = (r_idx + 30) + np.argmax(t_zone) if len(t_zone) > 0 else r_idx + 75
    t_amp = beat[t_idx]
    
    pr_interval = ((r_idx - p_idx) / fs) * 1000
    qrs_duration = ((s_idx - q_idx) / fs) * 1000
    qt_interval = ((t_idx - q_idx) / fs) * 1000
    
    return [
        p_amp, q_amp, r_amp, s_amp, t_amp,          
        (r_idx - p_idx), (r_idx - q_idx),           
        (s_idx - r_idx), (t_idx - r_idx),
        pr_interval, qrs_duration, qt_interval      
    ]

def extract_table_metrics(cm, classes_list):
    total_samples = cm.sum()
    row_sums = cm.sum(axis=1)
    
    global_acc = np.diag(cm).sum() / total_samples if total_samples > 0 else 0.0
    global_err = 1.0 - global_acc
    
    class_breakdown = {}
    for i, cls in enumerate(classes_list):
        acc = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0
        err = 1.0 - acc
        class_breakdown[cls] = {'Accuracy': acc, 'ErrorRate': err}
        
    return global_acc, global_err, class_breakdown

# =========================================================
# PIPELINE DATA PREPARATION
# =========================================================

features_list, labels_list = [], []
print("Step 1: Processing signals and extracting morphological features... Please wait...")

for rec in records:
    try:
        r_path = os.path.join(dataset_path, rec)
        record = wfdb.rdrecord(r_path); ann = wfdb.rdann(r_path, "atr")
        sig = butter_bandpass(record.p_signal[:, 0], fs=FS)
        
        for pos, sym in zip(ann.sample, ann.symbol):
            cls = map_class(sym)
            if cls and 0 <= pos - WINDOW and pos + WINDOW < len(sig):
                features_list.append(extract_morphological_features(sig[pos - WINDOW : pos + WINDOW], fs=FS))
                labels_list.append(cls)
    except: continue

X = np.array(features_list)
y = np.array([label_map[x] for x in labels_list])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# NEW CUSTOM HYPERPARAMETER CONFIGURATIONS (UPDATED)
# =========================================================
N_values = [200, 500, 800]  # Zdna l-500 hna kima bghit

# FIX: Initialisation dynamique kamla bech mathandalech f l-KeyError
tables_data_store = {N: {} for N in N_values}

linear_C_values = [0.1, 1, 10, 100]
rbf_C_values = [1, 10, 20]
rbf_gamma_values = [0.01, 0.1, 1]

svm_configurations = []

# 1. Linear SVM Grid
for c in linear_C_values:
    svm_configurations.append({
        'label': f'Linear C={c}', 
        'kernel': 'linear', 
        'C': c, 
        'gamma': 'scale'
    })

# 2. RBF SVM Grid
for c in rbf_C_values:
    for g in rbf_gamma_values:
        svm_configurations.append({
            'label': f'RBF C={c} G={g}', 
            'kernel': 'rbf', 
            'C': c, 
            'gamma': g
        })

# Core Training Loop (Silent Mode)
for N in N_values:
    idx = []
    for cls_idx in range(5):
        ids = np.where(y_train == cls_idx)[0]
        np.random.seed(42)
        idx.extend(np.random.choice(ids, min(N, len(ids)), replace=False))
    
    Xs, ys = X_train[idx], y_train[idx]
    
    for config in svm_configurations:
        label = config['label']
        
        clf = SVC(kernel=config['kernel'], C=config['C'], gamma=config['gamma'], random_state=42)
        clf.fit(Xs, ys)
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=range(5))
        
        g_acc, g_err, c_breakdown = extract_table_metrics(cm, CLASSES_LIST)
        
        tables_data_store[N][label] = {
            'global_acc': g_acc, 'global_err': g_err, 'classes': c_breakdown
        }

# =========================================================
# SEPARATE MASTER TABLES GENERATOR (SPACING OPTIMIZED)
# =========================================================
print("\nProcessing Complete! Displaying separated clean reports for Linear and RBF...")

target_classes = ['A', 'R', 'N', 'L', 'V']

for N in N_values:
    
    # -----------------------------------------------------
    # TABLE 1: LINEAR SVM PERFORMANCE REPORT
    # -----------------------------------------------------
    linear_table_rows = []
    for config in svm_configurations:
        if config['kernel'] == 'linear':
            label = config['label']
            m_set = tables_data_store[N][label]
            
            linear_table_rows.append([f"{label}", "global", "A", "R", "N", "L", "V"])
            
            acc_row = ["Accurcy", f"{m_set['global_acc']:.3f}"]
            for cls in target_classes:
                acc_row.append(f"{m_set['classes'][cls]['Accuracy']:.3f}")
            linear_table_rows.append(acc_row)
            
            err_row = ["Erreur rate", f"{m_set['global_err']:.3f}"]
            for cls in target_classes:
                err_row.append(f"{m_set['classes'][cls]['ErrorRate']:.3f}")
            linear_table_rows.append(err_row)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    rendered_linear_table = ax.table(cellText=linear_table_rows, loc='center', cellLoc='center')
    rendered_linear_table.auto_set_font_size(False)
    rendered_linear_table.set_fontsize(10)
    rendered_linear_table.scale(1.1, 1.6)
    
    for r_idx, r_content in enumerate(linear_table_rows):
        if "Linear" in str(r_content[0]):
            for c_idx in range(len(r_content)):
                cell = rendered_linear_table[r_idx, c_idx]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#dcdcdc')
        else:
            rendered_linear_table[r_idx, 0].set_text_props(weight='bold')
            
    plt.title(f"LINEAR KERNEL SVM PERFORMANCE REPORT (N = {N})", weight='bold', fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # TABLE 2: RBF SVM PERFORMANCE REPORT (ANTI-OVERLAP FIX)
    # -----------------------------------------------------
    rbf_table_rows = []
    for config in svm_configurations:
        if config['kernel'] == 'rbf':
            label = config['label']
            m_set = tables_data_store[N][label]
            
            rbf_table_rows.append([f"{label}", "global", "A", "R", "N", "L", "V"])
            
            acc_row = ["Accurcy", f"{m_set['global_acc']:.3f}"]
            for cls in target_classes:
                acc_row.append(f"{m_set['classes'][cls]['Accuracy']:.3f}")
            rbf_table_rows.append(acc_row)
            
            err_row = ["Erreur rate", f"{m_set['global_err']:.3f}"]
            for cls in target_classes:
                err_row.append(f"{m_set['classes'][cls]['ErrorRate']:.3f}")
            rbf_table_rows.append(err_row)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')
    rendered_rbf_table = ax.table(cellText=rbf_table_rows, loc='center', cellLoc='center')
    rendered_rbf_table.auto_set_font_size(False)
    rendered_rbf_table.set_fontsize(10) 
    rendered_rbf_table.scale(1.1, 1.6)    
    
    for r_idx, r_content in enumerate(rbf_table_rows):
        if "RBF" in str(r_content[0]):
            for c_idx in range(len(r_content)):
                cell = rendered_rbf_table[r_idx, c_idx]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e2e8f0') 
        else:
            rendered_rbf_table[r_idx, 0].set_text_props(weight='bold')
            
    plt.title(f"RBF KERNEL SVM PERFORMANCE REPORT (N = {N})", weight='bold', fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()