# =========================================================
# ECG MIT-BIH - SVM WITH MORPHOLOGICAL FEATURE EXTRACTION
# Performance Evaluation per N Subsets (200, 400, 600) with Dashboards
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

# Master storage to hold data for the final structural summary tables
tables_data_store = {200: {}, 400: {}, 600: {}}

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

# =========================================================
# METRICS COMPLIANCE EXTRACTOR
# =========================================================

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
# UNIFIED VISUALIZATION DASHBOARD
# =========================================================

def plot_evaluation_dashboard(cm, classes, title, global_acc, global_err, class_breakdown):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1)
    total_samples = cm.sum()
    
    labeled_classes = [f"{cls}\n({int(row_sums[i])})" for i, cls in enumerate(classes)]
    cm_percent = np.divide(cm.astype('float'), row_sums[:, None], out=np.zeros_like(cm, dtype=float), where=row_sums[:, None] != 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.1, 1.1]})

    # --- Panel A: Confusion Matrix Matrix ---
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks(np.arange(len(classes))); ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, weight='bold'); ax1.set_yticklabels(labeled_classes, weight='bold')
    ax1.set_xlabel("Predicted Class", weight='bold')
    ax1.set_ylabel("True Class (Sample Count)", weight='bold')
    ax1.set_title(f"Confusion Matrix (Total Test N = {total_samples})", weight='bold', fontsize=11)

    thresh = cm.max() * 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax1.text(j, i, f"{cm[i, j]}\n{cm_percent[i, j]*100:.1f}%", ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # --- Panel B: Dynamic Sub-Configuration Table ---
    ax2.axis('off')
    target_classes = ['A', 'R', 'N', 'L', 'V']
    
    table_rows = [
        ["Configuration", "global", "A", "R", "N", "L", "V"],
        ["Accurcy", f"{global_acc:.3f}"] + [f"{class_breakdown[c]['Accuracy']:.3f}" for c in target_classes],
        ["Erreur rate", f"{global_err:.3f}"] + [f"{class_breakdown[c]['ErrorRate']:.3f}" for c in target_classes]
    ]
    
    rendered_table = ax2.table(cellText=table_rows, loc='center', cellLoc='center')
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(11)
    rendered_table.scale(1.0, 2.5)
    
    # Shade structural header block grey
    for c_idx in range(len(table_rows[0])):
        cell = rendered_table[0, c_idx]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#dcdcdc')
    rendered_table[1, 0].set_text_props(weight='bold')
    rendered_table[2, 0].set_text_props(weight='bold')

    plt.suptitle(title, fontsize=13, weight='bold', y=0.98)
    plt.tight_layout(); plt.show()

# =========================================================
# PIPELINE DATA PREPARATION
# =========================================================

features_list, labels_list = [], []
print("Step 1: Processing signals and extracting morphological features...")

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
# EXPERIMENTAL TRAINING LOOPS (SVM PARADIGM)
# =========================================================
N_values = [200, 400, 600]

svm_configurations = [
    {'label': 'Linear C=0.1', 'kernel': 'linear', 'C': 0.1, 'gamma': 'scale'},
    {'label': 'Linear C=1.0', 'kernel': 'linear', 'C': 1.0, 'gamma': 'scale'},
    {'label': 'Linear C=10.0', 'kernel': 'linear', 'C': 10.0, 'gamma': 'scale'},
    {'label': 'RBF Gamma=0.01', 'kernel': 'rbf', 'C': 1.0, 'gamma': 0.01},
    {'label': 'RBF Gamma=0.1', 'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1},
    {'label': 'RBF Gamma=1.0', 'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0}
]

for N in N_values:
    idx = []
    for cls_idx in range(5):
        ids = np.where(y_train == cls_idx)[0]
        np.random.seed(42)
        idx.extend(np.random.choice(ids, min(N, len(ids)), replace=False))
    
    Xs, ys = X_train[idx], y_train[idx]
    
    for config in svm_configurations:
        label = config['label']
        print(f"Evaluating Model Setup -> N={N} | {label}...")
        
        clf = SVC(kernel=config['kernel'], C=config['C'], gamma=config['gamma'], random_state=42)
        clf.fit(Xs, ys)
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=range(5))
        
        g_acc, g_err, c_breakdown = extract_table_metrics(cm, CLASSES_LIST)
        
        # Save metrics keyed by configuration label for final tables compilation
        tables_data_store[N][label] = {
            'global_acc': g_acc, 'global_err': g_err, 'classes': c_breakdown
        }
        
        # Display figures + micro-table concurrently per loop execution
        plot_evaluation_dashboard(
            cm, CLASSES_LIST, 
            f"SVM Structural Configuration: Subset N={N} | Hyperparameter: {label}",
            g_acc, g_err, c_breakdown
        )

# =========================================================
# FINAL STACKED PRESENTATION TABLES (IMAGE_CE8400.PNG REPLICA)
# =========================================================
print("\nProcessing complete. Generating final macro-comparison summary tables...")

target_classes = ['A', 'R', 'N', 'L', 'V']

for N in N_values:
    macro_table_rows = []
    
    for config in svm_configurations:
        label = config['label']
        m_set = tables_data_store[N][label]
        
        # Row 1: Structural header band
        macro_table_rows.append([f"{label}", "global", "A", "R", "N", "L", "V"])
        
        # Row 2: Accuracy Row
        acc_row = ["Accurcy", f"{m_set['global_acc']:.3f}"]
        for cls in target_classes:
            acc_row.append(f"{m_set['classes'][cls]['Accuracy']:.3f}")
        macro_table_rows.append(acc_row)
        
        # Row 3: Error Rate Row
        err_row = ["Erreur rate", f"{m_set['global_err']:.3f}"]
        for cls in target_classes:
            err_row.append(f"{m_set['classes'][cls]['ErrorRate']:.3f}")
        macro_table_rows.append(err_row)

    # Render Stacked Table Window
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.axis('off')
    
    rendered_macro_table = ax.table(cellText=macro_table_rows, loc='center', cellLoc='center')
    rendered_macro_table.auto_set_font_size(False)
    rendered_macro_table.set_fontsize(10)
    rendered_macro_table.scale(1.1, 1.8)
    
    # Color structural sub-headers grey to match targets cleanly
    for r_idx, r_content in enumerate(macro_table_rows):
        if "Linear" in str(r_content[0]) or "RBF" in str(r_content[0]):
            for c_idx in range(len(r_content)):
                cell = rendered_macro_table[r_idx, c_idx]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#dcdcdc')
        else:
            rendered_macro_table[r_idx, 0].set_text_props(weight='bold')
            
    plt.title(f"CONSOLIDATED PERFORMANCE PROFILE MATRIX: DATASET SUBSET N = {N}", weight='bold', fontsize=12, pad=5)
    plt.tight_layout()
    plt.show()
