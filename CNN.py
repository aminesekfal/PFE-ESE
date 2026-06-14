# =========================================================
# ECG MIT-BIH - 1D CNN MULTI-ARCHITECTURE & SPLIT EVALUATION
# Performance Report per Layer Count and Test Proportions
# =========================================================

import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Framework ta3 Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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
WINDOW = 150  # Hadhi t'creyi segment ta3 300 points (150 gbel w 150 b3d l-R-peak)
FS = 360      

# =========================================================
# RAW SIGNAL PROCESSING (NO MANUAL FEATURE EXTRACTION)
# =========================================================

def butter_bandpass(sig, fs=360, low=0.5, high=40):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def map_class(sym):
    for k, v in CLASSES.items():
        if sym in v: return k
    return None

# Extraction ta3 les signaux kham (Raw Windows) direct pour le CNN
raw_signals_list, labels_list = [], []
print("Step 1: Extracting raw ECG segments for CNN... Please wait...")

for rec in records:
    try:
        r_path = os.path.join(dataset_path, rec)
        record = wfdb.rdrecord(r_path); ann = wfdb.rdann(r_path, "atr")
        sig = butter_bandpass(record.p_signal[:, 0], fs=FS)
        
        for pos, sym in zip(ann.sample, ann.symbol):
            cls = map_class(sym)
            if cls and 0 <= pos - WINDOW and pos + WINDOW < len(sig):
                # Standardisation locale ta3 l-beat bch n'stablisou l-CNN
                beat = sig[pos - WINDOW : pos + WINDOW]
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                raw_signals_list.append(beat)
                labels_list.append(cls)
    except: continue

X_raw = np.array(raw_signals_list)
# Ndiro reshape bch tweli (samples, steps, channels) li y'se7bha l-Conv1D
X_raw = np.expand_dims(X_raw, axis=-1) 
y_raw = np.array([label_map[x] for x in labels_list])

# =========================================================
# CNN ARCHITECTURES DEFINITION (Nombre de couches)
# =========================================================

def build_cnn(architecture_type, input_shape=(300, 1)):
    model = Sequential()
    
    if architecture_type == "Shallow_CNN_2_Layers":
        # Architecture sghira: 2 couches de Convolution
        model.add(Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        
    elif architecture_type == "Deep_CNN_4_Layers":
        # Architecture kbira: 4 couches de Convolution (Kima bgha l-prof)
        model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(64, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax')) # 5 classes output
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =========================================================
# METRICS EXTRACTION SYSTEM
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
# EXPERIMENTATION LOOP (ARCHITECTURES VS TEST SPLITS)
# =========================================================

# Les scénarios ta3 l-test (1/3 w 2/3)
test_splits = {
    "Split 1_3 (Test=33%)": 0.33,
    "Split 2_3 (Test=66%)": 0.66
}

architectures = ["Shallow_CNN_2_Layers", "Deep_CNN_4_Layers"]
cnn_results_store = {}

for split_label, test_ratio in test_splits.items():
    cnn_results_store[split_label] = {}
    
    # Ndiro split 3la hssab l-scénario ta3 l-test size
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_ratio, random_state=42, stratify=y_raw
    )
    
    # Convertir les labels en One-Hot encoding pour Keras
    y_train_cat = to_categorical(y_train, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    
    for arch in architectures:
        print(f"\nTraining {arch} on {split_label}...")
        
        # Build w train l-CNN
        model = build_cnn(arch, input_shape=(300, 1))
        
        # N'entrainou ghir 3 epochs bch l-code i'kamel fsa3 fi PC ta3ek (t9ed tzidhoum)
        model.fit(X_train, y_train_cat, epochs=3, batch_size=64, verbose=0)
        
        # Predictions
        preds = model.predict(X_test)
        pred_classes = np.argmax(preds, axis=1)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, pred_classes, labels=range(5))
        g_acc, g_err, c_breakdown = extract_table_metrics(cm, CLASSES_LIST)
        
        cnn_results_store[split_label][arch] = {
            'global_acc': g_acc, 'global_err': g_err, 'classes': c_breakdown
        }

# =========================================================
# CNN PERFORMANCE TABLES GENERATOR
# =========================================================
print("\nGenerating CNN Performance Reports...")
target_classes = ['A', 'R', 'N', 'L', 'V']

for split_label in test_splits.keys():
    table_rows = []
    
    for arch in architectures:
        m_set = cnn_results_store[split_label][arch]
        
        # Row header bl-ism ta3 l-architecture
        table_rows.append([f"{arch}", "global", "A", "R", "N", "L", "V"])
        
        # Accuracies
        acc_row = ["Accuracy", f"{m_set['global_acc']:.3f}"]
        for cls in target_classes:
            acc_row.append(f"{m_set['classes'][cls]['Accuracy']:.3f}")
        table_rows.append(acc_row)
        
        # Error Rates
        err_row = ["Error rate", f"{m_set['global_err']:.3f}"]
        for cls in target_classes:
            err_row.append(f"{m_set['classes'][cls]['ErrorRate']:.3f}")
        table_rows.append(err_row)

    # Plotting l-tableau clean kima l-sabeq
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    rendered_table = ax.table(cellText=table_rows, loc='center', cellLoc='center')
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(10)
    rendered_table.scale(1.1, 1.6)
    
    for r_idx, r_content in enumerate(table_rows):
        if "CNN" in str(r_content[0]):
            for c_idx in range(len(r_content)):
                cell = rendered_table[r_idx, c_idx]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0f2fe') # Couleur bleue claire pour distinguer l-CNN
        else:
            rendered_table[r_idx, 0].set_text_props(weight='bold')
            
    plt.title(f"CNN PERFORMANCE REPORT FOR: {split_label.upper()}", weight='bold', fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()