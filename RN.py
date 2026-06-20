# =========================================================
# ECG MIT-BIH - MLP (RN) EXPERIMENTAL METRICS BENCHMARK
# Structured Engine for College Final Project Evaluation
# =========================================================

import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# CONFIGURATION & GLOBAL VARIABLES
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

# =========================================================
# SIGNAL PROCESSING & BIOMEDICAL FEATURE EXTRACTION
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
    """
    Extracts P-Q-R-S-T geometric features and clinical time intervals.
    """
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
    
    features = [
        p_amp, q_amp, r_amp, s_amp, t_amp,          
        (r_idx - p_idx), (r_idx - q_idx),           
        (s_idx - r_idx), (t_idx - r_idx),
        pr_interval, qrs_duration, qt_interval      
    ]
    return features

# =========================================================
# CLINICAL EVALUATION DASHBOARD
# =========================================================

def plot_evaluation_dashboard(cm, classes, title):
    cm = np.array(cm)
    total_samples = cm.sum()
    num_classes = len(classes)
    
    # Storage arrays for comprehensive metrics
    accuracies = []    # Per-Class Accuracy
    precisions = []    # Precision (PPV)
    recalls = []       # Recall / Sensitivity
    f1_scores = []     # F1-Measure
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)
        
        # Calculations
        acc = (tp + tn) / total_samples if total_samples > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        accuracies.append(acc * 100)
        precisions.append(prec * 100)
        recalls.append(rec * 100)
        f1_scores.append(f1 * 100)

    # Format labels to include total counts on the matrix axis
    row_sums = cm.sum(axis=1)
    axis_labels = [f"{cls}\n({int(row_sums[i])})" for i, cls in enumerate(classes)]
    cm_percent = np.divide(cm.astype('float'), row_sums[:, None], out=np.zeros_like(cm, dtype=float), where=row_sums[:, None] != 0)

    # Balanced 1:1 width display window
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={'width_ratios': [1.1, 1.1]})

    # --- Panel A: Confusion Matrix Matrix ---
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks(np.arange(num_classes)); ax1.set_yticks(np.arange(num_classes))
    ax1.set_xticklabels(classes, weight='bold'); ax1.set_yticklabels(axis_labels, weight='bold')
    ax1.set_xlabel("Predicted Class Label", weight='bold', labelpad=10)
    ax1.set_ylabel("True Class Label (Sample Vol.)", weight='bold', labelpad=10)
    ax1.set_title(f"Confusion Matrix (N_test = {total_samples})", weight='bold', fontsize=12, pad=12)

    thresh = cm.max() * 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax1.text(j, i, f"{cm[i, j]}\n({cm_percent[i, j]*100:.1f}%)", ha="center", va="center", color=color, fontsize=10)
    
    # --- Panel B: Academic Metrics Report Table ---
    ax2.axis('off')
    metric_data = []
    for i in range(num_classes):
        # Ordered variables precisely to align with table_columns
        metric_data.append([
            classes[i], 
            f"{accuracies[i]:.1f}%",
            f"{precisions[i]:.1f}%", 
            f"{recalls[i]:.1f}%", 
            f"{f1_scores[i]:.1f}%"
        ])
    
    table_columns = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_table = ax2.table(cellText=metric_data, colLabels=table_columns, loc='center', cellLoc='center')
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(11)
    metrics_table.scale(1.0, 2.5) 
    
    # Elegant Slate Blue headers
    for col_idx in range(len(table_columns)):
        cell = metrics_table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')

    plt.suptitle(title, fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(); plt.show()

# =========================================================
# DATA PREPARATION ENGINE
# =========================================================

features_list, labels_list = [], []
print("[Engine] Commencing batch filtering and morphological extraction...")

for rec in records:
    try:
        r_path = os.path.join(dataset_path, rec)
        record = wfdb.rdrecord(r_path); ann = wfdb.rdann(r_path, "atr")
        sig = butter_bandpass(record.p_signal[:, 0], fs=FS)
        
        for pos, sym in zip(ann.sample, ann.symbol):
            cls = map_class(sym)
            if cls and 0 <= pos - WINDOW and pos + WINDOW < len(sig):
                beat_segment = sig[pos - WINDOW : pos + WINDOW]
                features_list.append(extract_morphological_features(beat_segment, fs=FS))
                labels_list.append(cls)
    except: continue

X = np.array(features_list)
y = np.array([label_map[x] for x in labels_list])

# Stratified split to maintain real-world imbalance in validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# MACHINE LEARNING CORE PIPELINE
# =========================================================
results = []
M_values = [10, 20, 30]
N_values = [200, 400, 600]

def run_mlp(Xtr, Xte, ytr, yte, M):
    clf = MLPClassifier(hidden_layer_sizes=(M,), activation='logistic', solver='adam', max_iter=400, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), confusion_matrix(yte, pred, labels=range(5))

# 1. Balanced Subsampling Experiments (N Constraints)
for N in N_values:
    idx = []
    for cls_idx in range(5):
        ids = np.where(y_train == cls_idx)[0]
        np.random.seed(42)
        idx.extend(np.random.choice(ids, min(N, len(ids)), replace=False))
    
    Xs, ys = X_train[idx], y_train[idx]
    for M in M_values:
        print(f"[Training Model] Hyperparameters: Subsampled N={N}, Hidden Neurons M={M}...")
        acc, cm = run_mlp(Xs, X_test, ys, y_test, M)
        results.append([N, M, acc, 1-acc])
        plot_evaluation_dashboard(cm, CLASSES_LIST, f"Subsampled Configuration: N={N} | M={M}\nGlobal System Accuracy: {acc*100:.2f}%")

# 2. Imbalanced Global Control Group Runs
for M in M_values:
    print(f"[Training Model] Hyperparameters: Full GLOBAL Dataset, Hidden Neurons M={M}...")
    acc, cm = run_mlp(X_train, X_test, y_train, y_test, M)
    results.append(['Global', M, acc, 1-acc])
    plot_evaluation_dashboard(cm, CLASSES_LIST, f"GLOBAL Baseline Configuration: M={M}\nGlobal System Accuracy: {acc*100:.2f}%")

# =========================================================
# EXPERIMENTAL MATRIX REPORT
# =========================================================
df = pd.DataFrame(results, columns=['Training Subset (N)', 'Hidden Neurons (M)', 'Global Accuracy', 'System Error Rate'])
print("\n" + "="*60 + "\nFINAL RESEARCH METRICS LOG\n" + "="*60)
print(df.to_string(index=False))
print("="*60)
================================================================================================================
# ============================================================
# ECG CLASSIFICATION (SVM + NEURAL NETWORK)
# SAME FEATURES FOR FAIR COMPARISON
# ============================================================

!pip install wfdb scikit-learn numpy scipy matplotlib tensorflow --quiet

import wfdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ============================================================
# DATASET
# ============================================================
wfdb.dl_database('mitdb', dl_dir='/content/mitdb')
folder = '/content/mitdb/'

RECORDS = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230'
]

KEEP_SYMBOLS = {'N','L','R','A','a','J','S','V','E','F'}

# ============================================================
# CLASS MAP (5 classes SAME for all models)
# ============================================================
class_mapping = {
    'N': 0,
    'L': 1,
    'R': 2,
    'A': 3,
    'V': 4
}

names = ['N','L','R','A','V']
n_classes = 5

# ============================================================
# FILTER
# ============================================================
def bandpass(x, fs=360):
    b, a = butter(4, [0.5/(fs/2), 40/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

# ============================================================
# RR
# ============================================================
rr_hist = {}

def rr_interval(rec, r, fs=360):
    if rec not in rr_hist:
        rr_hist[rec] = []
    h = rr_hist[rec]
    rr = 800 if len(h) == 0 else ((r - h[-1]) / fs) * 1000
    h.append(r)
    return rr

# ============================================================
# FEATURE EXTRACTION (SVM FEATURES = SHARED)
# ============================================================
def extract(segment, rr, fs=360):

    r = 150

    q = np.argmin(segment[r-40:r]) + (r-40)
    s = np.argmin(segment[r:r+40]) + r
    p = np.argmax(segment[r-80:r-40]) + (r-80)

    t = np.argmax(segment[r+40:r+120]) + (r+40)

    QS = s - q
    PR = r - p
    ST = t - s
    RT = t - r

    P_amp = segment[p]
    Q_amp = segment[q]
    R_amp = segment[r]
    S_amp = segment[s]
    T_amp = segment[t]

    P_m = np.mean(segment[p-10:p+10])
    Q_m = np.mean(segment[q-5:q+5])
    R_m = np.mean(segment[r-5:r+5])
    S_m = np.mean(segment[s-5:s+5])
    T_m = np.mean(segment[t-10:t+10])

    return [
        QS, PR, ST, RT,
        P_amp, Q_amp, R_amp, S_amp, T_amp,
        P_m, Q_m, R_m, S_m, T_m
    ]

# ============================================================
# LOAD DATA
# ============================================================
X_all, y_all = [], []

def load():
    for rec in RECORDS:
        try:
            record = wfdb.rdrecord(folder + rec)
            ann = wfdb.rdann(folder + rec, 'atr')

            sig = z(bandpass(record.p_signal[:,0]))

            for r, sym in zip(ann.sample, ann.symbol):

                if sym not in KEEP_SYMBOLS:
                    continue

                if sym not in class_mapping:
                    continue

                if r < 150 or r + 150 >= len(sig):
                    continue

                seg = z(sig[r-150:r+150])
                rr = rr_interval(rec, r)

                X_all.append(extract(seg, rr))
                y_all.append(class_mapping[sym])

        except:
            pass

load()

X_all = np.array(X_all)
y_all = np.array(y_all)

print("Total:", len(y_all))
print("Distribution:", Counter(y_all))
print("\n==============================")
print("FEATURE + LABEL MATRICES PER CLASS")
print("==============================")

for c in np.unique(y_all):

    X_c = X_all[y_all == c]
    y_c = y_all[y_all == c]

    print("\nCLASS:", names[int(c)])

    print("Feature matrix shape:", X_c.shape)  # (beats, 14)
    print("Label matrix shape:", y_c.shape)    # (beats,)

    print("Example features (first 3 beats):")
    print(X_c[:3])

    print("Example labels (first 3 beats):")
    print(y_c[:3])

# ============================================================
# SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.30,
    random_state=42,
    stratify=y_all
)

# ============================================================
# SCALE (FOR SVM + NN)
# ============================================================
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# ONE HOT ENCODING
# ============================================================
y_train_cat = to_categorical(y_train, n_classes)
y_test_cat = to_categorical(y_test, n_classes)




M_values = [500, 1000, 2000, 5000]

for M in M_values:

    if M > len(X_train):
        continue

    print("\n================================")
    print("NUMBER OF SAMPLES M =", M)
    print("================================")

    idx = np.random.choice(len(X_train), M, replace=False)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    y_sub_cat = to_categorical(y_sub, n_classes)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_sub,
        y_sub_cat,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    cm = confusion_matrix(y_test, y_pred)

    print("Global Accuracy:", accuracy_score(y_test, y_pred))

    for i, cls in enumerate(names):
        acc_c = cm[i,i] / (np.sum(cm[i,:]) + 1e-6)
        print(f"{cls}: {acc_c:.4f}")

# ============================================================
# ================= SVM MODELS =================
# ============================================================
print("\n===================================")
print("LINEAR SVM - DIFFERENT C VALUES")
print("===================================")

C_values = [0.1, 1, 10, 100]

for C in C_values:

    model = SVC(
        kernel='linear',
        C=C,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    print(f"\nLINEAR SVM (C={C})")
    print("Global Accuracy:", accuracy_score(y_test, pred))

    for i, cls in enumerate(names):
        acc_c = cm[i, i] / (np.sum(cm[i, :]) + 1e-6)
        print(f"{cls}: {acc_c:.4f}")

print("\n===================================")
print("RBF SVM - DIFFERENT C AND GAMMA")
print("===================================")

C_values = [1, 10, 20, 50]
gamma_values = [0.001, 0.01, 0.1, 1]

for C in C_values:

    for gamma in gamma_values:

        model = SVC(
            kernel='rbf',
            C=C,
            gamma=gamma,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        cm = confusion_matrix(y_test, pred)

        print(f"\nRBF SVM (C={C}, gamma={gamma})")
        print("Global Accuracy:", accuracy_score(y_test, pred))

        for i, cls in enumerate(names):
            acc_c = cm[i, i] / (np.sum(cm[i, :]) + 1e-6)
            print(f"{cls}: {acc_c:.4f}")


linear_svm = SVC(kernel='linear', C=1, class_weight='balanced')
rbf_svm = SVC(kernel='rbf', C=20, gamma='scale', class_weight='balanced')

linear_svm.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)

pred_lin = linear_svm.predict(X_test)
pred_rbf = rbf_svm.predict(X_test)

# ============================================================
# ================= NEURAL NETWORK ===========================
# ============================================================
hidden_nodes_list = [64, 32, 16, 4]

results = {}

for nodes in hidden_nodes_list:

    print("\n\n======================================")
    print(f"NEURAL NETWORK - {nodes} NODES")
    print("======================================")

    model = Sequential([
        Dense(nodes, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[early],
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc_global = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("\nGLOBAL ACCURACY:", acc_global)

    print("\nPER-CLASS ACCURACY:")

    class_acc = {}

    for i, cls in enumerate(names):
        tp = cm[i, i]
        total = np.sum(cm[i, :])
        acc_c = tp / (total + 1e-6)

        class_acc[cls] = acc_c

        print(f"{cls}: {acc_c:.4f}")

    results[nodes] = {
        "global": acc_global,
        "class": class_acc
    }

# ============================================================
# ================= EVALUATION =================
# ============================================================
def evaluate(name, y_true, y_pred):
    print("\n====================")
    print(name)
    print("====================")

    print(classification_report(y_true, y_pred, target_names=names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    for i, c in enumerate(names):
        acc = cm[i,i] / (np.sum(cm[i,:]) + 1e-6)
        print(f"{c} accuracy: {acc:.4f}")

    return accuracy_score(y_true, y_pred)

acc1 = evaluate("Linear SVM", y_test, pred_lin)
acc2 = evaluate("RBF SVM", y_test, pred_rbf)
acc3 = evaluate("Neural Network", y_test, pred_nn)

# ============================================================
# FINAL COMPARISON
# ============================================================
plt.figure(figsize=(6,4))
plt.bar(
    ["Linear SVM","RBF SVM","Neural Net"],
    [acc1, acc2, acc3]
)
plt.ylim(0,1)
plt.title("Model Comparison")
plt.show()
