# scripts/spiral/3a_train_spiral_ml.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input

# ---------------------------
# Config / Paths / Seed
# ---------------------------
RND = 42
np.random.seed(RND)
tf.random.set_seed(RND)

ROOT_RESULTS = "results/spiral"
ROOT_MODELS = "models/spiral"
os.makedirs(ROOT_RESULTS, exist_ok=True)
os.makedirs(ROOT_MODELS, exist_ok=True)

FEATURES_CSV = "../../features/spiral_handcrafted.csv"  # input features file
SCALER_PATH = os.path.join(ROOT_MODELS, "spiral_scaler.pkl")

# ---------------------------
# Keras wrapper (sklearn-compatible)
# ---------------------------
class KerasClassifier:
    """
    Simple sklearn-compatible Keras wrapper.
    Save Keras model as HDF5; wrapper stores hyperparams locally.
    """
    def __init__(self, input_dim, epochs=50, batch_size=16, verbose=0):
        self.input_dim = int(input_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.verbose = int(verbose)
        self.model_ = None
        self.history_ = None

    def get_params(self, deep=True):
        return {"input_dim": self.input_dim, "epochs": self.epochs,
                "batch_size": self.batch_size, "verbose": self.verbose}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.15
        )
        return self

    def predict(self, X):
        probs = self.model_.predict(X, verbose=0)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        probs = self.model_.predict(X, verbose=0)
        return np.hstack([1 - probs, probs])

    def save(self, path_h5):
        if self.model_ is None:
            raise RuntimeError("No trained Keras model to save.")
        self.model_.save(path_h5)

    @classmethod
    def load(cls, path_h5, input_dim=None, epochs=None, batch_size=None, verbose=0):
        wrapper = cls(input_dim=input_dim or 0, epochs=epochs or 0, batch_size=batch_size or 16, verbose=verbose)
        wrapper.model_ = load_model(path_h5)
        return wrapper

# ---------------------------
# Utility functions
# ---------------------------
def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def plot_confusion(cm, labels, title, out_path, cmap="Blues", annotate=True):
    fig, ax = plt.subplots(figsize=(5,5))
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    ax.set_title(title)
    if annotate:
        # annotate with integer values in center of cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="black", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_single_heatmap(cm, labels, title, out_path, cmap="OrRd"):
    # Plot single heatmap with integer annotations (rounded)
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def compute_metrics(y_true, y_pred):
    """
    Compute metrics and also TN% and TP% as percentage of total samples.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.size != 4:
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp if (tn + fp + fn + tp) > 0 else 1
    acc = float(accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sensitivity = recall
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tn_pct = 100.0 * tn / total
    tp_pct = 100.0 * tp / total
    return {
        "Accuracy": acc,
        "Recall": recall,
        "F1-Score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN_pct": tn_pct,
        "TP_pct": tp_pct
    }

def save_classification_report(y_true, y_pred, out_path):
    txt = classification_report(y_true, y_pred, digits=6)
    with open(out_path, "w") as f:
        f.write(txt)

# ---------------------------
# Models definitions
# ---------------------------
def get_classical_models():
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RND),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RND),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RND),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=RND, eval_metric="logloss"),
    }

# ---------------------------
# Load dataset + scaler
# ---------------------------
if not os.path.exists(FEATURES_CSV):
    raise FileNotFoundError(f"Features CSV not found at: {FEATURES_CSV}")

df = pd.read_csv(FEATURES_CSV)
if "status" not in df.columns:
    raise ValueError("Expected 'status' column in features CSV")

X = df.drop(columns=["status"])
y = df["status"].astype(int)

# Scale and persist scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features. Saved scaler -> {SCALER_PATH}")

# ---------------------------
# Main experiment routine
# ---------------------------
SPLITS = [
    (0.2, "8020"),
    (0.3, "7030")
]

# Top-3 models fixed for ensemble: RandomForest, XGBoost, kNN
TOP3 = ["RandomForest", "XGBoost", "kNN"]

for test_size, suffix in SPLITS:
    print("\n" + "="*60)
    print(f"=== Running split {suffix} (test_size={test_size}) ===")
    print("="*60)

    # prepare output dirs for this split
    out_dir = os.path.join(ROOT_RESULTS, suffix)
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    make_dirs(out_dir); make_dirs(cm_dir)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=RND
    )

    pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(ROOT_MODELS, f"X_test_scaled_{suffix}.csv"), index=False)
    pd.DataFrame({"status": y_test.values}).to_csv(os.path.join(ROOT_MODELS, f"y_test_{suffix}.csv"), index=False)

    print(f" Train size: {len(X_train)}, Test size: {len(X_test)} (split={suffix})")

    # instantiate fresh models
    models = get_classical_models()
    fitted_models = {}
    cms_per_model = {}  # store confusion matrices for combining

    # Train classical models
    results_rows = {}
    for name, mdl in models.items():
        print(f"Training {name} on split {suffix}...")
        mdl.fit(X_train, y_train)
        fitted_models[name] = mdl
        # save classical model
        joblib.dump(mdl, os.path.join(ROOT_MODELS, f"{name.lower()}_{suffix}_model.pkl"))

        # evaluate on test set
        y_pred = mdl.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        results_rows[name] = metrics

        # store confusion matrix for this model
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        cms_per_model[name] = cm

        # save confusion matrix & report
        plot_confusion(cm, labels=[0,1],
                       title=f"{name} Confusion Matrix ({suffix})",
                       out_path=os.path.join(cm_dir, f"cm_{name}_{suffix}.png"))
        save_classification_report(y_test, y_pred, os.path.join(out_dir, f"classification_report_{name}_{suffix}.txt"))

    # Train MLP
    print(f"Training MLP on split {suffix}...")
    mlp = KerasClassifier(input_dim=X_train.shape[1], epochs=50, batch_size=16, verbose=0)
    mlp.fit(X_train, y_train)
    # save keras model (HDF5) and meta
    mlp_h5 = os.path.join(ROOT_MODELS, f"mlp_{suffix}.h5")
    mlp_meta = os.path.join(ROOT_MODELS, f"mlp_{suffix}_meta.json")
    mlp.save(mlp_h5)
    with open(mlp_meta, "w") as f:
        json.dump({"input_dim": mlp.input_dim, "epochs": mlp.epochs, "batch_size": mlp.batch_size}, f)

    # evaluate mlp
    y_pred_mlp = mlp.predict(X_test)
    metrics_mlp = compute_metrics(y_test, y_pred_mlp)
    results_rows["MLP"] = metrics_mlp
    cms_per_model["MLP"] = confusion_matrix(y_test, y_pred_mlp, labels=[0,1])

    cm = confusion_matrix(y_test, y_pred_mlp, labels=[0,1])
    plot_confusion(cm, labels=[0,1],
                   title=f"MLP Confusion Matrix ({suffix})",
                   out_path=os.path.join(cm_dir, f"cm_MLP_{suffix}.png"))
    save_classification_report(y_test, y_pred_mlp, os.path.join(out_dir, f"classification_report_MLP_{suffix}.txt"))

    # Save test metrics CSV
    metrics_df = pd.DataFrame(results_rows).T
    metrics_df.to_csv(os.path.join(out_dir, f"spiral_test_metrics_{suffix}.csv"), float_format="%.10f")

    # Save train metrics (optional)
    train_rows = {}
    for name, mdl in fitted_models.items():
        y_pred_train = mdl.predict(X_train)
        train_rows[name] = compute_metrics(y_train, y_pred_train)
    # MLP train metrics
    y_pred_mlp_train = mlp.predict(X_train)
    train_rows["MLP"] = compute_metrics(y_train, y_pred_mlp_train)
    train_metrics_df = pd.DataFrame(train_rows).T
    train_metrics_df.to_csv(os.path.join(out_dir, f"spiral_train_metrics_{suffix}.csv"), float_format="%.10f")

    # Plot Train vs Test Accuracy bar chart
    try:
        acc_train = train_metrics_df["Accuracy"]
        acc_test = metrics_df["Accuracy"]
        labels = metrics_df.index.tolist()
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x - width/2, acc_train[labels], width, label="Train", color="#FFB703", edgecolor="black")
        ax.bar(x + width/2, acc_test[labels], width, label="Test", color="#219EBC", edgecolor="black")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Train vs Test Accuracy ({suffix})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylim(0,1.05)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"train_test_accuracy_comparison_{suffix}.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Warning: could not plot train/test accuracy:", e)

        # ---------------------------
    # ---------------------------
# Combined confusion matrix (average of top3 models) + Unified 3-model heatmap
# ---------------------------
top3_cms = []
collected = {}

for nm in TOP3:
    # handle key mismatch (kNN vs knn)
    key = nm
    if key not in cms_per_model and nm.lower() in cms_per_model:
        key = nm.lower()

    if key in cms_per_model:
        cm = cms_per_model[key]
        top3_cms.append(cm)
        collected[nm] = cm
    else:
        print(f"Warning: missing CM for {nm} in split {suffix}.")

if len(top3_cms) >= 1:

    # ---------------------------
    # 1) Save combined (average) confusion matrix
    # ---------------------------
    avg_cm = np.mean(np.stack(top3_cms, axis=0), axis=0)
    avg_cm_rounded = np.rint(avg_cm).astype(int)

    combined_out = os.path.join(out_dir, f"combined_top3_confusion_{suffix}.png")
    plot_single_heatmap(
        avg_cm_rounded, labels=[0,1],
        title=f"Combined Top3 Confusion ({suffix})",
        out_path=combined_out
    )

    pd.DataFrame(
        avg_cm_rounded,
        index=["Actual_0","Actual_1"],
        columns=["Pred_0","Pred_1"]
    ).to_csv(os.path.join(out_dir, f"combined_top3_confusion_{suffix}.csv"))

    print(f"Saved combined confusion -> {combined_out}")

    # ---------------------------------------------------------
    # 2) NEW: ONE unified heatmap containing all 3 models
    # ---------------------------------------------------------
    import seaborn as sns

    # Create 6×2 grid → (RF rows 0–1, XGB rows 2–3, kNN rows 4–5)
    unified_cm = np.zeros((6, 2), dtype=int)
    row_labels = []

    model_order = ["RandomForest", "XGBoost", "kNN"]
    r = 0
    for m in model_order:
        if m in collected:
            unified_cm[r:r+2, :] = collected[m]
            row_labels += [f"{m} A0", f"{m} A1"]
        else:
            unified_cm[r:r+2, :] = np.array([[0,0],[0,0]])
            row_labels += [f"{m} A0 (NA)", f"{m} A1 (NA)"]
        r += 2

    plt.figure(figsize=(6, 10))
    sns.heatmap(
        unified_cm, annot=True, fmt="d",
        cmap="OrRd",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=row_labels,
        cbar=False
    )
    plt.title(f"Unified Confusion Heatmap – Top 3 Models ({suffix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual / Model")
    plt.tight_layout()

    unified_path = os.path.join(out_dir, f"top3_single_heatmap_{suffix}.png")
    plt.savefig(unified_path, dpi=300)
    plt.close()

    print(f"Saved unified 3-model confusion heatmap -> {unified_path}")

else:
    print("Not enough top3 confusion matrices available for split", suffix)


    # ---------------------------
    # Build ensemble for this split using fitted models (RF + XGBoost + kNN)
    # ---------------------------
    print(f"Building ensemble (RF+XGBoost+kNN) for split {suffix} from fitted models...")
    probs_list = []
    used = []
    for nm in TOP3:
        # classical fitted_models keys: 'SVM','RandomForest','LogisticRegression','kNN','XGBoost'
        key = nm
        # handle case of 'kNN' key
        if key not in fitted_models and nm.lower() in fitted_models:
            key = nm.lower()
        if key in fitted_models:
            mdl = fitted_models[key]
            try:
                if hasattr(mdl, "predict_proba"):
                    p2 = mdl.predict_proba(X_test)
                else:
                    p_ = mdl.predict(X_test)
                    p2 = np.vstack([1 - p_, p_]).T
                probs_list.append(p2)
                used.append(key)
            except Exception as e:
                print(f"Error getting probs from {key}: {e}")
        else:
            # try loading saved model from disk for this split
            path_try = os.path.join(ROOT_MODELS, f"{nm.lower()}_{suffix}_model.pkl")
            if os.path.exists(path_try):
                try:
                    mdl2 = joblib.load(path_try)
                    if hasattr(mdl2, "predict_proba"):
                        p2 = mdl2.predict_proba(X_test)
                    else:
                        p_ = mdl2.predict(X_test)
                        p2 = np.vstack([1 - p_, p_]).T
                    probs_list.append(p2)
                    used.append(nm)
                except Exception as e:
                    print(f"Error loading model {path_try}: {e}")

    if len(probs_list) >= 2:
        avg_prob = np.mean(np.stack(probs_list, axis=0), axis=0)
        y_ens = np.argmax(avg_prob, axis=1)
        metrics_ens = compute_metrics(y_test, y_ens)
        ens_out_dir = os.path.join(ROOT_RESULTS, f"ensemble_{suffix}")
        make_dirs(ens_out_dir)
        pd.DataFrame({"Prob_Healthy": avg_prob[:,0], "Prob_Parkinson": avg_prob[:,1]}).to_csv(os.path.join(ens_out_dir, "voting_test_probs.csv"), index=False)
        with open(os.path.join(ens_out_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(metrics_ens, f, indent=2)
        cm_ens = confusion_matrix(y_test, y_ens, labels=[0,1])
        plot_confusion(cm_ens, labels=[0,1], title=f"Ensemble Confusion Matrix ({suffix})", out_path=os.path.join(ens_out_dir, "cm_ensemble.png"))
        # also save combined top3 confusion in ensemble folder (rounded average)
        if len(top3_cms) >= 1:
            pd.DataFrame(avg_cm_rounded, index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"]).to_csv(os.path.join(ens_out_dir, "combined_top3_confusion_rounded.csv"))
            plot_single_heatmap(avg_cm_rounded, labels=[0,1], title=f"Combined Top3 Confusion ({suffix})", out_path=os.path.join(ens_out_dir, "combined_top3_confusion.png"))
        print(f"Saved ensemble results to {ens_out_dir}")
    else:
        print(f"Not enough models (>=2) for ensemble in split {suffix}. Found: {used}")

    print(f"Completed split {suffix}. Results saved to {out_dir}")

# ---------------------------
# ---------------------------
# ============================================================
# 5-fold CV (ALL MODELS + ENSEMBLE) — per-fold + mean metrics
# ============================================================
print("\n" + "="*60)
print("=== Running 5-Fold Cross-Validation on full dataset (with Ensemble) ===")
print("="*60)

cv_dir = os.path.join(ROOT_RESULTS, "cv5")
make_dirs(cv_dir)
folds_dir = os.path.join(cv_dir, "folds")
make_dirs(folds_dir)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
base_models = get_classical_models()

cv_summary_rows = []

# Store per-model confusion matrices
per_model_fold_cms = {name: [] for name in list(base_models.keys()) + ["MLP"]}

# ============================================================
# PART 1 — PROCESS CV FOR INDIVIDUAL MODELS
# ============================================================
for model_name in list(base_models.keys()) + ["MLP"]:
    print(f"\nProcessing CV for model: {model_name}")
    per_fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_scaled, y), start=1):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Train model
        if model_name != "MLP":
            mdl = clone(base_models[model_name])
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_val)
        else:
            mlp_cv = KerasClassifier(input_dim=X_tr.shape[1], epochs=50, batch_size=16, verbose=0)
            mlp_cv.fit(X_tr, y_tr)
            y_pred = mlp_cv.predict(X_val)

        # Metrics
        metrics = compute_metrics(y_val, y_pred)
        per_fold_metrics.append(metrics)

        # Save confusion matrix
        fold_folder = os.path.join(folds_dir, f"{model_name}_folds")
        make_dirs(fold_folder)

        cm = confusion_matrix(y_val, y_pred, labels=[0,1])
        per_model_fold_cms[model_name].append(cm)

        plot_confusion(
            cm, labels=[0,1],
            title=f"{model_name} - Fold {fold_idx}",
            out_path=os.path.join(fold_folder, f"cm_{model_name}_fold{fold_idx}.png")
        )

        save_classification_report(
            y_val, y_pred,
            os.path.join(fold_folder, f"classification_report_{model_name}_fold{fold_idx}.txt")
        )

    # Save per-fold metrics CSV
    df_folds = pd.DataFrame(per_fold_metrics)
    df_folds.to_csv(os.path.join(cv_dir, f"{model_name}_cv_folds.csv"), float_format="%.10f")

    # Compute and store mean metrics
    mean_row = df_folds.mean().to_dict()
    cv_summary_rows.append({
        "Model": model_name,
        "Mean_Accuracy": mean_row["Accuracy"],
        "Mean_Recall": mean_row["Recall"],
        "Mean_F1-Score": mean_row["F1-Score"],
        "Mean_Sensitivity": mean_row["Sensitivity"],
        "Mean_Specificity": mean_row["Specificity"],
        "Mean_FPR": mean_row["FPR"],
        "Mean_FNR": mean_row["FNR"],
        "Mean_TN_pct": mean_row["TN_pct"],
        "Mean_TP_pct": mean_row["TP_pct"],
    })

# ============================================================
# PART 2 — ENSEMBLE 5-FOLD CV (RF + XGB + kNN)
# ============================================================
print("\n=== Running Ensemble (RF + XGBoost + kNN) 5-Fold CV ===")

ensemble_fold_metrics = []
ensemble_fold_cms = []

for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_scaled, y), start=1):

    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Train top 3 models
    rf = clone(base_models["RandomForest"]).fit(X_tr, y_tr)
    xgb = clone(base_models["XGBoost"]).fit(X_tr, y_tr)
    knn = clone(base_models["kNN"]).fit(X_tr, y_tr)

    probs = []

    # Collect predictions
    for mdl in [rf, xgb, knn]:
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X_val)
        else:
            p_ = mdl.predict(X_val)
            p = np.vstack([1 - p_, p_]).T
        probs.append(p)

    # Average voting
    avg_prob = np.mean(np.stack(probs, axis=0), axis=0)
    y_pred_ens = np.argmax(avg_prob, axis=1)

    # Compute ensemble metrics
    m = compute_metrics(y_val, y_pred_ens)
    ensemble_fold_metrics.append(m)

    # Confusion matrix
    cm_ens = confusion_matrix(y_val, y_pred_ens, labels=[0,1])
    ensemble_fold_cms.append(cm_ens)

    # Save per-fold confusion
    ens_fold_dir = os.path.join(folds_dir, "Ensemble_folds")
    make_dirs(ens_fold_dir)

    plot_confusion(
        cm_ens, labels=[0,1],
        title=f"Ensemble - Fold {fold_idx}",
        out_path=os.path.join(ens_fold_dir, f"cm_ensemble_fold{fold_idx}.png")
    )

# Save ensemble fold metrics
df_ens = pd.DataFrame(ensemble_fold_metrics)
df_ens.to_csv(os.path.join(cv_dir, "ensemble_cv_folds.csv"), float_format="%.10f")

# Mean ensemble metrics
mean_ens = df_ens.mean().to_dict()
pd.DataFrame([mean_ens]).to_csv(
    os.path.join(cv_dir, "ensemble_cv_mean_metrics.csv"),
    index=False, float_format="%.10f"
)

print("\nSaved ensemble CV metrics -> ensemble_cv_mean_metrics.csv")

# ============================================================
# PART 3 — ENSEMBLE MEAN CONFUSION MATRIX
# ============================================================
if len(ensemble_fold_cms) > 0:
    avg_cm_ens = np.mean(np.stack(ensemble_fold_cms, axis=0), axis=0)
    avg_cm_ens_r = np.rint(avg_cm_ens).astype(int)

    plot_single_heatmap(
        avg_cm_ens_r, labels=[0,1],
        title="Ensemble Mean Confusion (CV-10)",
        out_path=os.path.join(cv_dir, "ensemble_cv_mean_confusion.png")
    )

    pd.DataFrame(
        avg_cm_ens_r,
        index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"]
    ).to_csv(os.path.join(cv_dir, "ensemble_cv_mean_confusion.csv"))

    print("Saved ensemble CV mean confusion matrix.")



# ---------------------------
# Additionally, build ensemble evaluation for both saved splits (8020 & 7030) and save results
# ---------------------------
for suffix in ["8020", "7030"]:
    ens_out_dir = os.path.join(ROOT_RESULTS, f"ensemble_{suffix}")
    make_dirs(ens_out_dir)
    # try load saved X_test/y_test for that suffix
    xt_path = os.path.join(ROOT_MODELS, f"X_test_scaled_{suffix}.csv")
    yt_path = os.path.join(ROOT_MODELS, f"y_test_{suffix}.csv")
    if os.path.exists(xt_path) and os.path.exists(yt_path):
        X_test_df = pd.read_csv(xt_path)
        y_test_df = pd.read_csv(yt_path)
        X_test_eval = X_test_df.values
        y_test_eval = y_test_df["status"].values
    else:
        # fallback random split
        _, X_test_eval, _, y_test_eval = train_test_split(X_scaled, y, test_size=0.2, random_state=RND, stratify=y)

    probs_list = []
    for nm in TOP3:
        # try load model pkl
        pkl_path = os.path.join(ROOT_MODELS, f"{nm.lower()}_{suffix}_model.pkl")
        if os.path.exists(pkl_path):
            try:
                mdl = joblib.load(pkl_path)
                if hasattr(mdl, "predict_proba"):
                    p2 = mdl.predict_proba(X_test_eval)
                else:
                    p_ = mdl.predict(X_test_eval)
                    p2 = np.vstack([1 - p_, p_]).T
                probs_list.append(p2)
            except Exception as e:
                print(f"Error loading/using model {pkl_path}: {e}")
        else:
            print(f"Warning: {pkl_path} not found for ensemble building.")

    if len(probs_list) >= 2:
        avg_prob = np.mean(np.stack(probs_list, axis=0), axis=0)
        y_ens = np.argmax(avg_prob, axis=1)
        metrics_ens = compute_metrics(y_test_eval, y_ens)
        pd.DataFrame({"Prob_Healthy": avg_prob[:,0], "Prob_Parkinson": avg_prob[:,1]}).to_csv(os.path.join(ens_out_dir, "voting_test_probs.csv"), index=False)
        with open(os.path.join(ens_out_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(metrics_ens, f, indent=2)
        cm_ens = confusion_matrix(y_test_eval, y_ens, labels=[0,1])
        plot_confusion(cm_ens, labels=[0,1], title=f"Ensemble Confusion Matrix ({suffix})", out_path=os.path.join(ens_out_dir, "cm_ensemble.png"))
        print(f"Saved ensemble metrics/CM for suffix {suffix} -> {ens_out_dir}")
    else:
        print(f"Not enough models available to build ensemble for suffix {suffix} (need >=2).")

print("\nAll experiments completed. Results saved under:", ROOT_RESULTS)
