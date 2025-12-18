import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# === Paths ===
scaler_path = "models/spiral/spiral_scaler.pkl"
model_paths = {
    "RandomForest": "models/spiral/randomforest_model.pkl",
    "KNN": "models/spiral/knn_model.pkl",
    "XGBoost": "models/spiral/xgboost_model.pkl"
}

# === Load models ===
models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f" Loaded model: {name}")
    else:
        print(f" Missing model file: {path}")

if len(models) < 2:
    raise RuntimeError("Not enough models available for ensemble prediction!")

# === Load scaler ===
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(" Scaler loaded successfully")
else:
    raise FileNotFoundError(" Scaler file not found!")

# === Feature extraction (same as training) ===
def extract_spiral_features(img_path):
    """Extract handcrafted statistical, texture, shape, and edge features."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0

    # Statistical
    mean, std, var, minv, maxv = np.mean(img), np.std(img), np.var(img), np.min(img), np.max(img)

    # Texture (GLCM)
    glcm = graycomatrix((img * 255).astype(np.uint8), distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # Shape (Hu moments)
    moments = cv2.moments((img * 255).astype(np.uint8))
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-9)

    # Edge density
    edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
    edge_density = np.mean(edges > 0)

    feats = [mean, std, var, minv, maxv,
             contrast, dissimilarity, homogeneity, energy, correlation,
             *hu_moments, edge_density]

    return np.array(feats).reshape(1, -1)

# === Prediction Function ===
def predict_spiral(img_path):
    print(f"\n Predicting for image: {img_path}")
    feats = extract_spiral_features(img_path)
    feats_scaled = scaler.transform(feats)

    # Ensemble weights
    weights = {"RandomForest": 1.0, "KNN": 1.0, "XGBoost": 1.0}

    weighted_sum = 0
    total_weight = 0

    # Collect weighted probabilities
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(feats_scaled)[:, 1]
        else:
            p = model.predict(feats_scaled).astype(float)

        w = weights.get(name, 1.0)
        weighted_sum += p * w
        total_weight += w

        print(f"🔹 {name:<12} → PD Probability: {p[0]:.4f}  | Weight: {w}")

    # Normalized weighted average
    avg_prob = (weighted_sum / total_weight)[0]
    prediction = "Parkinson" if avg_prob > 0.5 else "Healthy"

    print("\n" + "="*60)
    print(f" Final Ensemble Probability: {avg_prob:.4f}")
    print(f"🩺 Predicted Class: {prediction.upper()}")
    print("="*60)

    # === Show image with prediction ===
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {prediction} ({avg_prob*100:.2f}%)", fontsize=12, weight='bold')
    plt.axis('off')
    plt.show()


# === Run demo ===
sample_image = "../../data/spiral/Parkinson/Parkinson200.png" 
predict_spiral(sample_image)

