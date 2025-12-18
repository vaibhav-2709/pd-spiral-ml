# scripts/spiral/3a_spiral_handcrafted.py
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# === Setup paths ===
spiral_dir = "../../data/spiral"
out_path = "../../features/spiral_handcrafted.csv"
os.makedirs("../../features", exist_ok=True)

features, labels, file_names = [], [], []

# === Loop through classes ===
for label_name, label in [("healthy", 0), ("parkinson", 1)]:
    class_dir = os.path.join(spiral_dir, label_name)
    for f in os.listdir(class_dir):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(class_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize + Normalize
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0

        # === 1. Basic statistical features ===
        mean = np.mean(img)
        std = np.std(img)
        var = np.var(img)
        minv = np.min(img)
        maxv = np.max(img)

        # === 2. Texture features (GLCM) ===
        glcm = graycomatrix(
            (img * 255).astype(np.uint8),
            distances=[1, 2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True
        )
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        # === 3. Shape features (Hu Moments) ===
        moments = cv2.moments((img * 255).astype(np.uint8))
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log transform for better scaling stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-9)

        # === 4. Edge features ===
        edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
        edge_density = np.mean(edges > 0)

        # Combine all features
        feats = [
            mean, std, var, minv, maxv,
            contrast, dissimilarity, homogeneity, energy, correlation,
            *hu_moments, edge_density
        ]

        features.append(feats)
        labels.append(label)
        file_names.append(f)

# === Define columns ===
cols = [
    "mean", "std", "var", "min", "max",
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
    "edge_density"
]

# === Save final DataFrame ===
spiral_df = pd.DataFrame(features, columns=cols)
spiral_df["status"] = labels
spiral_df.to_csv(out_path, index=False)

# Save filenames too (for tracking)
pd.DataFrame({"file": file_names, "status": labels}).to_csv(
    "../../features/spiral_files.csv", index=False
)

print(f" Spiral handcrafted features saved to {out_path}")
print(f" Total features: {len(cols)} | Samples: {spiral_df.shape[0]}")
