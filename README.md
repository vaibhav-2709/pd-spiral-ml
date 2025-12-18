# Parkinson’s Disease Detection Using Spiral Drawing Analysis

## 📌 Overview
This project presents a machine learning–based framework for the early detection of Parkinson’s Disease (PD) using spiral-drawing images. The system leverages handcrafted feature extraction and multiple classification models, followed by a soft-voting ensemble strategy to improve diagnostic reliability.

The proposed approach emphasizes interpretability, robustness, and reproducibility, making it suitable for academic research and practical screening applications.

---

## 🎯 Objectives
- Detect Parkinson’s Disease from spiral-drawing patterns
- Extract interpretable handcrafted features capturing motor irregularities
- Compare multiple classical machine learning models
- Evaluate a proposed ensemble model using **5-fold cross-validation**
- Analyze performance using clinically relevant metrics

---

## 📂 Dataset
- **Input**: Handcrafted feature vectors extracted from spiral-drawing images  
- **Labels**:  
  - `0` → Healthy  
  - `1` → Parkinson’s Disease  

Features include:
- Statistical descriptors
- Texture features (GLCM)
- Shape descriptors (Hu Moments)
- Edge-based features

---

## 🧠 Models Implemented
- Support Vector Machine (SVM)
- Random Forest (RF)
- Logistic Regression (LR)
- k-Nearest Neighbors (kNN)
- XGBoost (XGB)
- Multi-Layer Perceptron (MLP)
- **Proposed Ensemble Model (RF + XGB + kNN)**

---

## 🔄 Experimental Setup
- **Data Normalization**: Z-score standardization
- **Train–Test Evaluation**:
  - 80–20 split
  - 70–30 split
- **Cross-Validation**:
  - **5-Fold Stratified Cross-Validation**
- **Ensemble Strategy**:
  - Soft-voting using averaged class probabilities

---

## 📊 Evaluation Metrics
The following metrics are reported for each model:
- Accuracy
- Recall (Sensitivity)
- F1-Score
- Specificity
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Confusion Matrix Analysis

Mean and per-fold results are reported for cross-validation experiments.
