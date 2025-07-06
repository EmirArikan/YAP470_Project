from PIL import Image, ImageFilter
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
from scipy.stats import randint, uniform, skew, kurtosis
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import joblib



# Pre-processing Fonksiyonları 

def noise_reduction(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def contrast_enhancement(image, level):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=level, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def zscore_standardization(image):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-8)

#-------------------------------------------------------------------------------------------------------------------------------

# Öznitelik Çıkarım Fonksiyonları

def lbp(image_tensor, radius=1):
    image = image_tensor.numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
    rows, cols = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = image[i, j]
            code = 0
            for n, (dx, dy) in enumerate([(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]):
                neighbor = image[i+dy, j+dx]
                code |= (neighbor >= center) << n
            lbp_image[i, j] = code
    return torch.from_numpy(lbp_image)

def lbp_histogram(lbp_img, bins=256):
    hist, _ = np.histogram(lbp_img.ravel(), bins=bins, range=(0, bins), density=True)
    return hist


def glcm(image_tensor):
    image_flat = image_tensor.ravel()
    
    mean = np.mean(image_flat)
    var = np.var(image_flat)
    skw = skew(image_flat)
    krt = kurtosis(image_flat)

    return torch.tensor([mean, var, skw, krt], dtype=torch.float32)

def glcm_matr(image, distance=1, angle=0, levels=256):
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    if angle == 0:
        dx, dy = distance, 0
    elif angle == 45:
        dx, dy = distance, -distance
    elif angle == 90:
        dx, dy = 0, -distance
    elif angle == 135:
        dx, dy = -distance, -distance
    else:
        raise ValueError("Only angles 0,45,90,135 are supported")

    img_norm = image.copy()
    if img_norm.dtype != np.uint8:
        img_norm = np.clip(img_norm, 0, 1)  
        img_norm = (img_norm * (levels - 1)).astype(np.uint8)
    else:
        img_norm = np.clip(img_norm, 0, levels - 1)

    for y in range(h):
        for x in range(w):
            x_n = x + dx
            y_n = y + dy
            if 0 <= x_n < w and 0 <= y_n < h:
                i = int(img_norm[y, x])
                j = int(img_norm[y_n, x_n])
                glcm[i, j] += 1

    glcm /= glcm.sum() + 1e-12
    return glcm

def glcm_features(glcm):
    levels = glcm.shape[0]
    I, J = np.ogrid[0:levels, 0:levels]

    contrast = np.sum(glcm * (I - J) ** 2)
    dissimilarity = np.sum(glcm * np.abs(I - J))
    homogeneity = np.sum(glcm / (1.0 + (I - J) ** 2))
    asm = np.sum(glcm ** 2)
    energy = np.sqrt(asm)

    mu_i = np.sum(I * glcm)
    mu_j = np.sum(J * glcm)
    sigma_i = np.sqrt(np.sum(((I - mu_i) ** 2) * glcm))
    sigma_j = np.sqrt(np.sum(((J - mu_j) ** 2) * glcm))

    correlation = np.sum((I - mu_i) * (J - mu_j) * glcm) / (sigma_i * sigma_j + 1e-12)

    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'ASM': asm,
        'energy': energy,
        'correlation': correlation
    }

def hog(image, orientations=9, pixels_per_cell=(8, 8)):
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    gx[:, 1:-1] = image[:, 2:].astype(np.float32) - image[:, :-2].astype(np.float32)
    gy[1:-1, :] = image[2:, :].astype(np.float32) - image[:-2, :].astype(np.float32)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180 / np.pi)
    angle[angle < 0] += 180  

    h, w = image.shape
    cell_h, cell_w = pixels_per_cell
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    bin_width = 180 / orientations
    hog_vector = []

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_mag = magnitude[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            cell_ang = angle[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            hist = np.zeros(orientations, dtype=np.float32)

            for i in range(cell_h):
                for j in range(cell_w):
                    mag = cell_mag[i, j]
                    ang = cell_ang[i, j]
                    bin_idx = int(ang // bin_width) % orientations
                    hist[bin_idx] += mag

            hist /= (np.linalg.norm(hist) + 1e-6)
            hog_vector.extend(hist)

    return np.array(hog_vector)


#-----------------------------------------------------------------------------------------------

#Öznitelik Seçim Fonksiyonu

def apply_pca(X, n_components=100):
    if X.shape[0] <= 1:
        raise ValueError("PCA için en az 2 örnek gerekir.")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(f"PCA sonrası boyut: {X_pca.shape}")
    print(f"Açıklanan varyans oranı toplamı: {pca.explained_variance_ratio_.sum():.4f}")

    return X_pca, pca



#---------------------------------------------------------------------------------------------

#Diğer Yardımcı Fonksiyonlar

def load_images_from_folder(folder_path, ext_list=None):
    if ext_list is None:
        ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

    images = []
    file_names = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in ext_list):
                img_path = os.path.join(root, filename)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                blurred = noise_reduction(img)
                result = contrast_enhancement(blurred, 1)
                normalize = zscore_standardization(result)

                if normalize is not None:
                    images.append(normalize)
                    file_names.append(os.path.splitext(filename)[0][:12])
    return images, file_names



def features_to_dataframe(texture_feats, hog_hist, lbp_hist=None, glcm_feats=None,
                          patient=None, histological_type=None):
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    texture_feats = to_np(texture_feats)
    hog_hist = to_np(hog_hist)
    if lbp_hist is not None:
        lbp_hist = to_np(lbp_hist)

    feature_dict = {}

    if patient is not None:
        feature_dict['Patient'] = patient

    if histological_type is not None:
        feature_dict['histological_type'] = histological_type
        
    feature_dict.update({
        'texture_mean': texture_feats[0],
        'texture_var': texture_feats[1],
        'texture_skew': texture_feats[2],
        'texture_kurtosis': texture_feats[3],
    })

    for i, val in enumerate(hog_hist):
        feature_dict[f'hog_{i}'] = val

    if lbp_hist is not None:
        for i, val in enumerate(lbp_hist):
            feature_dict[f'lbp_{i}'] = val

    if glcm_feats is not None:
        for key, val in glcm_feats.items():
            feature_dict[f'glcm_{key}'] = val

    

    return pd.DataFrame([feature_dict])

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Recall (macro): {recall:.4f}")
    print(f"  F1-Score (macro): {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.show()

    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        if len(classes) == 2:
            y_score = np.column_stack([1 - y_score, y_score])
    else:
        print(f"{name}: Model does not support probability outputs.")
        return

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    print(f"  AUC (micro-average): {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} ROC (AUC = {roc_auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.grid()
    plt.show()
