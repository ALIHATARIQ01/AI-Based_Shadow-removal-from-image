# 🖼️ AI-Based Shadow Removal from Images

This project is focused on removing shadows from natural images using computer vision techniques. It uses manual K-Means clustering, adaptive morphological filtering, and inpainting, achieving an **average SSIM score of over 80%** on the test set.

---

## 📁 Project Structure

.
├── dataset/ # Input shadow images
├── images_preprocessed/ # Preprocessed images (auto-generated)
├── result/ # Output: original, mask, shadow-removed images
├── preprocessing.py # Preprocessing and cross-correlation analysis
├── test.py # Shadow mask generation, removal, evaluation
├── cross_correlation_heatmap.jpg
├── mean_intensity_boxplot.jpg
---

## ✨ Features

- Shadow detection using HSV + K-Means clustering
- Morphological filtering and adaptive inpainting
- Structural Similarity Index (SSIM) evaluation
- Cross-correlation and statistical box plot analysis
- Minimal dependencies (only OpenCV and NumPy)

---

## 🎯 Accuracy

> ✅ **Average SSIM Score (Test Set): ~0.82**

The system effectively removes shadows while preserving important image features.

---

## ⚙️ How to Run

### Step 1: Preprocess the Images
```bash
python preprocessing.py
### Step 2: Run Shadow Removal and Evaluation
python test.py
The results (original, mask, and shadow-free output) will be saved in the result/ folder.
## 🔍 Techniques Used
HSV Color Space Conversion

Manual K-Means Clustering (k=3)

Morphological Operations (Erosion, Dilation)

Shadow Region Inpainting (cv2.inpaint)

SSIM-based Evaluation

Cross-Correlation Matrix

Statistical Boxplot Drawing in OpenCV

## 🖥️ Sample Output
Each output image in result/ includes:

Original Image

Detected Shadow Mask

Shadow-Removed Result
## Requirements
Python 3.x

OpenCV

NumPy

## Install dependencies:
pip install opencv-python numpy
## 🚀 Future Improvements
Use of Deep Learning models like U-Net or GANs

Deployment as a Web Application or API

Real-time shadow detection and removal for video

👩‍💻 Author
# Aliha Tariq
# Computer Science Enthusiast | Passionate about AI & Image Processing
