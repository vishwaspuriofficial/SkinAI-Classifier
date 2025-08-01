﻿# APS360-Project

---

## 🔍 Problem Statement

Skin cancer is one of the most common cancers worldwide. Early and accurate classification of skin lesions can help with timely treatment and improved outcomes. Our goal is to build a robust model that can classify images of skin lesions into one of seven known classes using deep learning.

---

## 🧠 Models

### 🔹 Baseline Model
- Custom CNN (3 convolutional layers + 2 FC layers)
- Evaluated on a small subset (~1000 training / 200 validation images)
- Achieved ~60–70% validation accuracy
- Helped confirm the model could learn from limited data

### 🔹 Primary Model
- Feature extractor using pretrained **ResNet18**
- Last FC layer replaced for 7-class output
- Extracted features fed into a smaller classifier
- Fast and more accurate (~70–80% validation accuracy)
- Trained using only a subset of full dataset to reduce runtime

---

## 📁 Dataset

- **HAM10000**: Human Against Machine with 10000 training images of skin lesions
- 7 classes:
  - Melanoma
  - Melanocytic nevus
  - Basal cell carcinoma
  - Actinic keratosis
  - Benign keratosis
  - Dermatofibroma
  - Vascular lesion
- Data Preprocessing:
  - Resized to 224×224
  - Normalized to ImageNet mean/std
  - Augmented with horizontal flips, color jitter, rotations

---

## 📊 Results

- ✅ **Quantitative**:
  - CNN (Baseline): ~65% validation accuracy on 1K images
  - ResNet Feature Extraction: Up to 78% accuracy on 1.2K images
- 👀 **Qualitative**:
  - Visual inspection of predictions on challenging samples
  - Confusion matrix shows which classes are harder to classify
  - Training/validation curves show effective learning

---

## 🧰 Technologies Used

- Python 3.11
- PyTorch
- torchvision
- NumPy
- matplotlib
- scikit-learn
- Google Colab


