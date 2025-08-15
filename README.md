# APS360-Project - AI Skin Cancer Classifier

## 🔍 Problem Statement

Skin cancer is one of the most common cancers worldwide. Early and accurate classification of skin lesions can help with timely treatment and improved outcomes. Our goal is to build a robust model that can classify images of skin lesions into one of seven known classes using deep learning.

## 🌟 Web Application Features

This project now includes an **impressive Streamlit web application** for real-time skin cancer classification:

- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Image Classification**: Upload and classify skin lesion images
- **Multiple Model Support**: ResNet18 and Encoder-Decoder CNN architectures
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Educational Content**: Detailed information about skin lesion types
- **Medical Recommendations**: Severity-based guidance for users

## 🚀 Quick Start - Web Application

### Prerequisites

- Python 3.8 or higher
- Your trained model file (.pth)

### Installation & Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained model file**

   - Copy your `.pth` model file to the project directory
   - Update the model path in `app.py` (line 236):
     ```python
     model_path = "your_model_name.pth"  # Update this path
     ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
APS360 Project/
├── app.py                          # Main Streamlit application
├── model_demo.py                   # Standalone demo script
├── requirements.txt                # Python dependencies
├── Skin_Cancer_Classification.ipynb # Training notebook
├── README.md                       # This file
└── your_model.pth                  # Your trained model file
```

## 🎯 Web Application Pages

### 🏠 Home Page

- Overview of the application and key features
- Model performance metrics display

### 🔍 Classifier Page

- **Image Upload**: Drag and drop or browse for skin lesion images
- **Model Selection**: Choose between ResNet18 and Encoder-Decoder CNN
- **Real-time Analysis**: Get instant predictions with confidence scores
- **Interactive Charts**: Visualize prediction confidence for all classes
- **Medical Guidance**: Severity-based recommendations

### 📊 About Model Page

- **Architecture Details**: Technical information about the models
- **Training Process**: Two-phase training strategy
- **Dataset Information**: HAM10000 dataset statistics

### ℹ️ Information Page

- **Skin Lesion Types**: Detailed descriptions of all 7 classes
- **Medical Information**: When to seek professional help

## 📊 Skin Lesion Classes

1. **AKIEC** - Actinic Keratoses and Intraepithelial Carcinoma (High Risk)
2. **BCC** - Basal Cell Carcinoma (Medium-High Risk)
3. **BKL** - Benign Keratosis-like Lesions (Low Risk)
4. **DF** - Dermatofibroma (Low Risk)
5. **MEL** - Melanoma (Very High Risk)
6. **NV** - Melanocytic Nevi (Low Risk)
7. **VASC** - Vascular Lesions (Low Risk)

## ⚠️ Important Medical Disclaimer

- This application is for **educational purposes only**
- **Not a substitute** for professional medical diagnosis
- Always consult qualified healthcare professionals
- Early detection is crucial for successful treatment

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
