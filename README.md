# AICTE-Garbage-Classification
# 🗑️ Garbage Image Classification using Deep Learning

This project implements a deep learning-based image classifier to categorize garbage into six classes using three pretrained CNN models: EfficientNetV2B2, EfficientNetV2B3, and DenseNet201. The goal is to improve waste management through automated image classification.

---

## 📌 Project Overview

- **Type**: Image Classification
- **Technique**: Transfer Learning with CNNs
- **Models Used**:
  - EfficientNetV2B2
  - EfficientNetV2B3
  - DenseNet201
- **Framework**: TensorFlow / Keras
- **Environment**: Google Colab

---

## 🎯 Objective

To classify waste images into the following categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 🛠️ Tools and Libraries

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Google Colab

---

## 📁 Dataset

- Dataset Source: [Kaggle Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- Number of Classes: 6
- Preprocessing:
  - Resized images to 124×124
  - Normalized pixel values
  - Training/Validation Split: 80/20

---

## 📊 Methodology

1. **Data Preprocessing**:
   - Resized and normalized all images.
   - Used `image_dataset_from_directory` for easy loading.

2. **Model Architecture**:
   - Imported pretrained base models without the top layer (`include_top=False`).
   - Added Global Average Pooling, Dropout, and Dense layer with softmax.

3. **Training**:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Epochs: 15
   - Batch size: 32

4. **Evaluation**:
   - Accuracy and loss plotted across epochs.
   - Final validation accuracy comparison:
     - EfficientNetV2B2: **88.67%**
     - EfficientNetV2B3: **91.80%**
     - DenseNet201: **91.02%**

---

## 📈 Results

| Model            | Validation Accuracy |
|------------------|---------------------|
| EfficientNetV2B2 | 88.67%              |
| EfficientNetV2B3 | 91.80%              |
| DenseNet201      | 91.02%              |

---

## ✅ Conclusion

- EfficientNetV2B2 performed best in terms of accuracy.
- All models were successfully trained using Google Colab.
- The project demonstrates that transfer learning is effective for small custom datasets like garbage classification.

---



