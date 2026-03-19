# Brain Tumor Classification using CNN 🧠

A deep learning model to classify brain MRI images as **Tumor** or **No Tumor** using a Convolutional Neural Network (CNN), built with TensorFlow and Keras.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 97.07% |
| Validation Accuracy | 95.0% |
| Training Loss | 0.05 |
| Validation Loss | 0.12 |
| AUC (ROC Curve) | 0.96 |
| Precision | 94.47% |
| Recall | 90.0% |
| F1 Score | 92.32% |

### Confusion Matrix (Test Set)
| | Predicted: No Tumor | Predicted: Tumor |
|---|---|---|
| **Actual: No Tumor** | 290 (TN) | 10 (FP) |
| **Actual: Tumor** | 8 (FN) | 292 (TP) |

---

## 📁 Dataset

- **Total Images:** 3000 (1500 per class)
- **Classes:** Tumor / No Tumor
- **Image Size:** 64 × 64 pixels (RGB)
- **Training Set:** 2400 images
- **Test Set:** 600 images
- **Split:** 80% training / 20% testing (via Scikit-learn `train_test_split`)

---

## 🏗️ Model Architecture

Built using Keras Sequential API:

- 3× Convolutional layers with **ReLU** activation
- 3× MaxPooling layers (dimensionality reduction)
- Flatten layer
- Fully Connected (Dense) layers
- Dropout layers (to reduce overfitting)
- Output layer with **Sigmoid** activation (binary classification)

**Optimizer:** Adam  
**Loss Function:** Binary Cross-Entropy  
**Epochs:** 10 | **Batch Size:** 16

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV (`cv2`)
- Scikit-learn
- Matplotlib / Seaborn
- NumPy

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Chelsi08/BrainTumor_Classification.git
cd BrainTumor_Classification
```

2. Install dependencies:
```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy
```

3. Run the training script:
```bash
python mainTrain.py
```

---

## 📈 Evaluation Outputs

The following visualizations are generated after training:

- `accuracy_plot.eps` — Training vs Validation Accuracy over epochs
- `loss_plot.eps` — Training vs Validation Loss over epochs
- `confusion_matrix.eps` — Confusion matrix heatmap
- `roc_curve.eps` — ROC curve with AUC score
- `random_test.eps` — Random test images with true vs predicted labels

---

## 📄 Research Paper

This project is part of a published research chapter:

**"Deep Learning for Brain Tumor Analysis: A Neural Network Approach"**    


---
