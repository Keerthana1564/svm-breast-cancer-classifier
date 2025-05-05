# SVM Binary Classification using Scikit-Learn

This project demonstrates how to use **Support Vector Machines (SVMs)** for binary classification using the **Breast Cancer Dataset** from `scikit-learn`. It covers both **linear** and **non-linear (RBF)** kernels, along with visualization, hyperparameter tuning, and evaluation.

---

## 🔍 Objective

- Use SVMs for linear and non-linear classification
- Visualize decision boundaries in 2D
- Tune hyperparameters (`C`, `gamma`) using GridSearchCV
- Evaluate model performance with cross-validation

---

## 📚 Technologies Used

- Python
- Scikit-learn
- NumPy
- Matplotlib
- PCA (for visualization)

---

## 📁 Dataset

The project uses the built-in `load_breast_cancer()` dataset from `scikit-learn`, which contains features computed from breast mass images.

- **Target:** Binary (0 = Malignant, 1 = Benign)
- **Features:** 30 numeric features like radius, texture, perimeter, etc.

---

## 🚀 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/svm-binary-classification.git
cd svm-binary-classification
