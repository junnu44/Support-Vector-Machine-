📌 Project Overview

This project applies Support Vector Machine (SVM), a powerful supervised machine learning algorithm, to classify tumors as Malignant (cancerous) or Benign (non-cancerous) using the Breast Cancer Wisconsin dataset.

SVM is particularly effective in classification tasks because it tries to find the optimal hyperplane that best separates two classes. With the use of kernels, SVM can also handle non-linear decision boundaries.

🎯 Objectives

Train and evaluate an SVM model for binary classification.

Apply preprocessing steps such as scaling features.

Compare model performance with different kernels (linear, rbf, poly).

Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.

Visualize the results with confusion matrix and ROC curve.

🛠️ Tech Stack

Programming Language: Python 🐍

Libraries:

NumPy → numerical computations

Pandas → data manipulation

Matplotlib & Seaborn → visualization

Scikit-learn → dataset, preprocessing, SVM model, evaluation

📂 Dataset

The Breast Cancer Wisconsin Dataset is included in scikit-learn.

Features (30):
Measurements such as mean radius, mean texture, mean smoothness, etc.

Target (Binary):

0 → Malignant (cancerous)

1 → Benign (non-cancerous)

Load dataset in Python:

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

🔎 Methodology

Load Dataset

Import from sklearn.datasets.

Data Preprocessing

Standardize features using StandardScaler.

Train-test split (80%-20%).

Model Training

Train an SVM classifier (sklearn.svm.SVC).

Experiment with linear, RBF, and polynomial kernels.

Model Evaluation

Accuracy, Precision, Recall, F1-score.

Confusion Matrix.

ROC Curve and AUC score.

Prediction

Test the model on unseen data.

📊 Results

Linear SVM achieves high accuracy (~96-97%).

RBF Kernel often provides slightly better performance in complex feature spaces.

Example Evaluation Metrics:

Accuracy: 0.97
Precision: 0.98
Recall: 0.96
F1-score: 0.97


Confusion Matrix shows classification distribution.

ROC Curve demonstrates high separability between classes.

🚀 How to Run the Project

Clone this repository:

git clone https://github.com/junnu44/Support-Vector-Machine.git
cd breast-cancer-svm


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook Breast_Cancer_SVM.ipynb

📌 Future Improvements

Apply GridSearchCV or RandomizedSearchCV for hyperparameter tuning (C, gamma, kernel).

Compare SVM with Logistic Regression, Random Forest, Gradient Boosting.

Build a Streamlit app to predict cancer outcomes from input features.

✨ Conclusion

This project demonstrates how Support Vector Machine (SVM) can effectively classify breast cancer tumors. With proper preprocessing and kernel selection, SVM provides high accuracy and robustness, making it a reliable choice for healthcare-related classification tasks.
