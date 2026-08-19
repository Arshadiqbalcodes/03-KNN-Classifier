
# 🧠 K-Nearest Neighbors (KNN) Classifier - Diabetes Prediction

This project demonstrates how to use the **K-Nearest Neighbors (KNN)** algorithm to classify whether a patient has diabetes or not based on medical features.

---

## 📌 What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for **classification and regression**.

It works by:
- Finding the **K nearest data points**
- Using majority vote (classification)
- Predicting based on similarity

📖 Reference: https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/

---

## ⚙️ Step 1: Install Required Libraries

pip install pandas numpy matplotlib scikit-learn seaborn

---

## 📥 Step 2: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

---

## 📊 Step 3: Load Dataset

df = pd.read_csv("diabetes.csv")
df.head()

Dataset contains medical attributes like:
- Pregnancies
- Glucose
- BloodPressure
- BMI
- Age
- Outcome (Target)

---

## 🎯 Step 4: Split Features & Target

X = df.drop('Outcome', axis=1)
y = df['Outcome']

---

## ✂️ Step 5: Train-Test Split (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

---

## 🤖 Step 6: Train KNN Model

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

---

## 🔮 Step 7: Make Predictions

y_pred = knn.predict(X_test)

---

## 📏 Step 8: Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score

Accuracy ≈ 66%

---

## 📊 Step 9: Confusion Matrix

Visualize correct vs incorrect predictions using heatmap.

---

## 🔍 Step 10: Find Best K Value

Test different values of K (1 to 20):

- Best accuracy found around **K = 4**
- Accuracy ≈ **72.7%**

---

## 📈 Step 11: Final Model (K=4)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

Final accuracy improved to ~73%

---

## 🧪 Step 12: Make Prediction on New Data

Sample input:

[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

Prediction:
- 1 → Diabetes detected
- 0 → No diabetes

---

## 🚀 Conclusion

- KNN is simple but powerful for classification
- Performance depends heavily on value of K
- Best accuracy achieved: ~73%
- Feature scaling can further improve results

---

## 📌 Future Improvements

- Apply feature scaling (StandardScaler)
- Try weighted KNN
- Compare with Logistic Regression or SVM
