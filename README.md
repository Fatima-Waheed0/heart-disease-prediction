# Heart Disease Prediction

## 🧠 Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection of individuals at risk can significantly reduce complications and save lives. This project aims to build a machine learning model that can predict the presence of heart disease based on various medical attributes of a patient. The goal is to assist medical professionals in making quicker, more accurate diagnoses.

---

## 📊 About the Dataset
This dataset, originally compiled in 1988, combines medical records from four databases: **Cleveland**, **Hungary**, **Switzerland**, and **Long Beach V**. While the dataset includes 76 attributes, only 14 are commonly used for analysis and modeling, as referenced in most published studies.

- Total records: 1025  
- Features used: 13  
- Target variable: 1 (indicates presence of heart disease) or 0 (no heart disease)

---

## 🔍 Features

| Feature        | Description |
|----------------|-------------|
| `age`          | Age of the patient |
| `sex`          | Sex (1 = male; 0 = female) |
| `cp`           | Chest pain type (0–3) |
| `trestbps`     | Resting blood pressure (mm Hg) |
| `chol`         | Serum cholesterol (mg/dl) |
| `fbs`          | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `restecg`      | Resting electrocardiographic results (0–2) |
| `thalach`      | Maximum heart rate achieved |
| `exang`        | Exercise-induced angina (1 = yes; 0 = no) |
| `oldpeak`      | ST depression induced by exercise relative to rest |
| `slope`        | Slope of the peak exercise ST segment |
| `ca`           | Number of major vessels colored by fluoroscopy (0–3) |
| `thal`         | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) |
| `target`       | 0 = No heart disease, 1 = Heart disease |

---

## ❓ Why This Dataset?
- It is a widely studied benchmark dataset in the medical and machine learning communities.
- Compact and clean for beginners but rich enough for meaningful insights.
- It provides a balanced classification problem with clear performance metrics.

---

## 📌 Source
Dataset link on Kaggle:  
[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?utm_source=chatgpt.com)

---

## ⚙️ Project Flow
1. **Data Cleaning**  
   - Removed duplicate rows (723 found).
2. **Exploratory Data Analysis (EDA)**  
   - Understanding feature distributions and relationships.
3. **Preprocessing**  
   - Encoding categorical variables and scaling numeric features using `ColumnTransformer`.
4. **Model Training**  
   - Tried Logistic Regression and Random Forest.
   - Final model: **Logistic Regression** (Accuracy: 81.97%).
5. **Model Saving**  
   - Final trained model saved using `joblib` for future use.

---

## ✅ Conclusion
This project demonstrates how machine learning can be applied to healthcare problems to assist professionals in identifying potential heart disease cases. The Logistic Regression model performed well and can serve as a foundational baseline for more advanced methods in future work.
