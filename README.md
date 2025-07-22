# -Heart-Failure-Prediction-Project

## Project Overview
This project is a Machine Learning-based web application that predicts the likelihood of heart failure based on clinical records. The app is built with **Flask** and trained using a classification model to assist medical professionals and patients in identifying risks early.
The model also achieves over 80% accuracy and includes a user-friendly Flask web application for real-time predictions.

## Objective

- Train a machine learning model to predict heart failure.
- Achieve at least **80% accuracy**.
- Deploy the model using Flask.
- Create a responsive web interface for user input.


Step 1:Dataset
The dataset used is the **Heart Failure Clinical Records Dataset**, which contains 13 clinical features such as age, ejection fraction, serum creatinine, and more.

--- [Source of Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Flask
- HTML/CSS (Custom styling)
- Jupyter Notebook

## Train the Model

Run the Jupyter notebook heart_failure_model.ipynb
This will generate:

- heart_failure_model.pkl
- feature_names.pkl
- scaler.pkl

## Model Performance

Primary Model: Random Forest Classifier (with hyperparameter tuning)
Accuracy: 80%+ (varies based on data split)
Features: 12 clinical parameters
Cross-validation: 5-fold CV for robust evaluation

Key Features Used:

age - Age of patient
anaemia - Decrease of red blood cells or hemoglobin
creatinine_phosphokinase - Level of CPK enzyme in blood
diabetes - If patient has diabetes
ejection_fraction - Percentage of blood leaving heart at each contraction
high_blood_pressure - If patient has hypertension
platelets - Platelet count in blood
serum_creatinine - Level of serum creatinine in blood
serum_sodium - Level of serum sodium in blood
sex - Woman or man
smoking - If patient smokes
time - Follow-up period

## Web Application Feature

Responsive Design: Works on desktop and mobile
Real-time Predictions: Instant results with confidence scores
Interactive UI: Modern design with animations and tooltips
Input Validation: Ensures all required fields are completed
Risk Visualization: Progress bars showing confidence levels
Error Handling: Graceful error messages and loading states

## Using the Web App

Fill in all 12 clinical parameters
Click "Predict Heart Failure Risk"
View the prediction result with confidence scores
Pro tip: Double-click the header title to fill sample data for testing

## Custom CSS Features
The web interface includes:

Gradient backgrounds and modern styling
Hover effects and transitions
Responsive grid layout
Custom progress bars for confidence visualization
Tooltip information for medical parameters
Loading animations
Risk-based color coding (red for high risk, blue for low risk)

## Prediction request
data = {
    'age': 75,
    'anaemia': 0,
    'creatinine_phosphokinase': 582,
    'diabetes': 0,
    'ejection_fraction': 20,
    'high_blood_pressure': 1,
    'platelets': 265000,
    'serum_creatinine': 1.9,
    'serum_sodium': 130,
    'sex': 1,
    'smoking': 0,
    'time': 4
}

response = requests.post('http://localhost:5000/predict', data=data)
result = response.json()
print(result)

## Model Training Process

Data Exploration: Comprehensive EDA with visualizations
Feature Engineering: Data preprocessing and scaling
Model Comparison: Testing multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression, SVM)
Hyperparameter Tuning: GridSearchCV for optimal parameters
Model Validation: Cross-validation and performance metrics
Feature Importance: Analysis of most predictive features

## Results & Insights

Most Important Features: Ejection fraction, serum creatinine, and age are typically the most predictive
Model Robustness: Cross-validation ensures stable performance
Real-world Application: Web interface makes the model accessible for practical use

## Acknowledgements
- Dev Town Bootcamp
- Dataset by Andras MVD on Kaggle


