# -Heart-Failure-Prediction-Project

## Project Overview
This project implements a machine learning model to predict heart failure risk using clinical parameters. The model achieves over 80% accuracy and includes a user-friendly Flask web application for real-time predictions.

## Project Structure
heart_failure_project/
│
├── heart_failure_model.ipynb     # Jupyter notebook with model training
├── app.py                        # Flask web application
├── heart_failure_model.pkl       # Trained model (generated after training)
├── feature_names.pkl            # Feature names (generated after training)
├── scaler.pkl                   # Data scaler (if needed, generated after training)
├── templates/
│   └── index.html               # Web interface with custom CSS
├── heart_failure_clinical_records_dataset (1).csv  # Dataset
└── README.md                    # This file

Step 1: Prepare the Dataset

Download the heart failure dataset
Place it in your project directory as heart_failure_clinical_records_dataset (1).csv

Step 2: Train the Model

Run the Jupyter notebook heart_failure_model.ipynb
This will generate:

heart_failure_model.pkl
feature_names.pkl
scaler.pkl

Step 3: Run the Flask App
bashpython app.py
The application will be available at http://localhost:5000
📊 Model Performance

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

Using the Web App

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

API Endpoints

GET / - Main web interface
POST /predict - Make prediction (returns JSON)
GET /api/features - Get feature names
GET /health - Health check endpoint

Example API Usage
pythonimport requests

# Prediction request
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
