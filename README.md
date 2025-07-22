# -Heart-Failure-Prediction-Project

## Project Overview
This project is a Machine Learning-based web application that predicts the likelihood of heart failure based on clinical records. The app is built with **Flask** and trained using a classification model to assist medical professionals and patients in identifying risks early.
The model also achieves over 80% accuracy and includes a user-friendly Flask web application for real-time predictions.

## Objective

- Train a machine learning model to predict heart failure.
- Achieve at least **80% accuracy**.
- Deploy the model using Flask.
- Create a responsive web interface for user input.


Dataset

The dataset used is the Heart Failure Clinical Records Dataset, which contains 13 clinical features such as age, ejection fraction, serum creatinine, and more.

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

## Prepare features and target
df = pd.read_csv('/content/heart_failure_clinical_records_dataset (1).csv')

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

print("\nFeature columns:")
print(X.columns.tolist())

# Train-test split
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train and evaluate models
results = {}
best_model = None
best_accuracy = 0
best_model_name = None
best_scaler = None

print("\nModel Performance Comparison:")
print("="*60)
 Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

for name, model in models.items():
    # Use scaled data for SVM and Logistic Regression
    if name in ['SVM', 'Logistic Regression']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
        current_scaler = scaler
    else:
        X_train_use = X_train
        X_test_use = X_test
        current_scaler = None

    # Train the model
    model.fit(X_train_use, y_train)

    # Make predictions
    y_pred = model.predict(X_test_use)
    y_pred_proba = model.predict_proba(X_test_use)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'accuracy': accuracy,
        'auc': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-"*60)

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_scaler = current_scaler

print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
```

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


