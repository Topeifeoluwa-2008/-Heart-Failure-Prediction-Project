from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
with open("heart_failure_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names
feature_names = joblib.load("feature_names.pkl")

# Retrieve feature importances if available (optional)
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
else:
    sorted_features = []

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve inputs from form
    input_data = [float(request.form.get(feature, 0)) for feature in feature_names]
    input_array = np.array(input_data).reshape(1, -1)

    # Scale inputs
    if scaler:
        input_array = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_array)
    result = 'High Risk of Heart Failure' if prediction[0] == 1 else 'Low Risk of Heart Failure'

    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)