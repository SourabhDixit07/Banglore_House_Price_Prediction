from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and feature names
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    print(f"Model loaded successfully with {len(feature_names)} features")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or feature_names is None:
        return render_template('index.html', prediction_text='Error: Model not loaded properly')
    
    try:
        # Get form data
        total_sqft = float(request.form['total_sqft'])
        bath = float(request.form['bath'])
        bhk = float(request.form['bhk'])
        location = request.form['location']
        
        # Create a DataFrame with the input features
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set the basic numeric features
        input_data['total_sqft'] = total_sqft
        input_data['bath'] = bath
        input_data['bhk'] = bhk
        input_data['price_per_sqft'] = 0  # This will be calculated by the model
        # If your model expects balcony, set to 0 or a default value
        if 'balcony' in input_data.columns:
            input_data['balcony'] = 0
        
        # Set the location (one-hot encoding)
        if location in feature_names:
            input_data[location] = 1
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Predicted House Price: ₹{prediction:.2f} Lakhs',
                             total_sqft=total_sqft,
                             bath=bath,
                             bhk=bhk,
                             location=location)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error in prediction: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)