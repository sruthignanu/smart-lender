# app.py - Complete Flask Application
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# IMPORT LIBRARIES AND LOAD MODEL
# ============================================
print("=" * 60)
print("SMART LENDER - LOADING MODEL...")
print("=" * 60)

# Create Flask application instance
app = Flask(__name__)

# Load the saved model and artifacts
try:
    # Load the saved model (using the name from the description: rdf.pkl or your model)
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    print("✅ Model and artifacts loaded successfully!")
    print(f"📊 Model type: XGBoost")
    
except FileNotFoundError:
    print("❌ Model file not found! Please ensure models are in the 'models/' folder.")
    model = None
    scaler = None
    label_encoders = None
    target_encoder = None

# ============================================
# PREPROCESSING FUNCTION
# ============================================
def preprocess_input(input_data):
    """Preprocess input data using saved encoders and scaler"""
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0
    
    # Scale numerical features
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df

# ============================================
# PREDICTION FUNCTION
# ============================================
def make_prediction(input_data):
    """Make prediction using the loaded model"""
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Decode result
        result = target_encoder.inverse_transform(prediction)[0]
        confidence = float(max(prediction_proba[0])) * 100
        prob_approved = float(prediction_proba[0][1]) * 100
        prob_rejected = float(prediction_proba[0][0]) * 100
        
        # Determine risk level
        if confidence > 80:
            risk_level = "Low"
        elif confidence > 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'prediction': result,
            'confidence': confidence,
            'probability_approved': prob_approved,
            'probability_rejected': prob_rejected,
            'risk_level': risk_level
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/predict')
def predict():
    """Render the prediction form page"""
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission and show results"""
    try:
        # Retrieve values from the UI using POST request
        input_data = {
            'Gender': request.form.get('gender'),
            'Married': request.form.get('married'),
            'Dependents': request.form.get('dependents'),
            'Education': request.form.get('education'),
            'Self_Employed': request.form.get('self_employed'),
            'ApplicantIncome': float(request.form.get('applicant_income', 0)),
            'CoapplicantIncome': float(request.form.get('coapplicant_income', 0)),
            'LoanAmount': float(request.form.get('loan_amount', 0)),
            'Loan_Amount_Term': float(request.form.get('loan_term', 360)),
            'Credit_History': float(request.form.get('credit_history', 1.0)),
            'Property_Area': request.form.get('property_area')
        }
        
        # Validate input
        if input_data['ApplicantIncome'] <= 0:
            return render_template('submit.html', 
                                 error="Applicant income must be greater than 0")
        
        if input_data['LoanAmount'] <= 0:
            return render_template('submit.html', 
                                 error="Loan amount must be greater than 0")
        
        # Store values in array and pass to model.predict()
        result = make_prediction(input_data)
        
        if 'error' in result:
            return render_template('submit.html', error=result['error'])
        
        # Render the submit.html results page with prediction
        return render_template('submit.html', 
                             prediction=result['prediction'],
                             confidence=f"{result['confidence']:.1f}",
                             risk_level=result['risk_level'],
                             probability_approved=f"{result['probability_approved']:.1f}",
                             probability_rejected=f"{result['probability_rejected']:.1f}")
    
    except Exception as e:
        return render_template('submit.html', error=f"An error occurred: {str(e)}")

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("SMART LENDER - STARTING APPLICATION")
    print("=" * 60)
    print("🌐 Open browser and navigate to: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)