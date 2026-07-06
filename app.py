# app.py - Complete Flask Application with XGBoost Fix
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================
# IMPORT LIBRARIES AND LOAD MODEL
# ============================================
print("=" * 60)
print("SMART LENDER - LOADING MODEL...")
print("=" * 60)

# Create Flask application instance
app = Flask(__name__)

# ============================================
# XGBOOST COMPATIBILITY FIX - MULTIPLE LAYERS
# ============================================
def fix_xgboost():
    """Apply multiple fixes for XGBoost compatibility"""
    try:
        import xgboost as xgb
        
        # Fix 1: Remove the attribute if it exists
        if hasattr(xgb.XGBClassifier, 'use_label_encoder'):
            try:
                del xgb.XGBClassifier.use_label_encoder
                print("✅ Removed use_label_encoder from XGBClassifier")
            except:
                pass
        
        # Fix 2: Set the attribute to False if it exists
        if hasattr(xgb.XGBClassifier, 'use_label_encoder'):
            xgb.XGBClassifier.use_label_encoder = False
        
        # Fix 3: Patch the class for loading
        class PatchedXGBClassifier(xgb.XGBClassifier):
            def __init__(self, *args, **kwargs):
                kwargs.pop('use_label_encoder', None)
                super().__init__(*args, **kwargs)
        
        # Store the patched class
        xgb.XGBClassifier = PatchedXGBClassifier
        
        print("✅ XGBoost compatibility fixes applied")
        return True
    except Exception as e:
        print(f"⚠️ XGBoost fix warning: {e}")
        return False

# Apply the fixes BEFORE loading the model
fix_xgboost()

# ============================================
# LOAD THE MODEL
# ============================================
model = None
scaler = None
label_encoders = None
target_encoder = None
feature_columns = None

def load_models():
    """Load all models with error handling"""
    global model, scaler, label_encoders, target_encoder, feature_columns
    
    try:
        # Try loading with patched class
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        print("✅ Model and artifacts loaded successfully!")
        print(f"📊 Model type: XGBoost")
        print(f"📊 Features: {len(feature_columns) if feature_columns else 0}")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Files: {os.listdir('.')}")
        if os.path.exists('models'):
            print(f"📁 Models folder: {os.listdir('models')}")
        return False
        
    except AttributeError as e:
        print(f"❌ Attribute error during load: {e}")
        # Try alternative loading method
        try:
            import pickle
            with open('models/xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            scaler = joblib.load('models/scaler.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            target_encoder = joblib.load('models/target_encoder.pkl')
            feature_columns = joblib.load('models/feature_columns.pkl')
            print("✅ Model loaded using alternative method!")
            return True
        except Exception as e2:
            print(f"❌ Alternative load also failed: {e2}")
            return False
            
    except Exception as e:
        print(f"❌ Unknown error loading model: {e}")
        return False

# Load models
model_loaded = load_models()

if not model_loaded:
    print("❌ Failed to load models. Prediction will not work.")
    print("⚠️ Please ensure 'models/' folder exists with all .pkl files")

# ============================================
# PREPROCESSING FUNCTION
# ============================================
def preprocess_input(input_data):
    """Preprocess input data using saved encoders and scaler"""
    if model is None:
        raise Exception("Model not loaded")
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        if col in label_encoders and col in input_df.columns:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0
            except Exception as e:
                print(f"⚠️ Encoding error for {col}: {e}")
                input_df[col] = 0
    
    # Scale numerical features
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    except Exception as e:
        print(f"⚠️ Scaling error: {e}")
        # Fallback: use the values as is
        pass
    
    # Ensure correct column order
    input_df = input_df[feature_columns]
    
    return input_df

# ============================================
# PREDICTION FUNCTION
# ============================================
def make_prediction(input_data):
    """Make prediction using the loaded model"""
    try:
        if model is None:
            return {'error': 'Model not loaded. Please check server logs.'}
        
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
        print(f"❌ Prediction error: {e}")
        return {'error': f"Prediction error: {str(e)}"}

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
        # Check if model is loaded
        if model is None:
            return render_template('submit.html', 
                                 error="Model not loaded. Please check server logs.")
        
        # Retrieve values from the UI using POST request
        input_data = {
            'Gender': request.form.get('gender', 'Male'),
            'Married': request.form.get('married', 'No'),
            'Dependents': request.form.get('dependents', '0'),
            'Education': request.form.get('education', 'Graduate'),
            'Self_Employed': request.form.get('self_employed', 'No'),
            'ApplicantIncome': float(request.form.get('applicant_income', 5000)),
            'CoapplicantIncome': float(request.form.get('coapplicant_income', 0)),
            'LoanAmount': float(request.form.get('loan_amount', 10000)),
            'Loan_Amount_Term': float(request.form.get('loan_term', 360)),
            'Credit_History': float(request.form.get('credit_history', 1.0)),
            'Property_Area': request.form.get('property_area', 'Urban')
        }
        
        # Validate input
        if input_data['ApplicantIncome'] <= 0:
            return render_template('submit.html', 
                                 error="Applicant income must be greater than 0")
        
        if input_data['LoanAmount'] <= 0:
            return render_template('submit.html', 
                                 error="Loan amount must be greater than 0")
        
        # Make prediction
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
        print(f"❌ Submit error: {e}")
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
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Models loaded: {model is not None}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
