from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Inisialisasi FastAPI
app = FastAPI(title="Loan Default Prediction API")

# Mengizinkan Vue.js mengakses API ini (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan scaler yang tadi di-download dari Google Colab
model = joblib.load("model_loan_default.pkl")
scaler = joblib.load("scaler.pkl")

# Menentukan struktur data input (harus sama persis dengan selected_features di Colab)
class LoanInput(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

@app.get("/")
def home():
    return {"message": "API Machine Learning Berjalan dengan Lancar!"}

@app.post("/predict")
def predict_default(data: LoanInput):
    # Menyusun data dari form Vue.js ke dalam bentuk array urut sesuai training
    input_data = np.array([[
        data.Age, 
        data.Income, 
        data.LoanAmount, 
        data.CreditScore, 
        data.MonthsEmployed, 
        data.InterestRate, 
        data.LoanTerm, 
        data.DTIRatio
    ]])
    
    # Lakukan scaling pada data input
    input_scaled = scaler.transform(input_data)
    
    # Prediksi menggunakan model Random Forest
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] # Probabilitas gagal bayar
    
    # Menentukan status hasil
    result_status = "Gagal Bayar (High Risk)" if prediction == 1 else "Lancar (Low Risk)"
    
    return {
        "prediction_code": int(prediction),
        "result_status": result_status,
        "default_probability": round(float(probability) * 100, 2)
    }