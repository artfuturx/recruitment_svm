# -----------------------------------------------------------------------------
# api.py

# Bu dosya FastAPI framework’ü ile yazılmış bir REST API servisidir.
# Eğitim aşamasında kaydedilen modelleri yükleyerek kullanıcıdan gelen aday verisine göre işe alım tahmini yapar.

# Kullanıcıdan alınan 'years_experience' ve 'technical_score' verileri
# önce scaler ile ölçeklenir, ardından istenilen SVM modeli ile tahmin yapılır.

# Tahmin sonucuna ek olarak, kullanıcıya basit bir rule-based yani sistemle yapılan değerlendirme sonucu da gösterilir.

# API terminalden şu komutla çalıştırılır: uvicorn api:app --reload --port 8001

# -----------------------------------------------------------------------------


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

scaler = joblib.load("models/scaler.pkl")
models = {
    "linear": joblib.load("models/svm_linear.pkl"),
    "rbf": joblib.load("models/svm_rbf.pkl"),
    "poly": joblib.load("models/svm_poly.pkl"),
    "sigmoid": joblib.load("models/svm_sigmoid.pkl"),
}

app = FastAPI(title="Recruitment SVM Prediction", description="Candidate Evaluation using SVM for Recruitment")

# Kullanıcıdan Bilgileri Aldığımız Değerler
class Candidate(BaseModel):
    years_experience: float
    technical_score: float

@app.post("/predict/{model_name}")
def predict(candidate: Candidate, model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Geçersiz model adı, linear, rbf, poly, sigmoid modellerinden birini giriniz.")
    
    model = models[model_name]
    data = np.array([[candidate.years_experience, candidate.technical_score]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]

# Model eğitimi sırasında veriler ölçeklendirildiği için, kullanıcıdan alınan veriler de aynı scaler ile dönüştürülmelidir. 
# Bu işlemden sonra model.predict() fonksiyonu kullanılarak tahmin yapılır.


    result = "Başvuru kabul edildi" if pred == 0 else "Başvuru reddedildi"
    rule = "Başvuru kabul edildi" if candidate.years_experience >= 2 or candidate.technical_score >= 60 else "Başvuru reddedildi"
    
    return {
        "model": model_name,
        "prediction": result,
        "rule_based": rule
    }

@app.get("/")
def home():
    return {"message": "API çalışıyor"}