from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load all saved brains
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('data_scaler.pkl')
encoder = joblib.load('data_encoder.pkl')
model_features = joblib.load('model_features.pkl')

@app.post("/predict")
def predict_fraud(data: dict):
    try:
        # 1. Convert incoming JSON request to a DataFrame
        df = pd.DataFrame([data])
        
        # 2. Run through Target Encoder
        df_encoded = encoder.transform(df)
        
        # 3. Ensure columns align exactly with what the model expects
        df_encoded = df_encoded[model_features]
        
        # 4. Scale features
        scaled_data = scaler.transform(df_encoded)
        
        # 5. Predict probability
        probability = model.predict_proba(scaled_data)[0][1]
        
        return {"fraud_probability": round(float(probability) * 100, 2)}
    except Exception as e:
        return {"error": str(e)}