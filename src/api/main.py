from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("artifacts/model.pkl")

@app.get("/")
def home():
    return "ML API is running"

def predict(data : list):
    prediction = model.predict([data])
    return {"prediction":prediction.tolist()}