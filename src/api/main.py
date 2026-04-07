from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("artifacts/model.pkl")

#Define request schema
class InputData(BaseModel):
    data: list

#@app.get("/")
#def home():
#   return "ML API is running"



@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>ML Prediction</h2>
            <input id="val" type="number"/>
            <button onclick="send()">Predict</button>
            <p id="res"></p>

            <script>
            async function send() {
                let v = document.getElementById("val").value;
                let res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({data: [Number(v)]})
                });
                let data = await res.json();
                document.getElementById("res").innerText = data.prediction;
            }
            </script>
        </body>
    </html>
    """

@app.post("/predict")
def predict(input : InputData):
    data =  np.array(input.data).reshape(1,-1)
    prediction = model.predict(data)
    return {"prediction":prediction.tolist()}