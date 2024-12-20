import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#Load the saved model
model = joblib.load("model/model.pkl")

# Instantiate FastAPI app
app = FastAPI()


class PredictionInput(BaseModel):
    ambient_temperature: float
    exhaust_vacuum: float
    ambient_pressure: float
    relative_humidity: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # prepare input data as a numpy array
    data = np.array([[
        input_data.ambient_temperature,
        input_data.exhaust_vacuum,
        input_data.exhaust_pressure,
        imput_data.relative_humidity
    ]])

    #make prediction
    prediction = model.predict(data)

    # return these prediction
    return {
        "prediction": preciction[0]
    }

# run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    Ambient
    Temperature(AT)
    Exhaust
    Vacuum(V)
    Ambient
    Pressure(AP)
    Relative
    Humidity(RH)