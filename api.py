import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load(r'C:\Users\Louisalex\Desktop\Projects\Development\linear_regression_model\model\model.pkl')

# Instantiate FastAPI app
app = FastAPI()

# Define the input data model
class PredictionInput(BaseModel):
    ambient_temperature: float
    exhaust_vacuum: float
    ambient_pressure: float
    relative_humidity: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predicts the output using the provided input data.
    """
    # Prepare input data as a numpy array
    data = np.array([[
        input_data.ambient_temperature,
        input_data.exhaust_vacuum,
        input_data.ambient_pressure,
        input_data.relative_humidity
    ]])

    # Make prediction
    prediction = model.predict(data)

    # Return the prediction
    return {
        "prediction": prediction[0]
    }

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)