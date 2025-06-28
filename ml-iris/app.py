from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)
# Define input schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Welcome to the Iris Prediction API!"}
@app.post("/predict")
def predict_species(data: IrisRequest):
    features = np.array([[data.sepal_length, data.sepal_width,
                          data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"predicted_species": int(prediction[0])}