from fastapi import FastAPI

from app.schemas import PredictionRequest
from app.inference import predict_next_words


app = FastAPI()



@app.get("/")
def home():

    return {
        "message":"Next Word Prediction API running"
    }



@app.post("/predict")
def predict(request: PredictionRequest):

    result = predict_next_words(
        request.text
    )


    return {
        "input": request.text,
        "prediction": result
    }