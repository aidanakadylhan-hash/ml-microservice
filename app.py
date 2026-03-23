from fastapi import FastAPI

app = FastAPI(title="Simple ML Microservice")

@app.get("/")
def home():
    return {
        "message": "Microservice is running"
    }

@app.get("/predict")
def predict():
    return {
        "prediction": "success"
    }
