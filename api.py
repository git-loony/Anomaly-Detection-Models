from fastapi import FastAPI, UploadFile, File
import pandas as pd
from predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Anomaly Detection API is running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        with open("temp.csv", "wb") as f:
            f.write(contents)

        df = predict("temp.csv")

        return {
            "rows": len(df),
            "anomalies": int(df["is_anomaly"].sum()),
            "data": df.head(10).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}