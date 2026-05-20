import httpx
import base64
import time
import json
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictionResponse

app = FastAPI(title="Food Classifier API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TF_SERVING_URL = os.getenv("TF_SERVING_URL", "http://localhost:8501")
PREDICT_URL    = f"{TF_SERVING_URL}/v1/models/efficientnet_food:predict"

# Make sure class_names.json is packaged with the API!
with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG/PNG images supported")

    image_bytes = await file.read()

    # Encode image as base64 for TF Serving REST API
    # TF Serving expects: {"instances": [{"b64": "<base64_string>"}]}
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    payload   = {"instances": [{"b64": b64_image}]}

    # Call TF Serving
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(PREDICT_URL, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise HTTPException(502, f"TF Serving error: {response.text}")

    # Parse predictions
    predictions = response.json()['predictions'][0]   # softmax scores, len=101
    predictions = np.array(predictions)

    top5_idx    = predictions.argsort()[-5:][::-1]
    top5        = [{"class": CLASS_NAMES[i], "confidence": round(float(predictions[i]), 4)}
                   for i in top5_idx]

    return PredictionResponse(
        predicted_class  = CLASS_NAMES[top5_idx[0]],
        confidence       = round(float(predictions[top5_idx[0]]), 4),
        top5             = top5,
        inference_time_ms= round(elapsed_ms, 2)
    )
