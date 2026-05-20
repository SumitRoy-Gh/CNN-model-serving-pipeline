from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top5: List[dict]
    inference_time_ms: float
