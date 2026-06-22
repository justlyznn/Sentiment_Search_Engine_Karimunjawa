from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks review yang akan diprediksi sentimennya")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Pantai yang sangat indah dan bersih, pasirnya putih!"}
            ]
        }
    }


# ── Response Models ───────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    text: str
    sentimen: str = Field(..., description="Hasil prediksi: positive / neutral / negative")
    confidence: float = Field(..., description="Skor kepercayaan model (0.0 - 1.0)")


class ReviewItem(BaseModel):
    title: str
    stars: int
    text: str
    sentimen: str


class SearchResponse(BaseModel):
    total: int
    results: List[ReviewItem]


class BeachRanking(BaseModel):
    rank: int
    name: str
    rating_mean: float
    review_count: int
    positive_percentage: float
    score: float


class RankingsResponse(BaseModel):
    total_beaches: int
    rankings: List[BeachRanking]


class BeachItem(BaseModel):
    name: str
    review_count: int


class BeachesResponse(BaseModel):
    total: int
    beaches: List[BeachItem]
