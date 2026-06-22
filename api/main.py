import os
import math
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline as hf_pipeline

from models import (
    PredictRequest, PredictResponse,
    SearchResponse, ReviewItem,
    RankingsResponse, BeachRanking,
    BeachesResponse, BeachItem,
)

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Search Engine Pantai Karimunjawa — API",
    description=(
        "Backend API untuk Search Engine Pantai Karimunjawa. "
        "Menyediakan prediksi sentimen dengan IndoBERT, pencarian review, "
        "dan ranking pantai berdasarkan rating & sentimen."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data & Model Loading ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "labelling", "label_data.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "w11wo/indonesian-roberta-base-sentiment-classifier")

# Global state
_df: Optional[pd.DataFrame] = None
_classifier = None


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        _df["text_processed"] = (
            _df["text"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
        )
    return _df


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = hf_pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1,          # CPU; ganti ke 0 jika ada GPU
            truncation=True,
            max_length=512,
        )
    return _classifier


@app.on_event("startup")
async def startup_event():
    """Pre-load data dan model saat server start."""
    get_df()
    get_classifier()


# ── Helper ────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # model kadang pakai label berbeda — normalisasi di sini
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}


def normalize_label(raw_label: str) -> str:
    return LABEL_MAP.get(raw_label.lower(), raw_label.lower())


def compute_rankings(df: pd.DataFrame, top_n: int = 3) -> list[BeachRanking]:
    stats = df.groupby("title").agg(
        stars_mean=("stars", "mean"),
        review_count=("stars", "count"),
        positive_count=("sentimen", lambda x: (x == "positive").sum()),
    ).reset_index()

    stats["positive_percentage"] = (stats["positive_count"] / stats["review_count"] * 100).round(2)
    stats["rating_mean"] = stats["stars_mean"].round(2)
    stats["score"] = (
        stats["rating_mean"]
        * (stats["positive_percentage"] / 100)
        * np.log1p(stats["review_count"])
    ).round(4)

    top = stats.sort_values("score", ascending=False).head(top_n)

    return [
        BeachRanking(
            rank=i + 1,
            name=row["title"],
            rating_mean=row["rating_mean"],
            review_count=int(row["review_count"]),
            positive_percentage=row["positive_percentage"],
            score=row["score"],
        )
        for i, (_, row) in enumerate(top.iterrows())
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Search Engine Pantai Karimunjawa API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Sentimen"])
def predict(body: PredictRequest):
    """
    Prediksi sentimen teks bebas menggunakan model IndoBERT
    (`w11wo/indonesian-roberta-base-sentiment-classifier`).
    """
    try:
        clf = get_classifier()
        result = clf(body.text)[0]
        label = normalize_label(result["label"])
        return PredictResponse(
            text=body.text,
            sentimen=label,
            confidence=round(result["score"], 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse, tags=["Data"])
def search(
    keyword: Optional[str] = Query(None, description="Kata kunci pencarian dalam teks review"),
    sentimen: Optional[str] = Query(None, description="Filter sentimen: positive / neutral / negative"),
    min_rating: int = Query(1, ge=1, le=5, description="Rating minimum (1-5)"),
    max_rating: int = Query(5, ge=1, le=5, description="Rating maksimum (1-5)"),
    limit: int = Query(50, ge=1, le=200, description="Jumlah hasil maksimal"),
):
    """
    Cari review pantai dengan kombinasi filter keyword, sentimen, dan rating.
    """
    df = get_df().copy()

    # Filter rating
    df = df[(df["stars"] >= min_rating) & (df["stars"] <= max_rating)]

    # Filter sentimen
    if sentimen:
        valid = {"positive", "neutral", "negative"}
        if sentimen.lower() not in valid:
            raise HTTPException(status_code=400, detail=f"sentimen harus salah satu dari: {valid}")
        df = df[df["sentimen"] == sentimen.lower()]

    # Filter keyword
    if keyword:
        kw = keyword.lower()
        df = df[df["text_processed"].str.contains(kw, na=False)]

    results = [
        ReviewItem(
            title=row["title"],
            stars=int(row["stars"]),
            text=str(row["text"]),
            sentimen=str(row["sentimen"]),
        )
        for _, row in df.head(limit).iterrows()
    ]

    return SearchResponse(total=len(df), results=results)


@app.get("/rankings", response_model=RankingsResponse, tags=["Data"])
def rankings(
    top_n: int = Query(3, ge=1, le=20, description="Jumlah pantai terbaik yang ditampilkan"),
    sentimen: Optional[str] = Query(None, description="Filter data berdasarkan sentimen sebelum ranking"),
    min_rating: int = Query(1, ge=1, le=5),
    max_rating: int = Query(5, ge=1, le=5),
):
    """
    Ambil ranking pantai terbaik berdasarkan algoritma:
    `Score = Rating_Mean × (Positive% / 100) × log(1 + jumlah_review)`
    """
    df = get_df().copy()
    df = df[(df["stars"] >= min_rating) & (df["stars"] <= max_rating)]

    if sentimen:
        df = df[df["sentimen"] == sentimen.lower()]

    if df.empty:
        return RankingsResponse(total_beaches=0, rankings=[])

    ranked = compute_rankings(df, top_n)
    total_beaches = df["title"].nunique()
    return RankingsResponse(total_beaches=total_beaches, rankings=ranked)


@app.get("/beaches", response_model=BeachesResponse, tags=["Data"])
def beaches():
    """Daftar semua pantai beserta jumlah review masing-masing."""
    df = get_df()
    counts = df.groupby("title").size().reset_index(name="review_count")
    counts = counts.sort_values("review_count", ascending=False)

    return BeachesResponse(
        total=len(counts),
        beaches=[
            BeachItem(name=row["title"], review_count=int(row["review_count"]))
            for _, row in counts.iterrows()
        ],
    )
