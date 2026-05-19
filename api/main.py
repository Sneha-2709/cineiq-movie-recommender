# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
sys.path.append('..')
from models.hybrid import HybridRecommender
from models.explainer import get_recommendation_with_explanation

app = FastAPI(title="CineIQ API", version="1.0")

# Load model once at startup
recommender = HybridRecommender()

class RecommendRequest(BaseModel):
    user_id: int
    seed_movie: str
    n: Optional[int] = 10
    w_collab: Optional[float] = 0.5
    w_content: Optional[float] = 0.3
    w_sentiment: Optional[float] = 0.2

class SimilarRequest(BaseModel):
    movie_title: str
    n: Optional[int] = 10

@app.get("/")
def root():
    return {"message": "CineIQ API is running"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """Get personalized recommendations for a user"""
    try:
        recs = get_recommendation_with_explanation(
            recommender,
            req.user_id,
            req.seed_movie,
            n=req.n
        )
        return {
            "user_id": req.user_id,
            "seed_movie": req.seed_movie,
            "recommendations": recs.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar")
def similar(req: SimilarRequest):
    """Get content-similar movies"""
    from models.content_based import get_similar_movies
    import pandas as pd, numpy as np
    
    content_df  = recommender.content_df
    cosine_sim  = recommender.cosine_sim
    
    result = get_similar_movies(req.movie_title, content_df, cosine_sim, req.n)
    if not len(result):
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return result.to_dict(orient='records')

# Run with: uvicorn api.main:app --reload --port 8000