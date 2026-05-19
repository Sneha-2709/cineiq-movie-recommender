# CineIQ — Project Report

## 1. Problem Statement
Content discovery on streaming platforms is opaque and biased toward 
promoted titles, trapping users in recommendation loops. CineIQ solves 
this with an open, explainable hybrid recommendation engine.

## 2. Methodology
### 2.1 Collaborative Filtering (SVD)
Matrix factorization on user-movie ratings to find latent preferences.

### 2.2 Content-Based Filtering (TF-IDF)
Cosine similarity on movie metadata — genres, cast, director, overview.

### 2.3 Sentiment Re-Ranking
VADER and DistilBERT analyze audience reviews to re-rank recommendations.

### 2.4 Hybrid Ensemble
Weighted combination: 50% collaborative + 30% content + 20% sentiment.

### 2.5 Explainability
LIME generates human-readable reasons for every recommendation.

## 3. Dataset
- MovieLens — user ratings data
- TMDB Metadata — cast, genres, keywords
- IMDB 50K Reviews — sentiment training

## 4. Results
| Metric | Value |
|---|---|
| SVD RMSE | 0.87 |
| SVD MAE | 0.67 |
| Catalog Coverage | 73% |

## 5. Tech Stack
Python, scikit-learn, Surprise (SVD), HuggingFace, VADER,
FastAPI, Streamlit, Plotly, MLflow, LIME

## 6. Conclusion
CineIQ successfully combines three recommendation strategies with 
explainability and sentiment awareness, achieving interpretable, 
personalized movie suggestions.