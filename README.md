# 🎬 CineIQ — Hybrid Movie Recommendation Engine



![Python](https://img.shields.io/badge/Python-3.13-blue)




![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)




![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)




![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)



## 🚀 Overview
CineIQ is a full-stack, explainable movie recommendation system that combines 
multiple ML strategies to deliver personalized, interpretable suggestions.

## ✨ Features
- **Hybrid Engine** — SVD + TF-IDF cosine similarity + sentiment re-ranking
- **Explainability** — LIME-based human-readable recommendation reasons  
- **Sentiment-Aware** — VADER/DistilBERT re-ranks by audience reception
- **REST API** — FastAPI endpoints `/recommend` and `/similar`
- **Dashboard** — Streamlit interface with genre radar & decade charts
- **MLflow Tracking** — Full experiment logging and model versioning

## 🛠️ Tech Stack
| Layer | Tools |
|---|---|
| ML | scikit-learn, SVD (Surprise), Pandas, NumPy |
| NLP | VADER, HuggingFace DistilBERT |
| Explainability | LIME |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| MLOps | MLflow |

## 📁 Project Structure
