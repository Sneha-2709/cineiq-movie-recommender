import pandas as pd

def get_recommendation_with_explanation(recommender, user_id, seed_movie, n=10):
    results = recommender.recommend(user_id, seed_movie, n)
    explained = []
    for r in results:
        explained.append({
            "movieId": r["movieId"],
            "title": r["title"],
            "genres": r["genres"],
            "score": r["score"],
            "explanation": f"Recommended based on similarity to '{seed_movie}' and your viewing history."
        })
    return pd.DataFrame(explained)
