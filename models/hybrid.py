import numpy as np
import pandas as pd
import pickle

class HybridRecommender:
    def __init__(self):
        with open("models/svd_model.pkl", "rb") as f:
            data = pickle.load(f)
        self.svd = data["svd"]
        self.matrix = data["matrix"]
        self.user_enc = data["user_enc"]
        self.movie_enc = data["movie_enc"]
        self.cosine_sim = np.load("models/cosine_sim.npy")
        self.content_df = pd.read_csv("data/processed/content_features.csv")
        self.sentiment_df = pd.read_csv("data/processed/movie_sentiment.csv")
        self.movies = pd.read_csv("data/raw/movies.csv")

    def get_collab_scores(self, user_id, movie_indices):
        try:
            user_idx = self.user_enc.transform([user_id])[0]
            user_vector = self.matrix[user_idx].reshape(1, -1)
            scores = self.svd.transform(user_vector)[0]
            return {i: float(scores[i % len(scores)]) for i in movie_indices}
        except:
            return {i: 0.0 for i in movie_indices}

    def get_content_scores(self, seed_movie, movie_indices):
        matches = self.movies[self.movies["title"].str.contains(seed_movie, case=False, na=False)]
        if matches.empty:
            return {i: 0.0 for i in movie_indices}
        seed_idx = matches.index[0]
        if seed_idx >= len(self.cosine_sim):
            return {i: 0.0 for i in movie_indices}
        sim_scores = self.cosine_sim[seed_idx]
        return {i: float(sim_scores[i]) if i < len(sim_scores) else 0.0 for i in movie_indices}

    def recommend(self, user_id, seed_movie, n=10):
        all_indices = list(range(min(len(self.movies), len(self.cosine_sim))))
        collab = self.get_collab_scores(user_id, all_indices)
        content = self.get_content_scores(seed_movie, all_indices)
        combined = {i: 0.5 * collab.get(i, 0) + 0.5 * content.get(i, 0) for i in all_indices}
        top = sorted(combined, key=combined.get, reverse=True)[:n]
        results = []
        for i in top:
            if i < len(self.movies):
                row = self.movies.iloc[i]
                results.append({
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "score": round(combined[i], 4)
                })
        return results
