import pandas as pd
import numpy as np

def get_similar_movies(movie_title, content_df, cosine_sim, n=10):
    matches = content_df[content_df["title"].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    if idx >= len(cosine_sim):
        return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    indices = [i[0] for i in sim_scores]
    results = content_df.iloc[indices][["title", "genres"]].copy()
    results["score"] = [round(i[1], 4) for i in sim_scores]
    return results
